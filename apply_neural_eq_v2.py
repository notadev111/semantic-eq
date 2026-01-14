"""
Apply Neural EQ V2 to Audio Files
==================================

Uses trained V2 model to generate semantic EQ and apply to real audio.

Usage:
    python apply_neural_eq_v2.py --input mix.wav --term warm
    python apply_neural_eq_v2.py --input mix.wav --term bright --output output_bright.wav
    python apply_neural_eq_v2.py --input mix.wav --interpolate warm bright 0.5
    python apply_neural_eq_v2.py --input mix.wav --term warm --intensity 0.7
"""

import torch
import torchaudio
import numpy as np
import argparse
from pathlib import Path
from typing import Tuple
import warnings
warnings.filterwarnings("ignore")

from core.neural_eq_morphing_safedb_v2 import NeuralEQMorphingSAFEDBV2

try:
    from dasp_pytorch import ParametricEQ
except ImportError:
    print("ERROR: dasp-pytorch not installed")
    print("Install with: pip install git+https://github.com/csteinmetz1/dasp-pytorch.git")
    exit(1)


def convert_safedb_to_dasp(eq_params_13: np.ndarray, sample_rate: int = 44100) -> torch.Tensor:
    """
    Convert SAFE-DB 13-parameter format to dasp-pytorch 18-parameter format

    SAFE-DB (13 params, 5 bands):
      Band 1 (Low Shelf):  [Gain, Freq]          (indices 0, 1)
      Band 2 (Bell):       [Gain, Freq, Q]       (indices 2, 3, 4)
      Band 3 (Bell):       [Gain, Freq, Q]       (indices 5, 6, 7)
      Band 4 (Bell):       [Gain, Freq, Q]       (indices 8, 9, 10)
      Band 5 (High Shelf): [Gain, Freq]          (indices 11, 12)

    dasp-pytorch (18 params, 6 bands):
      Each band: [Gain_norm, Freq_norm, Q_norm] × 6 bands

    Normalization for dasp-pytorch:
      Gain: [0, 1] where 0.5 = 0dB, range ±12dB
      Freq: [0, 1] log scale, 20Hz-20kHz
      Q: [0, 1] log scale, 0.1-10
    """

    dasp_params = torch.zeros(1, 18)

    # Extract SAFE-DB parameters
    gain1, freq1 = eq_params_13[0], eq_params_13[1]
    gain2, freq2, q2 = eq_params_13[2], eq_params_13[3], eq_params_13[4]
    gain3, freq3, q3 = eq_params_13[5], eq_params_13[6], eq_params_13[7]
    gain4, freq4, q4 = eq_params_13[8], eq_params_13[9], eq_params_13[10]
    gain5, freq5 = eq_params_13[11], eq_params_13[12]

    # Default Q for shelf filters
    q1 = 0.707  # Butterworth Q for shelf
    q5 = 0.707

    def normalize_gain(gain_db):
        """Normalize gain: [-12, 12] dB -> [0, 1]"""
        return np.clip((gain_db + 12) / 24, 0, 1)

    def normalize_freq(freq_hz):
        """Normalize frequency: [20, 20000] Hz -> [0, 1] (log scale)"""
        freq_hz = np.clip(freq_hz, 20, 20000)
        freq_log = np.log10(freq_hz / 20) / np.log10(20000 / 20)
        return np.clip(freq_log, 0, 1)

    def normalize_q(q_val):
        """Normalize Q: [0.1, 10] -> [0, 1] (log scale)"""
        q_val = np.clip(q_val, 0.1, 10)
        q_log = (np.log10(q_val) + 1) / 2  # log10(0.1) = -1, log10(10) = 1
        return np.clip(q_log, 0, 1)

    # Band 1: Low Shelf
    dasp_params[0, 0] = normalize_gain(gain1)
    dasp_params[0, 1] = normalize_freq(freq1)
    dasp_params[0, 2] = normalize_q(q1)

    # Band 2: Bell
    dasp_params[0, 3] = normalize_gain(gain2)
    dasp_params[0, 4] = normalize_freq(freq2)
    dasp_params[0, 5] = normalize_q(q2)

    # Band 3: Bell
    dasp_params[0, 6] = normalize_gain(gain3)
    dasp_params[0, 7] = normalize_freq(freq3)
    dasp_params[0, 8] = normalize_q(q3)

    # Band 4: Bell
    dasp_params[0, 9] = normalize_gain(gain4)
    dasp_params[0, 10] = normalize_freq(freq4)
    dasp_params[0, 11] = normalize_q(q4)

    # Band 5: High Shelf
    dasp_params[0, 12] = normalize_gain(gain5)
    dasp_params[0, 13] = normalize_freq(freq5)
    dasp_params[0, 14] = normalize_q(q5)

    # Band 6: Neutral (not in SAFE-DB, add for compatibility)
    dasp_params[0, 15] = 0.5  # 0 dB
    dasp_params[0, 16] = normalize_freq(10000)  # 10 kHz (neutral)
    dasp_params[0, 17] = normalize_q(0.707)

    return dasp_params


def load_audio(audio_path: str, target_sr: int = 44100) -> Tuple[torch.Tensor, int]:
    """Load and preprocess audio file"""

    print(f"\nLoading audio: {audio_path}")

    audio, sr = torchaudio.load(audio_path)
    print(f"  Original: {audio.shape} @ {sr} Hz")

    # Resample if needed
    if sr != target_sr:
        print(f"  Resampling: {sr} -> {target_sr} Hz")
        audio = torchaudio.functional.resample(audio, sr, target_sr)

    # Convert to stereo
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
        print(f"  Converted mono to stereo")
    elif audio.shape[0] > 2:
        audio = audio[:2]
        print(f"  Kept first 2 channels")

    duration = audio.shape[-1] / target_sr
    print(f"  Final: {audio.shape} ({duration:.2f}s)")

    return audio, target_sr


def apply_eq_to_audio(audio: torch.Tensor, eq_params_dasp: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Apply EQ using dasp-pytorch"""

    print("\nApplying EQ...")

    # Initialize EQ
    eq = ParametricEQ(sample_rate=sample_rate)

    # Add batch dimension if needed
    if audio.dim() == 2:
        audio = audio.unsqueeze(0)  # [C, T] -> [1, C, T]

    # Process audio
    processed = eq.process_normalized(audio, eq_params_dasp)

    # Remove batch dimension
    if processed.dim() == 3:
        processed = processed.squeeze(0)  # [1, C, T] -> [C, T]

    print(f"  Processed: {processed.shape}")

    return processed


def print_eq_summary(eq_params_13: np.ndarray):
    """Print human-readable EQ summary"""

    print("\nEQ Parameters:")
    print("  Band 1 (Low Shelf):  Gain={:6.2f}dB, Freq={:7.1f}Hz".format(
        eq_params_13[0], eq_params_13[1]))
    print("  Band 2 (Bell):       Gain={:6.2f}dB, Freq={:7.1f}Hz, Q={:.2f}".format(
        eq_params_13[2], eq_params_13[3], eq_params_13[4]))
    print("  Band 3 (Bell):       Gain={:6.2f}dB, Freq={:7.1f}Hz, Q={:.2f}".format(
        eq_params_13[5], eq_params_13[6], eq_params_13[7]))
    print("  Band 4 (Bell):       Gain={:6.2f}dB, Freq={:7.1f}Hz, Q={:.2f}".format(
        eq_params_13[8], eq_params_13[9], eq_params_13[10]))
    print("  Band 5 (High Shelf): Gain={:6.2f}dB, Freq={:7.1f}Hz".format(
        eq_params_13[11], eq_params_13[12]))


def compute_audio_stats(audio: torch.Tensor, label: str = "Audio"):
    """Compute and print audio statistics"""

    rms = torch.sqrt(torch.mean(audio ** 2))
    peak = torch.max(torch.abs(audio))
    crest_factor = 20 * torch.log10(peak / rms) if rms > 0 else 0

    print(f"\n{label} Stats:")
    print(f"  RMS: {20 * torch.log10(rms):.2f} dBFS")
    print(f"  Peak: {20 * torch.log10(peak):.2f} dBFS ({peak:.4f})")
    print(f"  Crest Factor: {crest_factor:.2f} dB")


def main():
    parser = argparse.ArgumentParser(description="Apply Neural EQ V2 to Audio")

    parser.add_argument('--input', '-i', required=True, help='Input audio file')
    parser.add_argument('--output', '-o', help='Output audio file (default: auto-generated)')
    parser.add_argument('--term', '-t', help='Semantic term (e.g., warm, bright)')
    parser.add_argument('--interpolate', nargs=3, metavar=('TERM1', 'TERM2', 'ALPHA'),
                       help='Interpolate between two terms (e.g., warm bright 0.5)')
    parser.add_argument('--intensity', type=float, default=1.0,
                       help='EQ intensity multiplier (default: 1.0, range: 0-2)')
    parser.add_argument('--model', default='neural_eq_safedb_v2.pt',
                       help='Path to trained model')
    parser.add_argument('--sample-rate', type=int, default=44100,
                       help='Target sample rate (default: 44100)')
    parser.add_argument('--no-limiter', action='store_true',
                       help='Disable output limiter (allow clipping)')

    args = parser.parse_args()

    # Validate arguments
    if not args.term and not args.interpolate:
        print("ERROR: Must specify either --term or --interpolate")
        return

    if args.term and args.interpolate:
        print("ERROR: Cannot use both --term and --interpolate")
        return

    # Check input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return

    print("="*70)
    print("NEURAL EQ V2 - AUDIO PROCESSING")
    print("="*70)

    # Load model
    print(f"\nLoading model: {args.model}")
    system = NeuralEQMorphingSAFEDBV2()
    system.load_model(args.model)
    system.load_dataset(min_examples=5)
    print(f"  Loaded {len(system.term_to_idx)} semantic terms")

    # Load audio
    audio, sr = load_audio(str(input_path), args.sample_rate)
    compute_audio_stats(audio, "Input")

    # Generate EQ parameters
    if args.term:
        print(f"\nGenerating EQ for term: '{args.term}'")
        if args.term not in system.term_to_idx:
            print(f"ERROR: Unknown term '{args.term}'")
            print(f"Available terms: {list(system.term_to_idx.keys())}")
            return

        eq_params_13 = system.generate_eq_from_term(args.term)
        output_suffix = args.term

    else:  # interpolate
        term1, term2, alpha_str = args.interpolate
        alpha = float(alpha_str)

        if term1 not in system.term_to_idx:
            print(f"ERROR: Unknown term '{term1}'")
            print(f"Available terms: {list(system.term_to_idx.keys())}")
            return

        if term2 not in system.term_to_idx:
            print(f"ERROR: Unknown term '{term2}'")
            print(f"Available terms: {list(system.term_to_idx.keys())}")
            return

        print(f"\nInterpolating: {term1} -> {term2} (alpha={alpha})")
        eq_params_13 = system.interpolate_terms(term1, term2, alpha)
        output_suffix = f"{term1}_{term2}_{alpha:.2f}"

    # Apply intensity scaling
    if args.intensity != 1.0:
        print(f"\nApplying intensity scaling: {args.intensity}x")
        # Scale only gains, not frequencies or Q
        eq_params_13[0] *= args.intensity   # Band 1 gain
        eq_params_13[2] *= args.intensity   # Band 2 gain
        eq_params_13[5] *= args.intensity   # Band 3 gain
        eq_params_13[8] *= args.intensity   # Band 4 gain
        eq_params_13[11] *= args.intensity  # Band 5 gain

    print_eq_summary(eq_params_13)

    # Convert to dasp format
    print("\nConverting SAFE-DB -> dasp-pytorch format...")
    eq_params_dasp = convert_safedb_to_dasp(eq_params_13, sr)

    # Apply EQ
    processed_audio = apply_eq_to_audio(audio, eq_params_dasp, sr)
    compute_audio_stats(processed_audio, "Output")

    # Compute RMS change
    rms_before = torch.sqrt(torch.mean(audio ** 2))
    rms_after = torch.sqrt(torch.mean(processed_audio ** 2))
    rms_change_db = 20 * torch.log10(rms_after / rms_before) if rms_before > 0 else 0
    print(f"\nRMS Change: {rms_change_db:+.2f} dB")

    # Apply limiter if needed
    peak_after = torch.max(torch.abs(processed_audio))
    if peak_after > 0.99 and not args.no_limiter:
        scale = 0.99 / peak_after
        processed_audio = processed_audio * scale
        reduction_db = 20 * torch.log10(scale)
        print(f"\nApplied limiter: {reduction_db:.2f} dB reduction")
        print(f"  (Use --no-limiter to disable)")

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = input_path.parent / "processed"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{input_path.stem}_{output_suffix}.wav"

    # Save
    print(f"\nSaving: {output_path}")
    torchaudio.save(str(output_path), processed_audio, sr)

    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"\nInput:  {input_path}")
    print(f"Output: {output_path}")
    print(f"\nTo listen:")
    print(f"  Original: {input_path}")
    print(f"  Processed: {output_path}")


if __name__ == "__main__":
    main()
