"""
Apply Neural EQ V2 to Audio Files (Simple Version - No dasp-pytorch)
====================================================================

Uses trained V2 model + scipy for EQ filtering (no dasp-pytorch dependency!)

Usage:
    python apply_neural_eq_v2_simple.py --input mix.wav --term warm
    python apply_neural_eq_v2_simple.py --input mix.wav --interpolate warm bright 0.5
"""

import torch
import torchaudio
import numpy as np
import argparse
from pathlib import Path
from scipy import signal
from typing import Tuple
import warnings
warnings.filterwarnings("ignore")

from core.neural_eq_morphing_safedb_v2 import NeuralEQMorphingSAFEDBV2


def apply_biquad_filter(audio: np.ndarray, b: np.ndarray, a: np.ndarray, sr: int) -> np.ndarray:
    """Apply biquad filter to audio"""
    # Apply to each channel
    filtered = np.zeros_like(audio)
    for ch in range(audio.shape[0]):
        filtered[ch] = signal.lfilter(b, a, audio[ch])
    return filtered


def low_shelf(gain_db: float, freq_hz: float, q: float, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """Design low shelf filter"""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq_hz / sr
    alpha = np.sin(w0) / (2 * q)

    b0 = A * ((A+1) - (A-1)*np.cos(w0) + 2*np.sqrt(A)*alpha)
    b1 = 2 * A * ((A-1) - (A+1)*np.cos(w0))
    b2 = A * ((A+1) - (A-1)*np.cos(w0) - 2*np.sqrt(A)*alpha)
    a0 = (A+1) + (A-1)*np.cos(w0) + 2*np.sqrt(A)*alpha
    a1 = -2 * ((A-1) + (A+1)*np.cos(w0))
    a2 = (A+1) + (A-1)*np.cos(w0) - 2*np.sqrt(A)*alpha

    return np.array([b0/a0, b1/a0, b2/a0]), np.array([1.0, a1/a0, a2/a0])


def high_shelf(gain_db: float, freq_hz: float, q: float, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """Design high shelf filter"""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq_hz / sr
    alpha = np.sin(w0) / (2 * q)

    b0 = A * ((A+1) + (A-1)*np.cos(w0) + 2*np.sqrt(A)*alpha)
    b1 = -2 * A * ((A-1) + (A+1)*np.cos(w0))
    b2 = A * ((A+1) + (A-1)*np.cos(w0) - 2*np.sqrt(A)*alpha)
    a0 = (A+1) - (A-1)*np.cos(w0) + 2*np.sqrt(A)*alpha
    a1 = 2 * ((A-1) - (A+1)*np.cos(w0))
    a2 = (A+1) - (A-1)*np.cos(w0) - 2*np.sqrt(A)*alpha

    return np.array([b0/a0, b1/a0, b2/a0]), np.array([1.0, a1/a0, a2/a0])


def peaking_eq(gain_db: float, freq_hz: float, q: float, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """Design peaking (bell) filter"""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq_hz / sr
    alpha = np.sin(w0) / (2 * q)

    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A

    return np.array([b0/a0, b1/a0, b2/a0]), np.array([1.0, a1/a0, a2/a0])


def apply_safedb_eq(audio: np.ndarray, eq_params_13: np.ndarray, sr: int) -> np.ndarray:
    """
    Apply 5-band EQ directly from SAFE-DB parameters

    SAFE-DB format (13 params):
      Band 1 (Low Shelf):  [Gain, Freq]
      Band 2 (Bell):       [Gain, Freq, Q]
      Band 3 (Bell):       [Gain, Freq, Q]
      Band 4 (Bell):       [Gain, Freq, Q]
      Band 5 (High Shelf): [Gain, Freq]
    """

    # Extract parameters
    gain1, freq1 = eq_params_13[0], eq_params_13[1]
    gain2, freq2, q2 = eq_params_13[2], eq_params_13[3], eq_params_13[4]
    gain3, freq3, q3 = eq_params_13[5], eq_params_13[6], eq_params_13[7]
    gain4, freq4, q4 = eq_params_13[8], eq_params_13[9], eq_params_13[10]
    gain5, freq5 = eq_params_13[11], eq_params_13[12]

    # Default Q for shelf filters
    q1 = 0.707
    q5 = 0.707

    # Clip frequencies to valid range
    freq1 = np.clip(freq1, 20, sr/2 - 1000)
    freq2 = np.clip(freq2, 20, sr/2 - 1000)
    freq3 = np.clip(freq3, 20, sr/2 - 1000)
    freq4 = np.clip(freq4, 20, sr/2 - 1000)
    freq5 = np.clip(freq5, 20, sr/2 - 1000)

    processed = audio.copy()

    # Band 1: Low Shelf
    if abs(gain1) > 0.01:  # Only apply if gain is significant
        b, a = low_shelf(gain1, freq1, q1, sr)
        processed = apply_biquad_filter(processed, b, a, sr)

    # Band 2: Bell
    if abs(gain2) > 0.01:
        b, a = peaking_eq(gain2, freq2, q2, sr)
        processed = apply_biquad_filter(processed, b, a, sr)

    # Band 3: Bell
    if abs(gain3) > 0.01:
        b, a = peaking_eq(gain3, freq3, q3, sr)
        processed = apply_biquad_filter(processed, b, a, sr)

    # Band 4: Bell
    if abs(gain4) > 0.01:
        b, a = peaking_eq(gain4, freq4, q4, sr)
        processed = apply_biquad_filter(processed, b, a, sr)

    # Band 5: High Shelf
    if abs(gain5) > 0.01:
        b, a = high_shelf(gain5, freq5, q5, sr)
        processed = apply_biquad_filter(processed, b, a, sr)

    return processed


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
    parser = argparse.ArgumentParser(description="Apply Neural EQ V2 to Audio (Simple Version)")

    parser.add_argument('--input', '-i', required=True, help='Input audio file')
    parser.add_argument('--output', '-o', help='Output audio file (default: auto-generated)')
    parser.add_argument('--term', '-t', help='Semantic term (e.g., warm, bright)')
    parser.add_argument('--interpolate', nargs=3, metavar=('TERM1', 'TERM2', 'ALPHA'),
                       help='Interpolate between two terms')
    parser.add_argument('--intensity', type=float, default=1.0,
                       help='EQ intensity multiplier (default: 1.0)')
    parser.add_argument('--model', default='neural_eq_safedb_v2.pt',
                       help='Path to trained model')
    parser.add_argument('--sample-rate', type=int, default=44100,
                       help='Target sample rate (default: 44100)')
    parser.add_argument('--no-limiter', action='store_true',
                       help='Disable output limiter')

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
    print("NEURAL EQ V2 - AUDIO PROCESSING (SIMPLE)")
    print("="*70)
    print("\nUsing scipy for EQ (no dasp-pytorch required)")

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

        if term1 not in system.term_to_idx or term2 not in system.term_to_idx:
            print(f"ERROR: Unknown term")
            print(f"Available terms: {list(system.term_to_idx.keys())}")
            return

        print(f"\nInterpolating: {term1} -> {term2} (alpha={alpha})")
        eq_params_13 = system.interpolate_terms(term1, term2, alpha)
        output_suffix = f"{term1}_{term2}_{alpha:.2f}"

    # Apply intensity scaling
    if args.intensity != 1.0:
        print(f"\nApplying intensity scaling: {args.intensity}x")
        eq_params_13[0] *= args.intensity   # Band 1 gain
        eq_params_13[2] *= args.intensity   # Band 2 gain
        eq_params_13[5] *= args.intensity   # Band 3 gain
        eq_params_13[8] *= args.intensity   # Band 4 gain
        eq_params_13[11] *= args.intensity  # Band 5 gain

    print_eq_summary(eq_params_13)

    # Apply EQ directly (no conversion needed!)
    print("\nApplying 5-band EQ...")
    audio_np = audio.numpy()
    processed_np = apply_safedb_eq(audio_np, eq_params_13, sr)
    processed_audio = torch.from_numpy(processed_np).float()

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


if __name__ == "__main__":
    main()
