"""
Demo Application: Adaptive Semantic EQ
======================================

Interactive demo that showcases the adaptive EQ system.

Features:
1. Analyze input audio and show semantic profile
2. Generate adaptive EQ for semantic targets
3. Apply with adjustable intensity
4. Compare before/after audio
5. Export processed audio

Usage:
    # Analyze audio
    python demo_adaptive_eq.py --input mix.wav --analyze

    # Apply adaptive EQ
    python demo_adaptive_eq.py --input mix.wav --target warm --intensity 0.7

    # Process with auto-intensity
    python demo_adaptive_eq.py --input mix.wav --target bright --auto-intensity
"""

import torch
import torchaudio
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple
import warnings
warnings.filterwarnings("ignore")

from core.adaptive_eq_generator import AdaptiveEQGenerator
from core.training_data_synthesis import BiquadEQFilter


class AdaptiveEQDemo:
    """
    Demo application for Adaptive Semantic EQ
    """

    def __init__(self,
                 v2_model_path: str = 'neural_eq_safedb_v2.pt',
                 audio_encoder_path: str = 'audio_encoder_best.pt'):
        """
        Args:
            v2_model_path: Path to V2 model
            audio_encoder_path: Path to Audio Encoder
        """
        print("="*70)
        print("ADAPTIVE SEMANTIC EQ - DEMO APPLICATION")
        print("="*70)

        self.generator = AdaptiveEQGenerator(
            v2_model_path=v2_model_path,
            audio_encoder_path=audio_encoder_path
        )

        self.eq_filter = BiquadEQFilter()

        print("\nDemo ready!")
        print(f"Available semantic terms: {list(self.generator.semantic_embeddings.keys())}")

    def load_audio(self, audio_path: str, target_sr: int = 44100) -> Tuple[torch.Tensor, int]:
        """Load and preprocess audio"""

        print(f"\nLoading audio: {audio_path}")

        audio, sr = torchaudio.load(audio_path)
        print(f"  Original: {audio.shape} @ {sr} Hz")

        # Resample
        if sr != target_sr:
            print(f"  Resampling: {sr} -> {target_sr} Hz")
            audio = torchaudio.functional.resample(audio, sr, target_sr)

        # Convert to stereo
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:
            audio = audio[:2]

        duration = audio.shape[-1] / target_sr
        print(f"  Final: {audio.shape} ({duration:.2f}s)")

        return audio, target_sr

    def analyze_audio(self, audio: torch.Tensor):
        """
        Analyze audio and display semantic profile
        """
        print("\n" + "="*70)
        print("SEMANTIC PROFILE ANALYSIS")
        print("="*70)

        # Get semantic profile
        profile = self.generator.get_semantic_profile(audio, top_k=10)

        print(f"\nTop 10 semantic characteristics of input audio:")
        print("-" * 50)
        for i, (term, similarity) in enumerate(profile, 1):
            bar = "█" * int(similarity * 40)
            print(f"  {i:2d}. {term:15s} [{similarity:5.3f}] {bar}")

        return profile

    def generate_and_apply_eq(self,
                             audio: torch.Tensor,
                             semantic_target: str,
                             intensity: float,
                             sample_rate: int = 44100,
                             auto_intensity: bool = False) -> Tuple[torch.Tensor, np.ndarray, float]:
        """
        Generate adaptive EQ and apply to audio

        Args:
            audio: Input audio tensor
            semantic_target: Target semantic term
            intensity: EQ intensity (ignored if auto_intensity=True)
            sample_rate: Sample rate
            auto_intensity: If True, auto-suggest intensity

        Returns:
            processed_audio: Processed audio tensor
            eq_params: EQ parameters used
            actual_intensity: Actual intensity used
        """
        print("\n" + "="*70)
        print("GENERATING ADAPTIVE EQ")
        print("="*70)

        # Auto-suggest intensity if requested
        if auto_intensity:
            intensity = self.generator.suggest_intensity(audio, semantic_target)
            print(f"\nAuto-suggested intensity: {intensity:.2f}")

        print(f"\nTarget: '{semantic_target}' with intensity {intensity:.2f}")

        # Generate adaptive EQ
        eq_params, similarity = self.generator.generate_adaptive_eq(
            audio,
            semantic_target=semantic_target,
            intensity=intensity,
            return_similarity=True
        )

        print(f"Current similarity to '{semantic_target}': {similarity:.3f}")

        # Print EQ parameters
        print(f"\nGenerated EQ Parameters:")
        print("-" * 50)
        print(f"  Band 1 (Low Shelf):  Gain={eq_params[0]:+6.2f}dB, Freq={eq_params[1]:7.1f}Hz")
        print(f"  Band 2 (Bell):       Gain={eq_params[2]:+6.2f}dB, Freq={eq_params[3]:7.1f}Hz, Q={eq_params[4]:.2f}")
        print(f"  Band 3 (Bell):       Gain={eq_params[5]:+6.2f}dB, Freq={eq_params[6]:7.1f}Hz, Q={eq_params[7]:.2f}")
        print(f"  Band 4 (Bell):       Gain={eq_params[8]:+6.2f}dB, Freq={eq_params[9]:7.1f}Hz, Q={eq_params[10]:.2f}")
        print(f"  Band 5 (High Shelf): Gain={eq_params[11]:+6.2f}dB, Freq={eq_params[12]:7.1f}Hz")

        # Apply EQ to audio
        print("\nApplying EQ to audio...")
        audio_np = audio.numpy()
        processed_np = self.eq_filter.apply_eq(audio_np[0], eq_params, sample_rate)  # Mono processing

        # If stereo, apply to both channels
        if audio_np.shape[0] == 2:
            processed_np_r = self.eq_filter.apply_eq(audio_np[1], eq_params, sample_rate)
            processed_np = np.stack([processed_np, processed_np_r], axis=0)
        else:
            processed_np = processed_np[np.newaxis, :]

        processed_audio = torch.from_numpy(processed_np).float()

        # Apply limiter if needed
        peak = torch.max(torch.abs(processed_audio))
        if peak > 0.99:
            scale = 0.99 / peak
            processed_audio = processed_audio * scale
            print(f"Applied limiter: {20*torch.log10(scale):.2f}dB reduction")

        return processed_audio, eq_params, intensity

    def visualize_eq_curve(self, eq_params: np.ndarray, sample_rate: int = 44100, save_path: str = None):
        """
        Visualize EQ frequency response

        Args:
            eq_params: [13] EQ parameters
            sample_rate: Sample rate
            save_path: Path to save figure (optional)
        """
        print("\nGenerating EQ curve visualization...")

        # Compute frequency response
        frequencies = np.logspace(np.log10(20), np.log10(20000), 1000)
        response_db = np.zeros_like(frequencies)

        # Extract parameters
        gain1, freq1 = eq_params[0], eq_params[1]
        gain2, freq2, q2 = eq_params[2], eq_params[3], eq_params[4]
        gain3, freq3, q3 = eq_params[5], eq_params[6], eq_params[7]
        gain4, freq4, q4 = eq_params[8], eq_params[9], eq_params[10]
        gain5, freq5 = eq_params[11], eq_params[12]

        q1, q5 = 0.707, 0.707

        # Compute response for each band (simplified)
        for i, f in enumerate(frequencies):
            # Band 1: Low shelf (simplified approximation)
            if f < freq1:
                response_db[i] += gain1 * (1 - f / freq1)

            # Band 2-4: Bells (simplified)
            for gain, fc, q in [(gain2, freq2, q2), (gain3, freq3, q3), (gain4, freq4, q4)]:
                bw = fc / q
                if abs(f - fc) < 2 * bw:
                    response_db[i] += gain * np.exp(-0.5 * ((f - fc) / (bw/2))**2)

            # Band 5: High shelf
            if f > freq5:
                response_db[i] += gain5 * (f / freq5 - 1) / (f / freq5)

        # Plot
        plt.figure(figsize=(12, 6))
        plt.semilogx(frequencies, response_db, linewidth=2, color='#2E86AB')
        plt.grid(True, alpha=0.3)
        plt.xlabel('Frequency (Hz)', fontsize=12)
        plt.ylabel('Gain (dB)', fontsize=12)
        plt.title('Adaptive EQ Frequency Response', fontsize=14, fontweight='bold')
        plt.xlim(20, 20000)
        plt.ylim(-15, 15)

        # Mark band center frequencies
        band_freqs = [freq1, freq2, freq3, freq4, freq5]
        band_gains = [gain1, gain2, gain3, gain4, gain5]
        plt.scatter(band_freqs, band_gains, c='red', s=100, zorder=5, label='Band Centers')
        plt.legend()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved to: {save_path}")

        plt.close()

    def compare_audio_stats(self, original: torch.Tensor, processed: torch.Tensor):
        """Compare audio statistics before/after"""

        print("\n" + "="*70)
        print("AUDIO STATISTICS COMPARISON")
        print("="*70)

        def compute_stats(audio):
            rms = torch.sqrt(torch.mean(audio ** 2))
            peak = torch.max(torch.abs(audio))
            rms_db = 20 * torch.log10(rms) if rms > 0 else -np.inf
            peak_db = 20 * torch.log10(peak) if peak > 0 else -np.inf
            crest_db = peak_db - rms_db
            return rms_db.item(), peak_db.item(), crest_db

        orig_rms, orig_peak, orig_crest = compute_stats(original)
        proc_rms, proc_peak, proc_crest = compute_stats(processed)

        print(f"\n{'Metric':<20s} {'Original':>12s} {'Processed':>12s} {'Change':>12s}")
        print("-" * 60)
        print(f"{'RMS Level':<20s} {orig_rms:>11.2f}dB {proc_rms:>11.2f}dB {proc_rms-orig_rms:>+11.2f}dB")
        print(f"{'Peak Level':<20s} {orig_peak:>11.2f}dB {proc_peak:>11.2f}dB {proc_peak-orig_peak:>+11.2f}dB")
        print(f"{'Crest Factor':<20s} {orig_crest:>11.2f}dB {proc_crest:>11.2f}dB {proc_crest-orig_crest:>+11.2f}dB")


def main():
    parser = argparse.ArgumentParser(description="Adaptive Semantic EQ Demo")

    parser.add_argument('--input', '-i', required=True, help='Input audio file')
    parser.add_argument('--output', '-o', help='Output audio file (auto-generated if not specified)')
    parser.add_argument('--analyze', action='store_true', help='Only analyze audio, don\'t process')
    parser.add_argument('--target', '-t', help='Semantic target (e.g., warm, bright)')
    parser.add_argument('--intensity', type=float, default=0.7, help='EQ intensity [0-1] (default: 0.7)')
    parser.add_argument('--auto-intensity', action='store_true', help='Auto-suggest intensity')
    parser.add_argument('--visualize', action='store_true', help='Generate EQ curve visualization')
    parser.add_argument('--v2-model', default='neural_eq_safedb_v2.pt', help='V2 model path')
    parser.add_argument('--audio-encoder', default='audio_encoder_best.pt', help='Audio Encoder path')

    args = parser.parse_args()

    # Check input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return

    # Check models
    if not Path(args.v2_model).exists():
        print(f"ERROR: V2 model not found: {args.v2_model}")
        print("Please train V2 model first: python train_neural_eq_v2.py")
        return

    if not Path(args.audio_encoder).exists():
        print(f"WARNING: Audio Encoder not found: {args.audio_encoder}")
        print("Using random weights (results will be poor!)")
        print("Train Audio Encoder first: python train_audio_encoder.py")

    # Create demo
    demo = AdaptiveEQDemo(
        v2_model_path=args.v2_model,
        audio_encoder_path=args.audio_encoder
    )

    # Load audio
    audio, sr = demo.load_audio(str(input_path))

    # Analyze
    profile = demo.analyze_audio(audio)

    # If only analyzing, stop here
    if args.analyze:
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        return

    # Check target is specified
    if not args.target:
        print("\nERROR: Must specify --target for processing")
        print(f"Available targets: {list(demo.generator.semantic_embeddings.keys())}")
        return

    # Generate and apply EQ
    processed_audio, eq_params, actual_intensity = demo.generate_and_apply_eq(
        audio,
        semantic_target=args.target,
        intensity=args.intensity,
        sample_rate=sr,
        auto_intensity=args.auto_intensity
    )

    # Compare stats
    demo.compare_audio_stats(audio, processed_audio)

    # Visualize EQ curve
    if args.visualize:
        output_dir = input_path.parent / "processed"
        output_dir.mkdir(exist_ok=True)
        viz_path = output_dir / f"{input_path.stem}_{args.target}_eq_curve.png"
        demo.visualize_eq_curve(eq_params, sr, str(viz_path))

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = input_path.parent / "processed"
        output_dir.mkdir(exist_ok=True)
        intensity_str = f"{actual_intensity:.2f}".replace('.', '_')
        output_path = output_dir / f"{input_path.stem}_{args.target}_{intensity_str}.wav"

    # Save
    print(f"\nSaving processed audio: {output_path}")
    torchaudio.save(str(output_path), processed_audio, sr)

    print("\n" + "="*70)
    print("PROCESSING COMPLETE!")
    print("="*70)
    print(f"\nInput:  {input_path}")
    print(f"Output: {output_path}")
    print(f"\nTarget: '{args.target}' with intensity {actual_intensity:.2f}")


if __name__ == "__main__":
    main()
