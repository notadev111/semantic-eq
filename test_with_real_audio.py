"""
Test Adaptive EQ System with Real Audio
========================================

Comprehensive testing script that:
1. Analyzes audio semantic profile (genre/style characteristics)
2. Generates adaptive EQ curves
3. Creates publication-quality visualizations
4. Compares before/after audio

Usage:
    # Single file analysis
    python test_with_real_audio.py --input song.wav

    # Multiple files (batch)
    python test_with_real_audio.py --input-dir ./test_audio/

    # Apply specific semantic target
    python test_with_real_audio.py --input song.wav --target warm --intensity 0.7
"""

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import List, Tuple
import warnings
warnings.filterwarnings("ignore")

from core.adaptive_eq_generator import AdaptiveEQGenerator
from core.training_data_synthesis import BiquadEQFilter


class AudioAnalyzer:
    """Comprehensive audio analysis and visualization"""

    def __init__(self,
                 v2_model_path: str = 'neural_eq_safedb_v2.pt',
                 audio_encoder_path: str = 'audio_encoder_best.pt'):

        print("="*70)
        print("ADAPTIVE EQ SYSTEM - REAL AUDIO TESTING")
        print("="*70)

        self.generator = AdaptiveEQGenerator(
            v2_model_path=v2_model_path,
            audio_encoder_path=audio_encoder_path
        )

        self.eq_filter = BiquadEQFilter()

        # Color scheme for plots
        self.colors = {
            'warm': '#D32F2F',
            'bright': '#FBC02D',
            'clear': '#1976D2',
            'muddy': '#795548',
            'full': '#7B1FA2',
            'thin': '#0097A7',
            'smooth': '#388E3C',
            'harsh': '#E64A19',
            'punchy': '#C2185B',
            'soft': '#00796B',
            'deep': '#303F9F',
            'airy': '#0288D1',
            'nasal': '#F57C00',
            'boomy': '#5D4037'
        }

        print("\nSystem ready!")
        print(f"Available semantic terms: {len(self.generator.semantic_embeddings)}")

    def load_audio(self, audio_path: str, max_duration: float = 30.0) -> Tuple[torch.Tensor, int, dict]:
        """Load audio and extract metadata"""

        print(f"\nLoading: {Path(audio_path).name}")

        # Handle different file formats
        file_ext = Path(audio_path).suffix.lower()

        if file_ext == '.mp3':
            # Use pydub for MP3 files
            try:
                from pydub import AudioSegment
                import numpy as np

                print(f"  Loading MP3 with pydub...")
                audio_segment = AudioSegment.from_mp3(audio_path)

                # Convert to numpy array
                samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                samples = samples / (2**15)  # Normalize to [-1, 1]

                sr = audio_segment.frame_rate
                n_channels = audio_segment.channels

                # Reshape based on channels
                if n_channels == 2:
                    samples = samples.reshape((-1, 2)).T  # [2, samples]
                else:
                    samples = samples.reshape((1, -1))  # [1, samples]

                audio = torch.from_numpy(samples)

            except Exception as e:
                print(f"  Error with pydub: {e}")
                print(f"  Please convert to WAV: ffmpeg -i {audio_path} {Path(audio_path).stem}.wav")
                raise
        else:
            # Use soundfile for WAV, FLAC, etc.
            try:
                import soundfile as sf
                audio_data, sr = sf.read(audio_path, dtype='float32')
                # Convert to torch tensor [channels, samples]
                if audio_data.ndim == 1:
                    audio = torch.from_numpy(audio_data).unsqueeze(0)  # [1, samples]
                else:
                    audio = torch.from_numpy(audio_data.T)  # [channels, samples]
            except Exception as e:
                print(f"  Error loading with soundfile: {e}")
                raise

        # Get metadata
        metadata = {
            'filename': Path(audio_path).name,
            'channels': audio.shape[0],
            'sample_rate': sr,
            'duration': audio.shape[-1] / sr,
            'samples': audio.shape[-1]
        }

        # Limit duration for analysis
        if metadata['duration'] > max_duration:
            n_samples = int(max_duration * sr)
            # Take middle section
            start = (audio.shape[-1] - n_samples) // 2
            audio = audio[:, start:start + n_samples]
            metadata['analyzed_duration'] = max_duration
        else:
            metadata['analyzed_duration'] = metadata['duration']

        # Convert to mono for analysis
        if audio.shape[0] > 1:
            audio_mono = audio.mean(dim=0, keepdim=True)
        else:
            audio_mono = audio

        print(f"  Duration: {metadata['duration']:.2f}s (analyzing {metadata['analyzed_duration']:.2f}s)")
        print(f"  Sample rate: {sr} Hz")
        print(f"  Channels: {metadata['channels']}")

        return audio_mono, sr, metadata

    def analyze_semantic_profile(self, audio: torch.Tensor, top_k: int = 14) -> List[Tuple[str, float]]:
        """Analyze audio semantic profile"""

        print("\n" + "="*70)
        print("SEMANTIC PROFILE ANALYSIS")
        print("="*70)

        profile = self.generator.get_semantic_profile(audio, top_k=top_k)

        print(f"\nTop {min(top_k, len(profile))} semantic characteristics:")
        print("-" * 70)

        for i, (term, similarity) in enumerate(profile, 1):
            bar_length = int(similarity * 50)
            bar = "█" * bar_length
            print(f"  {i:2d}. {term:15s} [{similarity:5.3f}] {bar}")

        return profile

    def generate_eq_for_target(self, audio: torch.Tensor,
                               semantic_target: str,
                               intensity: float = 0.7) -> Tuple[np.ndarray, float]:
        """Generate adaptive EQ for semantic target"""

        print("\n" + "="*70)
        print(f"GENERATING ADAPTIVE EQ: '{semantic_target}' @ {intensity:.2f}")
        print("="*70)

        eq_params, similarity = self.generator.generate_adaptive_eq(
            audio,
            semantic_target=semantic_target,
            intensity=intensity,
            return_similarity=True
        )

        print(f"\nCurrent similarity to '{semantic_target}': {similarity:.3f}")
        print(f"\nGenerated EQ Parameters:")
        print("-" * 70)
        print(f"  Band 1 (Low Shelf):  Gain={eq_params[0]:+6.2f}dB, Freq={eq_params[1]:7.1f}Hz")
        print(f"  Band 2 (Bell):       Gain={eq_params[2]:+6.2f}dB, Freq={eq_params[3]:7.1f}Hz, Q={eq_params[4]:.2f}")
        print(f"  Band 3 (Bell):       Gain={eq_params[5]:+6.2f}dB, Freq={eq_params[6]:7.1f}Hz, Q={eq_params[7]:.2f}")
        print(f"  Band 4 (Bell):       Gain={eq_params[8]:+6.2f}dB, Freq={eq_params[9]:7.1f}Hz, Q={eq_params[10]:.2f}")
        print(f"  Band 5 (High Shelf): Gain={eq_params[11]:+6.2f}dB, Freq={eq_params[12]:7.1f}Hz")

        return eq_params, similarity

    def plot_semantic_profile(self, profile: List[Tuple[str, float]],
                             output_path: str):
        """Create semantic profile visualization"""

        terms, similarities = zip(*profile)

        fig, ax = plt.subplots(figsize=(12, 8))

        # Get colors for terms
        colors = [self.colors.get(term, '#757575') for term in terms]

        # Horizontal bar chart
        y_pos = np.arange(len(terms))
        bars = ax.barh(y_pos, similarities, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

        # Customize
        ax.set_yticks(y_pos)
        ax.set_yticklabels(terms, fontsize=12, fontweight='bold')
        ax.set_xlabel('Similarity Score', fontsize=14, fontweight='bold')
        ax.set_title('Audio Semantic Profile', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlim(0, 1.0)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.invert_yaxis()

        # Add value labels
        for i, (bar, sim) in enumerate(zip(bars, similarities)):
            ax.text(sim + 0.02, i, f'{sim:.3f}',
                   va='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n✓ Saved semantic profile: {output_path}")

    def plot_eq_response(self, eq_params: np.ndarray,
                        semantic_target: str,
                        intensity: float,
                        output_path: str,
                        sample_rate: int = 44100):
        """Plot EQ frequency response curve"""

        # Compute frequency response
        frequencies = np.logspace(np.log10(20), np.log10(20000), 1000)
        response_db = np.zeros_like(frequencies)

        # Extract parameters
        gain1, freq1 = eq_params[0], eq_params[1]
        gain2, freq2, q2 = eq_params[2], eq_params[3], eq_params[4]
        gain3, freq3, q3 = eq_params[5], eq_params[6], eq_params[7]
        gain4, freq4, q4 = eq_params[8], eq_params[9], eq_params[10]
        gain5, freq5 = eq_params[11], eq_params[12]

        # Approximate response (simplified)
        for i, f in enumerate(frequencies):
            # Low shelf
            if f < freq1 * 1.5:
                factor = 1 - np.exp(-(f - 20) / (freq1 * 0.5))
                response_db[i] += gain1 * factor

            # Bells (Gaussian approximation)
            for gain, fc, q in [(gain2, freq2, q2), (gain3, freq3, q3), (gain4, freq4, q4)]:
                bw = fc / q
                response_db[i] += gain * np.exp(-0.5 * ((f - fc) / (bw * 0.5))**2)

            # High shelf
            if f > freq5 * 0.67:
                factor = 1 - np.exp(-(20000 - f) / (freq5 * 0.3))
                response_db[i] += gain5 * factor

        # Plot
        fig, ax = plt.subplots(figsize=(14, 6))

        color = self.colors.get(semantic_target, '#1976D2')
        ax.semilogx(frequencies, response_db, linewidth=3, color=color, label=f'{semantic_target} (intensity={intensity:.2f})')
        ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)

        # Mark band centers
        band_freqs = [freq1, freq2, freq3, freq4, freq5]
        band_gains = [gain1, gain2, gain3, gain4, gain5]
        ax.scatter(band_freqs, band_gains, s=200, c='red', zorder=5,
                  edgecolors='black', linewidths=2, label='Band Centers')

        # Annotate bands
        for i, (f, g) in enumerate(zip(band_freqs, band_gains), 1):
            ax.annotate(f'B{i}\n{f:.0f}Hz\n{g:+.1f}dB',
                       xy=(f, g), xytext=(10, 10),
                       textcoords='offset points',
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

        ax.set_xlabel('Frequency (Hz)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Gain (dB)', fontsize=14, fontweight='bold')
        ax.set_title(f'Adaptive EQ Response: {semantic_target}', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlim(20, 20000)
        ax.set_ylim(-12, 12)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=12, loc='upper left')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved EQ response: {output_path}")

    def plot_comparison_grid(self, audio_path: str, profile: List[Tuple[str, float]],
                            top_targets: List[str], output_path: str):
        """Create comparison grid showing EQ curves for top semantic targets"""

        # Load audio
        audio, sr, _ = self.load_audio(audio_path)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        frequencies = np.logspace(np.log10(20), np.log10(20000), 1000)

        for idx, target in enumerate(top_targets[:4]):
            ax = axes[idx]

            # Generate EQ
            eq_params, similarity = self.generator.generate_adaptive_eq(
                audio, target, intensity=0.7, return_similarity=True
            )

            # Compute response (same as above)
            response_db = np.zeros_like(frequencies)
            gain1, freq1 = eq_params[0], eq_params[1]
            gain2, freq2, q2 = eq_params[2], eq_params[3], eq_params[4]
            gain3, freq3, q3 = eq_params[5], eq_params[6], eq_params[7]
            gain4, freq4, q4 = eq_params[8], eq_params[9], eq_params[10]
            gain5, freq5 = eq_params[11], eq_params[12]

            for i, f in enumerate(frequencies):
                if f < freq1 * 1.5:
                    response_db[i] += gain1 * (1 - np.exp(-(f - 20) / (freq1 * 0.5)))
                for gain, fc, q in [(gain2, freq2, q2), (gain3, freq3, q3), (gain4, freq4, q4)]:
                    bw = fc / q
                    response_db[i] += gain * np.exp(-0.5 * ((f - fc) / (bw * 0.5))**2)
                if f > freq5 * 0.67:
                    response_db[i] += gain5 * (1 - np.exp(-(20000 - f) / (freq5 * 0.3)))

            color = self.colors.get(target, '#1976D2')
            ax.semilogx(frequencies, response_db, linewidth=3, color=color)
            ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)

            # Band centers
            band_freqs = [freq1, freq2, freq3, freq4, freq5]
            band_gains = [gain1, gain2, gain3, gain4, gain5]
            ax.scatter(band_freqs, band_gains, s=100, c='red', zorder=5, edgecolors='black', linewidths=1.5)

            ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Gain (dB)', fontsize=12, fontweight='bold')
            ax.set_title(f'{target.upper()} (similarity: {similarity:.3f})',
                        fontsize=14, fontweight='bold', color=color)
            ax.set_xlim(20, 20000)
            ax.set_ylim(-12, 12)
            ax.grid(True, alpha=0.3, which='both')

        plt.suptitle(f'Adaptive EQ Comparison\n{Path(audio_path).name}',
                    fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved comparison grid: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test Adaptive EQ with Real Audio")

    parser.add_argument('--input', '-i', help='Input audio file')
    parser.add_argument('--input-dir', help='Directory with multiple audio files')
    parser.add_argument('--target', '-t', help='Semantic target for EQ')
    parser.add_argument('--intensity', type=float, default=0.7, help='EQ intensity [0-1]')
    parser.add_argument('--output-dir', '-o', default='./analysis_results', help='Output directory')
    parser.add_argument('--v2-model', default='neural_eq_safedb_v2.pt')
    parser.add_argument('--audio-encoder', default='audio_encoder_best.pt')

    args = parser.parse_args()

    # Check models
    if not Path(args.v2_model).exists():
        print(f"ERROR: V2 model not found: {args.v2_model}")
        return

    if not Path(args.audio_encoder).exists():
        print(f"ERROR: Audio Encoder not found: {args.audio_encoder}")
        print("Train it first: python train_audio_encoder.py")
        return

    # Create analyzer
    analyzer = AudioAnalyzer(args.v2_model, args.audio_encoder)

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Get input files
    if args.input:
        audio_files = [Path(args.input)]
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        audio_files = list(input_dir.glob('*.wav')) + list(input_dir.glob('*.mp3'))
    else:
        print("ERROR: Must specify --input or --input-dir")
        return

    print(f"\nFound {len(audio_files)} audio file(s) to analyze")

    # Process each file
    for audio_path in audio_files:
        print("\n" + "="*70)
        print(f"PROCESSING: {audio_path.name}")
        print("="*70)

        # Load audio
        audio, sr, metadata = analyzer.load_audio(str(audio_path))

        # Output prefix
        output_prefix = output_dir / audio_path.stem

        # 1. Analyze semantic profile
        profile = analyzer.analyze_semantic_profile(audio, top_k=14)
        analyzer.plot_semantic_profile(profile, f"{output_prefix}_semantic_profile.png")

        # 2. Generate comparison grid for top 4 characteristics
        top_4_terms = [term for term, _ in profile[:4]]
        analyzer.plot_comparison_grid(
            str(audio_path),
            profile,
            top_4_terms,
            f"{output_prefix}_eq_comparison.png"
        )

        # 3. If specific target requested, generate detailed EQ
        if args.target:
            eq_params, similarity = analyzer.generate_eq_for_target(
                audio, args.target, args.intensity
            )
            analyzer.plot_eq_response(
                eq_params,
                args.target,
                args.intensity,
                f"{output_prefix}_{args.target}_eq_response.png",
                sr
            )

            # Apply EQ and save
            audio_np = audio.numpy()[0]
            processed = analyzer.eq_filter.apply_eq(audio_np, eq_params, sr)

            # Save processed audio using soundfile
            output_audio = output_dir / f"{audio_path.stem}_{args.target}_{args.intensity:.2f}.wav"
            import soundfile as sf
            sf.write(str(output_audio), processed, sr)
            print(f"✓ Saved processed audio: {output_audio}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - *_semantic_profile.png - Audio characteristics analysis")
    print("  - *_eq_comparison.png - Top 4 adaptive EQ curves")
    if args.target:
        print(f"  - *_{args.target}_eq_response.png - Detailed EQ curve")
        print(f"  - *_{args.target}_*.wav - Processed audio")


if __name__ == "__main__":
    main()
