"""
Unified Demo Script for Semantic Mastering System
==================================================

Consolidated demo showcasing all system capabilities.

Usage:
    python demo.py                    # Run all demos
    python demo.py --analysis         # Analysis capabilities only
    python demo.py --llm-comparison   # LLM vs Dataset comparison
    python demo.py --variance         # Variance problem explanation
    python demo.py --neural           # Neural EQ morphing with SocialFX

Author: Semantic Mastering System
"""

import sys
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def demo_analysis():
    """
    Demo: EQ Profile Analysis
    ==========================
    Quick demonstration of analysis capabilities without requiring audio files.
    Generates synthetic test signals and demonstrates all analysis features.
    """

    import torch
    import torchaudio
    import numpy as np
    import matplotlib.pyplot as plt

    def create_test_audio(duration: float = 5.0, sample_rate: int = 44100) -> torch.Tensor:
        """Create synthetic test audio with multiple frequency components"""

        t = torch.linspace(0, duration, int(sample_rate * duration))

        # Create complex test signal
        frequencies = [220, 440, 880, 1320, 1760]  # A3 and harmonics
        amplitudes = [0.3, 0.4, 0.2, 0.1, 0.05]

        audio = torch.zeros(2, len(t))  # Stereo

        # Add harmonic content
        for freq, amp in zip(frequencies, amplitudes):
            sine_wave = amp * torch.sin(2 * np.pi * freq * t)
            audio[0] += sine_wave
            audio[1] += sine_wave

        # Add broadband content
        noise = 0.02 * torch.randn_like(t)
        noise_filtered = torch.conv1d(noise.unsqueeze(0).unsqueeze(0),
                                     torch.ones(1, 1, 50) / 50, padding=25)[0, 0]

        audio[0] += noise_filtered[:len(t)]
        audio[1] += noise_filtered[:len(t)]
        audio[1] *= 0.9  # Stereo separation

        # Normalize
        peak = torch.max(torch.abs(audio))
        if peak > 0.8:
            audio = audio * (0.8 / peak)

        return audio

    print("\n" + "="*60)
    print("DEMO: EQ PROFILE ANALYSIS")
    print("="*60)

    try:
        from core.analyze_eq_profiles import EQAnalyzer
        from tests.test_eq_profiles import EQProfileTester

        # Initialize analyzer
        analyzer = EQAnalyzer(sample_rate=44100)
        available = analyzer.get_available_profiles()
        print(f"Available presets: {available['presets']}")

        if available['presets']:
            test_profiles = available['presets'][:3]
            Path("./outputs/plots").mkdir(parents=True, exist_ok=True)

            # Frequency response
            analyzer.plot_frequency_responses(
                test_profiles,
                save_path="./outputs/plots/demo_frequency_response.png",
                show_plot=False
            )
            print("✓ Frequency response plot saved")

            # Audio testing with synthetic signal
            print("\nCreating synthetic test audio...")
            test_audio = create_test_audio(duration=5.0, sample_rate=44100)
            test_audio_path = Path("./outputs/demo_test_audio.wav")
            torchaudio.save(str(test_audio_path), test_audio, 44100)
            print(f"✓ Test audio saved: {test_audio_path}")

            # Run tests
            tester = EQProfileTester(sample_rate=44100)
            if tester.mastering_system and tester.mastering_system.presets:
                presets = list(tester.mastering_system.presets.keys())

                if len(presets) >= 2:
                    results = tester.ab_compare_profiles(
                        str(test_audio_path),
                        presets[0],
                        presets[1],
                        output_dir="./outputs"
                    )
                    if results:
                        print("✓ A/B comparison completed")

            # Cleanup
            if test_audio_path.exists():
                test_audio_path.unlink()

        print("\n✓ Analysis demo complete!")

    except Exception as e:
        print(f"Error in analysis demo: {e}")


def demo_llm_comparison():
    """
    Demo: LLM vs Dataset vs Adaptive EQ Generation
    ===============================================
    Compare different approaches for generating EQ parameters from semantic terms.
    No API keys required - uses rule-based LLM simulation.
    """

    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from typing import Dict

    try:
        from core.semantic_mastering import EQProfile
        SEMANTIC_AVAILABLE = True
    except ImportError:
        SEMANTIC_AVAILABLE = False
        from dataclasses import dataclass

        @dataclass
        class EQProfile:
            name: str
            params_dasp: torch.Tensor
            params_original: np.ndarray
            reasoning: str
            n_examples: int
            confidence: float = 1.0

    class EQMethodDemo:
        """Demo different EQ generation approaches without requiring API keys"""

        def __init__(self):
            # Simulated dataset statistics
            self.dataset_stats = {
                'warm': {
                    'examples': 64,
                    'mean_eq': [1.4, 1.4, 0.3, -0.2, -0.8, -1.5],
                    'variance': [1.0, 0.8, 0.4, 0.5, 0.7, 0.9],
                    'strategies': {
                        'gentle': [0.5, 0.5, 0.5, 0.0, 0.0, -0.5],
                        'aggressive': [2.5, 2.0, 0.0, -1.0, -2.0, -3.0],
                        'balanced': [1.0, 1.0, 0.0, 0.0, -0.5, -1.0]
                    }
                },
                'bright': {
                    'examples': 19,
                    'mean_eq': [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
                    'variance': [0.8, 0.6, 0.9, 1.2, 0.8, 1.1],
                    'strategies': {
                        'gentle': [0.0, 0.0, 0.0, 0.5, 1.0, 1.0],
                        'aggressive': [-1.0, -0.5, 1.0, 2.0, 3.0, 3.5],
                        'balanced': [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
                    }
                }
            }

            # LLM simulation rules
            self.llm_rules = {
                'warm': {
                    'gpt4': [1.2, 0.8, 0.0, -0.3, -1.0, -1.8],
                    'claude': [1.5, 1.0, 0.2, -0.5, -1.2, -2.0],
                    'reasoning': "Warm audio benefits from enhanced lows and gentle high rolloff"
                },
                'bright': {
                    'gpt4': [-0.3, 0.0, 0.8, 1.5, 2.0, 1.8],
                    'claude': [-0.5, 0.2, 1.0, 1.8, 2.2, 2.0],
                    'reasoning': "Brightness through high-frequency enhancement"
                }
            }

        def compare_all_methods(self, term: str) -> Dict:
            """Compare all methods for a given term"""

            print(f"\n{'='*70}")
            print(f"COMPARING EQ GENERATION METHODS: {term.upper()}")
            print(f"{'='*70}")

            if term not in self.dataset_stats:
                print(f"Term '{term}' not available in demo")
                return {}

            stats = self.dataset_stats[term]
            results = {}

            # Method 1: Dataset averaging
            print("\n1. Dataset Averaging")
            profile = EQProfile(
                name=f"{term}_dataset",
                params_dasp=torch.tensor([stats['mean_eq']]),
                params_original=np.array(stats['mean_eq']),
                reasoning=f"Average of {stats['examples']} examples",
                n_examples=stats['examples'],
                confidence=0.7
            )
            print(f"   Examples: {stats['examples']}")
            print(f"   EQ curve: {stats['mean_eq']}")
            results['dataset'] = {'method': 'Dataset Averaging', 'profile': profile}

            # Method 2: Adaptive
            print("\n2. Adaptive Audio-Informed Selection")
            adaptive_eq = stats['strategies']['balanced']
            profile = EQProfile(
                name=f"{term}_adaptive",
                params_dasp=torch.tensor([adaptive_eq]),
                params_original=np.array(adaptive_eq),
                reasoning="Selected balanced strategy based on audio analysis",
                n_examples=stats['examples'] // 3,
                confidence=0.85
            )
            print(f"   Strategy: balanced")
            print(f"   EQ curve: {adaptive_eq}")
            results['adaptive'] = {'method': 'Adaptive Selection', 'profile': profile}

            # Method 3: LLM
            print("\n3. LLM Generation (simulated)")
            llm_eq = self.llm_rules[term]['gpt4']
            profile = EQProfile(
                name=f"{term}_llm",
                params_dasp=torch.tensor([llm_eq]),
                params_original=np.array(llm_eq),
                reasoning=self.llm_rules[term]['reasoning'],
                n_examples=1,
                confidence=0.6
            )
            print(f"   Reasoning: {self.llm_rules[term]['reasoning']}")
            print(f"   EQ curve: {llm_eq}")
            results['llm'] = {'method': 'LLM Generation', 'profile': profile}

            return results

    print("\n" + "="*60)
    print("DEMO: LLM VS DATASET VS ADAPTIVE COMPARISON")
    print("="*60)

    demo = EQMethodDemo()

    for term in ['warm', 'bright']:
        results = demo.compare_all_methods(term)

    print("\n✓ LLM comparison demo complete!")


def demo_variance():
    """
    Demo: Why Averaging EQ Parameters Can Be Problematic
    =====================================================
    Shows the key issue with averaging multiple EQ examples for semantic terms.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    print("\n" + "="*60)
    print("DEMO: WHY SIMPLE EQ AVERAGING CAN BE PROBLEMATIC")
    print("="*60)

    # Simulate 5 different "warm" EQ curves
    warm_examples = {
        "Engineer 1 - Dark Rock": [2.0, 1.5, 0.0, -0.5, -1.0, -2.0],
        "Engineer 2 - Thin Vocal": [1.0, 2.5, 1.0, 0.5, -0.5, -1.0],
        "Engineer 3 - Bright Pop": [0.5, 0.5, 0.0, -1.0, -2.0, -3.0],
        "Engineer 4 - Bass-heavy": [3.0, 2.0, 0.0, 0.0, -0.5, -1.0],
        "Engineer 5 - Gentle": [0.5, 0.5, 0.5, 0.0, 0.0, -0.5]
    }

    band_names = ['Sub\n60Hz', 'Bass\n200Hz', 'LowMid\n500Hz', 'Mid\n1kHz', 'HighMid\n3kHz', 'Treble\n8kHz']

    examples_array = np.array(list(warm_examples.values()))
    averaged_eq = np.mean(examples_array, axis=0)
    std_dev = np.std(examples_array, axis=0)
    min_vals = np.min(examples_array, axis=0)
    max_vals = np.max(examples_array, axis=0)

    print(f"\nAnalyzing {len(warm_examples)} 'warm' EQ examples:")
    print("-" * 60)

    for i, band in enumerate(band_names):
        print(f"{band:12}: Avg={averaged_eq[i]:+5.1f}dB  Range=[{min_vals[i]:+4.1f}, {max_vals[i]:+4.1f}]  Std={std_dev[i]:.1f}")

    high_variance_bands = np.where(std_dev > 1.0)[0]

    print(f"\nPROBLEMS with simple averaging:")
    print("-" * 40)
    print(f"• {len(high_variance_bands)}/{len(band_names)} bands have high variance (>1dB)")
    print(f"• Averaged EQ might not match ANY real engineer's decision")
    print(f"• Context of original audio is completely lost")

    # Visualization
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        x = np.arange(len(band_names))
        colors = ['red', 'blue', 'green', 'orange', 'purple']

        # Plot individual examples
        for i, (engineer, eq_curve) in enumerate(warm_examples.items()):
            ax1.plot(x, eq_curve, 'o-', color=colors[i], alpha=0.7,
                    label=engineer, linewidth=2, markersize=6)

        ax1.plot(x, averaged_eq, 'k-', linewidth=4, marker='s', markersize=8,
                label='AVERAGED (Current)', alpha=0.8)

        ax1.set_title('Individual "Warm" EQ Examples vs Average', fontweight='bold')
        ax1.set_xlabel('Frequency Band')
        ax1.set_ylabel('Gain (dB)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(band_names)
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # Plot variance
        ax2.bar(x, std_dev, color='red', alpha=0.7)
        ax2.axhline(y=1.0, color='orange', linestyle='--', linewidth=2,
                   label='High Variance Threshold')
        ax2.set_title('Parameter Variance Analysis', fontweight='bold')
        ax2.set_xlabel('Frequency Band')
        ax2.set_ylabel('Standard Deviation (dB)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(band_names)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()

        Path("./outputs/plots").mkdir(parents=True, exist_ok=True)
        plt.savefig("./outputs/plots/variance_demo.png", dpi=300, bbox_inches='tight')
        print(f"\nPlot saved: ./outputs/plots/variance_demo.png")
        plt.close()

    except Exception as e:
        print(f"Visualization error: {e}")

    print("\n✓ Variance demo complete!")


def demo_neural():
    """
    Demo: Neural EQ Morphing with SocialFX Dataset
    ===============================================
    Demo using real SocialFX dataset with filtered English semantic terms.
    """

    import numpy as np
    import pandas as pd
    from collections import Counter
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsRegressor
    from dataclasses import dataclass
    from typing import Dict, List, Optional

    @dataclass
    class EQSetting:
        """Complete EQ setting with metadata"""
        parameters: Dict[str, float]
        semantic_label: str
        source_dataset: str
        audio_context: Optional[str] = None
        engineer_id: Optional[str] = None
        confidence: float = 1.0

    print("\n" + "="*60)
    print("DEMO: NEURAL EQ MORPHING WITH SOCIALFX")
    print("="*60)

    try:
        # Load SocialFX dataset
        print("Loading SocialFX dataset...")
        df = pd.read_parquet("hf://datasets/seungheondoh/socialfx-original/data/eq-00000-of-00001.parquet")
        print(f"Loaded {len(df)} raw examples")

        # Filter for English terms
        semantic_terms = [str(text).lower().strip() for text in df['text']]
        term_counts = Counter(semantic_terms)

        min_examples = 8
        english_terms = {
            term: count for term, count in term_counts.items()
            if term.isascii() and count >= min_examples and len(term) > 2
        }

        print(f"Found {len(english_terms)} English terms with >={min_examples} examples")
        print(f"Top terms: {dict(sorted(english_terms.items(), key=lambda x: x[1], reverse=True)[:10])}")

        # Filter dataset
        eq_settings = []
        for idx, row in df.iterrows():
            semantic_label = str(row['text']).lower().strip()

            if semantic_label in english_terms:
                param_keys = row['param_keys']
                param_values = row['param_values']

                parameters = {}
                for key, value in zip(param_keys, param_values):
                    parameters[key] = float(value)

                eq_setting = EQSetting(
                    parameters=parameters,
                    semantic_label=semantic_label,
                    source_dataset="socialfx",
                    audio_context=row['id'],
                    engineer_id=f"engineer_{idx}",
                    confidence=1.0
                )
                eq_settings.append(eq_setting)

        print(f"Filtered to {len(eq_settings)} examples")

        # Train simple PCA model
        print("\nTraining PCA model...")

        param_names = list(eq_settings[0].parameters.keys())
        n_params = len(param_names)

        params_array = np.zeros((len(eq_settings), n_params))
        labels = []

        for i, setting in enumerate(eq_settings):
            for j, param_name in enumerate(param_names):
                params_array[i, j] = setting.parameters.get(param_name, 0.0)
            labels.append(setting.semantic_label)

        # Train PCA
        scaler = StandardScaler()
        params_scaled = scaler.fit_transform(params_array)

        pca = PCA(n_components=12)
        latent_features = pca.fit_transform(params_scaled)

        print(f"PCA variance explained: {np.sum(pca.explained_variance_ratio_):.1%}")

        # Show example generation for 'warm' if available
        unique_labels = list(set(labels))
        if 'warm' in unique_labels:
            print("\nGenerating 'warm' EQ variations...")
            warm_indices = [i for i, lbl in enumerate(labels) if lbl == 'warm']
            warm_latent = latent_features[warm_indices]

            centroid = np.mean(warm_latent, axis=0)
            print(f"Based on {len(warm_indices)} warm examples")
            print(f"Latent centroid (first 5 dims): {centroid[:5]}")

        print("\n✓ Neural EQ demo complete!")
        print(f"Successfully processed {len(eq_settings)} examples")
        print(f"Semantic terms: {len(unique_labels)}")

    except Exception as e:
        print(f"Error in neural demo: {e}")
        print("Note: This demo requires internet connection to download SocialFX dataset")


def main():
    """Main entry point with argument parsing"""

    parser = argparse.ArgumentParser(
        description='Unified Demo for Semantic Mastering System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py                    Run all demos
  python demo.py --analysis         Analysis capabilities only
  python demo.py --llm-comparison   LLM vs Dataset comparison
  python demo.py --variance         Variance problem explanation
  python demo.py --neural           Neural EQ morphing
        """
    )

    parser.add_argument('--analysis', action='store_true', help='Run analysis demo')
    parser.add_argument('--llm-comparison', action='store_true', help='Run LLM comparison demo')
    parser.add_argument('--variance', action='store_true', help='Run variance demo')
    parser.add_argument('--neural', action='store_true', help='Run neural EQ demo')

    args = parser.parse_args()

    # If no specific demo selected, run all
    run_all = not (args.analysis or args.llm_comparison or args.variance or args.neural)

    print("="*60)
    print("SEMANTIC MASTERING SYSTEM - UNIFIED DEMO")
    print("="*60)

    if run_all or args.analysis:
        demo_analysis()

    if run_all or args.llm_comparison:
        demo_llm_comparison()

    if run_all or args.variance:
        demo_variance()

    if run_all or args.neural:
        demo_neural()

    print("\n" + "="*60)
    print("ALL DEMOS COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("📁 ./outputs/plots/ - Visualization plots")
    print("📁 ./outputs/ - Processed audio and reports")
    print("\nNext steps:")
    print("1. Explore production code in core/")
    print("2. Run tests with your audio in tests/")
    print("3. Try research tools in research/")


if __name__ == '__main__':
    main()
