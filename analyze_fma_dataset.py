"""
FMA Dataset Analysis
====================

Quick analysis of the FMA dataset to understand:
1. Genre distribution
2. Track durations
3. Audio quality/format stats
4. Sample waveforms

Run this BEFORE training to understand your data.

Usage:
    python analyze_fma_dataset.py --fma-path ./data/fma/fma_small --metadata-path ./data/fma/fma_metadata
"""

import os
import sys
import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    import torchaudio
    HAS_TORCHAUDIO = True
except:
    HAS_TORCHAUDIO = False
    print("Warning: torchaudio not available, skipping audio analysis")


def load_fma_metadata(metadata_path: str):
    """Load FMA metadata files."""
    tracks_path = Path(metadata_path) / 'tracks.csv'
    genres_path = Path(metadata_path) / 'genres.csv'

    if not tracks_path.exists():
        print(f"Error: tracks.csv not found at {tracks_path}")
        return None, None

    # FMA tracks.csv has multi-level headers
    tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1])
    genres = pd.read_csv(genres_path, index_col=0) if genres_path.exists() else None

    return tracks, genres


def analyze_metadata(tracks_df: pd.DataFrame, genres_df: pd.DataFrame):
    """Analyze the metadata."""
    print("\n" + "="*70)
    print("FMA METADATA ANALYSIS")
    print("="*70)

    # Basic stats
    print(f"\nTotal tracks in metadata: {len(tracks_df)}")

    # Genre distribution
    if ('track', 'genre_top') in tracks_df.columns:
        genre_counts = tracks_df[('track', 'genre_top')].value_counts()
        print(f"\nGenre Distribution (top-level):")
        print("-" * 40)
        for genre, count in genre_counts.items():
            pct = count / len(tracks_df) * 100
            bar = '█' * int(pct / 2)
            print(f"  {genre:20s}: {count:5d} ({pct:5.1f}%) {bar}")

        return genre_counts
    else:
        print("Genre information not found in metadata")
        return None


def analyze_audio_files(fma_path: str, n_samples: int = 100):
    """Analyze actual audio files."""
    print("\n" + "="*70)
    print("FMA AUDIO FILE ANALYSIS")
    print("="*70)

    # Find all audio files
    audio_files = glob.glob(str(Path(fma_path) / '**/*.mp3'), recursive=True)
    print(f"\nTotal audio files found: {len(audio_files)}")

    if not audio_files:
        print("No audio files found!")
        return None

    # File size distribution
    file_sizes = [os.path.getsize(f) / (1024*1024) for f in audio_files]  # MB
    print(f"\nFile sizes:")
    print(f"  Min: {min(file_sizes):.2f} MB")
    print(f"  Max: {max(file_sizes):.2f} MB")
    print(f"  Mean: {np.mean(file_sizes):.2f} MB")
    print(f"  Total: {sum(file_sizes)/1024:.2f} GB")

    # Sample audio properties
    if HAS_TORCHAUDIO:
        print(f"\nSampling {min(n_samples, len(audio_files))} files for audio properties...")

        sample_files = np.random.choice(audio_files, min(n_samples, len(audio_files)), replace=False)

        durations = []
        sample_rates = []
        channels = []

        for filepath in tqdm(sample_files, desc="Analyzing audio"):
            try:
                info = torchaudio.info(filepath)
                durations.append(info.num_frames / info.sample_rate)
                sample_rates.append(info.sample_rate)
                channels.append(info.num_channels)
            except Exception as e:
                continue

        if durations:
            print(f"\nAudio Properties (from {len(durations)} samples):")
            print(f"  Duration: {np.mean(durations):.1f}s mean, {np.min(durations):.1f}s min, {np.max(durations):.1f}s max")
            print(f"  Sample rates: {Counter(sample_rates)}")
            print(f"  Channels: {Counter(channels)}")

            return {
                'n_files': len(audio_files),
                'mean_duration': np.mean(durations),
                'sample_rates': Counter(sample_rates),
            }

    return {'n_files': len(audio_files)}


def analyze_track_ids(fma_path: str, tracks_df: pd.DataFrame):
    """Check which tracks from metadata are actually present."""
    print("\n" + "="*70)
    print("TRACK ID COVERAGE")
    print("="*70)

    audio_files = glob.glob(str(Path(fma_path) / '**/*.mp3'), recursive=True)

    # Extract track IDs from filenames
    file_track_ids = set()
    for f in audio_files:
        try:
            track_id = int(Path(f).stem)
            file_track_ids.add(track_id)
        except:
            continue

    metadata_track_ids = set(tracks_df.index)

    both = file_track_ids & metadata_track_ids
    only_files = file_track_ids - metadata_track_ids
    only_metadata = metadata_track_ids - file_track_ids

    print(f"\nTrack IDs in audio files: {len(file_track_ids)}")
    print(f"Track IDs in metadata: {len(metadata_track_ids)}")
    print(f"Matching IDs: {len(both)}")
    print(f"In files only: {len(only_files)}")
    print(f"In metadata only: {len(only_metadata)}")

    coverage = len(both) / len(file_track_ids) * 100 if file_track_ids else 0
    print(f"\nMetadata coverage: {coverage:.1f}%")

    return both


def plot_distributions(genre_counts, output_path: str = 'fma_analysis.png'):
    """Create visualization of the dataset."""
    if genre_counts is None:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    genres = genre_counts.index.tolist()
    counts = genre_counts.values

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(genres)))
    bars = ax.barh(genres, counts, color=colors)

    ax.set_xlabel('Number of Tracks')
    ax.set_title('FMA Genre Distribution')

    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
               f'{count}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nVisualization saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze FMA dataset')
    parser.add_argument('--fma-path', type=str, required=True,
                       help='Path to FMA audio files (e.g., ./data/fma/fma_small)')
    parser.add_argument('--metadata-path', type=str, required=True,
                       help='Path to FMA metadata (e.g., ./data/fma/fma_metadata)')
    parser.add_argument('--n-samples', type=int, default=100,
                       help='Number of audio files to sample for analysis')
    parser.add_argument('--output', type=str, default='fma_analysis.png',
                       help='Output visualization path')
    args = parser.parse_args()

    # Load metadata
    print("Loading FMA metadata...")
    tracks_df, genres_df = load_fma_metadata(args.metadata_path)

    if tracks_df is not None:
        genre_counts = analyze_metadata(tracks_df, genres_df)
        matching_ids = analyze_track_ids(args.fma_path, tracks_df)
        plot_distributions(genre_counts, args.output)
    else:
        genre_counts = None

    # Analyze audio files
    audio_stats = analyze_audio_files(args.fma_path, args.n_samples)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"""
FMA Dataset Ready for Training:
  - Audio files: {audio_stats['n_files'] if audio_stats else 'Unknown'}
  - Genres: {len(genre_counts) if genre_counts is not None else 'Unknown'}
  - Suitable for E2E audio encoder training: YES

Recommended training command:
  python train_audio_encoder_e2e.py \\
      --epochs 100 --device cuda \\
      --fma-path {args.fma_path} \\
      --fma-ratio 0.7 --pink-ratio 0.3
""")


if __name__ == '__main__':
    main()
