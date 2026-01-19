"""
Multi-Source Training Dataset
=============================

Combines multiple audio sources for robust Audio Encoder training:
1. Pink noise (synthetic) - Clear EQ effects, unlimited data
2. FMA dataset (real music) - Genre diversity, real-world generalization
3. MUSDB stems (isolated instruments) - Clean source separation

This addresses the domain gap between synthetic training data and real music.

References:
    - FMA: Defferrard et al. "FMA: A Dataset For Music Analysis" ISMIR 2017
    - MUSDB18: Rafii et al. "MUSDB18 - a corpus for music separation" 2017
    - Pink noise synthesis: Voss-McCartney algorithm

Usage:
    from core.multi_source_dataset import MultiSourceDataset, DatasetConfig

    config = DatasetConfig(
        fma_path="path/to/fma_small",
        musdb_path="path/to/musdb18",
        pink_noise_ratio=0.3,
        fma_ratio=0.5,
        musdb_ratio=0.2,
    )
    dataset = MultiSourceDataset(eq_settings, config)
"""

import os
import glob
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from core.training_data_synthesis import PinkNoiseGenerator, BiquadEQFilter


@dataclass
class DatasetConfig:
    """Configuration for multi-source dataset."""

    # Dataset paths (None = don't use)
    fma_path: Optional[str] = None
    musdb_path: Optional[str] = None

    # Source ratios (must sum to 1.0)
    pink_noise_ratio: float = 0.3
    fma_ratio: float = 0.5
    musdb_ratio: float = 0.2

    # Audio settings
    sample_rate: int = 44100
    duration: float = 2.0  # seconds

    # Augmentation
    n_augmentations: int = 3
    apply_random_eq: bool = True  # Apply random EQ from SAFE-DB

    # Data limits (for debugging)
    max_fma_files: Optional[int] = None
    max_musdb_files: Optional[int] = None

    def __post_init__(self):
        total = self.pink_noise_ratio + self.fma_ratio + self.musdb_ratio
        if abs(total - 1.0) > 0.01:
            # Normalize ratios
            self.pink_noise_ratio /= total
            self.fma_ratio /= total
            self.musdb_ratio /= total


class FMALoader:
    """
    Loader for Free Music Archive (FMA) dataset.

    FMA provides Creative Commons-licensed music tracks.
    Download from: https://github.com/mdeff/fma

    Expected structure:
        fma_small/
            000/
                000002.mp3
                000005.mp3
                ...
            001/
                ...
    """

    def __init__(
        self,
        root_path: str,
        sample_rate: int = 44100,
        duration: float = 2.0,
        max_files: Optional[int] = None,
    ):
        self.root_path = Path(root_path)
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)

        # Find all audio files
        self.audio_files = []
        for ext in ['*.mp3', '*.wav', '*.flac']:
            self.audio_files.extend(glob.glob(str(self.root_path / '**' / ext), recursive=True))

        if max_files:
            self.audio_files = self.audio_files[:max_files]

        print(f"FMA: Found {len(self.audio_files)} audio files")

    def load_random_clip(self) -> Optional[np.ndarray]:
        """Load a random clip from FMA."""
        if not self.audio_files:
            return None

        # Try up to 5 files in case some fail to load
        for _ in range(5):
            try:
                filepath = np.random.choice(self.audio_files)
                waveform, sr = torchaudio.load(filepath)

                # Resample if needed
                if sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                    waveform = resampler(waveform)

                # Convert to mono
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                # Random crop
                if waveform.shape[1] > self.n_samples:
                    start = np.random.randint(0, waveform.shape[1] - self.n_samples)
                    waveform = waveform[:, start:start + self.n_samples]
                elif waveform.shape[1] < self.n_samples:
                    # Pad if too short
                    pad = self.n_samples - waveform.shape[1]
                    waveform = F.pad(waveform, (0, pad))

                # Normalize
                waveform = waveform / (waveform.abs().max() + 1e-8)
                waveform = waveform * 0.5  # Leave headroom

                return waveform.squeeze(0).numpy()

            except Exception as e:
                continue

        return None


class MUSDBLoader:
    """
    Loader for MUSDB18 dataset (isolated stems).

    MUSDB18 provides professionally-produced music with isolated stems:
    - drums, bass, vocals, other

    Download from: https://sigsep.github.io/datasets/musdb.html

    Expected structure:
        musdb18/
            train/
                A Classic Education - NightOwl/
                    mixture.wav
                    drums.wav
                    bass.wav
                    vocals.wav
                    other.wav
                ...
    """

    STEMS = ['drums', 'bass', 'vocals', 'other', 'mixture']

    def __init__(
        self,
        root_path: str,
        sample_rate: int = 44100,
        duration: float = 2.0,
        max_files: Optional[int] = None,
        stems: List[str] = None,
    ):
        self.root_path = Path(root_path)
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        self.stems = stems or self.STEMS

        # Find all track directories
        self.track_dirs = []
        for split in ['train', 'test']:
            split_path = self.root_path / split
            if split_path.exists():
                self.track_dirs.extend([d for d in split_path.iterdir() if d.is_dir()])

        if max_files:
            self.track_dirs = self.track_dirs[:max_files]

        print(f"MUSDB: Found {len(self.track_dirs)} tracks")

    def load_random_clip(self, stem: str = None) -> Optional[np.ndarray]:
        """Load a random clip from MUSDB."""
        if not self.track_dirs:
            return None

        stem = stem or np.random.choice(self.stems)

        for _ in range(5):
            try:
                track_dir = np.random.choice(self.track_dirs)
                stem_path = track_dir / f"{stem}.wav"

                if not stem_path.exists():
                    continue

                waveform, sr = torchaudio.load(str(stem_path))

                # Resample if needed
                if sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                    waveform = resampler(waveform)

                # Convert to mono
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                # Random crop
                if waveform.shape[1] > self.n_samples:
                    start = np.random.randint(0, waveform.shape[1] - self.n_samples)
                    waveform = waveform[:, start:start + self.n_samples]
                elif waveform.shape[1] < self.n_samples:
                    pad = self.n_samples - waveform.shape[1]
                    waveform = torch.nn.functional.pad(waveform, (0, pad))

                # Normalize
                waveform = waveform / (waveform.abs().max() + 1e-8)
                waveform = waveform * 0.5

                return waveform.squeeze(0).numpy()

            except Exception as e:
                continue

        return None


class MultiSourceDataset(Dataset):
    """
    Multi-source dataset for Audio Encoder training.

    Combines:
    1. Pink noise + EQ (synthetic, controlled)
    2. FMA clips + EQ (real music diversity)
    3. MUSDB stems + EQ (isolated instruments)

    All sources have SAFE-DB EQ applied to create consistent training pairs.
    """

    def __init__(
        self,
        eq_settings: List,
        config: DatasetConfig,
        eq_loader = None,  # For denormalization
    ):
        self.eq_settings = eq_settings
        self.config = config
        self.eq_loader = eq_loader

        self.sample_rate = config.sample_rate
        self.n_samples = int(config.sample_rate * config.duration)

        # Source generators
        self.pink_generator = PinkNoiseGenerator()
        self.eq_filter = BiquadEQFilter()

        # Real audio loaders
        self.fma_loader = None
        self.musdb_loader = None

        if config.fma_path and Path(config.fma_path).exists():
            self.fma_loader = FMALoader(
                config.fma_path,
                sample_rate=config.sample_rate,
                duration=config.duration,
                max_files=config.max_fma_files,
            )
        else:
            print(f"FMA not available (path: {config.fma_path})")
            # Redistribute ratio to pink noise
            config.pink_noise_ratio += config.fma_ratio
            config.fma_ratio = 0

        if config.musdb_path and Path(config.musdb_path).exists():
            self.musdb_loader = MUSDBLoader(
                config.musdb_path,
                sample_rate=config.sample_rate,
                duration=config.duration,
                max_files=config.max_musdb_files,
            )
        else:
            print(f"MUSDB not available (path: {config.musdb_path})")
            config.pink_noise_ratio += config.musdb_ratio
            config.musdb_ratio = 0

        # Create example indices
        n_total = len(eq_settings) * config.n_augmentations
        self.n_pink = int(n_total * config.pink_noise_ratio)
        self.n_fma = int(n_total * config.fma_ratio)
        self.n_musdb = int(n_total * config.musdb_ratio)

        # Ensure we have at least n_total examples
        self.n_total = self.n_pink + self.n_fma + self.n_musdb

        print(f"\nMultiSourceDataset created:")
        print(f"  Pink noise: {self.n_pink} examples ({config.pink_noise_ratio*100:.0f}%)")
        print(f"  FMA: {self.n_fma} examples ({config.fma_ratio*100:.0f}%)")
        print(f"  MUSDB: {self.n_musdb} examples ({config.musdb_ratio*100:.0f}%)")
        print(f"  Total: {self.n_total} examples")

    def __len__(self) -> int:
        return self.n_total

    def _get_source_audio(self, idx: int) -> Tuple[np.ndarray, str]:
        """Get source audio based on index."""
        if idx < self.n_pink:
            # Pink noise
            audio = self.pink_generator.generate(self.n_samples, self.sample_rate)
            source = 'pink_noise'
        elif idx < self.n_pink + self.n_fma:
            # FMA
            audio = self.fma_loader.load_random_clip() if self.fma_loader else None
            if audio is None:
                audio = self.pink_generator.generate(self.n_samples, self.sample_rate)
                source = 'pink_noise'
            else:
                source = 'fma'
        else:
            # MUSDB
            audio = self.musdb_loader.load_random_clip() if self.musdb_loader else None
            if audio is None:
                audio = self.pink_generator.generate(self.n_samples, self.sample_rate)
                source = 'pink_noise'
            else:
                source = 'musdb'

        return audio, source

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training example.

        Returns dict with:
            - audio_clean: Source audio before EQ [1, samples]
            - audio_eq: Audio with EQ applied [1, samples]
            - eq_params_norm: Normalized EQ parameters [13]
            - semantic_label: Semantic term string
            - source_type: Source identifier string
        """
        # Get source audio
        audio_clean, source_type = self._get_source_audio(idx)

        # Random EQ setting from SAFE-DB
        setting = np.random.choice(self.eq_settings)

        # Denormalize EQ params for filtering
        if self.eq_loader is not None:
            eq_params_denorm = self.eq_loader.denormalize_params(
                np.array(setting.eq_params_normalized)
            )
        else:
            # Assume already denormalized or use as-is
            eq_params_denorm = np.array(setting.eq_params_normalized) * 24 - 12  # Rough denorm

        # Apply EQ
        audio_eq = self.eq_filter.apply_eq(audio_clean, eq_params_denorm, self.sample_rate)

        # Convert to tensors
        audio_clean_tensor = torch.FloatTensor(audio_clean).unsqueeze(0)  # [1, samples]
        audio_eq_tensor = torch.FloatTensor(audio_eq).unsqueeze(0)  # [1, samples]
        eq_params_tensor = torch.FloatTensor(setting.eq_params_normalized)  # [13]

        return {
            'audio_clean': audio_clean_tensor,
            'audio_eq': audio_eq_tensor,
            'eq_params_norm': eq_params_tensor,
            'semantic_label': setting.semantic_label,
            'source_type': source_type,
        }


def create_dataset_from_config(
    eq_settings: List,
    eq_loader,
    fma_path: str = None,
    musdb_path: str = None,
    pink_ratio: float = 0.3,
    fma_ratio: float = 0.5,
    musdb_ratio: float = 0.2,
    **kwargs,
) -> MultiSourceDataset:
    """
    Convenience function to create dataset.

    Args:
        eq_settings: List of EQ settings from SAFE-DB
        eq_loader: Loader for parameter denormalization
        fma_path: Path to FMA dataset
        musdb_path: Path to MUSDB18 dataset
        pink_ratio: Ratio of pink noise examples
        fma_ratio: Ratio of FMA examples
        musdb_ratio: Ratio of MUSDB examples
        **kwargs: Additional config options

    Returns:
        MultiSourceDataset instance
    """
    config = DatasetConfig(
        fma_path=fma_path,
        musdb_path=musdb_path,
        pink_noise_ratio=pink_ratio,
        fma_ratio=fma_ratio,
        musdb_ratio=musdb_ratio,
        **kwargs,
    )

    return MultiSourceDataset(eq_settings, config, eq_loader)


def test_multi_source_dataset():
    """Test the multi-source dataset."""
    print("Testing MultiSourceDataset...")

    # Mock EQ settings
    class MockSetting:
        def __init__(self):
            self.eq_params_normalized = np.random.rand(13).tolist()
            self.semantic_label = np.random.choice(['warm', 'bright', 'muddy'])

    eq_settings = [MockSetting() for _ in range(100)]

    # Create config (pink noise only for testing)
    config = DatasetConfig(
        fma_path=None,
        musdb_path=None,
        pink_noise_ratio=1.0,
        fma_ratio=0.0,
        musdb_ratio=0.0,
        n_augmentations=2,
    )

    dataset = MultiSourceDataset(eq_settings, config)

    # Test loading
    sample = dataset[0]
    print(f"  audio_clean: {sample['audio_clean'].shape}")
    print(f"  audio_eq: {sample['audio_eq'].shape}")
    print(f"  eq_params_norm: {sample['eq_params_norm'].shape}")
    print(f"  semantic_label: {sample['semantic_label']}")
    print(f"  source_type: {sample['source_type']}")

    print("Test complete!")


if __name__ == "__main__":
    test_multi_source_dataset()
