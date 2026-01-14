"""
Training Data Synthesis for Audio Encoder
=========================================

Since SAFE-DB has EQ parameters but NO audio files, we synthesize training data by:
1. Generate pink noise (1/f spectrum, similar to music)
2. Apply SAFE-DB EQ settings to noise
3. Create (audio, EQ_params) pairs

This allows us to train the Audio Encoder to recognize EQ characteristics in audio.

Usage:
    from core.training_data_synthesis import PinkNoiseGenerator, TrainingDataSynthesizer

    synthesizer = TrainingDataSynthesizer()
    dataset = synthesizer.create_dataset(eq_settings, n_samples=1000)
"""

import torch
import numpy as np
from scipy import signal
from typing import Tuple, List
import warnings
warnings.filterwarnings("ignore")


class PinkNoiseGenerator:
    """
    Generate pink noise (1/f spectrum)

    Pink noise has equal energy per octave, similar to music.
    This makes it ideal for training the audio encoder to recognize EQ characteristics.
    """

    @staticmethod
    def generate(n_samples: int, sample_rate: int = 44100) -> np.ndarray:
        """
        Generate pink noise using Voss-McCartney algorithm

        Args:
            n_samples: Number of samples to generate
            sample_rate: Sample rate (Hz)

        Returns:
            pink_noise: [n_samples] mono pink noise
        """
        # Number of random sources (more = better quality, slower)
        num_rows = 16

        # Initialize state
        array = np.zeros((num_rows, n_samples))
        values = np.zeros(num_rows)

        for i in range(n_samples):
            # Determine which rows to update (based on trailing zeros in binary)
            update_mask = (i & (i + 1)) == 0
            update_indices = np.where([update_mask >> j & 1 for j in range(num_rows)])[0]

            # Update random values
            values[update_indices] = np.random.randn(len(update_indices))

            # Sum all values
            array[:, i] = values

        # Sum rows to get pink noise
        pink_noise = np.sum(array, axis=0)

        # Normalize to [-1, 1]
        pink_noise = pink_noise / np.max(np.abs(pink_noise))
        pink_noise *= 0.5  # Scale to moderate level

        return pink_noise.astype(np.float32)


class BiquadEQFilter:
    """
    Apply 5-band EQ using biquad filters (same as apply_neural_eq_v2_simple.py)

    SAFE-DB format (13 parameters):
        Band 1 (Low Shelf):  [Gain, Freq]
        Band 2 (Bell):       [Gain, Freq, Q]
        Band 3 (Bell):       [Gain, Freq, Q]
        Band 4 (Bell):       [Gain, Freq, Q]
        Band 5 (High Shelf): [Gain, Freq]
    """

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def apply_eq(audio: np.ndarray, eq_params_13: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """
        Apply 5-band EQ to audio

        Args:
            audio: [n_samples] mono audio
            eq_params_13: [13] SAFE-DB EQ parameters
            sample_rate: Sample rate (Hz)

        Returns:
            processed: [n_samples] EQ'd audio
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
        nyquist = sample_rate / 2
        freq1 = np.clip(freq1, 20, nyquist - 1000)
        freq2 = np.clip(freq2, 20, nyquist - 1000)
        freq3 = np.clip(freq3, 20, nyquist - 1000)
        freq4 = np.clip(freq4, 20, nyquist - 1000)
        freq5 = np.clip(freq5, 20, nyquist - 1000)

        processed = audio.copy()

        # Band 1: Low Shelf
        if abs(gain1) > 0.01:
            b, a = BiquadEQFilter.low_shelf(gain1, freq1, q1, sample_rate)
            processed = signal.lfilter(b, a, processed)

        # Band 2: Bell
        if abs(gain2) > 0.01:
            b, a = BiquadEQFilter.peaking_eq(gain2, freq2, q2, sample_rate)
            processed = signal.lfilter(b, a, processed)

        # Band 3: Bell
        if abs(gain3) > 0.01:
            b, a = BiquadEQFilter.peaking_eq(gain3, freq3, q3, sample_rate)
            processed = signal.lfilter(b, a, processed)

        # Band 4: Bell
        if abs(gain4) > 0.01:
            b, a = BiquadEQFilter.peaking_eq(gain4, freq4, q4, sample_rate)
            processed = signal.lfilter(b, a, processed)

        # Band 5: High Shelf
        if abs(gain5) > 0.01:
            b, a = BiquadEQFilter.high_shelf(gain5, freq5, q5, sample_rate)
            processed = signal.lfilter(b, a, processed)

        # Normalize to prevent clipping
        max_val = np.max(np.abs(processed))
        if max_val > 1.0:
            processed = processed / max_val * 0.99

        return processed


class TrainingDataSynthesizer:
    """
    Synthesize (audio, EQ) training pairs from SAFE-DB EQ settings

    Process:
        1. Generate pink noise
        2. Apply EQ setting to noise
        3. Store (audio_with_eq, eq_params_normalized) pair
    """

    def __init__(self, sample_rate: int = 44100, audio_duration: float = 2.0):
        """
        Args:
            sample_rate: Sample rate for generated audio
            audio_duration: Duration of audio clips in seconds
        """
        self.sample_rate = sample_rate
        self.audio_duration = audio_duration
        self.n_samples = int(sample_rate * audio_duration)

        self.pink_generator = PinkNoiseGenerator()
        self.eq_filter = BiquadEQFilter()

    def synthesize_single(self, eq_params_13: np.ndarray, eq_params_norm: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate single training example

        Args:
            eq_params_13: [13] raw EQ parameters (for applying filter)
            eq_params_norm: [13] normalized EQ parameters (for training target)

        Returns:
            audio_tensor: [1, n_samples] audio with EQ applied
            eq_tensor: [13] normalized EQ parameters
        """
        # Generate pink noise
        pink_noise = self.pink_generator.generate(self.n_samples, self.sample_rate)

        # Apply EQ
        audio_with_eq = self.eq_filter.apply_eq(pink_noise, eq_params_13, self.sample_rate)

        # Convert to tensors
        audio_tensor = torch.FloatTensor(audio_with_eq).unsqueeze(0)  # [1, n_samples]
        eq_tensor = torch.FloatTensor(eq_params_norm)  # [13]

        return audio_tensor, eq_tensor

    def create_dataset(self, eq_settings: List, n_augmentations: int = 3) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Create full training dataset from EQ settings

        Args:
            eq_settings: List of SAFEEQSetting objects from SAFE-DB
            n_augmentations: Number of audio variations per EQ setting

        Returns:
            dataset: List of (audio, eq_params_norm) tuples
        """
        dataset = []

        print(f"\nSynthesizing training data...")
        print(f"  EQ settings: {len(eq_settings)}")
        print(f"  Augmentations per setting: {n_augmentations}")
        print(f"  Total examples: {len(eq_settings) * n_augmentations}")

        for idx, eq_setting in enumerate(eq_settings):
            # Generate multiple audio examples for each EQ setting
            for aug_idx in range(n_augmentations):
                audio, eq_norm = self.synthesize_single(
                    eq_setting.eq_params,
                    eq_setting.eq_params_normalized
                )
                dataset.append((audio, eq_norm, eq_setting.semantic_label))

            # Progress
            if (idx + 1) % 100 == 0:
                print(f"  Synthesized: {idx + 1}/{len(eq_settings)} settings...")

        print(f"\nDataset created: {len(dataset)} examples")

        return dataset


class SynthesizedAudioDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapper for synthesized audio-EQ pairs
    """

    def __init__(self, data_list: List[Tuple[torch.Tensor, torch.Tensor, str]]):
        """
        Args:
            data_list: List of (audio, eq_params_norm, semantic_label) tuples
        """
        self.data = data_list

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        return self.data[idx]


def test_synthesis():
    """Test training data synthesis"""

    print("="*70)
    print("TESTING TRAINING DATA SYNTHESIS")
    print("="*70)

    # Create synthesizer
    synthesizer = TrainingDataSynthesizer(sample_rate=44100, audio_duration=2.0)

    # Test pink noise generation
    print("\nGenerating pink noise...")
    pink_noise = synthesizer.pink_generator.generate(44100 * 2)
    print(f"  Shape: {pink_noise.shape}")
    print(f"  Range: [{pink_noise.min():.3f}, {pink_noise.max():.3f}]")

    # Test EQ application
    print("\nTesting EQ application...")
    eq_params_raw = np.array([
        3.0, 200.0,           # Band 1: Low shelf
        2.0, 1000.0, 1.5,     # Band 2: Bell
        -1.0, 3000.0, 2.0,    # Band 3: Bell
        1.5, 8000.0, 1.0,     # Band 4: Bell
        2.0, 12000.0          # Band 5: High shelf
    ], dtype=np.float32)

    eq_params_norm = (eq_params_raw - eq_params_raw.mean()) / (eq_params_raw.std() + 1e-8)

    audio, eq = synthesizer.synthesize_single(eq_params_raw, eq_params_norm)

    print(f"  Audio shape: {audio.shape}")
    print(f"  EQ shape: {eq.shape}")
    print(f"  Audio range: [{audio.min():.3f}, {audio.max():.3f}]")

    # Test batch creation
    print("\nTesting dataset creation...")
    from dataclasses import dataclass

    @dataclass
    class DummyEQSetting:
        eq_params: np.ndarray
        eq_params_normalized: np.ndarray
        semantic_label: str

    dummy_settings = [
        DummyEQSetting(eq_params_raw, eq_params_norm, "warm"),
        DummyEQSetting(eq_params_raw, eq_params_norm, "bright"),
    ]

    dataset = synthesizer.create_dataset(dummy_settings, n_augmentations=2)

    print(f"\nDataset size: {len(dataset)}")
    print(f"First example:")
    print(f"  Audio: {dataset[0][0].shape}")
    print(f"  EQ: {dataset[0][1].shape}")
    print(f"  Label: {dataset[0][2]}")

    print("\n" + "="*70)
    print("SYNTHESIS TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    test_synthesis()
