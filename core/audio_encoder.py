"""
Audio Encoder for Adaptive Semantic EQ
======================================

Maps audio input to 32D latent space (same space as EQ Encoder).

Architecture:
    1. Mel Spectrogram (64 bins, 1024 FFT)
    2. Depthwise Separable Convolutions (MobileNet-style)
    3. Global Average Pooling
    4. Projection to 32D latent space

Optimized for real-time performance (<2ms inference time).

Usage:
    encoder = FastAudioEncoder(latent_dim=32)
    audio = torch.randn(1, 1, 44100)  # 1 second mono
    z_audio = encoder(audio)  # [1, 32]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import Tuple


class DepthwiseSeparableConv2d(nn.Module):
    """
    Depthwise Separable Convolution (MobileNet-style)

    Standard conv: C_in × C_out × K_h × K_w operations
    Depthwise separable: (C_in × K_h × K_w) + (C_in × C_out × 1 × 1)

    Result: ~10x fewer operations!
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()

        # Depthwise: Each input channel processed separately
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels  # Key: groups = in_channels
        )

        # Pointwise: 1x1 conv to mix channels
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class FastAudioEncoder(nn.Module):
    """
    Lightweight Audio Encoder for Real-Time Adaptive EQ

    Input: Audio waveform [batch, channels, samples]
    Output: 32D latent vector [batch, latent_dim]

    Design choices for speed:
    - Mel spectrogram (64 bins) instead of full STFT
    - Depthwise separable convolutions (10x faster)
    - Average pooling instead of max (smoother, faster)
    - Small model (few channels)

    Target latency: <2ms on CPU
    """

    def __init__(self,
                 latent_dim: int = 32,
                 sample_rate: int = 44100,
                 n_fft: int = 1024,
                 hop_length: int = 512,
                 n_mels: int = 64,
                 audio_duration: float = 2.0):
        """
        Args:
            latent_dim: Output dimension (must match EQ encoder = 32)
            sample_rate: Audio sample rate
            n_fft: FFT size (reduced from typical 2048 for speed)
            hop_length: Hop length for STFT
            n_mels: Number of mel bins (reduced from typical 128 for speed)
            audio_duration: Analysis window in seconds
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.sample_rate = sample_rate
        self.audio_duration = audio_duration
        self.n_samples = int(sample_rate * audio_duration)

        # Mel Spectrogram (lightweight)
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
            normalized=True
        )

        # Log scaling for better dynamic range
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')

        # CNN Feature Extractor (depthwise separable for speed)
        # Input: [batch, 1, n_mels=64, time_frames]
        self.conv1 = DepthwiseSeparableConv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.AvgPool2d((2, 2))  # [32, 32, time/2]

        self.conv2 = DepthwiseSeparableConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.AvgPool2d((2, 2))  # [64, 16, time/4]

        self.conv3 = DepthwiseSeparableConv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.AvgPool2d((2, 2))  # [128, 8, time/8]

        # Global Average Pooling (temporal and frequency)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Projection to latent space
        self.to_latent = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, latent_dim),
            nn.Tanh()  # Match EQ encoder's bounded latent space
        )

    def preprocess_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Preprocess audio waveform to fixed length

        Args:
            audio: [batch, channels, samples] or [batch, samples]

        Returns:
            audio: [batch, 1, n_samples] (mono, fixed length)
        """
        # Handle channel dimension
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # [batch, 1, samples]
        elif audio.dim() == 3 and audio.shape[1] > 1:
            # Convert to mono if stereo
            audio = audio.mean(dim=1, keepdim=True)

        batch_size = audio.shape[0]
        current_samples = audio.shape[2]

        # Pad or truncate to fixed length
        if current_samples < self.n_samples:
            # Pad with zeros
            padding = self.n_samples - current_samples
            audio = F.pad(audio, (0, padding))
        elif current_samples > self.n_samples:
            # Take center crop
            start = (current_samples - self.n_samples) // 2
            audio = audio[:, :, start:start + self.n_samples]

        return audio

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode audio to latent vector

        Args:
            audio: [batch, channels, samples] raw waveform

        Returns:
            z_audio: [batch, latent_dim] latent vector
        """
        # Preprocess to fixed length mono
        audio = self.preprocess_audio(audio)  # [batch, 1, n_samples]

        # Extract mel spectrogram
        mel = self.mel_spec(audio)  # [batch, 1, n_mels, time_frames]
        mel_db = self.amplitude_to_db(mel)  # Convert to dB scale

        # Normalize to [-1, 1] range
        mel_db = (mel_db + 80) / 80  # Assume range [-80, 0] dB
        mel_db = torch.clamp(mel_db, -1, 1)

        # CNN feature extraction
        x = self.conv1(mel_db)  # [batch, 32, n_mels/2, time/2]
        x = self.pool1(x)

        x = self.conv2(x)  # [batch, 64, n_mels/4, time/4]
        x = self.pool2(x)

        x = self.conv3(x)  # [batch, 128, n_mels/8, time/8]
        x = self.pool3(x)

        # Global pooling
        x = self.global_pool(x)  # [batch, 128, 1, 1]
        x = x.view(x.size(0), -1)  # [batch, 128]

        # Project to latent space
        z_audio = self.to_latent(x)  # [batch, latent_dim]

        return z_audio

    def get_num_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AudioEncoderConfig:
    """Configuration for Audio Encoder"""

    # Standard configuration (real-time optimized)
    STANDARD = {
        'latent_dim': 32,
        'sample_rate': 44100,
        'n_fft': 1024,
        'hop_length': 512,
        'n_mels': 64,
        'audio_duration': 2.0
    }

    # High quality (slower, more detailed)
    HIGH_QUALITY = {
        'latent_dim': 32,
        'sample_rate': 44100,
        'n_fft': 2048,
        'hop_length': 256,
        'n_mels': 128,
        'audio_duration': 3.0
    }

    # Ultra-fast (for embedded systems)
    ULTRA_FAST = {
        'latent_dim': 32,
        'sample_rate': 44100,
        'n_fft': 512,
        'hop_length': 512,
        'n_mels': 32,
        'audio_duration': 1.0
    }


def test_audio_encoder():
    """Test Audio Encoder with dummy data"""

    print("="*70)
    print("TESTING AUDIO ENCODER")
    print("="*70)

    # Create encoder
    encoder = FastAudioEncoder(**AudioEncoderConfig.STANDARD)
    print(f"\nModel parameters: {encoder.get_num_parameters():,}")

    # Test with different input shapes
    test_cases = [
        (1, 1, 44100),      # 1 second mono
        (4, 1, 88200),      # 2 seconds mono (batch=4)
        (2, 2, 132300),     # 3 seconds stereo
        (1, 1, 22050),      # 0.5 seconds (will be padded)
    ]

    print("\nTesting input shapes:")
    for i, shape in enumerate(test_cases, 1):
        audio = torch.randn(*shape)

        # Time inference
        import time
        start = time.time()
        with torch.no_grad():
            z_audio = encoder(audio)
        elapsed = (time.time() - start) * 1000  # Convert to ms

        print(f"\n  Test {i}: Input {audio.shape}")
        print(f"    Output: {z_audio.shape}")
        print(f"    Latency: {elapsed:.2f}ms")

        # Check output range (should be bounded by tanh)
        print(f"    Range: [{z_audio.min().item():.3f}, {z_audio.max().item():.3f}]")

    print("\n" + "="*70)
    print("AUDIO ENCODER TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    test_audio_encoder()
