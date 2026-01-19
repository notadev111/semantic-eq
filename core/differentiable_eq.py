"""
Differentiable EQ Module
========================

Wrapper around dasp-pytorch to provide differentiable parametric EQ
that matches the SAFE-DB 13-parameter format.

This enables end-to-end training with gradients flowing through the EQ application.

References:
    - dasp-pytorch: https://github.com/csteinmetz1/dasp-pytorch
    - Steinmetz et al. "Style Transfer of Audio Effects with Differentiable
      Signal Processing" (2022)
    - SAFE-DB: Stables et al. "SAFE: A system for the extraction and retrieval
      of semantic audio descriptors" ISMIR 2014

Usage:
    from core.differentiable_eq import DifferentiableEQ

    eq = DifferentiableEQ(sample_rate=44100)
    output = eq(audio, eq_params_13)  # Fully differentiable!

Requirements:
    pip install dasp-pytorch auraloss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

try:
    import dasp_pytorch
    import dasp_pytorch.functional as daspF
    DASP_AVAILABLE = True
except ImportError:
    DASP_AVAILABLE = False
    # Only print warning when module is actually used
    dasp_pytorch = None
    daspF = None


class DifferentiableEQ(nn.Module):
    """
    Differentiable 5-band parametric EQ matching SAFE-DB format.

    Converts SAFE-DB 13-parameter format to dasp-pytorch 18-parameter format
    and applies EQ differentiably.

    SAFE-DB format (13 params, denormalized):
        Band 1 (Low Shelf):  [Gain_dB, Freq_Hz]
        Band 2 (Bell):       [Gain_dB, Freq_Hz, Q]
        Band 3 (Bell):       [Gain_dB, Freq_Hz, Q]
        Band 4 (Bell):       [Gain_dB, Freq_Hz, Q]
        Band 5 (High Shelf): [Gain_dB, Freq_Hz]

    Normalized format (for training):
        All values in [0, 1] range
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        # Gain range (dB)
        min_gain_db: float = -12.0,
        max_gain_db: float = 12.0,
        # Frequency range (Hz) - log scale
        min_freq_hz: float = 20.0,
        max_freq_hz: float = 20000.0,
        # Q range - log scale
        min_q: float = 0.1,
        max_q: float = 10.0,
        # Default Q for shelf filters
        shelf_q: float = 0.707,
    ):
        super().__init__()

        if not DASP_AVAILABLE:
            raise ImportError("dasp-pytorch is required. Install with: pip install dasp-pytorch")

        self.sample_rate = sample_rate
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db
        self.min_freq_hz = min_freq_hz
        self.max_freq_hz = max_freq_hz
        self.min_q = min_q
        self.max_q = max_q
        self.shelf_q = shelf_q

        # Log frequency range for normalization
        self.log_min_freq = torch.log10(torch.tensor(min_freq_hz))
        self.log_max_freq = torch.log10(torch.tensor(max_freq_hz))
        self.log_min_q = torch.log10(torch.tensor(min_q))
        self.log_max_q = torch.log10(torch.tensor(max_q))

    def denormalize_params(self, params_norm: torch.Tensor) -> dict:
        """
        Convert normalized [0,1] parameters to dasp-pytorch format.

        Args:
            params_norm: [batch, 13] normalized parameters

        Returns:
            dict with dasp-pytorch parameter names
        """
        batch_size = params_norm.shape[0]
        device = params_norm.device

        # Extract normalized parameters
        # Band 1 (Low Shelf): [gain, freq]
        low_shelf_gain_norm = params_norm[:, 0]
        low_shelf_freq_norm = params_norm[:, 1]

        # Band 2 (Bell 0): [gain, freq, q]
        band0_gain_norm = params_norm[:, 2]
        band0_freq_norm = params_norm[:, 3]
        band0_q_norm = params_norm[:, 4]

        # Band 3 (Bell 1): [gain, freq, q]
        band1_gain_norm = params_norm[:, 5]
        band1_freq_norm = params_norm[:, 6]
        band1_q_norm = params_norm[:, 7]

        # Band 4 (Bell 2): [gain, freq, q]
        band2_gain_norm = params_norm[:, 8]
        band2_freq_norm = params_norm[:, 9]
        band2_q_norm = params_norm[:, 10]

        # Band 5 (High Shelf): [gain, freq]
        high_shelf_gain_norm = params_norm[:, 11]
        high_shelf_freq_norm = params_norm[:, 12]

        # Denormalize gains (linear scale)
        low_shelf_gain_db = low_shelf_gain_norm * (self.max_gain_db - self.min_gain_db) + self.min_gain_db
        band0_gain_db = band0_gain_norm * (self.max_gain_db - self.min_gain_db) + self.min_gain_db
        band1_gain_db = band1_gain_norm * (self.max_gain_db - self.min_gain_db) + self.min_gain_db
        band2_gain_db = band2_gain_norm * (self.max_gain_db - self.min_gain_db) + self.min_gain_db
        high_shelf_gain_db = high_shelf_gain_norm * (self.max_gain_db - self.min_gain_db) + self.min_gain_db

        # Denormalize frequencies (log scale)
        log_min = self.log_min_freq.to(device)
        log_max = self.log_max_freq.to(device)
        low_shelf_freq = 10 ** (low_shelf_freq_norm * (log_max - log_min) + log_min)
        band0_freq = 10 ** (band0_freq_norm * (log_max - log_min) + log_min)
        band1_freq = 10 ** (band1_freq_norm * (log_max - log_min) + log_min)
        band2_freq = 10 ** (band2_freq_norm * (log_max - log_min) + log_min)
        high_shelf_freq = 10 ** (high_shelf_freq_norm * (log_max - log_min) + log_min)

        # Denormalize Q (log scale)
        log_min_q = self.log_min_q.to(device)
        log_max_q = self.log_max_q.to(device)
        band0_q = 10 ** (band0_q_norm * (log_max_q - log_min_q) + log_min_q)
        band1_q = 10 ** (band1_q_norm * (log_max_q - log_min_q) + log_min_q)
        band2_q = 10 ** (band2_q_norm * (log_max_q - log_min_q) + log_min_q)

        # Default Q for shelf filters and unused band
        shelf_q = torch.full((batch_size,), self.shelf_q, device=device)
        band3_gain_db = torch.zeros(batch_size, device=device)  # Unused band
        band3_freq = torch.full((batch_size,), 1000.0, device=device)
        band3_q = torch.ones(batch_size, device=device)

        return {
            'low_shelf_gain_db': low_shelf_gain_db,
            'low_shelf_cutoff_freq': low_shelf_freq,
            'low_shelf_q_factor': shelf_q,
            'band0_gain_db': band0_gain_db,
            'band0_cutoff_freq': band0_freq,
            'band0_q_factor': band0_q,
            'band1_gain_db': band1_gain_db,
            'band1_cutoff_freq': band1_freq,
            'band1_q_factor': band1_q,
            'band2_gain_db': band2_gain_db,
            'band2_cutoff_freq': band2_freq,
            'band2_q_factor': band2_q,
            'band3_gain_db': band3_gain_db,
            'band3_cutoff_freq': band3_freq,
            'band3_q_factor': band3_q,
            'high_shelf_gain_db': high_shelf_gain_db,
            'high_shelf_cutoff_freq': high_shelf_freq,
            'high_shelf_q_factor': shelf_q,
        }

    def forward(
        self,
        audio: torch.Tensor,
        params_norm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply parametric EQ differentiably.

        Args:
            audio: [batch, channels, samples] input audio
            params_norm: [batch, 13] normalized EQ parameters

        Returns:
            output: [batch, channels, samples] EQ'd audio
        """
        # Convert to dasp format
        params = self.denormalize_params(params_norm)

        # Apply EQ using dasp-pytorch functional interface
        output = dasp_pytorch.functional.parametric_eq(
            audio,
            self.sample_rate,
            **params
        )

        return output


class SemanticEQLoss(nn.Module):
    """
    Loss function for semantic EQ training.

    Combines:
    1. Semantic consistency loss - output should be closer to target semantic
    2. Parameter supervision loss - predicted params should match SAFE-DB ground truth
    3. Audio quality loss - prevent extreme/unnatural EQ
    """

    def __init__(
        self,
        semantic_weight: float = 1.0,
        param_weight: float = 0.5,
        quality_weight: float = 0.1,
        use_stft_loss: bool = True,
    ):
        super().__init__()

        self.semantic_weight = semantic_weight
        self.param_weight = param_weight
        self.quality_weight = quality_weight

        if use_stft_loss:
            try:
                from auraloss.freq import MultiResolutionSTFTLoss
                self.stft_loss = MultiResolutionSTFTLoss(
                    fft_sizes=[512, 1024, 2048],
                    hop_sizes=[256, 512, 1024],
                    win_lengths=[512, 1024, 2048],
                    w_sc=0.0,
                    w_phs=0.0,
                    w_lin_mag=1.0,
                    w_log_mag=1.0,
                )
            except ImportError:
                print("Warning: auraloss not installed. Using MSE for quality loss.")
                self.stft_loss = None
        else:
            self.stft_loss = None

    def forward(
        self,
        z_output: torch.Tensor,
        z_target: torch.Tensor,
        params_pred: Optional[torch.Tensor] = None,
        params_gt: Optional[torch.Tensor] = None,
        audio_output: Optional[torch.Tensor] = None,
        audio_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.

        Args:
            z_output: [batch, latent_dim] encoded output audio
            z_target: [batch, latent_dim] target semantic embedding
            params_pred: [batch, 13] predicted EQ parameters (optional)
            params_gt: [batch, 13] ground truth EQ parameters (optional)
            audio_output: [batch, channels, samples] output audio (optional)
            audio_input: [batch, channels, samples] input audio (optional)

        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        loss_dict = {}

        # 1. Semantic consistency loss (always computed)
        semantic_loss = F.mse_loss(z_output, z_target)
        loss_dict['semantic'] = semantic_loss.item()

        total_loss = self.semantic_weight * semantic_loss

        # 2. Parameter supervision loss (optional)
        if params_pred is not None and params_gt is not None and self.param_weight > 0:
            param_loss = F.mse_loss(params_pred, params_gt)
            loss_dict['param'] = param_loss.item()
            total_loss = total_loss + self.param_weight * param_loss

        # 3. Audio quality loss (optional)
        if audio_output is not None and audio_input is not None and self.quality_weight > 0:
            if self.stft_loss is not None:
                quality_loss = self.stft_loss(audio_output, audio_input)
            else:
                quality_loss = F.mse_loss(audio_output, audio_input)
            loss_dict['quality'] = quality_loss.item()
            total_loss = total_loss + self.quality_weight * quality_loss

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


def test_differentiable_eq():
    """Test the differentiable EQ module."""
    print("Testing DifferentiableEQ...")

    if not DASP_AVAILABLE:
        print("dasp-pytorch not installed, skipping test")
        return

    # Create module
    eq = DifferentiableEQ(sample_rate=44100)

    # Create test data
    batch_size = 4
    audio = torch.randn(batch_size, 1, 88200)  # 2 seconds mono
    params = torch.rand(batch_size, 13)  # Random normalized params

    # Ensure requires_grad
    params.requires_grad_(True)

    # Forward pass
    output = eq(audio, params)

    print(f"  Input shape: {audio.shape}")
    print(f"  Params shape: {params.shape}")
    print(f"  Output shape: {output.shape}")

    # Test gradient flow
    loss = output.mean()
    loss.backward()

    if params.grad is not None:
        print(f"  Gradients computed: {params.grad.shape}")
        print("  [OK] Gradients flow through EQ!")
    else:
        print("  [FAIL] No gradients!")

    print("Test complete!")


if __name__ == "__main__":
    test_differentiable_eq()
