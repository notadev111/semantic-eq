"""
Real-Time Streaming Adaptive EQ Processor
=========================================

Implements real-time adaptive EQ with frame-based processing and parameter smoothing.

Key Features:
- Frame-based processing (update EQ every N frames)
- Analysis buffer (maintain context)
- Parameter smoothing (avoid clicks/pops)
- Target latency: 2-5ms per frame

Architecture:
    Audio Input → Analysis Buffer → Audio Encoder (every N frames)
                                 ↓
                            Latent Traversal
                                 ↓
                            EQ Parameters → Smooth → Apply EQ

Usage:
    processor = StreamingAdaptiveEQ(
        v2_model_path='neural_eq_safedb_v2.pt',
        audio_encoder_path='audio_encoder_best.pt'
    )

    # Set semantic target
    processor.set_target('warm', intensity=0.7)

    # Process audio frames
    for frame in audio_stream:
        processed_frame = processor.process_frame(frame)
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy import signal
from typing import Optional, Tuple
from collections import deque
import warnings
warnings.filterwarnings("ignore")

from core.adaptive_eq_generator import AdaptiveEQGenerator


class ParameterSmoother:
    """
    Smooth EQ parameter transitions to avoid clicks/pops

    Uses exponential moving average:
        current = alpha * current + (1 - alpha) * target

    where alpha controls smoothing time constant
    """

    def __init__(self, n_params: int = 13, smoothing_factor: float = 0.9):
        """
        Args:
            n_params: Number of parameters to smooth
            smoothing_factor: Smoothing coefficient [0, 1]
                            0 = instant change (no smoothing)
                            0.9 = smooth (10 frame time constant)
                            0.95 = very smooth (20 frame time constant)
        """
        self.n_params = n_params
        self.alpha = smoothing_factor

        self.current_params = None

    def reset(self, initial_params: np.ndarray):
        """Reset smoother with initial parameters"""
        self.current_params = initial_params.copy()

    def smooth(self, target_params: np.ndarray) -> np.ndarray:
        """
        Smooth transition from current to target parameters

        Args:
            target_params: [13] target EQ parameters

        Returns:
            smoothed_params: [13] smoothed EQ parameters
        """
        if self.current_params is None:
            self.current_params = target_params.copy()
            return self.current_params

        # Exponential moving average
        self.current_params = self.alpha * self.current_params + (1 - self.alpha) * target_params

        return self.current_params.copy()


class BiquadEQProcessor:
    """
    Real-time biquad EQ processor

    Applies 5-band EQ using cascaded biquad filters with state preservation
    for continuous processing.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

        # Filter states for continuous processing
        self.filter_states = [None] * 5  # 5 bands

    def reset(self):
        """Reset filter states"""
        self.filter_states = [None] * 5

    @staticmethod
    def _design_low_shelf(gain_db: float, freq_hz: float, q: float, sr: int):
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
    def _design_high_shelf(gain_db: float, freq_hz: float, q: float, sr: int):
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
    def _design_peaking_eq(gain_db: float, freq_hz: float, q: float, sr: int):
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

    def process_frame(self, audio_frame: np.ndarray, eq_params_13: np.ndarray) -> np.ndarray:
        """
        Process single audio frame with EQ

        Args:
            audio_frame: [n_channels, n_samples] audio frame
            eq_params_13: [13] SAFE-DB EQ parameters

        Returns:
            processed_frame: [n_channels, n_samples] EQ'd audio
        """
        # Extract parameters
        gain1, freq1 = eq_params_13[0], eq_params_13[1]
        gain2, freq2, q2 = eq_params_13[2], eq_params_13[3], eq_params_13[4]
        gain3, freq3, q3 = eq_params_13[5], eq_params_13[6], eq_params_13[7]
        gain4, freq4, q4 = eq_params_13[8], eq_params_13[9], eq_params_13[10]
        gain5, freq5 = eq_params_13[11], eq_params_13[12]

        q1 = 0.707
        q5 = 0.707

        # Clip frequencies
        nyquist = self.sample_rate / 2
        freq1 = np.clip(freq1, 20, nyquist - 1000)
        freq2 = np.clip(freq2, 20, nyquist - 1000)
        freq3 = np.clip(freq3, 20, nyquist - 1000)
        freq4 = np.clip(freq4, 20, nyquist - 1000)
        freq5 = np.clip(freq5, 20, nyquist - 1000)

        processed = audio_frame.copy()

        # Process each channel
        n_channels = processed.shape[0]

        for ch in range(n_channels):
            # Band 1: Low Shelf
            if abs(gain1) > 0.01:
                b, a = self._design_low_shelf(gain1, freq1, q1, self.sample_rate)
                processed[ch], self.filter_states[0] = signal.lfilter(
                    b, a, processed[ch], zi=self.filter_states[0]
                )

            # Band 2: Bell
            if abs(gain2) > 0.01:
                b, a = self._design_peaking_eq(gain2, freq2, q2, self.sample_rate)
                processed[ch], self.filter_states[1] = signal.lfilter(
                    b, a, processed[ch], zi=self.filter_states[1]
                )

            # Band 3: Bell
            if abs(gain3) > 0.01:
                b, a = self._design_peaking_eq(gain3, freq3, q3, self.sample_rate)
                processed[ch], self.filter_states[2] = signal.lfilter(
                    b, a, processed[ch], zi=self.filter_states[2]
                )

            # Band 4: Bell
            if abs(gain4) > 0.01:
                b, a = self._design_peaking_eq(gain4, freq4, q4, self.sample_rate)
                processed[ch], self.filter_states[3] = signal.lfilter(
                    b, a, processed[ch], zi=self.filter_states[3]
                )

            # Band 5: High Shelf
            if abs(gain5) > 0.01:
                b, a = self._design_high_shelf(gain5, freq5, q5, self.sample_rate)
                processed[ch], self.filter_states[4] = signal.lfilter(
                    b, a, processed[ch], zi=self.filter_states[4]
                )

        return processed


class StreamingAdaptiveEQ:
    """
    Real-Time Streaming Adaptive EQ Processor

    Processes audio in frames, updating EQ parameters adaptively based on
    input audio characteristics and semantic target.
    """

    def __init__(self,
                 v2_model_path: str = 'neural_eq_safedb_v2.pt',
                 audio_encoder_path: str = 'audio_encoder_best.pt',
                 sample_rate: int = 44100,
                 frame_size: int = 512,
                 update_interval: int = 4,
                 smoothing_factor: float = 0.9):
        """
        Args:
            v2_model_path: Path to V2 model
            audio_encoder_path: Path to Audio Encoder
            sample_rate: Sample rate (Hz)
            frame_size: Frame size in samples
            update_interval: Update EQ every N frames (e.g., 4 = ~23ms @ 44.1kHz)
            smoothing_factor: Parameter smoothing coefficient [0, 1]
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.update_interval = update_interval

        # Load adaptive EQ generator
        print(f"Initializing Streaming Adaptive EQ...")
        self.generator = AdaptiveEQGenerator(
            v2_model_path=v2_model_path,
            audio_encoder_path=audio_encoder_path
        )

        # EQ processor
        self.eq_processor = BiquadEQProcessor(sample_rate=sample_rate)

        # Parameter smoother
        self.param_smoother = ParameterSmoother(n_params=13, smoothing_factor=smoothing_factor)

        # Analysis buffer (2 seconds for Audio Encoder)
        self.analysis_buffer_size = int(sample_rate * 2.0)
        self.analysis_buffer = deque(maxlen=self.analysis_buffer_size)

        # Frame counter
        self.frame_count = 0

        # Current target
        self.semantic_target = None
        self.intensity = 1.0
        self.bypass = False

        # Current EQ parameters
        self.current_eq_params = np.zeros(13, dtype=np.float32)
        self.target_eq_params = np.zeros(13, dtype=np.float32)

        print(f"  Frame size: {frame_size} samples ({frame_size/sample_rate*1000:.2f}ms)")
        print(f"  Update interval: {update_interval} frames ({update_interval*frame_size/sample_rate*1000:.2f}ms)")
        print(f"  Ready for streaming!")

    def set_target(self, semantic_target: str, intensity: float = 1.0):
        """
        Set semantic target and intensity

        Args:
            semantic_target: Target semantic term (e.g., 'warm', 'bright')
            intensity: EQ intensity [0, 1]
        """
        if semantic_target not in self.generator.semantic_embeddings:
            raise ValueError(f"Unknown semantic target: {semantic_target}")

        self.semantic_target = semantic_target
        self.intensity = np.clip(intensity, 0.0, 1.0)
        self.bypass = False

        print(f"\nTarget set: '{semantic_target}' with intensity {intensity:.2f}")

    def set_bypass(self, bypass: bool):
        """Enable/disable bypass"""
        self.bypass = bypass

    def reset(self):
        """Reset processor state"""
        self.analysis_buffer.clear()
        self.frame_count = 0
        self.eq_processor.reset()
        self.current_eq_params = np.zeros(13, dtype=np.float32)
        self.target_eq_params = np.zeros(13, dtype=np.float32)

    def process_frame(self, audio_frame: np.ndarray) -> np.ndarray:
        """
        Process single audio frame

        Args:
            audio_frame: [n_channels, frame_size] audio frame

        Returns:
            processed_frame: [n_channels, frame_size] processed audio
        """
        # Bypass if no target set or bypass enabled
        if self.bypass or self.semantic_target is None:
            return audio_frame.copy()

        # Add to analysis buffer (convert to mono for analysis)
        if audio_frame.shape[0] > 1:
            mono_frame = audio_frame.mean(axis=0)
        else:
            mono_frame = audio_frame[0]

        self.analysis_buffer.extend(mono_frame)

        # Update EQ parameters every N frames
        if self.frame_count % self.update_interval == 0 and len(self.analysis_buffer) >= self.analysis_buffer_size:
            # Get buffered audio for analysis
            analysis_audio = np.array(list(self.analysis_buffer), dtype=np.float32)
            analysis_audio_tensor = torch.FloatTensor(analysis_audio).unsqueeze(0).unsqueeze(0)  # [1, 1, samples]

            # Generate adaptive EQ
            self.target_eq_params = self.generator.generate_adaptive_eq(
                analysis_audio_tensor,
                semantic_target=self.semantic_target,
                intensity=self.intensity
            )

        # Smooth parameter transitions
        self.current_eq_params = self.param_smoother.smooth(self.target_eq_params)

        # Apply EQ
        processed_frame = self.eq_processor.process_frame(audio_frame, self.current_eq_params)

        self.frame_count += 1

        return processed_frame

    def get_current_params(self) -> np.ndarray:
        """Get current EQ parameters"""
        return self.current_eq_params.copy()

    def get_latency_ms(self) -> float:
        """Estimate total latency in milliseconds"""
        # Frame processing + EQ computation + update interval
        frame_latency = self.frame_size / self.sample_rate * 1000
        update_latency = self.update_interval * frame_latency
        total_latency = frame_latency + update_latency / 2  # Average
        return total_latency


def test_streaming_eq():
    """Test streaming adaptive EQ"""

    print("="*70)
    print("TESTING STREAMING ADAPTIVE EQ")
    print("="*70)

    from pathlib import Path

    # Check models
    v2_path = Path('neural_eq_safedb_v2.pt')
    audio_encoder_path = Path('audio_encoder_best.pt')

    if not v2_path.exists():
        print(f"\nERROR: V2 model not found: {v2_path}")
        return

    if not audio_encoder_path.exists():
        print(f"\nWARNING: Audio Encoder not found, using random weights")

    # Create processor
    processor = StreamingAdaptiveEQ(
        v2_model_path=str(v2_path),
        audio_encoder_path=str(audio_encoder_path),
        frame_size=512,
        update_interval=4
    )

    print(f"\nEstimated latency: {processor.get_latency_ms():.2f}ms")

    # Set target
    processor.set_target('warm', intensity=0.7)

    # Simulate streaming
    print("\nSimulating 2 seconds of streaming audio...")

    n_frames = int(2.0 * 44100 / 512)  # 2 seconds
    frame_size = 512

    for i in range(n_frames):
        # Generate random audio frame (stereo)
        audio_frame = np.random.randn(2, frame_size).astype(np.float32) * 0.1

        # Process
        processed_frame = processor.process_frame(audio_frame)

        # Show params every 20 frames
        if i % 20 == 0:
            params = processor.get_current_params()
            print(f"  Frame {i:3d}: Band1_Gain={params[0]:+5.2f}dB, "
                  f"Band2_Gain={params[2]:+5.2f}dB, "
                  f"Band5_Gain={params[11]:+5.2f}dB")

    print("\n" + "="*70)
    print("STREAMING TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    test_streaming_eq()
