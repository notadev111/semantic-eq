"""
Adaptive EQ Generation Module
==============================

Implements adaptive semantic EQ generation using latent space traversal.

Key Concept:
    z_final = z_audio + intensity * (z_target - z_audio)

    where:
    - z_audio: Current audio characteristics (from Audio Encoder)
    - z_target: Target semantic embedding (e.g., "warm")
    - intensity: How much to traverse (0=no change, 1=full semantic target)

Usage:
    from core.adaptive_eq_generator import AdaptiveEQGenerator

    generator = AdaptiveEQGenerator(
        v2_model_path='neural_eq_safedb_v2.pt',
        audio_encoder_path='audio_encoder_best.pt'
    )

    eq_params = generator.generate_adaptive_eq(
        audio_input,
        semantic_target='warm',
        intensity=0.7
    )
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

from core.neural_eq_morphing_safedb_v2 import NeuralEQMorphingSAFEDBV2
from core.audio_encoder import FastAudioEncoder, AudioEncoderConfig


class AdaptiveEQGenerator:
    """
    Generate adaptive semantic EQ based on input audio characteristics

    The system:
    1. Analyzes input audio → z_audio
    2. Gets target semantic embedding → z_target
    3. Computes adaptive EQ: z_final = z_audio + intensity * (z_target - z_audio)
    4. Decodes to EQ parameters
    """

    def __init__(self,
                 v2_model_path: str = 'neural_eq_safedb_v2.pt',
                 audio_encoder_path: str = 'audio_encoder_best.pt',
                 audio_encoder_config: dict = None,
                 device: str = None):
        """
        Args:
            v2_model_path: Path to pre-trained V2 model (EQ encoder/decoder)
            audio_encoder_path: Path to trained Audio Encoder
            audio_encoder_config: Audio Encoder configuration
            device: Device to run on (auto-detect if None)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load V2 system (EQ encoder/decoder)
        print(f"Loading V2 system: {v2_model_path}")
        self.v2_system = NeuralEQMorphingSAFEDBV2()
        self.v2_system.load_model(v2_model_path)
        self.v2_system.load_dataset(min_examples=5)
        self.v2_system.encoder.eval()
        self.v2_system.decoder.eval()

        print(f"  Available semantic terms: {list(self.v2_system.term_to_idx.keys())}")

        # Load Audio Encoder
        print(f"\nLoading Audio Encoder: {audio_encoder_path}")
        if audio_encoder_config is None:
            audio_encoder_config = AudioEncoderConfig.STANDARD

        self.audio_encoder = FastAudioEncoder(**audio_encoder_config).to(self.device)

        # Load weights
        if Path(audio_encoder_path).exists():
            checkpoint = torch.load(audio_encoder_path, map_location=self.device)
            self.audio_encoder.load_state_dict(checkpoint['audio_encoder_state_dict'])
            print(f"  Loaded trained weights")
        else:
            print(f"  WARNING: Audio Encoder checkpoint not found, using random weights!")

        self.audio_encoder.eval()

        # Pre-compute semantic embeddings for all terms
        self._precompute_semantic_embeddings()

    def _precompute_semantic_embeddings(self):
        """
        Pre-compute latent embeddings for all semantic terms

        This allows fast lookup during inference instead of encoding each time.
        """
        print(f"\nPre-computing semantic embeddings...")

        self.semantic_embeddings = {}

        with torch.no_grad():
            for term, idx in self.v2_system.term_to_idx.items():
                # Get all EQ settings for this term
                term_settings = [s for s in self.v2_system.eq_settings if s.semantic_label == term]

                if not term_settings:
                    continue

                # Encode all settings and average
                term_latents = []
                for setting in term_settings:
                    eq_params = torch.FloatTensor(setting.eq_params_normalized).unsqueeze(0).to(self.device)
                    z, _ = self.v2_system.encoder(eq_params)
                    term_latents.append(z)

                # Average latent (centroid of semantic term)
                avg_latent = torch.stack(term_latents).mean(dim=0)  # [1, 32]
                self.semantic_embeddings[term] = avg_latent

        print(f"  Computed {len(self.semantic_embeddings)} semantic embeddings")

    def analyze_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Analyze audio and return latent representation

        Args:
            audio: [batch, channels, samples] or [channels, samples]

        Returns:
            z_audio: [batch, 32] latent vector
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)  # Add batch dimension

        audio = audio.to(self.device)

        with torch.no_grad():
            z_audio = self.audio_encoder(audio)

        return z_audio

    def compute_semantic_similarity(self, z_audio: torch.Tensor, semantic_term: str) -> float:
        """
        Compute similarity between audio and semantic term

        Returns value in [0, 1] where:
        - 1.0 = audio already matches semantic target perfectly
        - 0.0 = audio is very different from semantic target

        Args:
            z_audio: [1, 32] audio latent vector
            semantic_term: Target semantic term (e.g., 'warm')

        Returns:
            similarity: Cosine similarity in [0, 1]
        """
        if semantic_term not in self.semantic_embeddings:
            raise ValueError(f"Unknown semantic term: {semantic_term}")

        z_target = self.semantic_embeddings[semantic_term]

        # Cosine similarity
        similarity = F.cosine_similarity(z_audio, z_target, dim=1).item()

        # Map from [-1, 1] to [0, 1]
        similarity = (similarity + 1) / 2

        return similarity

    def generate_adaptive_eq(self,
                           audio: torch.Tensor,
                           semantic_target: str,
                           intensity: float = 1.0,
                           return_similarity: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        """
        Generate adaptive EQ based on input audio and semantic target

        Process:
            1. Encode audio → z_audio
            2. Get target embedding → z_target
            3. Compute difference: delta_z = z_target - z_audio
            4. Scale by intensity: delta_z_scaled = intensity * delta_z
            5. Traverse latent space: z_final = z_audio + delta_z_scaled
            6. Decode to EQ parameters

        Args:
            audio: [channels, samples] or [batch, channels, samples]
            semantic_target: Target semantic term (e.g., 'warm', 'bright')
            intensity: Traversal intensity in [0, 1]
                      0 = no change (stay at z_audio)
                      1 = full semantic target (z_target)
            return_similarity: If True, also return semantic similarity

        Returns:
            eq_params_13: [13] EQ parameters (denormalized)
            similarity (optional): Current similarity to target [0, 1]
        """
        if semantic_target not in self.semantic_embeddings:
            raise ValueError(f"Unknown semantic term: {semantic_target}. "
                           f"Available: {list(self.semantic_embeddings.keys())}")

        # Step 1: Analyze input audio
        z_audio = self.analyze_audio(audio)  # [1, 32]

        # Step 2: Get target semantic embedding
        z_target = self.semantic_embeddings[semantic_target]  # [1, 32]

        # Step 3: Compute semantic similarity (optional)
        if return_similarity:
            similarity = self.compute_semantic_similarity(z_audio, semantic_target)

        # Step 4: Compute difference vector
        delta_z = z_target - z_audio  # [1, 32]

        # Step 5: Scale by intensity
        delta_z_scaled = intensity * delta_z

        # Step 6: Traverse latent space
        z_final = z_audio + delta_z_scaled  # [1, 32]

        # Step 7: Decode to EQ parameters
        with torch.no_grad():
            eq_params_norm = self.v2_system.decoder(z_final)  # [1, 13]

        # Denormalize
        eq_params_norm_np = eq_params_norm.cpu().numpy()[0]
        eq_params_13 = self.v2_system.loader.denormalize_params(eq_params_norm_np)

        if return_similarity:
            return eq_params_13, similarity
        else:
            return eq_params_13

    def interpolate_semantics(self,
                            audio: torch.Tensor,
                            term1: str,
                            term2: str,
                            alpha: float = 0.5,
                            intensity: float = 1.0) -> np.ndarray:
        """
        Generate EQ that blends two semantic targets

        Args:
            audio: Input audio
            term1: First semantic term
            term2: Second semantic term
            alpha: Blend factor [0, 1]
                   0 = term1 only
                   1 = term2 only
            intensity: Overall intensity of EQ application

        Returns:
            eq_params_13: [13] EQ parameters
        """
        if term1 not in self.semantic_embeddings or term2 not in self.semantic_embeddings:
            raise ValueError(f"Unknown semantic term")

        # Analyze audio
        z_audio = self.analyze_audio(audio)

        # Get semantic embeddings
        z_term1 = self.semantic_embeddings[term1]
        z_term2 = self.semantic_embeddings[term2]

        # Interpolate semantic targets
        z_target = (1 - alpha) * z_term1 + alpha * z_term2

        # Compute adaptive EQ
        delta_z = z_target - z_audio
        z_final = z_audio + intensity * delta_z

        # Decode
        with torch.no_grad():
            eq_params_norm = self.v2_system.decoder(z_final)

        eq_params_13 = self.v2_system.loader.denormalize_params(
            eq_params_norm.cpu().numpy()[0]
        )

        return eq_params_13

    def get_semantic_profile(self, audio: torch.Tensor, top_k: int = 5) -> list:
        """
        Analyze audio and return top-k matching semantic terms

        Args:
            audio: Input audio
            top_k: Number of top terms to return

        Returns:
            List of (term, similarity) tuples, sorted by similarity
        """
        z_audio = self.analyze_audio(audio)

        similarities = {}
        for term in self.semantic_embeddings.keys():
            sim = self.compute_semantic_similarity(z_audio, term)
            similarities[term] = sim

        # Sort by similarity
        sorted_terms = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        return sorted_terms[:top_k]

    def suggest_intensity(self, audio: torch.Tensor, semantic_target: str) -> float:
        """
        Suggest optimal intensity based on current audio characteristics

        If audio is already similar to target → lower intensity
        If audio is very different → higher intensity

        Args:
            audio: Input audio
            semantic_target: Target semantic term

        Returns:
            suggested_intensity: Value in [0.3, 1.0]
        """
        similarity = self.compute_semantic_similarity(
            self.analyze_audio(audio),
            semantic_target
        )

        # If already similar (>0.7), use lower intensity
        if similarity > 0.7:
            return 0.3 + 0.4 * (1 - similarity)  # [0.3, 0.42]
        else:
            # If very different, use higher intensity
            return 0.5 + 0.5 * (1 - similarity)  # [0.5, 1.0]


def test_adaptive_generator():
    """Test Adaptive EQ Generator"""

    print("="*70)
    print("TESTING ADAPTIVE EQ GENERATOR")
    print("="*70)

    # Check if models exist
    v2_path = Path('neural_eq_safedb_v2.pt')
    audio_encoder_path = Path('audio_encoder_best.pt')

    if not v2_path.exists():
        print(f"\nERROR: V2 model not found: {v2_path}")
        print("Please train V2 model first:")
        print("  python train_neural_eq_v2.py")
        return

    if not audio_encoder_path.exists():
        print(f"\nWARNING: Audio Encoder not found: {audio_encoder_path}")
        print("Using random weights for testing...")

    # Create generator
    generator = AdaptiveEQGenerator(
        v2_model_path=str(v2_path),
        audio_encoder_path=str(audio_encoder_path)
    )

    # Test with synthetic audio
    print("\n" + "="*70)
    print("TEST 1: Generate Adaptive EQ")
    print("="*70)

    audio = torch.randn(1, 1, 88200)  # 2 seconds mono

    eq_params, similarity = generator.generate_adaptive_eq(
        audio,
        semantic_target='warm',
        intensity=0.7,
        return_similarity=True
    )

    print(f"\nInput audio: {audio.shape}")
    print(f"Target: 'warm' with intensity 0.7")
    print(f"Current similarity to 'warm': {similarity:.3f}")
    print(f"\nGenerated EQ parameters:")
    print(f"  Band 1 (Low Shelf):  Gain={eq_params[0]:6.2f}dB, Freq={eq_params[1]:7.1f}Hz")
    print(f"  Band 2 (Bell):       Gain={eq_params[2]:6.2f}dB, Freq={eq_params[3]:7.1f}Hz, Q={eq_params[4]:.2f}")
    print(f"  Band 3 (Bell):       Gain={eq_params[5]:6.2f}dB, Freq={eq_params[6]:7.1f}Hz, Q={eq_params[7]:.2f}")
    print(f"  Band 4 (Bell):       Gain={eq_params[8]:6.2f}dB, Freq={eq_params[9]:7.1f}Hz, Q={eq_params[10]:.2f}")
    print(f"  Band 5 (High Shelf): Gain={eq_params[11]:6.2f}dB, Freq={eq_params[12]:7.1f}Hz")

    # Test semantic profile
    print("\n" + "="*70)
    print("TEST 2: Semantic Profile Analysis")
    print("="*70)

    profile = generator.get_semantic_profile(audio, top_k=5)
    print(f"\nTop 5 semantic matches for input audio:")
    for i, (term, sim) in enumerate(profile, 1):
        print(f"  {i}. {term:15s}: {sim:.3f}")

    # Test intensity suggestion
    print("\n" + "="*70)
    print("TEST 3: Intensity Suggestion")
    print("="*70)

    suggested = generator.suggest_intensity(audio, 'warm')
    print(f"\nSuggested intensity for 'warm': {suggested:.2f}")

    print("\n" + "="*70)
    print("ADAPTIVE EQ GENERATOR TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    test_adaptive_generator()
