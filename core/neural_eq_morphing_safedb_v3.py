"""
Neural EQ Morphing System V3 - SAFE-DB Dataset
===============================================

FIXED NORMALIZATION BOUNDS - addresses V2 limited frequency range issue.

Key improvements from V2:
1. FIXED normalization bounds instead of data-dependent
   - Frequencies: 20-20,000 Hz (full audible range)
   - Q values: 0.1-10 (standard EQ range)
   - Gains: -12 to +12 dB (unchanged)
2. Allows model to utilize full frequency spectrum
3. V2 only generated 83-7,901 Hz (dataset had up to 20,000 Hz!)

All other V2 improvements retained (log-scale, annealing, etc.)

Usage:
    from core.neural_eq_morphing_safedb_v3 import NeuralEQMorphingSAFEDBV3

    system = NeuralEQMorphingSAFEDBV3()
    system.load_dataset()
    system.train(epochs=150)
    system.save_model("neural_eq_safedb_v3.pt")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import silhouette_score, davies_bouldin_score


@dataclass
class SAFEEQSetting:
    """Container for SAFE-DB EQ setting"""
    setting_id: int
    semantic_label: str
    eq_params: np.ndarray  # 13 parameters (raw)
    eq_params_normalized: Optional[np.ndarray] = None  # Normalized for training


class SAFEDBDatasetLoaderV3:
    """
    V3 Data Loader with FIXED normalization bounds

    Key changes from V2:
    - FIXED frequency bounds: 20-20,000 Hz (not data-dependent)
    - FIXED Q bounds: 0.1-10 (standard EQ range)
    - Allows full frequency spectrum utilization
    """

    def __init__(self, data_dir: str = "research/data"):
        self.data_dir = Path(data_dir)
        self.user_data_path = self.data_dir / "SAFEEqualiserUserData.csv"

        self.eq_settings = []
        self.user_df = None

        # FIXED normalization bounds (NOT data-dependent)
        self.gain_min = -12.0
        self.gain_max = 12.0

        # Fixed frequency bounds: 20-20,000 Hz
        self.freq_min = 20.0
        self.freq_max = 20000.0
        self.freq_log_min = np.log10(self.freq_min)  # ~1.30
        self.freq_log_max = np.log10(self.freq_max)  # ~4.30

        # Fixed Q bounds: 0.1-10
        self.q_min = 0.1
        self.q_max = 10.0
        self.q_log_min = np.log10(self.q_min)  # -1.0
        self.q_log_max = np.log10(self.q_max)  # 1.0

    def load_dataset(self, min_examples: int = 5) -> List[SAFEEQSetting]:
        """Load and preprocess SAFE-DB dataset with FIXED normalization bounds"""

        print("="*70)
        print("LOADING SAFE-DB DATASET V3 (FIXED NORMALIZATION BOUNDS)")
        print("="*70)

        # Load CSV
        print(f"\nLoading UserData from: {self.user_data_path}")
        self.user_df = pd.read_csv(self.user_data_path, header=None)
        print(f"  Loaded {len(self.user_df)} EQ settings")

        # Parse raw EQ settings
        print(f"\nParsing raw EQ parameters...")
        raw_settings = []

        for idx, row in self.user_df.iterrows():
            setting_id = int(row[0])
            semantic_label = str(row[1]).strip().lower()
            eq_params = row[5:18].values.astype(np.float32)  # 13 params

            # Skip invalid
            if np.any(np.isnan(eq_params)):
                continue

            setting = SAFEEQSetting(
                setting_id=setting_id,
                semantic_label=semantic_label,
                eq_params=eq_params
            )
            raw_settings.append(setting)

        print(f"  Parsed {len(raw_settings)} valid settings")

        # Filter by minimum examples
        if min_examples > 1:
            term_counts = Counter(s.semantic_label for s in raw_settings)
            valid_terms = {t for t, c in term_counts.items() if c >= min_examples}
            raw_settings = [s for s in raw_settings if s.semantic_label in valid_terms]
            print(f"  Filtered to {len(raw_settings)} settings ({len(valid_terms)} terms with >={min_examples} examples)")

        # V3: Normalization bounds are FIXED (no computation needed)
        print(f"\nUsing FIXED normalization bounds (not data-dependent):")
        print(f"  Frequencies: {self.freq_min:.0f}-{self.freq_max:.0f} Hz")
        print(f"  Q values: {self.q_min:.1f}-{self.q_max:.1f}")
        print(f"  Gains: {self.gain_min:.0f} to +{self.gain_max:.0f} dB")

        # Normalize all settings
        print(f"\nNormalizing parameters...")
        for setting in raw_settings:
            setting.eq_params_normalized = self._normalize_params(setting.eq_params)

        self.eq_settings = raw_settings

        # Show stats
        self._show_normalization_stats()

        return self.eq_settings

    def _show_normalization_stats(self):
        """Display normalization statistics"""
        print(f"\nNormalization ranges:")
        print(f"  Gains: [{self.gain_min:.1f}, {self.gain_max:.1f}] dB -> [0, 1]")
        print(f"  Freqs (log): [{self.freq_log_min:.2f}, {self.freq_log_max:.2f}] -> [0, 1]")
        print(f"  Q (log): [{self.q_log_min:.2f}, {self.q_log_max:.2f}] -> [0, 1]")

    def _normalize_params(self, params: np.ndarray) -> np.ndarray:
        """
        Normalize 13 parameters to [0, 1] range

        Structure: [G,F, G,F,Q, G,F,Q, G,F,Q, G,F]
        Indices:   [0,1, 2,3,4, 5,6,7, 8,9,10, 11,12]
        """
        params_norm = np.zeros_like(params)

        # Gain indices: 0, 2, 5, 8, 11
        gain_indices = [0, 2, 5, 8, 11]
        for idx in gain_indices:
            # Min-max normalization: [-12, 12] → [0, 1]
            params_norm[idx] = (params[idx] - self.gain_min) / (self.gain_max - self.gain_min)

        # Frequency indices: 1, 3, 6, 9, 12
        freq_indices = [1, 3, 6, 9, 12]
        for idx in freq_indices:
            # Log-scale then min-max
            freq_log = np.log10(params[idx] + 1)
            params_norm[idx] = (freq_log - self.freq_log_min) / (self.freq_log_max - self.freq_log_min)

        # Q indices: 4, 7, 10
        q_indices = [4, 7, 10]
        for idx in q_indices:
            # Log-scale then min-max
            q_log = np.log10(params[idx])
            params_norm[idx] = (q_log - self.q_log_min) / (self.q_log_max - self.q_log_min)

        return params_norm

    def denormalize_params(self, params_norm: np.ndarray) -> np.ndarray:
        """Convert normalized [0,1] params back to original scale"""
        params = np.zeros_like(params_norm)

        # Gains
        gain_indices = [0, 2, 5, 8, 11]
        for idx in gain_indices:
            params[idx] = params_norm[idx] * (self.gain_max - self.gain_min) + self.gain_min

        # Frequencies
        freq_indices = [1, 3, 6, 9, 12]
        for idx in freq_indices:
            freq_log = params_norm[idx] * (self.freq_log_max - self.freq_log_min) + self.freq_log_min
            params[idx] = 10 ** freq_log - 1  # Back to linear

        # Q values
        q_indices = [4, 7, 10]
        for idx in q_indices:
            q_log = params_norm[idx] * (self.q_log_max - self.q_log_min) + self.q_log_min
            params[idx] = 10 ** q_log

        return params

    def analyze_dataset(self) -> Dict:
        """Show dataset statistics"""
        if not self.eq_settings:
            return {}

        print(f"\n{'='*70}")
        print("DATASET STATISTICS")
        print("="*70)

        term_counts = Counter(s.semantic_label for s in self.eq_settings)
        print(f"\nTotal settings: {len(self.eq_settings)}")
        print(f"Unique terms: {len(term_counts)}")
        print(f"\nTop terms:")
        for term, count in term_counts.most_common(10):
            print(f"  {term:15s}: {count:3d} examples")

        return {'n_settings': len(self.eq_settings), 'term_counts': term_counts}


# ========== Neural Network Components ==========

class ResidualBlock(nn.Module):
    """Residual block (unchanged from V1)"""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()

        self.main_path = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main_path(x) + self.skip(x))


class NeuralEQEncoderV3(nn.Module):
    """Encoder V3 (same as V2, normalization fixed upstream)"""

    def __init__(self, latent_dim: int = 32, hidden_dims: List[int] = None):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 128, 64]

        self.input_dim = 13
        self.latent_dim = latent_dim

        # Encoder layers
        layers = []
        prev_dim = self.input_dim
        for hidden_dim in hidden_dims:
            layers.append(ResidualBlock(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        self.encoder_layers = nn.ModuleList(layers)

        # To latent space
        self.to_latent = nn.Sequential(
            nn.Linear(prev_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Tanh()  # Bounded latent space
        )

        # Semantic projection
        self.semantic_proj = nn.Linear(latent_dim, 128)

    def forward(self, eq_params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = eq_params

        for layer in self.encoder_layers:
            x = layer(x)

        latent = self.to_latent(x)
        semantic_emb = self.semantic_proj(latent)

        return latent, semantic_emb


class NeuralEQDecoderV3(nn.Module):
    """
    IMPROVED Decoder V2

    Key changes:
    - Outputs NORMALIZED params [0, 1]
    - Uses sigmoid activation (safe for [0,1])
    - Hard clipping for extra safety
    - Denormalization happens outside decoder
    """

    def __init__(self, latent_dim: int = 32, hidden_dims: List[int] = None):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 128, 64]

        self.latent_dim = latent_dim
        self.output_dim = 13

        # Decoder layers
        layers = []
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.append(ResidualBlock(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        self.decoder_layers = nn.ModuleList(layers)

        # Single output head for all 13 parameters
        # All params are normalized to [0,1], so use sigmoid
        self.output_head = nn.Sequential(
            nn.Linear(prev_dim, prev_dim),
            nn.ReLU(),
            nn.Linear(prev_dim, self.output_dim),
            nn.Sigmoid()  # All outputs in [0, 1]
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Output: [batch, 13] parameters in NORMALIZED [0,1] range
        Denormalization happens in the main system
        """
        x = latent

        for layer in self.decoder_layers:
            x = layer(x)

        # All params in [0, 1]
        params_norm = self.output_head(x)

        # Hard clipping for safety (shouldn't be needed with sigmoid, but just in case)
        params_norm = torch.clamp(params_norm, 0.0, 1.0)

        return params_norm


class ContrastiveLoss(nn.Module):
    """Contrastive loss (unchanged from V1)"""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, dim=1)
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        labels = labels.unsqueeze(0)
        mask = torch.eq(labels, labels.T).float()
        mask = mask - torch.eye(mask.shape[0], device=mask.device)

        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        loss = -mean_log_prob_pos.mean()

        return loss


# ========== Main Training System V2 ==========

class NeuralEQMorphingSAFEDBV3:
    """
    IMPROVED Neural EQ Morphing System V2

    Key improvements:
    - Proper log-scale normalization
    - Better decoder
    - Annealed contrastive weight
    - Learning rate scheduling
    """

    def __init__(self, latent_dim: int = 32, device: str = None):
        self.latent_dim = latent_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Models
        self.encoder = NeuralEQEncoderV3(latent_dim=latent_dim).to(self.device)
        self.decoder = NeuralEQDecoderV3(latent_dim=latent_dim).to(self.device)

        # Data
        self.loader = SAFEDBDatasetLoaderV3()
        self.eq_settings = []
        self.term_to_idx = {}
        self.idx_to_term = {}

        # Training history
        self.history = {
            'reconstruction_loss': [],
            'contrastive_loss': [],
            'total_loss': []
        }

    def load_dataset(self, min_examples: int = 5):
        """Load and preprocess dataset"""
        print(f"\nLoading SAFE-DB dataset V3...")
        self.eq_settings = self.loader.load_dataset(min_examples=min_examples)

        # Build vocabulary
        unique_terms = sorted(set(s.semantic_label for s in self.eq_settings))
        self.term_to_idx = {term: idx for idx, term in enumerate(unique_terms)}
        self.idx_to_term = {idx: term for term, idx in self.term_to_idx.items()}

        print(f"\nVocabulary: {len(unique_terms)} unique semantic terms")

        # Analyze
        self.loader.analyze_dataset()

    def train(self,
              epochs: int = 150,
              batch_size: int = 64,
              learning_rate: float = 0.001,
              contrastive_weight_start: float = 0.1,
              contrastive_weight_end: float = 0.5,
              save_every: int = 30):
        """
        Train with improvements:
        - Larger batch size (64 vs 32)
        - Annealed contrastive weight
        - More epochs (150 vs 100)
        """

        print(f"\n{'='*70}")
        print("TRAINING NEURAL EQ MORPHING SYSTEM V2")
        print("="*70)
        print(f"\nHyperparameters:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Contrastive weight: {contrastive_weight_start} → {contrastive_weight_end} (annealed)")
        print(f"  Latent dimension: {self.latent_dim}")
        print(f"  Device: {self.device}")

        # Prepare dataset
        params_list = []
        labels_list = []

        for setting in self.eq_settings:
            params_list.append(setting.eq_params_normalized)  # Already normalized!
            labels_list.append(self.term_to_idx[setting.semantic_label])

        params_tensor = torch.tensor(np.array(params_list), dtype=torch.float32, device=self.device)
        labels_tensor = torch.tensor(labels_list, dtype=torch.long, device=self.device)

        # Optimizer
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20
        )

        # Loss functions
        contrastive_loss_fn = ContrastiveLoss(temperature=0.1)
        reconstruction_loss_fn = nn.MSELoss()

        # Training loop
        n_samples = len(params_list)
        n_batches = (n_samples + batch_size - 1) // batch_size

        print(f"\nStarting training...")
        print(f"  Total samples: {n_samples}")
        print(f"  Batches per epoch: {n_batches}")

        for epoch in range(epochs):
            # Annealed contrastive weight
            progress = epoch / max(epochs - 1, 1)
            contrastive_weight = contrastive_weight_start + progress * (contrastive_weight_end - contrastive_weight_start)

            epoch_recon_loss = 0.0
            epoch_contr_loss = 0.0
            epoch_total_loss = 0.0

            # Shuffle
            perm = torch.randperm(n_samples, device=self.device)
            params_shuffled = params_tensor[perm]
            labels_shuffled = labels_tensor[perm]

            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)

                batch_params = params_shuffled[start_idx:end_idx]
                batch_labels = labels_shuffled[start_idx:end_idx]

                # Forward
                latent, semantic_emb = self.encoder(batch_params)
                params_recon = self.decoder(latent)

                # Losses (all params already normalized to [0,1], so balanced!)
                recon_loss = reconstruction_loss_fn(params_recon, batch_params)
                contr_loss = contrastive_loss_fn(semantic_emb, batch_labels)
                total_loss = recon_loss + contrastive_weight * contr_loss

                # Backward
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_recon_loss += recon_loss.item()
                epoch_contr_loss += contr_loss.item()
                epoch_total_loss += total_loss.item()

            # Average losses
            epoch_recon_loss /= n_batches
            epoch_contr_loss /= n_batches
            epoch_total_loss /= n_batches

            self.history['reconstruction_loss'].append(epoch_recon_loss)
            self.history['contrastive_loss'].append(epoch_contr_loss)
            self.history['total_loss'].append(epoch_total_loss)

            # Update learning rate
            scheduler.step(epoch_total_loss)

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs}: "
                      f"Total={epoch_total_loss:.4f}, "
                      f"Recon={epoch_recon_loss:.4f}, "
                      f"Contr={epoch_contr_loss:.4f}, "
                      f"λ={contrastive_weight:.3f}")

            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_model(f"neural_eq_safedb_v2_epoch{epoch+1}.pt")

        print(f"\nTraining complete!")
        self.evaluate_clustering()

    def evaluate_clustering(self):
        """Evaluate clustering quality"""
        print(f"\n{'='*70}")
        print("EVALUATING LATENT SPACE CLUSTERING")
        print("="*70)

        self.encoder.eval()
        with torch.no_grad():
            params_list = []
            labels_list = []

            for setting in self.eq_settings:
                params_list.append(setting.eq_params_normalized)
                labels_list.append(self.term_to_idx[setting.semantic_label])

            params_tensor = torch.tensor(np.array(params_list), dtype=torch.float32, device=self.device)
            latent, _ = self.encoder(params_tensor)
            latent_np = latent.cpu().numpy()
            labels_np = np.array(labels_list)

        silhouette = silhouette_score(latent_np, labels_np)
        davies_bouldin = davies_bouldin_score(latent_np, labels_np)

        print(f"\nClustering Metrics:")
        print(f"  Silhouette Score: {silhouette:.4f} (higher is better, range [-1, 1])")
        print(f"  Davies-Bouldin Index: {davies_bouldin:.4f} (lower is better)")

        if silhouette > 0.5:
            print(f"  EXCELLENT clustering!")
        elif silhouette > 0.3:
            print(f"  GOOD clustering")
        elif silhouette > 0.0:
            print(f"  MODERATE clustering")
        else:
            print(f"  POOR clustering")

        return {'silhouette': silhouette, 'davies_bouldin': davies_bouldin}

    def save_model(self, path: str):
        """Save model checkpoint"""
        checkpoint = {
            'encoder_state': self.encoder.state_dict(),
            'decoder_state': self.decoder.state_dict(),
            'term_to_idx': self.term_to_idx,
            'idx_to_term': self.idx_to_term,
            'loader_state': {
                'gain_min': self.loader.gain_min,
                'gain_max': self.loader.gain_max,
                'freq_log_min': self.loader.freq_log_min,
                'freq_log_max': self.loader.freq_log_max,
                'q_log_min': self.loader.q_log_min,
                'q_log_max': self.loader.q_log_max,
            },
            'latent_dim': self.latent_dim,
            'history': self.history
        }
        torch.save(checkpoint, path)
        print(f"\nModel saved to: {path}")

    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(checkpoint['encoder_state'])
        self.decoder.load_state_dict(checkpoint['decoder_state'])
        self.term_to_idx = checkpoint['term_to_idx']
        self.idx_to_term = checkpoint['idx_to_term']

        # Restore loader normalization stats
        loader_state = checkpoint['loader_state']
        self.loader.gain_min = loader_state['gain_min']
        self.loader.gain_max = loader_state['gain_max']
        self.loader.freq_log_min = loader_state['freq_log_min']
        self.loader.freq_log_max = loader_state['freq_log_max']
        self.loader.q_log_min = loader_state['q_log_min']
        self.loader.q_log_max = loader_state['q_log_max']

        self.history = checkpoint['history']
        print(f"\nModel loaded from: {path}")

    def generate_eq_from_term(self, semantic_term: str) -> np.ndarray:
        """Generate EQ parameters from semantic term"""
        if semantic_term not in self.term_to_idx:
            raise ValueError(f"Unknown term: {semantic_term}")

        # Find all examples
        examples = [s for s in self.eq_settings if s.semantic_label == semantic_term]

        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            latents = []
            for ex in examples:
                params = torch.tensor(ex.eq_params_normalized, dtype=torch.float32, device=self.device).unsqueeze(0)
                latent, _ = self.encoder(params)
                latents.append(latent)

            avg_latent = torch.stack(latents).mean(dim=0)
            eq_params_norm = self.decoder(avg_latent)
            eq_params_norm_np = eq_params_norm.cpu().numpy()[0]

        # Denormalize
        eq_params = self.loader.denormalize_params(eq_params_norm_np)

        return eq_params

    def interpolate_terms(self, term1: str, term2: str, alpha: float = 0.5) -> np.ndarray:
        """Interpolate between two terms"""
        params1 = self.generate_eq_from_term(term1)
        params2 = self.generate_eq_from_term(term2)

        params_interp = (1 - alpha) * params1 + alpha * params2

        return params_interp


# ========== CLI Entry Point ==========

if __name__ == "__main__":
    print("Neural EQ Morphing System V2 - SAFE-DB")
    print("="*70)

    system = NeuralEQMorphingSAFEDBV2(latent_dim=32)
    system.load_dataset(min_examples=5)
    system.train(epochs=150, batch_size=64, contrastive_weight_start=0.1, contrastive_weight_end=0.5)
    system.save_model("neural_eq_safedb_v2.pt")

    print("\n" + "="*70)
    print("Training complete! Model saved to: neural_eq_safedb_v2.pt")
    print("="*70)
