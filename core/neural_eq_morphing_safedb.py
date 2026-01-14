"""
Neural EQ Morphing System - SAFE-DB Dataset
============================================

Neural Residual Networks + Contrastive Learning for semantic EQ generation.
Trained on SAFE-DB dataset with 13 EQ parameters (5 bands).

Key Differences from SocialFX version:
- Uses SAFE-DB dataset (1,700 examples, 368 unique terms)
- 13 parameters instead of 40
- Better term distribution (warm: 457, bright: 421)
- Includes audio features for evaluation

Dataset Structure:
- SAFEEqualiserUserData.csv: ID, semantic_term, EQ_params (13 values)
- SAFEEqualiserAudioFeatureData.csv: ID, processed/unprocessed, audio_features (80 values)

EQ Parameter Structure (13 params = 5 bands):
  Band 1 (Low shelf):    Gain, Freq
  Band 2 (Bell):         Gain, Freq, Q
  Band 3 (Bell):         Gain, Freq, Q
  Band 4 (Bell):         Gain, Freq, Q
  Band 5 (High shelf):   Gain, Freq

Usage:
    from core.neural_eq_morphing_safedb import NeuralEQMorphingSAFEDB

    system = NeuralEQMorphingSAFEDB()
    system.load_dataset()
    system.train(epochs=100)
    system.save_model("neural_eq_safedb.pt")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import Counter
import json
import warnings
warnings.filterwarnings("ignore")

# ML components
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


@dataclass
class SAFEEQSetting:
    """Container for SAFE-DB EQ setting"""
    setting_id: int
    semantic_label: str
    eq_params: np.ndarray  # 13 parameters
    audio_features_processed: Optional[np.ndarray] = None  # 80 features
    audio_features_unprocessed: Optional[np.ndarray] = None  # 80 features
    ip_address: Optional[str] = None
    filter_types: Optional[Tuple[int, int]] = None


class SAFEDBDatasetLoader:
    """
    Load and process SAFE-DB dataset

    Combines UserData (EQ parameters) with AudioFeatureData (before/after features)
    """

    def __init__(self, data_dir: str = "research/data"):
        self.data_dir = Path(data_dir)
        self.user_data_path = self.data_dir / "SAFEEqualiserUserData.csv"
        self.audio_data_path = self.data_dir / "SAFEEqualiserAudioFeatureData.csv"

        self.eq_settings = []
        self.user_df = None
        self.audio_df = None

    def load_dataset(self, include_audio_features: bool = True) -> List[SAFEEQSetting]:
        """
        Load SAFE-DB dataset

        Args:
            include_audio_features: Whether to load audio features (slower but more complete)

        Returns:
            List of SAFEEQSetting objects
        """
        print("="*70)
        print("LOADING SAFE-DB DATASET")
        print("="*70)

        # Load UserData
        print(f"\nLoading UserData from: {self.user_data_path}")
        self.user_df = pd.read_csv(self.user_data_path, header=None)
        print(f"  Loaded {len(self.user_df)} EQ settings")

        # Load AudioFeatureData if requested
        if include_audio_features:
            print(f"\nLoading AudioFeatureData from: {self.audio_data_path}")
            self.audio_df = pd.read_csv(self.audio_data_path, header=None)
            print(f"  Loaded {len(self.audio_df)} audio feature rows (2x UserData)")

        # Parse EQ settings
        print(f"\nParsing EQ settings...")
        self.eq_settings = []

        for idx, row in self.user_df.iterrows():
            setting_id = int(row[0])
            semantic_label = str(row[1]).strip().lower()
            ip_address = str(row[2])

            # Filter types (columns 3-4)
            filter_types = (int(row[3]), int(row[4]))

            # EQ parameters (columns 5-17 = 13 params)
            eq_params = row[5:18].values.astype(np.float32)

            # Handle NaN values
            if np.any(np.isnan(eq_params)):
                continue  # Skip invalid settings

            # Get audio features if available
            audio_feat_proc = None
            audio_feat_unproc = None

            if include_audio_features and self.audio_df is not None:
                matching_rows = self.audio_df[self.audio_df[0] == setting_id]

                if len(matching_rows) == 2:
                    # Find processed and unprocessed
                    proc_row = matching_rows[matching_rows[1] == 'processed']
                    unproc_row = matching_rows[matching_rows[1] == 'unprocessed']

                    if len(proc_row) > 0:
                        audio_feat_proc = proc_row.iloc[0, 2:].values.astype(np.float32)
                    if len(unproc_row) > 0:
                        audio_feat_unproc = unproc_row.iloc[0, 2:].values.astype(np.float32)

            setting = SAFEEQSetting(
                setting_id=setting_id,
                semantic_label=semantic_label,
                eq_params=eq_params,
                audio_features_processed=audio_feat_proc,
                audio_features_unprocessed=audio_feat_unproc,
                ip_address=ip_address,
                filter_types=filter_types
            )

            self.eq_settings.append(setting)

        print(f"\n  Successfully parsed {len(self.eq_settings)} valid EQ settings")

        return self.eq_settings

    def analyze_dataset(self) -> Dict:
        """Analyze dataset statistics"""
        if not self.eq_settings:
            return {"error": "No EQ settings loaded"}

        print(f"\n{'='*70}")
        print("SAFE-DB DATASET ANALYSIS")
        print(f"{'='*70}")

        n_settings = len(self.eq_settings)
        semantic_terms = [s.semantic_label for s in self.eq_settings]
        unique_terms = list(set(semantic_terms))

        print(f"\nDataset Size:")
        print(f"  Total EQ settings: {n_settings}")
        print(f"  Unique semantic terms: {len(unique_terms)}")
        print(f"  Parameters per setting: 13")
        print(f"  EQ bands: 5 (2 shelves + 3 bells)")

        # Term distribution
        term_counts = Counter(semantic_terms)
        print(f"\nTop 20 semantic terms:")
        for term, count in term_counts.most_common(20):
            print(f"  {term:20s}: {count:4d} examples")

        # Parameter statistics
        all_params = np.array([s.eq_params for s in self.eq_settings])

        print(f"\nParameter Statistics:")
        param_names = [
            "Band1_Gain", "Band1_Freq",
            "Band2_Gain", "Band2_Freq", "Band2_Q",
            "Band3_Gain", "Band3_Freq", "Band3_Q",
            "Band4_Gain", "Band4_Freq", "Band4_Q",
            "Band5_Gain", "Band5_Freq"
        ]

        for i, name in enumerate(param_names):
            vals = all_params[:, i]
            print(f"  {name:15s}: [{vals.min():8.2f}, {vals.max():8.2f}] "
                  f"(mean={vals.mean():7.2f}, std={vals.std():6.2f})")

        # Audio features
        with_audio = sum(1 for s in self.eq_settings
                        if s.audio_features_processed is not None)
        print(f"\nAudio Features:")
        print(f"  Settings with audio features: {with_audio}/{n_settings}")

        return {
            'n_settings': n_settings,
            'unique_terms': unique_terms,
            'term_counts': term_counts,
            'param_stats': {
                'mean': all_params.mean(axis=0),
                'std': all_params.std(axis=0),
                'min': all_params.min(axis=0),
                'max': all_params.max(axis=0)
            }
        }

    def filter_by_min_examples(self, min_examples: int = 10) -> List[SAFEEQSetting]:
        """
        Filter dataset to only include terms with sufficient examples

        This improves clustering by removing rare terms
        """
        term_counts = Counter(s.semantic_label for s in self.eq_settings)
        valid_terms = {term for term, count in term_counts.items()
                      if count >= min_examples}

        filtered = [s for s in self.eq_settings if s.semantic_label in valid_terms]

        print(f"\nFiltering dataset:")
        print(f"  Original: {len(self.eq_settings)} settings, {len(term_counts)} terms")
        print(f"  Filtered (>={min_examples} examples): {len(filtered)} settings, {len(valid_terms)} terms")

        return filtered


# ========== Neural Network Components ==========

class ResidualBlock(nn.Module):
    """Residual block for stable deep learning"""

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

        if in_dim != out_dim:
            self.skip = nn.Linear(in_dim, out_dim)
        else:
            self.skip = nn.Identity()

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main_path(x) + self.skip(x))


class NeuralEQEncoder(nn.Module):
    """
    Encoder: 13 EQ params -> latent representation

    Uses residual networks for stable training
    """

    def __init__(self, latent_dim: int = 32, hidden_dims: List[int] = None):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 128, 64]

        self.input_dim = 13  # SAFE-DB has 13 parameters
        self.latent_dim = latent_dim

        # Build encoder layers
        layers = []
        prev_dim = self.input_dim

        for hidden_dim in hidden_dims:
            layers.append(ResidualBlock(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        self.encoder_layers = nn.ModuleList(layers)

        # Project to latent space
        self.to_latent = nn.Sequential(
            nn.Linear(prev_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Tanh()  # Bound latent space to [-1, 1]
        )

        # Project to semantic embedding space (for contrastive learning)
        self.semantic_proj = nn.Linear(latent_dim, 128)

    def forward(self, eq_params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            eq_params: [batch, 13] EQ parameters

        Returns:
            latent: [batch, latent_dim] latent representation
            semantic_emb: [batch, 128] semantic embedding
        """
        x = eq_params

        # Encode through residual blocks
        for layer in self.encoder_layers:
            x = layer(x)

        # Get latent representation
        latent = self.to_latent(x)

        # Get semantic embedding
        semantic_emb = self.semantic_proj(latent)

        return latent, semantic_emb


class NeuralEQDecoder(nn.Module):
    """
    Decoder: latent representation -> 13 EQ params

    Specialized output heads for gain/freq/Q constraints
    """

    def __init__(self, latent_dim: int = 32, hidden_dims: List[int] = None):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 128, 64]

        self.latent_dim = latent_dim
        self.output_dim = 13  # SAFE-DB has 13 parameters

        # Build decoder layers
        layers = []
        prev_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.append(ResidualBlock(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        self.decoder_layers = nn.ModuleList(layers)

        # Specialized output heads for different parameter types
        # Band structure: [G,F, G,F,Q, G,F,Q, G,F,Q, G,F]

        self.gain_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 6),  # 6 gain values
            nn.Tanh()  # Range [-1, 1], scale to [-12, 12] dB
        )

        self.freq_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 5),  # 5 frequency values
            nn.Sigmoid()  # Range [0, 1], map to Hz ranges
        )

        self.q_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # 3 Q values (bands 2, 3, 4)
            nn.Sigmoid()  # Range [0, 1], scale to [0.1, 10]
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: [batch, latent_dim]

        Returns:
            eq_params: [batch, 13] reconstructed EQ parameters
        """
        x = latent

        # Decode through residual blocks
        for layer in self.decoder_layers:
            x = layer(x)

        # Get specialized outputs
        gains = self.gain_head(x) * 12.0  # Scale to [-12, 12] dB
        freqs = self.freq_head(x)  # [0, 1], will map to Hz ranges
        qs = self.q_head(x) * 9.9 + 0.1  # Scale to [0.1, 10]

        # Map frequencies to appropriate ranges for each band
        # Band 1: [22, 1000] Hz
        # Band 2: [82, 3900] Hz
        # Band 3: [180, 4700] Hz
        # Band 4: [220, 10000] Hz
        # Band 5: [580, 20000] Hz

        batch_size = latent.shape[0]
        eq_params = torch.zeros(batch_size, 13, device=latent.device)

        # Band 1: Gain, Freq
        eq_params[:, 0] = gains[:, 0]
        eq_params[:, 1] = freqs[:, 0] * (1000 - 22) + 22

        # Band 2: Gain, Freq, Q
        eq_params[:, 2] = gains[:, 1]
        eq_params[:, 3] = freqs[:, 1] * (3900 - 82) + 82
        eq_params[:, 4] = qs[:, 0]

        # Band 3: Gain, Freq, Q
        eq_params[:, 5] = gains[:, 2]
        eq_params[:, 6] = freqs[:, 2] * (4700 - 180) + 180
        eq_params[:, 7] = qs[:, 1]

        # Band 4: Gain, Freq, Q
        eq_params[:, 8] = gains[:, 3]
        eq_params[:, 9] = freqs[:, 3] * (10000 - 220) + 220
        eq_params[:, 10] = qs[:, 2]

        # Band 5: Gain, Freq
        eq_params[:, 11] = gains[:, 4]
        eq_params[:, 12] = freqs[:, 4] * (20000 - 580) + 580

        return eq_params


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for semantic clustering

    Encourages similar semantic terms to cluster in latent space
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [batch, 128] semantic embeddings
            labels: [batch] semantic label indices

        Returns:
            loss: scalar contrastive loss
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create mask for positive pairs (same label)
        labels = labels.unsqueeze(0)
        mask = torch.eq(labels, labels.T).float()

        # Remove diagonal (self-similarity)
        mask = mask - torch.eye(mask.shape[0], device=mask.device)

        # Compute loss
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))

        # Mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        loss = -mean_log_prob_pos.mean()

        return loss


# ========== Main Training System ==========

class NeuralEQMorphingSAFEDB:
    """
    Complete neural EQ morphing system for SAFE-DB

    Combines data loading, training, and inference
    """

    def __init__(self, latent_dim: int = 32, device: str = None):
        self.latent_dim = latent_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Models
        self.encoder = NeuralEQEncoder(latent_dim=latent_dim).to(self.device)
        self.decoder = NeuralEQDecoder(latent_dim=latent_dim).to(self.device)

        # Data
        self.loader = SAFEDBDatasetLoader()
        self.eq_settings = []
        self.term_to_idx = {}
        self.idx_to_term = {}

        # Normalization stats
        self.param_mean = None
        self.param_std = None

        # Training history
        self.history = {
            'reconstruction_loss': [],
            'contrastive_loss': [],
            'total_loss': []
        }

    def load_dataset(self, min_examples: int = 10, include_audio: bool = False):
        """Load and preprocess SAFE-DB dataset"""
        print(f"\nLoading SAFE-DB dataset...")
        self.eq_settings = self.loader.load_dataset(include_audio_features=include_audio)

        # Filter to well-represented terms
        if min_examples > 1:
            self.eq_settings = self.loader.filter_by_min_examples(min_examples)

        # Analyze dataset
        self.loader.analyze_dataset()

        # Build term vocabulary
        unique_terms = sorted(set(s.semantic_label for s in self.eq_settings))
        self.term_to_idx = {term: idx for idx, term in enumerate(unique_terms)}
        self.idx_to_term = {idx: term for term, idx in self.term_to_idx.items()}

        print(f"\nVocabulary: {len(unique_terms)} unique semantic terms")

        # Compute normalization statistics
        all_params = np.array([s.eq_params for s in self.eq_settings])
        self.param_mean = torch.tensor(all_params.mean(axis=0), dtype=torch.float32, device=self.device)
        self.param_std = torch.tensor(all_params.std(axis=0) + 1e-8, dtype=torch.float32, device=self.device)

        print(f"\nNormalization computed:")
        print(f"  Mean: {self.param_mean.cpu().numpy()}")
        print(f"  Std:  {self.param_std.cpu().numpy()}")

    def normalize_params(self, params: torch.Tensor) -> torch.Tensor:
        """Z-score normalization"""
        return (params - self.param_mean) / self.param_std

    def denormalize_params(self, params_norm: torch.Tensor) -> torch.Tensor:
        """Reverse z-score normalization"""
        return params_norm * self.param_std + self.param_mean

    def train(self,
              epochs: int = 100,
              batch_size: int = 32,
              learning_rate: float = 0.001,
              contrastive_weight: float = 0.5,
              save_every: int = 20):
        """
        Train the neural EQ morphing system

        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            contrastive_weight: Weight for contrastive loss (higher = better clustering)
            save_every: Save model every N epochs
        """
        print(f"\n{'='*70}")
        print("TRAINING NEURAL EQ MORPHING SYSTEM (SAFE-DB)")
        print(f"{'='*70}")
        print(f"\nHyperparameters:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Contrastive weight: {contrastive_weight}")
        print(f"  Latent dimension: {self.latent_dim}")
        print(f"  Device: {self.device}")

        # Prepare dataset
        params_list = []
        labels_list = []

        for setting in self.eq_settings:
            params_list.append(setting.eq_params)
            labels_list.append(self.term_to_idx[setting.semantic_label])

        params_tensor = torch.tensor(np.array(params_list), dtype=torch.float32, device=self.device)
        labels_tensor = torch.tensor(labels_list, dtype=torch.long, device=self.device)

        # Normalize parameters
        params_tensor = self.normalize_params(params_tensor)

        # Optimizer and losses
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate
        )

        contrastive_loss_fn = ContrastiveLoss(temperature=0.1)
        reconstruction_loss_fn = nn.MSELoss()

        # Training loop
        n_samples = len(params_list)
        n_batches = (n_samples + batch_size - 1) // batch_size

        print(f"\nStarting training...")
        print(f"  Total samples: {n_samples}")
        print(f"  Batches per epoch: {n_batches}")

        for epoch in range(epochs):
            epoch_recon_loss = 0.0
            epoch_contr_loss = 0.0
            epoch_total_loss = 0.0

            # Shuffle data
            perm = torch.randperm(n_samples, device=self.device)
            params_shuffled = params_tensor[perm]
            labels_shuffled = labels_tensor[perm]

            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)

                batch_params = params_shuffled[start_idx:end_idx]
                batch_labels = labels_shuffled[start_idx:end_idx]

                # Forward pass
                latent, semantic_emb = self.encoder(batch_params)
                params_recon = self.decoder(latent)

                # Compute losses
                recon_loss = reconstruction_loss_fn(params_recon, batch_params)
                contr_loss = contrastive_loss_fn(semantic_emb, batch_labels)
                total_loss = recon_loss + contrastive_weight * contr_loss

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Track losses
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

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs}: "
                      f"Total={epoch_total_loss:.4f}, "
                      f"Recon={epoch_recon_loss:.4f}, "
                      f"Contr={epoch_contr_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_model(f"neural_eq_safedb_epoch{epoch+1}.pt")

        print(f"\nTraining complete!")

        # Evaluate clustering
        self.evaluate_clustering()

    def evaluate_clustering(self):
        """Evaluate latent space clustering quality"""
        print(f"\n{'='*70}")
        print("EVALUATING LATENT SPACE CLUSTERING")
        print(f"{'='*70}")

        # Get all latent representations
        self.encoder.eval()
        with torch.no_grad():
            params_list = []
            labels_list = []

            for setting in self.eq_settings:
                params_list.append(setting.eq_params)
                labels_list.append(self.term_to_idx[setting.semantic_label])

            params_tensor = torch.tensor(np.array(params_list), dtype=torch.float32, device=self.device)
            params_tensor = self.normalize_params(params_tensor)

            latent, _ = self.encoder(params_tensor)
            latent_np = latent.cpu().numpy()
            labels_np = np.array(labels_list)

        # Compute clustering metrics
        silhouette = silhouette_score(latent_np, labels_np)
        davies_bouldin = davies_bouldin_score(latent_np, labels_np)

        print(f"\nClustering Metrics:")
        print(f"  Silhouette Score: {silhouette:.4f} (higher is better, range [-1, 1])")
        print(f"  Davies-Bouldin Index: {davies_bouldin:.4f} (lower is better)")

        if silhouette > 0.3:
            print(f"  ✓ Good clustering! (silhouette > 0.3)")
        elif silhouette > 0.0:
            print(f"  ~ Moderate clustering (0 < silhouette < 0.3)")
        else:
            print(f"  ✗ Poor clustering (silhouette < 0)")

        return {'silhouette': silhouette, 'davies_bouldin': davies_bouldin}

    def save_model(self, path: str):
        """Save model checkpoint"""
        checkpoint = {
            'encoder_state': self.encoder.state_dict(),
            'decoder_state': self.decoder.state_dict(),
            'term_to_idx': self.term_to_idx,
            'idx_to_term': self.idx_to_term,
            'param_mean': self.param_mean,
            'param_std': self.param_std,
            'latent_dim': self.latent_dim,
            'history': self.history
        }
        torch.save(checkpoint, path)
        print(f"\nModel saved to: {path}")

    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state'])
        self.decoder.load_state_dict(checkpoint['decoder_state'])
        self.term_to_idx = checkpoint['term_to_idx']
        self.idx_to_term = checkpoint['idx_to_term']
        self.param_mean = checkpoint['param_mean']
        self.param_std = checkpoint['param_std']
        self.history = checkpoint['history']
        print(f"\nModel loaded from: {path}")

    def generate_eq_from_term(self, semantic_term: str) -> np.ndarray:
        """Generate EQ parameters from semantic term"""
        if semantic_term not in self.term_to_idx:
            raise ValueError(f"Unknown term: {semantic_term}")

        # Find all examples of this term
        examples = [s for s in self.eq_settings if s.semantic_label == semantic_term]

        # Average their latent representations
        self.encoder.eval()
        with torch.no_grad():
            latents = []
            for ex in examples:
                params = torch.tensor(ex.eq_params, dtype=torch.float32, device=self.device).unsqueeze(0)
                params = self.normalize_params(params)
                latent, _ = self.encoder(params)
                latents.append(latent)

            avg_latent = torch.stack(latents).mean(dim=0)
            eq_params = self.decoder(avg_latent)
            eq_params = self.denormalize_params(eq_params)

        return eq_params.cpu().numpy()[0]

    def interpolate_terms(self, term1: str, term2: str, alpha: float = 0.5) -> np.ndarray:
        """
        Interpolate between two semantic terms

        Args:
            term1: First semantic term
            term2: Second semantic term
            alpha: Interpolation factor [0, 1] (0=term1, 1=term2)

        Returns:
            eq_params: Interpolated EQ parameters
        """
        params1 = self.generate_eq_from_term(term1)
        params2 = self.generate_eq_from_term(term2)

        # Interpolate in parameter space
        params_interp = (1 - alpha) * params1 + alpha * params2

        return params_interp


# ========== CLI Entry Point ==========

if __name__ == "__main__":
    print("Neural EQ Morphing System - SAFE-DB")
    print("="*70)

    # Initialize system
    system = NeuralEQMorphingSAFEDB(latent_dim=32)

    # Load dataset
    system.load_dataset(min_examples=10, include_audio=False)

    # Train
    system.train(
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        contrastive_weight=0.5
    )

    # Save model
    system.save_model("neural_eq_safedb.pt")

    print("\n" + "="*70)
    print("Training complete! Model saved to: neural_eq_safedb.pt")
    print("="*70)
