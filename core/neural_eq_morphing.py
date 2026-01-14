"""
Neural EQ Morphing System - SocialFX Dataset
=============================================

Neural Residual Networks + Contrastive Learning for semantic EQ generation.
Trained on real engineer data from SocialFX dataset.

Usage:
    python neural_eq_morphing.py
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
import json
import warnings
warnings.filterwarnings("ignore")

# Advanced neural network components
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import pytorch_lightning as pl
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    print("PyTorch Lightning not available, using basic training")


@dataclass
class EQSetting:
    parameters: Dict[str, float]
    semantic_label: str
    source_dataset: str
    audio_context: Optional[str] = None
    engineer_id: Optional[str] = None
    confidence: float = 1.0


class SocialFXDatasetLoader:

    def __init__(self, cache_dir: str = "./eq_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def load_socialfx_dataset(self) -> List[EQSetting]:
        print("Loading SocialFX EQ dataset from HuggingFace...")

        import pandas as pd
        df = pd.read_parquet("hf://datasets/seungheondoh/socialfx-original/data/eq-00000-of-00001.parquet")

        print(f"Successfully loaded {len(df)} examples")
        print(f"Columns: {list(df.columns)}")

        eq_settings = []

        for idx, row in df.iterrows():
            param_keys = row['param_keys']
            param_values = row['param_values']

            parameters = {}
            for key, value in zip(param_keys, param_values):
                parameters[key] = float(value)

            semantic_label = str(row['text']).lower().strip()
            extra_info = eval(row['extra']) if isinstance(row['extra'], str) else {}

            eq_setting = EQSetting(
                parameters=parameters,
                semantic_label=semantic_label,
                source_dataset="socialfx_original",
                audio_context=row['id'],
                engineer_id=f"socialfx_engineer_{idx}",
                confidence=extra_info.get('ratings_consistency', 1.0)
            )

            eq_settings.append(eq_setting)

        return eq_settings

    def analyze_dataset(self, eq_settings: List[EQSetting]) -> Dict:
        if not eq_settings:
            return {"error": "No EQ settings to analyze"}

        print(f"\n{'='*60}")
        print("EQ DATASET ANALYSIS")
        print(f"{'='*60}")

        n_settings = len(eq_settings)
        semantic_terms = [setting.semantic_label for setting in eq_settings]
        unique_terms = list(set(semantic_terms))

        print(f"Total EQ settings: {n_settings}")
        print(f"Unique semantic terms: {len(unique_terms)}")
        print(f"Parameters per setting: {len(eq_settings[0].parameters)}")

        from collections import Counter
        term_counts = Counter(semantic_terms)
        print(f"\\nTerm distribution:")
        for term, count in term_counts.most_common(10):
            print(f"  {term}: {count} examples")

        all_params = list(eq_settings[0].parameters.keys())
        param_stats = {}

        for param in all_params:
            values = [setting.parameters[param] for setting in eq_settings
                     if param in setting.parameters]
            param_stats[param] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }

        print(f"\\nParameter ranges:")
        for param, stats in param_stats.items():
            print(f"  {param}: {stats['min']:.2f} to {stats['max']:.2f} "
                  f"(mean={stats['mean']:.2f}, std={stats['std']:.2f})")

        return {
            'n_settings': n_settings,
            'unique_terms': unique_terms,
            'term_counts': term_counts,
            'param_stats': param_stats,
            'parameter_names': all_params
        }


class NeuralResidualEQEncoder(nn.Module):

    def __init__(self, input_dim: int, latent_dim: int = 64, hidden_dims: List[int] = None):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 256, 128]

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(ResidualBlock(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        self.encoder_layers = nn.ModuleList(layers)

        self.to_latent = nn.Sequential(
            nn.Linear(prev_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Tanh()
        )

        self.semantic_proj = nn.Linear(latent_dim, 128)

    def forward(self, eq_params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = eq_params

        for layer in self.encoder_layers:
            x = layer(x)

        latent = self.to_latent(x)
        semantic_emb = self.semantic_proj(latent)

        return latent, semantic_emb


class ResidualBlock(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.main_path = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
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


class NeuralResidualEQDecoder(nn.Module):

    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: List[int] = None):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 256, 128]

        self.latent_dim = latent_dim
        self.output_dim = output_dim

        layers = []
        prev_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.append(ResidualBlock(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        self.decoder_layers = nn.ModuleList(layers)

        self.to_params = nn.Sequential(
            nn.Linear(prev_dim, output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim)
        )

        self.gain_head = nn.Sequential(nn.Linear(output_dim, output_dim // 3), nn.Tanh())
        self.freq_head = nn.Sequential(nn.Linear(output_dim, output_dim // 3), nn.Sigmoid())
        self.q_head = nn.Sequential(nn.Linear(output_dim, output_dim // 3), nn.Sigmoid())

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        x = latent

        for layer in self.decoder_layers:
            x = layer(x)

        base_params = self.to_params(x)

        gains = self.gain_head(base_params) * 12.0
        freqs = self.freq_head(base_params)
        qs = self.q_head(base_params) * 9.9 + 0.1

        n_bands = self.output_dim // 3
        eq_params = torch.zeros_like(base_params)

        for i in range(n_bands):
            eq_params[:, i*3] = gains[:, i]
            eq_params[:, i*3 + 1] = freqs[:, i]
            eq_params[:, i*3 + 2] = qs[:, i]

        return eq_params


class TransformerEQEncoder(nn.Module):
    """
    Transformer-based EQ encoder for sequence modeling
    
    Treats EQ bands as sequences and captures inter-band relationships
    """
    
    def __init__(self, n_bands: int = 5, latent_dim: int = 64):
        super().__init__()
        
        self.n_bands = n_bands
        self.latent_dim = latent_dim
        
        # Input embedding for EQ parameters (gain, freq, Q per band)
        self.param_embedding = nn.Linear(3, 64)  # 3 params per band
        
        # Positional encoding for band positions
        self.pos_encoding = nn.Embedding(n_bands, 64)
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=64,
            nhead=8,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=4)
        
        # Output projection
        self.to_latent = nn.Sequential(
            nn.Linear(64 * n_bands, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Tanh()
        )
    
    def forward(self, eq_params: torch.Tensor) -> torch.Tensor:
        """
        Forward pass treating EQ bands as sequence
        
        Args:
            eq_params: [batch_size, n_bands * 3] EQ parameters
            
        Returns:
            latent: [batch_size, latent_dim] latent representation
        """
        
        batch_size = eq_params.shape[0]
        
        # Reshape to sequence format [batch_size, n_bands, 3]
        eq_sequence = eq_params.view(batch_size, self.n_bands, 3)
        
        # Embed parameters
        param_emb = self.param_embedding(eq_sequence)  # [batch, n_bands, 64]
        
        # Add positional encoding
        positions = torch.arange(self.n_bands, device=eq_params.device)
        pos_emb = self.pos_encoding(positions).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Combine embeddings
        x = param_emb + pos_emb
        
        # Apply transformer
        x = self.transformer(x)  # [batch, n_bands, 64]
        
        # Flatten and project to latent
        x = x.view(batch_size, -1)  # [batch, n_bands * 64]
        latent = self.to_latent(x)
        
        return latent


class ContrastiveEQLoss(nn.Module):
    """
    Contrastive loss for learning semantic EQ embeddings
    
    Pulls together EQ settings with same semantic labels,
    pushes apart those with different labels
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss
        
        Args:
            embeddings: [batch_size, embedding_dim] normalized embeddings
            labels: [batch_size] semantic label indices
            
        Returns:
            loss: scalar contrastive loss
        """
        
        batch_size = embeddings.shape[0]
        device = embeddings.device
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create label mask
        labels = labels.unsqueeze(1)
        label_mask = (labels == labels.T).float()
        
        # Remove self-similarities
        identity_mask = torch.eye(batch_size, device=device)
        label_mask = label_mask * (1 - identity_mask)
        
        # Compute contrastive loss
        exp_sim = torch.exp(similarity)
        
        # Positive pairs (same label)
        pos_sim = exp_sim * label_mask
        
        # All pairs for normalization
        all_sim = exp_sim.sum(dim=1, keepdim=True) - exp_sim * identity_mask
        
        # Avoid division by zero
        pos_sim = pos_sim.sum(dim=1)
        pos_count = label_mask.sum(dim=1)
        
        # Only compute loss where we have positive pairs
        valid_mask = (pos_count > 0).float()
        
        loss = -torch.log((pos_sim + 1e-8) / (all_sim.squeeze() + 1e-8))
        loss = (loss * valid_mask).mean()
        
        return loss


class NeuralEQMorphingSystem:
    """
    Complete Neural EQ Morphing System with training and inference
    """
    
    def __init__(self, latent_dim: int = 64, device: str = 'cpu'):
        self.device = device
        self.latent_dim = latent_dim
        
        # Will be initialized when dataset is loaded
        self.encoder = None
        self.decoder = None
        self.transformer_encoder = None
        
        self.semantic_to_idx = {}
        self.idx_to_semantic = {}
        
        # Training state
        self.training_data = None
        self.is_trained = False
        
    def load_dataset(self, eq_settings: List[EQSetting]) -> bool:
        """Load and prepare dataset for training"""
        
        if not eq_settings:
            print("Error: No EQ settings provided")
            return False
        
        print(f"Loading {len(eq_settings)} EQ settings...")
        
        # Build semantic label mapping
        unique_labels = list(set(setting.semantic_label for setting in eq_settings))
        self.semantic_to_idx = {label: i for i, label in enumerate(unique_labels)}
        self.idx_to_semantic = {i: label for label, i in self.semantic_to_idx.items()}
        
        print(f"Semantic terms: {unique_labels}")
        
        # Extract parameters and normalize
        param_names = list(eq_settings[0].parameters.keys())
        n_params = len(param_names)
        
        # Convert to numpy arrays
        params_array = np.zeros((len(eq_settings), n_params))
        labels_array = np.zeros(len(eq_settings), dtype=int)
        
        for i, setting in enumerate(eq_settings):
            for j, param_name in enumerate(param_names):
                params_array[i, j] = setting.parameters.get(param_name, 0.0)
            labels_array[i] = self.semantic_to_idx[setting.semantic_label]
        
        # Normalize parameters
        self.param_means = np.mean(params_array, axis=0)
        self.param_stds = np.std(params_array, axis=0) + 1e-6  # Avoid division by zero
        params_normalized = (params_array - self.param_means) / self.param_stds
        
        # Convert to tensors
        self.training_data = {
            'params': torch.FloatTensor(params_normalized).to(self.device),
            'labels': torch.LongTensor(labels_array).to(self.device),
            'param_names': param_names,
            'n_params': n_params,
            'n_classes': len(unique_labels)
        }
        
        # Initialize models
        self._initialize_models()
        
        return True
    
    def _initialize_models(self):
        """Initialize neural network models"""
        
        n_params = self.training_data['n_params']
        
        # Neural Residual Encoder/Decoder
        self.encoder = NeuralResidualEQEncoder(
            input_dim=n_params,
            latent_dim=self.latent_dim
        ).to(self.device)
        
        self.decoder = NeuralResidualEQDecoder(
            latent_dim=self.latent_dim,
            output_dim=n_params
        ).to(self.device)
        
        # Transformer Encoder (alternative)
        n_bands = n_params // 3  # Assume 3 params per band
        if n_params % 3 == 0 and n_bands > 0:
            self.transformer_encoder = TransformerEQEncoder(
                n_bands=n_bands,
                latent_dim=self.latent_dim
            ).to(self.device)
        
        print(f"Initialized models for {n_params} parameters, {self.latent_dim}D latent space")
    
    def train(self, epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001):
        """Train the neural EQ morphing system"""
        
        if self.training_data is None:
            print("Error: No training data loaded")
            return
        
        print(f"Training neural EQ morphing system...")
        print(f"Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {learning_rate}")
        
        # Prepare data
        dataset = torch.utils.data.TensorDataset(
            self.training_data['params'],
            self.training_data['labels']
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        # Optimizers
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate
        )
        
        # Loss functions
        reconstruction_loss = nn.MSELoss()
        contrastive_loss = ContrastiveEQLoss()
        
        # Training loop
        self.encoder.train()
        self.decoder.train()
        
        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0
            
            for batch_params, batch_labels in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                latent, semantic_emb = self.encoder(batch_params)
                reconstructed = self.decoder(latent)
                
                # Losses
                recon_loss = reconstruction_loss(reconstructed, batch_params)
                contrast_loss = contrastive_loss(semantic_emb, batch_labels)
                
                # Combined loss
                loss = recon_loss + 0.1 * contrast_loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}/{epochs}: Loss = {avg_loss:.4f}")
        
        self.is_trained = True
        print(f"Training completed!")
        
        # Evaluation mode
        self.encoder.eval()
        self.decoder.eval()
    
    def generate_eq_from_semantic(self, semantic_term: str, variations: int = 5) -> List[Dict]:
        """Generate EQ parameters from semantic term"""
        
        if not self.is_trained:
            print("Error: System not trained yet")
            return []
        
        if semantic_term not in self.semantic_to_idx:
            print(f"Error: Unknown semantic term '{semantic_term}'")
            print(f"Available terms: {list(self.semantic_to_idx.keys())}")
            return []
        
        print(f"Generating {variations} variations for '{semantic_term}'...")
        
        # Find examples of this semantic term in training data
        label_idx = self.semantic_to_idx[semantic_term]
        mask = (self.training_data['labels'] == label_idx)
        examples = self.training_data['params'][mask]
        
        if len(examples) == 0:
            print(f"Error: No examples found for '{semantic_term}'")
            return []
        
        # Encode examples to latent space
        with torch.no_grad():
            latent_examples, _ = self.encoder(examples)
            
            # Compute centroid in latent space
            centroid = torch.mean(latent_examples, dim=0, keepdim=True)
            
            # Generate variations around centroid
            results = []
            for i in range(variations):
                # Add noise for variation
                noise = torch.randn_like(centroid) * 0.1
                varied_latent = centroid + noise
                
                # Decode to parameters
                eq_params = self.decoder(varied_latent)
                
                # Denormalize
                eq_params_np = eq_params.cpu().numpy().flatten()
                denormalized = eq_params_np * self.param_stds + self.param_means
                
                # Convert to dictionary
                param_dict = {}
                for j, param_name in enumerate(self.training_data['param_names']):
                    param_dict[param_name] = float(denormalized[j])
                
                results.append({
                    'parameters': param_dict,
                    'semantic_term': semantic_term,
                    'variation': i + 1,
                    'method': 'neural_residual'
                })
        
        return results
    
    def morph_between_terms(self, term1: str, term2: str, steps: int = 10) -> List[Dict]:
        """Morph between two semantic terms in latent space"""
        
        if not self.is_trained:
            print("Error: System not trained yet")
            return []
        
        # Validate terms
        for term in [term1, term2]:
            if term not in self.semantic_to_idx:
                print(f"Error: Unknown semantic term '{term}'")
                return []
        
        print(f"Morphing from '{term1}' to '{term2}' in {steps} steps...")
        
        # Get centroids for both terms
        with torch.no_grad():
            centroids = {}
            
            for term in [term1, term2]:
                label_idx = self.semantic_to_idx[term]
                mask = (self.training_data['labels'] == label_idx)
                examples = self.training_data['params'][mask]
                
                if len(examples) == 0:
                    print(f"Error: No examples for '{term}'")
                    return []
                
                latent_examples, _ = self.encoder(examples)
                centroids[term] = torch.mean(latent_examples, dim=0)
            
            # Interpolate between centroids
            results = []
            centroid1, centroid2 = centroids[term1], centroids[term2]
            
            for i in range(steps):
                alpha = i / (steps - 1)  # 0 to 1
                interpolated = (1 - alpha) * centroid1 + alpha * centroid2
                interpolated = interpolated.unsqueeze(0)
                
                # Decode
                eq_params = self.decoder(interpolated)
                eq_params_np = eq_params.cpu().numpy().flatten()
                denormalized = eq_params_np * self.param_stds + self.param_means
                
                # Convert to dictionary
                param_dict = {}
                for j, param_name in enumerate(self.training_data['param_names']):
                    param_dict[param_name] = float(denormalized[j])
                
                results.append({
                    'parameters': param_dict,
                    'semantic_term': f"{term1}→{term2}",
                    'morph_step': i + 1,
                    'morph_alpha': alpha,
                    'method': 'neural_morph'
                })
        
        return results
    
    def interpolate_semantic_terms(self, term1: str, term2: str, alpha: float) -> Dict:
        """
        Real-time semantic interpolation between two terms

        This is the interactive "Semantic Mode" - smoothly blend between two
        semantic descriptors using a single interpolation parameter.

        Args:
            term1: First semantic term (e.g., "warm")
            term2: Second semantic term (e.g., "bright")
            alpha: Interpolation factor [0, 1]
                   0 = 100% term1, 1 = 100% term2, 0.5 = 50/50 blend

        Returns:
            Dictionary with interpolated EQ parameters
        """

        if not self.is_trained:
            print("Error: System not trained yet")
            return {}

        # Validate terms
        for term in [term1, term2]:
            if term not in self.semantic_to_idx:
                print(f"Error: Unknown semantic term '{term}'")
                print(f"Available: {list(self.semantic_to_idx.keys())}")
                return {}

        # Clamp alpha to [0, 1]
        alpha = max(0.0, min(1.0, alpha))

        # Get semantic centroids (cached for performance)
        if not hasattr(self, '_semantic_centroids'):
            self._cache_semantic_centroids()

        centroid1 = self._semantic_centroids[term1]
        centroid2 = self._semantic_centroids[term2]

        # Interpolate in latent space
        with torch.no_grad():
            interpolated = (1 - alpha) * centroid1 + alpha * centroid2
            interpolated = interpolated.unsqueeze(0)

            # Decode to EQ parameters
            eq_params = self.decoder(interpolated)
            eq_params_np = eq_params.cpu().numpy().flatten()
            denormalized = eq_params_np * self.param_stds + self.param_means

            # Convert to dictionary
            param_dict = {}
            for j, param_name in enumerate(self.training_data['param_names']):
                param_dict[param_name] = float(denormalized[j])

        return {
            'parameters': param_dict,
            'term_a': term1,
            'term_b': term2,
            'interpolation': alpha,
            'blend_percentage': {
                term1: f"{(1-alpha)*100:.1f}%",
                term2: f"{alpha*100:.1f}%"
            },
            'description': f"{term1} ({(1-alpha)*100:.0f}%) → {term2} ({alpha*100:.0f}%)",
            'method': 'semantic_interpolation'
        }

    def _cache_semantic_centroids(self):
        """Pre-compute and cache semantic term centroids for fast interpolation"""

        print("Caching semantic centroids for fast interpolation...")
        self._semantic_centroids = {}

        with torch.no_grad():
            for term, label_idx in self.semantic_to_idx.items():
                mask = (self.training_data['labels'] == label_idx)
                examples = self.training_data['params'][mask]

                if len(examples) > 0:
                    latent_examples, _ = self.encoder(examples)
                    self._semantic_centroids[term] = torch.mean(latent_examples, dim=0)

        print(f"Cached {len(self._semantic_centroids)} semantic centroids")

    def explore_latent_space(self, n_samples: int = 20) -> List[Dict]:
        """Randomly sample from latent space to explore novel EQ settings"""
        
        if not self.is_trained:
            print("Error: System not trained yet")
            return []
        
        print(f"Exploring latent space with {n_samples} random samples...")
        
        results = []
        
        with torch.no_grad():
            for i in range(n_samples):
                # Sample from latent space (assuming Tanh output, so [-1, 1])
                random_latent = torch.tanh(torch.randn(1, self.latent_dim)).to(self.device)
                
                # Decode to parameters
                eq_params = self.decoder(random_latent)
                eq_params_np = eq_params.cpu().numpy().flatten()
                denormalized = eq_params_np * self.param_stds + self.param_means
                
                # Convert to dictionary
                param_dict = {}
                for j, param_name in enumerate(self.training_data['param_names']):
                    param_dict[param_name] = float(denormalized[j])
                
                results.append({
                    'parameters': param_dict,
                    'semantic_term': 'random_exploration',
                    'sample': i + 1,
                    'method': 'latent_exploration'
                })
        
        return results
    
    def analyze_latent_space(self):
        """Analyze the learned latent space structure"""
        
        if not self.is_trained:
            print("Error: System not trained yet")
            return
        
        print("Analyzing latent space structure...")
        
        # Encode all training data
        with torch.no_grad():
            all_latent, all_semantic = self.encoder(self.training_data['params'])
            all_latent_np = all_latent.cpu().numpy()
            all_labels = self.training_data['labels'].cpu().numpy()
        
        # Dimensionality reduction for visualization
        if all_latent_np.shape[0] > 50:  # Only if we have enough samples
            try:
                # t-SNE
                tsne = TSNE(n_components=2, random_state=42)
                latent_2d = tsne.fit_transform(all_latent_np)
                
                # Plot
                plt.figure(figsize=(12, 5))
                
                plt.subplot(1, 2, 1)
                for label_idx in range(self.training_data['n_classes']):
                    mask = (all_labels == label_idx)
                    if np.any(mask):
                        label_name = self.idx_to_semantic[label_idx]
                        plt.scatter(latent_2d[mask, 0], latent_2d[mask, 1], 
                                   label=label_name, alpha=0.7)
                
                plt.title("t-SNE of Latent Space")
                plt.xlabel("t-SNE 1")
                plt.ylabel("t-SNE 2")
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
                # Show latent space statistics
                plt.subplot(1, 2, 2)
                latent_norms = np.linalg.norm(all_latent_np, axis=1)
                plt.hist(latent_norms, bins=20, alpha=0.7)
                plt.title("Latent Vector Norms")
                plt.xlabel("L2 Norm")
                plt.ylabel("Count")
                
                plt.tight_layout()
                plt.show()
                
                print(f"Latent space analysis complete:")
                print(f"  Dimension: {self.latent_dim}")
                print(f"  Mean norm: {np.mean(latent_norms):.3f}")
                print(f"  Std norm: {np.std(latent_norms):.3f}")
                
            except Exception as e:
                print(f"Visualization error: {e}")
                print("Skipping visualization, but latent space is working")


def main():
    """Demo the complete neural EQ morphing system"""
    
    print("Neural EQ Morphing System - Complete Implementation")
    print("=" * 60)
    
    # Initialize SocialFX dataset loader  
    loader = SocialFXDatasetLoader()
    
    # Load real SocialFX dataset
    print("Loading real SocialFX EQ dataset...")
    eq_settings = loader.load_socialfx_dataset()
    
    # Analyze dataset
    analysis = loader.analyze_dataset(eq_settings)
    
    if 'error' in analysis:
        print("Error loading dataset")
        return
    
    # Initialize neural system
    print(f"\\nInitializing Neural EQ Morphing System...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    system = NeuralEQMorphingSystem(latent_dim=32, device=device)
    
    # Load data into system
    if not system.load_dataset(eq_settings):
        print("Failed to load dataset into system")
        return
    
    # Train the system
    print(f"\\nTraining neural networks...")
    system.train(epochs=50, batch_size=16, learning_rate=0.001)
    
    # Demo semantic generation
    print(f"\\n" + "="*50)
    print("DEMO: SEMANTIC EQ GENERATION")
    print(f"="*50)
    
    test_terms = ['warm', 'bright', 'punchy']
    
    for term in test_terms:
        if term in system.semantic_to_idx:
            print(f"\\nGenerating '{term}' EQ variations:")
            variations = system.generate_eq_from_semantic(term, variations=3)
            
            for i, variation in enumerate(variations, 1):
                print(f"  Variation {i}:")
                params = variation['parameters']
                # Show first few parameters as example
                for param_name in list(params.keys())[:6]:
                    print(f"    {param_name}: {params[param_name]:.3f}")
                if len(params) > 6:
                    print(f"    ... and {len(params) - 6} more parameters")
    
    # Demo morphing
    print(f"\\n" + "="*50)
    print("DEMO: SEMANTIC MORPHING")
    print(f"="*50)
    
    if 'warm' in system.semantic_to_idx and 'bright' in system.semantic_to_idx:
        print(f"\\nMorphing from 'warm' to 'bright':")
        morph_steps = system.morph_between_terms('warm', 'bright', steps=5)
        
        for step in morph_steps:
            alpha = step['morph_alpha']
            print(f"  Step {step['morph_step']} (α={alpha:.2f}):")
            params = step['parameters']
            # Show key parameters
            key_params = ['band1_gain', 'band5_gain'] if 'band1_gain' in params else list(params.keys())[:2]
            for param in key_params:
                if param in params:
                    print(f"    {param}: {params[param]:.3f}")
    
    # Demo NEW FEATURE: Interactive Semantic Interpolation
    print(f"\\n" + "="*50)
    print("DEMO: INTERACTIVE SEMANTIC INTERPOLATION")
    print(f"="*50)
    print("This is the 'Semantic Mode' - blend between semantic descriptors")

    if 'warm' in system.semantic_to_idx and 'bright' in system.semantic_to_idx:
        print(f"\\nInterpolating between 'warm' and 'bright':")
        print("(Simulating slider positions from 0.0 to 1.0)")

        # Simulate different slider positions
        slider_positions = [0.0, 0.25, 0.5, 0.75, 1.0]

        for alpha in slider_positions:
            result = system.interpolate_semantic_terms('warm', 'bright', alpha)

            if result:
                print(f"\\n  Slider @ {alpha:.2f}: {result['description']}")
                params = result['parameters']

                # Show key EQ bands to demonstrate the blend
                key_params = ['band1_gain', 'band5_gain'] if 'band1_gain' in params else list(params.keys())[:2]
                for param in key_params:
                    if param in params:
                        print(f"    {param}: {params[param]:+.2f} dB")

        print(f"\\n  💡 This allows real-time EQ exploration with a single slider!")
        print(f"     α=0.0 → 100% warm")
        print(f"     α=0.5 → 50% warm, 50% bright")
        print(f"     α=1.0 → 100% bright")

    # Demo latent exploration
    print(f"\\n" + "="*50)
    print("DEMO: LATENT SPACE EXPLORATION")
    print(f"="*50)
    
    print(f"\\nExploring novel EQ settings from latent space:")
    explorations = system.explore_latent_space(n_samples=3)
    
    for i, exploration in enumerate(explorations, 1):
        print(f"  Random sample {i}:")
        params = exploration['parameters']
        # Show sample of parameters
        sample_params = list(params.keys())[:4]
        for param in sample_params:
            print(f"    {param}: {params[param]:.3f}")
    
    # Analyze latent space
    print(f"\\n" + "="*50)
    print("LATENT SPACE ANALYSIS")
    print(f"="*50)
    
    system.analyze_latent_space()
    
    print(f"\\n" + "="*60)
    print("SYSTEM READY FOR INTERACTIVE USE!")
    print(f"="*60)
    print(f"Available semantic terms: {list(system.semantic_to_idx.keys())}")
    print(f"Latent dimension: {system.latent_dim}")
    print(f"Parameter count: {system.training_data['n_params']}")
    print(f"\\nThe neural EQ morphing system is now trained and ready!")
    print(f"You can:")
    print(f"  1. Generate EQ variations for any semantic term")
    print(f"  2. Morph smoothly between different semantic styles")
    print(f"  3. 🆕 INTERPOLATE between terms with a single slider (Semantic Mode)")
    print(f"  4. Explore novel EQ settings via latent space sampling")
    print(f"  5. Analyze the learned semantic-to-parameter relationships")
    print(f"\\nExample usage for Semantic Interpolation:")
    print(f"  result = system.interpolate_semantic_terms('warm', 'bright', alpha=0.5)")
    print(f"  # Returns 50/50 blend of warm and bright characteristics")


if __name__ == '__main__':
    main()