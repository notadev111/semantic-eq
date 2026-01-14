"""
Neural EQ Morphing System - Lite Version
========================================

Simplified version that works without PyTorch dependencies.
Demonstrates the core concepts using traditional ML approaches.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor


@dataclass
class EQParameter:
    """Container for EQ parameter information"""
    name: str
    value: float
    range_min: float
    range_max: float
    is_frequency: bool = False
    is_gain: bool = False
    is_q: bool = False


@dataclass 
class EQSetting:
    """Complete EQ setting with metadata"""
    parameters: Dict[str, float]
    semantic_label: str
    source_dataset: str
    audio_context: Optional[str] = None
    engineer_id: Optional[str] = None
    confidence: float = 1.0


class SocialFXDatasetLoader:
    """
    Real SocialFX dataset loader with fallback to synthetic data
    """
    
    def __init__(self, cache_dir: str = "./eq_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def load_socialfx_dataset(self) -> List[EQSetting]:
        """Load real SocialFX EQ dataset from HuggingFace"""
        
        print("Loading SocialFX EQ dataset from HuggingFace...")
        
        try:
            import pandas as pd
            df = pd.read_parquet("hf://datasets/seungheondoh/socialfx-original/data/eq-00000-of-00001.parquet")
            
            print(f"Successfully loaded {len(df)} examples")
            
            # Convert to our EQSetting format
            eq_settings = []
            
            for idx, row in df.iterrows():
                # Parse parameters
                param_keys = row['param_keys']
                param_values = row['param_values']
                
                # Create parameter dictionary
                parameters = {}
                for key, value in zip(param_keys, param_values):
                    parameters[key] = float(value)
                
                # Parse semantic label
                semantic_label = str(row['text']).lower().strip()
                
                # Parse extra info
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
            
            print(f"Converted to EQSettings: {len(eq_settings)} examples")
            
            # Show semantic distribution
            from collections import Counter
            semantic_counts = Counter([s.semantic_label for s in eq_settings])
            print(f"Top semantic terms: {dict(semantic_counts.most_common(10))}")
            
            return eq_settings
            
        except Exception as e:
            print(f"Error loading SocialFX dataset: {e}")
            print("Falling back to synthetic dataset...")
            return self.generate_synthetic_fallback()
    
    def generate_synthetic_fallback(self, n_samples_per_term: int = 70) -> List[EQSetting]:
        """Generate synthetic EQ dataset based on real engineering practices"""
        
        print("Generating synthetic EQ dataset for research...")
        
        # Standard 5-band parametric EQ structure
        eq_structure = {
            'band1_gain': (-12, 12),
            'band1_freq': (20, 200),
            'band1_q': (0.1, 2.0),
            'band2_gain': (-12, 12),
            'band2_freq': (200, 800),
            'band2_q': (0.1, 10.0),
            'band3_gain': (-12, 12),
            'band3_freq': (800, 3200),
            'band3_q': (0.1, 10.0),
            'band4_gain': (-12, 12),
            'band4_freq': (3200, 12800),
            'band4_q': (0.1, 10.0),
            'band5_gain': (-12, 12),
            'band5_freq': (8000, 20000),
            'band5_q': (0.1, 2.0),
        }
        
        # Semantic terms and their typical EQ characteristics
        semantic_templates = {
            'warm': {
                'band1_gain': (2, 6),    # Boost low shelf
                'band2_gain': (0, 3),    # Slight low-mid boost
                'band3_gain': (-2, 1),   # Slight mid cut/boost
                'band4_gain': (-3, 0),   # Cut upper mids
                'band5_gain': (-4, -1),  # Cut high shelf
            },
            'bright': {
                'band1_gain': (-3, 0),   # Cut/neutral low shelf
                'band2_gain': (-2, 1),   # Slight low-mid variation
                'band3_gain': (-1, 2),   # Slight mid boost
                'band4_gain': (1, 4),    # Boost upper mids
                'band5_gain': (2, 6),    # Boost high shelf
            },
            'heavy': {
                'band1_gain': (4, 8),    # Strong low boost
                'band2_gain': (2, 5),    # Low-mid boost
                'band3_gain': (-1, 2),   # Mid variation
                'band4_gain': (-2, 1),   # Upper mid variation
                'band5_gain': (-3, 0),   # Cut highs
            },
            'soft': {
                'band1_gain': (0, 2),    # Gentle low boost
                'band2_gain': (-1, 1),   # Subtle low-mid
                'band3_gain': (-2, 0),   # Cut harsh mids
                'band4_gain': (-4, -1),  # Cut upper mids
                'band5_gain': (-5, -2),  # Cut high shelf
            },
            'sharp': {
                'band1_gain': (-2, 1),   # Low variation
                'band2_gain': (0, 3),    # Low-mid boost
                'band3_gain': (2, 5),    # Mid boost
                'band4_gain': (3, 6),    # Upper mid boost
                'band5_gain': (1, 4),    # High boost
            },
            'smooth': {
                'band1_gain': (-1, 2),   # Gentle low
                'band2_gain': (-1, 1),   # Subtle low-mid
                'band3_gain': (-2, 1),   # Gentle mid
                'band4_gain': (-3, 0),   # Cut harsh upper
                'band5_gain': (-2, 1),   # Gentle high
            },
            'punchy': {
                'band1_gain': (1, 4),    # Low punch
                'band2_gain': (3, 6),    # Low-mid punch
                'band3_gain': (2, 4),    # Mid punch
                'band4_gain': (-1, 2),   # Upper mid variation
                'band5_gain': (-2, 1),   # High variation
            },
            'hollow': {
                'band1_gain': (0, 3),    # Low boost
                'band2_gain': (-4, -1),  # Cut low-mid
                'band3_gain': (-5, -2),  # Cut mid
                'band4_gain': (-3, 0),   # Cut upper mid
                'band5_gain': (1, 4),    # High boost
            }
        }
        
        eq_settings = []
        np.random.seed(42)  # Reproducible results
        
        for semantic_term, gain_ranges in semantic_templates.items():
            # Generate multiple variations for each term
            n_variations = n_samples_per_term
            
            for i in range(n_variations):
                parameters = {}
                
                # Generate parameters for each band
                for band_idx in range(1, 6):
                    # Gain
                    gain_key = f'band{band_idx}_gain'
                    if gain_key in gain_ranges:
                        gain_min, gain_max = gain_ranges[gain_key]
                        parameters[gain_key] = np.random.uniform(gain_min, gain_max)
                    else:
                        parameters[gain_key] = np.random.uniform(-2, 2)
                    
                    # Frequency
                    freq_key = f'band{band_idx}_freq'
                    freq_min, freq_max = eq_structure[freq_key]
                    log_min, log_max = np.log10(freq_min), np.log10(freq_max)
                    parameters[freq_key] = 10 ** np.random.uniform(log_min, log_max)
                    
                    # Q factor
                    q_key = f'band{band_idx}_q'
                    q_min, q_max = eq_structure[q_key]
                    log_q_min, log_q_max = np.log10(q_min), np.log10(q_max)
                    parameters[q_key] = 10 ** np.random.uniform(log_q_min, log_q_max)
                
                # Create EQ setting
                eq_setting = EQSetting(
                    parameters=parameters,
                    semantic_label=semantic_term,
                    source_dataset="synthetic_neural_eq",
                    audio_context=f"synthetic_context_{i}",
                    confidence=0.8 + 0.2 * np.random.random()
                )
                
                eq_settings.append(eq_setting)
        
        print(f"Generated {len(eq_settings)} synthetic EQ settings")
        print(f"Terms: {list(semantic_templates.keys())}")
        print(f"Parameters per setting: {len(eq_settings[0].parameters)}")
        
        return eq_settings


class LatentEQMorphingSystem:
    """
    Latent EQ Morphing System using traditional ML
    
    Alternative to neural networks - uses PCA for latent space
    and k-NN regression for semantic generation
    """
    
    def __init__(self, latent_dim: int = 8):
        self.latent_dim = latent_dim
        
        # ML Models
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=latent_dim)
        self.semantic_models = {}  # One model per semantic term
        
        # Data
        self.semantic_to_idx = {}
        self.idx_to_semantic = {}
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
        
        # Extract parameters
        param_names = list(eq_settings[0].parameters.keys())
        n_params = len(param_names)
        
        # Convert to arrays
        params_array = np.zeros((len(eq_settings), n_params))
        labels_array = np.zeros(len(eq_settings), dtype=int)
        
        for i, setting in enumerate(eq_settings):
            for j, param_name in enumerate(param_names):
                params_array[i, j] = setting.parameters.get(param_name, 0.0)
            labels_array[i] = self.semantic_to_idx[setting.semantic_label]
        
        self.training_data = {
            'params': params_array,
            'labels': labels_array,
            'param_names': param_names,
            'n_params': n_params,
            'n_classes': len(unique_labels)
        }
        
        return True
    
    def train(self):
        """Train the latent space and semantic models"""
        
        if self.training_data is None:
            print("Error: No training data loaded")
            return
        
        print("Training latent EQ morphing system...")
        
        params = self.training_data['params']
        labels = self.training_data['labels']
        
        # Normalize parameters
        params_scaled = self.scaler.fit_transform(params)
        
        # Create latent space using PCA
        latent_features = self.pca.fit_transform(params_scaled)
        
        print(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_[:5]}")
        print(f"Total variance explained: {np.sum(self.pca.explained_variance_ratio_):.3f}")
        
        # Train one model per semantic term for generation
        for semantic_term, term_idx in self.semantic_to_idx.items():
            # Get examples for this term
            term_mask = (labels == term_idx)
            term_latent = latent_features[term_mask]
            term_params = params_scaled[term_mask]
            
            if len(term_latent) < 5:  # Need minimum examples
                print(f"Warning: Only {len(term_latent)} examples for '{semantic_term}'")
                continue
            
            # Train k-NN regressor from latent to parameters
            model = KNeighborsRegressor(n_neighbors=min(5, len(term_latent)))
            model.fit(term_latent, term_params)
            
            self.semantic_models[semantic_term] = {
                'model': model,
                'centroid': np.mean(term_latent, axis=0),
                'std': np.std(term_latent, axis=0),
                'n_examples': len(term_latent)
            }
        
        self.is_trained = True
        print(f"Training completed! Models for {len(self.semantic_models)} semantic terms")
    
    def generate_eq_from_semantic(self, semantic_term: str, variations: int = 5) -> List[Dict]:
        """Generate EQ parameters from semantic term"""
        
        if not self.is_trained:
            print("Error: System not trained yet")
            return []
        
        if semantic_term not in self.semantic_models:
            print(f"Error: Unknown semantic term '{semantic_term}'")
            print(f"Available terms: {list(self.semantic_models.keys())}")
            return []
        
        model_data = self.semantic_models[semantic_term]
        model = model_data['model']
        centroid = model_data['centroid']
        std = model_data['std']
        
        print(f"Generating {variations} variations for '{semantic_term}'...")
        
        results = []
        for i in range(variations):
            # Sample around centroid in latent space
            noise = np.random.normal(0, 0.2, size=centroid.shape)
            latent_sample = centroid + noise * std
            latent_sample = latent_sample.reshape(1, -1)
            
            # Predict parameters
            params_scaled = model.predict(latent_sample)[0]
            
            # Denormalize
            params_original = self.scaler.inverse_transform(params_scaled.reshape(1, -1))[0]
            
            # Convert to dictionary
            param_dict = {}
            for j, param_name in enumerate(self.training_data['param_names']):
                param_dict[param_name] = float(params_original[j])
            
            results.append({
                'parameters': param_dict,
                'semantic_term': semantic_term,
                'variation': i + 1,
                'method': 'latent_pca_knn',
                'n_examples': model_data['n_examples']
            })
        
        return results
    
    def morph_between_terms(self, term1: str, term2: str, steps: int = 10) -> List[Dict]:
        """Morph between two semantic terms in latent space"""
        
        if not self.is_trained:
            print("Error: System not trained yet")
            return []
        
        # Validate terms
        for term in [term1, term2]:
            if term not in self.semantic_models:
                print(f"Error: Unknown semantic term '{term}'")
                return []
        
        print(f"Morphing from '{term1}' to '{term2}' in {steps} steps...")
        
        centroid1 = self.semantic_models[term1]['centroid']
        centroid2 = self.semantic_models[term2]['centroid']
        
        results = []
        
        for i in range(steps):
            alpha = i / (steps - 1)  # 0 to 1
            interpolated = (1 - alpha) * centroid1 + alpha * centroid2
            interpolated = interpolated.reshape(1, -1)
            
            # Use model from closer term for prediction
            if alpha < 0.5:
                model = self.semantic_models[term1]['model']
            else:
                model = self.semantic_models[term2]['model']
            
            # Predict parameters
            params_scaled = model.predict(interpolated)[0]
            params_original = self.scaler.inverse_transform(params_scaled.reshape(1, -1))[0]
            
            # Convert to dictionary
            param_dict = {}
            for j, param_name in enumerate(self.training_data['param_names']):
                param_dict[param_name] = float(params_original[j])
            
            results.append({
                'parameters': param_dict,
                'semantic_term': f"{term1}→{term2}",
                'morph_step': i + 1,
                'morph_alpha': alpha,
                'method': 'latent_morph'
            })
        
        return results
    
    def explore_latent_space(self, n_samples: int = 20) -> List[Dict]:
        """Randomly sample from latent space"""
        
        if not self.is_trained:
            print("Error: System not trained yet")
            return []
        
        print(f"Exploring latent space with {n_samples} random samples...")
        
        # Sample from latent space (based on PCA distribution)
        params_scaled = self.scaler.transform(self.training_data['params'])
        latent_all = self.pca.transform(params_scaled)
        
        # Estimate latent space distribution
        latent_mean = np.mean(latent_all, axis=0)
        latent_cov = np.cov(latent_all.T)
        
        results = []
        
        for i in range(n_samples):
            # Sample from multivariate normal
            latent_sample = np.random.multivariate_normal(latent_mean, latent_cov)
            latent_sample = latent_sample.reshape(1, -1)
            
            # Convert back to parameter space
            params_scaled = self.pca.inverse_transform(latent_sample)
            params_original = self.scaler.inverse_transform(params_scaled)[0]
            
            # Convert to dictionary
            param_dict = {}
            for j, param_name in enumerate(self.training_data['param_names']):
                param_dict[param_name] = float(params_original[j])
            
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
        
        # Transform all data to latent space
        params_scaled = self.scaler.transform(self.training_data['params'])
        latent_features = self.pca.transform(params_scaled)
        labels = self.training_data['labels']
        
        # Plot latent space if 2D or use t-SNE
        if self.latent_dim >= 2:
            try:
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                
                # Plot first two latent dimensions
                ax1 = axes[0]
                for label_idx in range(self.training_data['n_classes']):
                    mask = (labels == label_idx)
                    if np.any(mask):
                        label_name = self.idx_to_semantic[label_idx]
                        ax1.scatter(latent_features[mask, 0], latent_features[mask, 1], 
                                   label=label_name, alpha=0.7, s=30)
                
                ax1.set_title("PCA Latent Space (First 2 Components)")
                ax1.set_xlabel(f"PC1 ({self.pca.explained_variance_ratio_[0]:.2%} var)")
                ax1.set_ylabel(f"PC2 ({self.pca.explained_variance_ratio_[1]:.2%} var)")
                ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
                # Show semantic term centroids
                ax2 = axes[1]
                for term, model_data in self.semantic_models.items():
                    centroid = model_data['centroid']
                    ax2.scatter(centroid[0], centroid[1], s=200, alpha=0.8, 
                               label=f"{term} (n={model_data['n_examples']})")
                    ax2.annotate(term, (centroid[0], centroid[1]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
                
                ax2.set_title("Semantic Term Centroids")
                ax2.set_xlabel("PC1")
                ax2.set_ylabel("PC2")
                ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
                plt.tight_layout()
                plt.show()
                
                print(f"Latent space analysis complete:")
                print(f"  Latent dimension: {self.latent_dim}")
                print(f"  Total variance explained: {np.sum(self.pca.explained_variance_ratio_):.3f}")
                print(f"  Semantic models trained: {len(self.semantic_models)}")
                
            except Exception as e:
                print(f"Visualization error: {e}")
                print("Analysis complete without visualization")


def main():
    """Demo the latent EQ morphing system"""
    
    print("Neural EQ Morphing System - Lite Version (Traditional ML)")
    print("=" * 60)
    
    # Initialize SocialFX dataset loader
    loader = SocialFXDatasetLoader()
    
    # Load real SocialFX dataset (or fallback to synthetic)
    print("Loading EQ dataset...")
    eq_settings = loader.load_socialfx_dataset()
    
    # Initialize latent system
    print(f"\nInitializing Latent EQ Morphing System...")
    system = LatentEQMorphingSystem(latent_dim=8)
    
    # Load data
    if not system.load_dataset(eq_settings):
        print("Failed to load dataset")
        return
    
    # Train the system
    system.train()
    
    # Demo semantic generation
    print(f"\n" + "="*50)
    print("DEMO: SEMANTIC EQ GENERATION")
    print(f"="*50)
    
    test_terms = ['warm', 'bright', 'punchy']
    
    for term in test_terms:
        if term in system.semantic_models:
            print(f"\nGenerating '{term}' EQ variations:")
            variations = system.generate_eq_from_semantic(term, variations=3)
            
            for i, variation in enumerate(variations, 1):
                print(f"  Variation {i} (from {variation['n_examples']} examples):")
                params = variation['parameters']
                # Show first few parameters
                for param_name in list(params.keys())[:6]:
                    print(f"    {param_name}: {params[param_name]:.3f}")
    
    # Demo morphing
    print(f"\n" + "="*50)
    print("DEMO: SEMANTIC MORPHING")
    print(f"="*50)
    
    if 'warm' in system.semantic_models and 'bright' in system.semantic_models:
        print(f"\nMorphing from 'warm' to 'bright':")
        morph_steps = system.morph_between_terms('warm', 'bright', steps=5)
        
        for step in morph_steps:
            alpha = step['morph_alpha']
            print(f"  Step {step['morph_step']} (alpha={alpha:.2f}):")
            params = step['parameters']
            # Show key parameters
            key_params = ['band1_gain', 'band5_gain']
            for param in key_params:
                if param in params:
                    print(f"    {param}: {params[param]:.3f}")
    
    # Demo exploration
    print(f"\n" + "="*50)
    print("DEMO: LATENT SPACE EXPLORATION")
    print(f"="*50)
    
    explorations = system.explore_latent_space(n_samples=3)
    for i, exploration in enumerate(explorations, 1):
        print(f"  Random sample {i}:")
        params = exploration['parameters']
        for param in ['band1_gain', 'band3_gain', 'band5_gain']:
            if param in params:
                print(f"    {param}: {params[param]:.3f}")
    
    # Analyze latent space
    print(f"\n" + "="*50)
    print("LATENT SPACE ANALYSIS")
    print(f"="*50)
    
    system.analyze_latent_space()
    
    print(f"\n" + "="*60)
    print("LITE SYSTEM DEMONSTRATION COMPLETE!")
    print(f"="*60)
    print(f"This demonstrates the core concepts of neural EQ morphing")
    print(f"using traditional ML approaches (PCA + k-NN regression).")
    print(f"\nAvailable semantic terms: {list(system.semantic_models.keys())}")
    print(f"Latent space dimension: {system.latent_dim}")
    print(f"Total variance explained: {np.sum(system.pca.explained_variance_ratio_):.3f}")


if __name__ == '__main__':
    main()