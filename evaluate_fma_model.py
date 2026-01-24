"""
FMA-Trained Model Evaluation Suite
==================================

Comprehensive evaluation of the FMA-trained E2E audio encoder:

1. Genre-Semantic Clustering: Do genres cluster meaningfully in latent space?
2. Semantic Consistency: Does the model produce consistent semantic predictions?
3. Cross-Domain Generalization: Does FMA training improve on unseen audio?
4. Latent Space Quality: Is the learned space well-structured?

Usage:
    python evaluate_fma_model.py \
        --encoder audio_encoder_fma_e2e.pt \
        --v2-model neural_eq_safedb_v2.pt \
        --fma-path ./data/fma/fma_small \
        --metadata-path ./data/fma/fma_metadata \
        --output eval_results/

References:
    - FMA Dataset: https://github.com/mdeff/fma
    - t-SNE: van der Maaten & Hinton (2008)
"""

import os
import sys
import argparse
import json
import glob
import numpy as np
import pandas as pd
import torch
import torchaudio
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from datetime import datetime


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.audio_encoder import FastAudioEncoder, AudioEncoderConfig
from core.neural_eq_morphing_safedb_v2 import NeuralEQMorphingSAFEDBV2


def load_encoder(encoder_path: str, device: str) -> FastAudioEncoder:
    """Load trained audio encoder."""
    checkpoint = torch.load(encoder_path, map_location=device, weights_only=False)

    if 'config' in checkpoint:
        config = AudioEncoderConfig(**checkpoint['config'])
        encoder = FastAudioEncoder(config)
    else:
        encoder = FastAudioEncoder()

    if 'audio_encoder_state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['audio_encoder_state_dict'])
    elif 'model_state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['state_dict'])
    else:
        encoder.load_state_dict(checkpoint)

    encoder.to(device)
    encoder.eval()
    return encoder


def load_v2_model(v2_model_path: str, device: str) -> NeuralEQMorphingSAFEDBV2:
    """Load V2 semantic EQ model."""
    model = NeuralEQMorphingSAFEDBV2(latent_dim=32, device=device)
    model.load_dataset(min_examples=3)
    model.load_model(v2_model_path)
    return model


def load_fma_metadata(metadata_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load FMA metadata for genre labels."""
    tracks_path = Path(metadata_path) / 'tracks.csv'
    genres_path = Path(metadata_path) / 'genres.csv'

    # Load tracks with genre info
    tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1])
    genres = pd.read_csv(genres_path, index_col=0)

    return tracks, genres


def get_track_genre(track_id: int, tracks_df: pd.DataFrame) -> Optional[str]:
    """Get genre for a track ID."""
    try:
        genre_id = tracks_df.loc[track_id, ('track', 'genre_top')]
        return genre_id if pd.notna(genre_id) else None
    except:
        return None


def load_audio_clip(filepath: str, sample_rate: int = 44100, duration: float = 3.0) -> Optional[torch.Tensor]:
    """Load and preprocess an audio clip."""
    try:
        waveform, sr = torchaudio.load(filepath)

        # Resample if needed
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Take center clip
        n_samples = int(sample_rate * duration)
        if waveform.shape[1] > n_samples:
            start = (waveform.shape[1] - n_samples) // 2
            waveform = waveform[:, start:start + n_samples]
        elif waveform.shape[1] < n_samples:
            pad = n_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        # Normalize
        waveform = waveform / (waveform.abs().max() + 1e-8)

        return waveform.squeeze(0)
    except:
        return None


def compute_semantic_centroids(v2_model: NeuralEQMorphingSAFEDBV2, device: str) -> Dict[str, torch.Tensor]:
    """Compute semantic centroids from V2 model."""
    centroids = {}

    for term in v2_model.term_to_idx.keys():
        examples = [s for s in v2_model.eq_settings if s.semantic_label == term]
        if len(examples) < 3:
            continue

        eq_params_list = [torch.from_numpy(ex.eq_params_normalized).float() for ex in examples[:20]]
        eq_batch = torch.stack(eq_params_list).to(device)

        with torch.no_grad():
            z_semantic, _ = v2_model.encoder(eq_batch)
            centroids[term] = z_semantic.mean(dim=0)

    return centroids


def get_semantic_scores(z_audio: torch.Tensor, centroids: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Compute cosine similarity to each semantic centroid."""
    scores = {}
    for term, centroid in centroids.items():
        sim = torch.nn.functional.cosine_similarity(z_audio.unsqueeze(0), centroid.unsqueeze(0))
        scores[term] = sim.item()
    return scores


# ============================================================================
# EVALUATION 1: Genre-Semantic Clustering
# ============================================================================

def evaluate_genre_clustering(
    encoder: FastAudioEncoder,
    v2_model: NeuralEQMorphingSAFEDBV2,
    fma_path: str,
    metadata_path: str,
    device: str,
    n_samples_per_genre: int = 50,
    output_dir: str = 'eval_results'
) -> Dict:
    """
    Evaluate whether different genres cluster in latent space.

    Good clustering indicates the model learns musically meaningful features.
    """
    print("\n" + "="*70)
    print("EVALUATION 1: Genre-Semantic Clustering")
    print("="*70)

    # Load metadata
    tracks_df, genres_df = load_fma_metadata(metadata_path)

    # Find audio files and their genres
    audio_files = glob.glob(str(Path(fma_path) / '**/*.mp3'), recursive=True)
    print(f"Found {len(audio_files)} audio files")

    # Group by genre
    genre_files = defaultdict(list)
    for filepath in audio_files:
        track_id = int(Path(filepath).stem)
        genre = get_track_genre(track_id, tracks_df)
        if genre:
            genre_files[genre].append(filepath)

    print(f"Genres found: {list(genre_files.keys())}")

    # Sample files per genre
    latent_vectors = []
    genre_labels = []

    centroids = compute_semantic_centroids(v2_model, device)

    for genre, files in tqdm(genre_files.items(), desc="Processing genres"):
        sampled = np.random.choice(files, min(n_samples_per_genre, len(files)), replace=False)

        for filepath in sampled:
            audio = load_audio_clip(filepath)
            if audio is None:
                continue

            with torch.no_grad():
                audio = audio.unsqueeze(0).to(device)
                z = encoder(audio)
                latent_vectors.append(z.cpu().numpy().squeeze())
                genre_labels.append(genre)

    latent_vectors = np.array(latent_vectors)
    genre_labels = np.array(genre_labels)

    print(f"\nProcessed {len(latent_vectors)} samples across {len(set(genre_labels))} genres")

    # Compute clustering metrics
    unique_genres = list(set(genre_labels))
    genre_to_idx = {g: i for i, g in enumerate(unique_genres)}
    label_indices = np.array([genre_to_idx[g] for g in genre_labels])

    # Silhouette score (higher = better clustering)
    silhouette = silhouette_score(latent_vectors, label_indices)
    print(f"\nSilhouette Score: {silhouette:.4f}")
    print("  (>0.5 = strong clustering, 0.25-0.5 = reasonable, <0.25 = weak)")

    # Compute genre centroids in latent space
    genre_centroids = {}
    for genre in unique_genres:
        mask = genre_labels == genre
        genre_centroids[genre] = latent_vectors[mask].mean(axis=0)

    # Inter-genre distances
    print("\nInter-genre distances (higher = more distinct):")
    distances = []
    for i, g1 in enumerate(unique_genres):
        for g2 in unique_genres[i+1:]:
            dist = np.linalg.norm(genre_centroids[g1] - genre_centroids[g2])
            distances.append((g1, g2, dist))
    distances.sort(key=lambda x: -x[2])
    for g1, g2, dist in distances[:5]:
        print(f"  {g1} <-> {g2}: {dist:.4f}")

    # Visualize with t-SNE
    print("\nGenerating t-SNE visualization...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    latent_2d = tsne.fit_transform(latent_vectors)

    plt.figure(figsize=(12, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_genres)))

    for i, genre in enumerate(unique_genres):
        mask = genre_labels == genre
        plt.scatter(latent_2d[mask, 0], latent_2d[mask, 1],
                   c=[colors[i]], label=genre, alpha=0.6, s=50)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f'Genre Clustering in Latent Space\n(Silhouette: {silhouette:.3f})')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/genre_clustering_tsne.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Genre-Semantic correlation
    print("\nGenre-Semantic Correlations:")
    genre_semantic_scores = {}
    for genre in unique_genres:
        mask = genre_labels == genre
        genre_latents = latent_vectors[mask]

        # Average semantic scores for this genre
        scores = defaultdict(list)
        for z in genre_latents:
            z_tensor = torch.from_numpy(z).float().to(device)
            s = get_semantic_scores(z_tensor, centroids)
            for term, score in s.items():
                scores[term].append(score)

        avg_scores = {term: np.mean(vals) for term, vals in scores.items()}
        top_semantics = sorted(avg_scores.items(), key=lambda x: -x[1])[:3]
        genre_semantic_scores[genre] = top_semantics
        print(f"  {genre}: {', '.join([f'{t}({s:.2f})' for t,s in top_semantics])}")

    results = {
        'silhouette_score': silhouette,
        'n_samples': len(latent_vectors),
        'n_genres': len(unique_genres),
        'genre_semantic_correlations': {g: [(t, float(s)) for t,s in scores]
                                        for g, scores in genre_semantic_scores.items()},
    }

    return results


# ============================================================================
# EVALUATION 2: Semantic Consistency
# ============================================================================

def evaluate_semantic_consistency(
    encoder: FastAudioEncoder,
    v2_model: NeuralEQMorphingSAFEDBV2,
    fma_path: str,
    device: str,
    n_samples: int = 100,
    output_dir: str = 'eval_results'
) -> Dict:
    """
    Evaluate semantic consistency across different clips from same track.

    If the model is consistent, different windows of the same track should
    have similar semantic profiles.
    """
    print("\n" + "="*70)
    print("EVALUATION 2: Semantic Consistency (Temporal)")
    print("="*70)

    audio_files = glob.glob(str(Path(fma_path) / '**/*.mp3'), recursive=True)
    sampled_files = np.random.choice(audio_files, min(n_samples, len(audio_files)), replace=False)

    centroids = compute_semantic_centroids(v2_model, device)

    consistencies = []

    for filepath in tqdm(sampled_files, desc="Evaluating consistency"):
        try:
            waveform, sr = torchaudio.load(filepath)
            if sr != 44100:
                resampler = torchaudio.transforms.Resample(sr, 44100)
                waveform = resampler(waveform)

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)

            # Extract 3 non-overlapping windows
            window_size = 44100 * 3  # 3 seconds
            if waveform.shape[0] < window_size * 3:
                continue

            windows = [
                waveform[:window_size],
                waveform[window_size:window_size*2],
                waveform[window_size*2:window_size*3],
            ]

            # Get semantic scores for each window
            window_scores = []
            for w in windows:
                w = w / (w.abs().max() + 1e-8)
                with torch.no_grad():
                    z = encoder(w.unsqueeze(0).to(device))
                    scores = get_semantic_scores(z.squeeze(0), centroids)
                    window_scores.append(scores)

            # Compute consistency (average pairwise correlation)
            terms = list(window_scores[0].keys())
            vectors = np.array([[s[t] for t in terms] for s in window_scores])

            # Pairwise correlations
            corrs = []
            for i in range(3):
                for j in range(i+1, 3):
                    corr = np.corrcoef(vectors[i], vectors[j])[0, 1]
                    corrs.append(corr)

            consistencies.append(np.mean(corrs))

        except Exception as e:
            continue

    mean_consistency = np.mean(consistencies)
    std_consistency = np.std(consistencies)

    print(f"\nSemantic Consistency Score: {mean_consistency:.4f} ± {std_consistency:.4f}")
    print("  (>0.8 = excellent, 0.6-0.8 = good, <0.6 = concerning)")

    # Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(consistencies, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(mean_consistency, color='red', linestyle='--', label=f'Mean: {mean_consistency:.3f}')
    plt.xlabel('Temporal Consistency (Correlation)')
    plt.ylabel('Count')
    plt.title('Semantic Consistency Across Track Windows')
    plt.legend()
    plt.savefig(f'{output_dir}/semantic_consistency.png', dpi=150)
    plt.close()

    return {
        'mean_consistency': float(mean_consistency),
        'std_consistency': float(std_consistency),
        'n_tracks': len(consistencies),
    }


# ============================================================================
# EVALUATION 3: Model Comparison (FMA vs Pink Noise trained)
# ============================================================================

def evaluate_model_comparison(
    fma_encoder: FastAudioEncoder,
    baseline_encoder: FastAudioEncoder,
    v2_model: NeuralEQMorphingSAFEDBV2,
    test_audio_paths: List[str],
    device: str,
    output_dir: str = 'eval_results'
) -> Dict:
    """
    Compare FMA-trained model vs baseline (pink noise trained).

    We expect FMA-trained model to:
    1. Have more stable predictions on real music
    2. Better distinguish between different musical content
    """
    print("\n" + "="*70)
    print("EVALUATION 3: FMA vs Baseline Model Comparison")
    print("="*70)

    centroids = compute_semantic_centroids(v2_model, device)

    fma_scores_list = []
    baseline_scores_list = []

    for filepath in tqdm(test_audio_paths, desc="Comparing models"):
        audio = load_audio_clip(filepath)
        if audio is None:
            continue

        with torch.no_grad():
            audio_tensor = audio.unsqueeze(0).to(device)

            z_fma = fma_encoder(audio_tensor)
            z_baseline = baseline_encoder(audio_tensor)

            scores_fma = get_semantic_scores(z_fma.squeeze(0), centroids)
            scores_baseline = get_semantic_scores(z_baseline.squeeze(0), centroids)

            fma_scores_list.append(scores_fma)
            baseline_scores_list.append(scores_baseline)

    # Compare distributions
    terms = list(fma_scores_list[0].keys())

    print("\nTop semantic differences (FMA - Baseline):")
    diffs = {}
    for term in terms:
        fma_mean = np.mean([s[term] for s in fma_scores_list])
        baseline_mean = np.mean([s[term] for s in baseline_scores_list])
        diffs[term] = fma_mean - baseline_mean

    sorted_diffs = sorted(diffs.items(), key=lambda x: abs(x[1]), reverse=True)
    for term, diff in sorted_diffs[:10]:
        print(f"  {term}: {diff:+.4f}")

    # Variance comparison (is FMA more consistent?)
    print("\nVariance comparison (lower = more consistent):")
    for term in terms[:5]:
        fma_var = np.var([s[term] for s in fma_scores_list])
        baseline_var = np.var([s[term] for s in baseline_scores_list])
        print(f"  {term}: FMA={fma_var:.4f}, Baseline={baseline_var:.4f}")

    return {
        'n_samples': len(fma_scores_list),
        'semantic_diffs': {t: float(d) for t, d in sorted_diffs},
    }


# ============================================================================
# EVALUATION 4: Semantic-Feature Correlation (NEW)
# ============================================================================

def evaluate_feature_correlation(
    encoder: FastAudioEncoder,
    v2_model: NeuralEQMorphingSAFEDBV2,
    fma_path: str,
    metadata_path: str,
    device: str,
    n_samples: int = 500,
    output_dir: str = 'eval_results'
) -> Dict:
    """
    Correlate learned semantic embeddings with standard audio features.

    This shows whether our "bright", "warm", etc. relate to:
    - Spectral features (centroid, bandwidth, rolloff)
    - Echo Nest descriptors (energy, acousticness, danceability)

    Novel research contribution: Semantic EQ descriptors vs standard features.
    """
    print("\n" + "="*70)
    print("EVALUATION 4: Semantic-Feature Correlation")
    print("="*70)

    # Load FMA features
    features_path = Path(metadata_path) / 'features.csv'
    echonest_path = Path(metadata_path) / 'echonest.csv'

    features_df = None
    echonest_df = None

    if features_path.exists():
        print("Loading features.csv...")
        features_df = pd.read_csv(features_path, index_col=0, header=[0, 1, 2])
        print(f"  Loaded {len(features_df)} track features")
    else:
        print("  features.csv not found")

    if echonest_path.exists():
        print("Loading echonest.csv...")
        echonest_df = pd.read_csv(echonest_path, index_col=0, header=[0, 1])
        print(f"  Loaded {len(echonest_df)} Echo Nest features")
    else:
        print("  echonest.csv not found")

    if features_df is None and echonest_df is None:
        print("No feature files found, skipping correlation analysis")
        return {}

    # Get audio files
    audio_files = glob.glob(str(Path(fma_path) / '**/*.mp3'), recursive=True)

    # Build track_id -> filepath mapping
    track_id_to_path = {}
    for f in audio_files:
        try:
            track_id = int(Path(f).stem)
            track_id_to_path[track_id] = f
        except:
            continue

    # Compute semantic centroids
    centroids = compute_semantic_centroids(v2_model, device)
    semantic_terms = list(centroids.keys())

    # Collect data
    semantic_scores_all = []
    spectral_features_all = []
    echonest_features_all = []
    valid_track_ids = []

    # Sample tracks that have both audio and features
    if features_df is not None:
        available_ids = list(set(features_df.index) & set(track_id_to_path.keys()))
    elif echonest_df is not None:
        available_ids = list(set(echonest_df.index) & set(track_id_to_path.keys()))
    else:
        available_ids = list(track_id_to_path.keys())

    sampled_ids = np.random.choice(available_ids, min(n_samples, len(available_ids)), replace=False)

    print(f"\nAnalyzing {len(sampled_ids)} tracks...")

    for track_id in tqdm(sampled_ids, desc="Computing correlations"):
        filepath = track_id_to_path.get(track_id)
        if not filepath:
            continue

        # Load audio and get semantic scores
        audio = load_audio_clip(filepath)
        if audio is None:
            continue

        with torch.no_grad():
            audio_tensor = audio.unsqueeze(0).to(device)
            z = encoder(audio_tensor)
            scores = get_semantic_scores(z.squeeze(0), centroids)

        semantic_scores_all.append([scores[t] for t in semantic_terms])
        valid_track_ids.append(track_id)

        # Get spectral features
        if features_df is not None and track_id in features_df.index:
            try:
                # Extract key spectral features
                row = features_df.loc[track_id]
                spectral = {
                    'spectral_centroid': row[('spectral_centroid', 'mean', '01')],
                    'spectral_bandwidth': row[('spectral_bandwidth', 'mean', '01')],
                    'spectral_rolloff': row[('spectral_rolloff', 'mean', '01')],
                    'zero_crossing_rate': row[('zcr', 'mean', '01')],
                    'rmse': row[('rmse', 'mean', '01')],
                }
                spectral_features_all.append(spectral)
            except:
                spectral_features_all.append(None)
        else:
            spectral_features_all.append(None)

        # Get Echo Nest features
        if echonest_df is not None and track_id in echonest_df.index:
            try:
                row = echonest_df.loc[track_id]
                echonest = {
                    'acousticness': row[('echonest', 'audio_features')].get('acousticness', np.nan),
                    'danceability': row[('echonest', 'audio_features')].get('danceability', np.nan),
                    'energy': row[('echonest', 'audio_features')].get('energy', np.nan),
                    'instrumentalness': row[('echonest', 'audio_features')].get('instrumentalness', np.nan),
                    'speechiness': row[('echonest', 'audio_features')].get('speechiness', np.nan),
                    'tempo': row[('echonest', 'audio_features')].get('tempo', np.nan),
                    'valence': row[('echonest', 'audio_features')].get('valence', np.nan),
                }
                echonest_features_all.append(echonest)
            except:
                echonest_features_all.append(None)
        else:
            echonest_features_all.append(None)

    semantic_scores_all = np.array(semantic_scores_all)

    # Compute correlations
    results = {'n_samples': len(valid_track_ids)}

    # Semantic vs Spectral correlations
    if any(s is not None for s in spectral_features_all):
        print("\n--- Semantic vs Spectral Feature Correlations ---")

        spectral_keys = ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'zero_crossing_rate', 'rmse']
        spectral_matrix = []
        valid_idx = []

        for i, s in enumerate(spectral_features_all):
            if s is not None:
                spectral_matrix.append([s[k] for k in spectral_keys])
                valid_idx.append(i)

        if spectral_matrix:
            spectral_matrix = np.array(spectral_matrix)
            semantic_subset = semantic_scores_all[valid_idx]

            correlations_spectral = {}
            for i, sem_term in enumerate(semantic_terms):
                for j, spec_feat in enumerate(spectral_keys):
                    corr = np.corrcoef(semantic_subset[:, i], spectral_matrix[:, j])[0, 1]
                    if not np.isnan(corr):
                        correlations_spectral[f"{sem_term}_vs_{spec_feat}"] = corr

            # Print strongest correlations
            sorted_corrs = sorted(correlations_spectral.items(), key=lambda x: abs(x[1]), reverse=True)
            print("\nStrongest Semantic-Spectral correlations:")
            for pair, corr in sorted_corrs[:10]:
                print(f"  {pair}: {corr:+.3f}")

            results['spectral_correlations'] = {k: float(v) for k, v in sorted_corrs[:20]}

            # Highlight expected patterns
            print("\n  Expected patterns check:")
            bright_centroid = correlations_spectral.get('bright_vs_spectral_centroid', 0)
            print(f"    'bright' vs spectral_centroid: {bright_centroid:+.3f} (expect positive)")
            warm_centroid = correlations_spectral.get('warm_vs_spectral_centroid', 0)
            print(f"    'warm' vs spectral_centroid: {warm_centroid:+.3f} (expect negative)")

    # Semantic vs Echo Nest correlations
    if any(s is not None for s in echonest_features_all):
        print("\n--- Semantic vs Echo Nest Correlations ---")

        echonest_keys = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'speechiness', 'valence']
        echonest_matrix = []
        valid_idx = []

        for i, e in enumerate(echonest_features_all):
            if e is not None:
                row = [e.get(k, np.nan) for k in echonest_keys]
                if not any(np.isnan(row)):
                    echonest_matrix.append(row)
                    valid_idx.append(i)

        if echonest_matrix:
            echonest_matrix = np.array(echonest_matrix)
            semantic_subset = semantic_scores_all[valid_idx]

            correlations_echonest = {}
            for i, sem_term in enumerate(semantic_terms):
                for j, en_feat in enumerate(echonest_keys):
                    corr = np.corrcoef(semantic_subset[:, i], echonest_matrix[:, j])[0, 1]
                    if not np.isnan(corr):
                        correlations_echonest[f"{sem_term}_vs_{en_feat}"] = corr

            sorted_corrs = sorted(correlations_echonest.items(), key=lambda x: abs(x[1]), reverse=True)
            print("\nStrongest Semantic-EchoNest correlations:")
            for pair, corr in sorted_corrs[:10]:
                print(f"  {pair}: {corr:+.3f}")

            results['echonest_correlations'] = {k: float(v) for k, v in sorted_corrs[:20]}

    # Create correlation heatmap
    if 'spectral_correlations' in results:
        print("\nGenerating correlation heatmap...")

        # Build correlation matrix for visualization
        top_semantics = ['warm', 'bright', 'thin', 'muddy', 'punchy', 'clear']
        top_semantics = [s for s in top_semantics if s in semantic_terms]

        spectral_keys = ['spectral_centroid', 'spectral_bandwidth', 'rmse']

        corr_matrix = np.zeros((len(top_semantics), len(spectral_keys)))
        for i, sem in enumerate(top_semantics):
            for j, spec in enumerate(spectral_keys):
                key = f"{sem}_vs_{spec}"
                corr_matrix[i, j] = results['spectral_correlations'].get(key, 0)

        plt.figure(figsize=(10, 6))
        plt.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
        plt.colorbar(label='Correlation')
        plt.xticks(range(len(spectral_keys)), spectral_keys, rotation=45, ha='right')
        plt.yticks(range(len(top_semantics)), top_semantics)
        plt.title('Semantic Descriptors vs Spectral Features')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/semantic_spectral_correlation.png', dpi=150)
        plt.close()

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate FMA-trained model')
    parser.add_argument('--encoder', type=str, required=True, help='Path to FMA-trained encoder')
    parser.add_argument('--baseline-encoder', type=str, default='audio_encoder_e2e.pt',
                       help='Path to baseline encoder for comparison')
    parser.add_argument('--v2-model', type=str, default='neural_eq_safedb_v2.pt',
                       help='Path to V2 model')
    parser.add_argument('--fma-path', type=str, required=True, help='Path to FMA dataset')
    parser.add_argument('--metadata-path', type=str, required=True, help='Path to FMA metadata')
    parser.add_argument('--output', type=str, default=None, help='Output directory (default: results/eval_YYYY-MM-DD)')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--n-samples', type=int, default=50, help='Samples per genre')
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dated output directory
    if args.output is None:
        date_str = datetime.now().strftime('%Y-%m-%d')
        output_dir = f'results/eval_{date_str}'
    else:
        output_dir = args.output

    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}/")

    # Load models
    print("\nLoading models...")
    encoder = load_encoder(args.encoder, device)
    v2_model = load_v2_model(args.v2_model, device)

    results = {}

    # Evaluation 1: Genre Clustering
    results['genre_clustering'] = evaluate_genre_clustering(
        encoder, v2_model, args.fma_path, args.metadata_path,
        device, n_samples_per_genre=args.n_samples, output_dir=output_dir
    )

    # Evaluation 2: Semantic Consistency
    results['semantic_consistency'] = evaluate_semantic_consistency(
        encoder, v2_model, args.fma_path, device,
        n_samples=100, output_dir=output_dir
    )

    # Evaluation 3: Model Comparison (if baseline exists)
    if Path(args.baseline_encoder).exists() and args.baseline_encoder != args.encoder:
        baseline_encoder = load_encoder(args.baseline_encoder, device)
        audio_files = glob.glob(str(Path(args.fma_path) / '**/*.mp3'), recursive=True)[:100]
        results['model_comparison'] = evaluate_model_comparison(
            encoder, baseline_encoder, v2_model, audio_files,
            device, output_dir=output_dir
        )

    # Evaluation 4: Feature Correlation (semantic vs spectral/echonest)
    results['feature_correlation'] = evaluate_feature_correlation(
        encoder, v2_model, args.fma_path, args.metadata_path,
        device, n_samples=500, output_dir=output_dir
    )

    # Save results with timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    results['metadata'] = {
        'timestamp': timestamp,
        'encoder': args.encoder,
        'v2_model': args.v2_model,
        'fma_path': args.fma_path,
        'device': device,
    }

    with open(f'{output_dir}/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"\nGenre Clustering Silhouette: {results['genre_clustering']['silhouette_score']:.4f}")
    print(f"Semantic Consistency: {results['semantic_consistency']['mean_consistency']:.4f}")
    if 'feature_correlation' in results and results['feature_correlation']:
        print(f"Feature Correlations: {results['feature_correlation'].get('n_samples', 0)} tracks analyzed")
    print(f"\nResults saved to: {output_dir}/")
    print("  - genre_clustering_tsne.png")
    print("  - semantic_consistency.png")
    print("  - semantic_spectral_correlation.png (if features available)")
    print("  - evaluation_results.json")


if __name__ == '__main__':
    main()
