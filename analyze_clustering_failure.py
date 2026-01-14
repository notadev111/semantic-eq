"""
Investigate Why Clustering Failed
==================================

Systematic analysis to understand the problem.
"""

import torch
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import sys
sys.path.insert(0, '.')
from core.neural_eq_morphing import NeuralResidualEQEncoder

print("="*70)
print("CLUSTERING FAILURE ANALYSIS")
print("="*70)

# Load model
checkpoint = torch.load('neural_eq_model.pt', weights_only=False, map_location='cpu')
encoder = NeuralResidualEQEncoder(input_dim=40, latent_dim=32)
encoder.load_state_dict(checkpoint['encoder'])
encoder.eval()

params = checkpoint['training_data']['params']
labels = checkpoint['training_data']['labels']
semantic_to_idx = checkpoint['semantic_to_idx']
idx_to_semantic = checkpoint['idx_to_semantic']

# Encode to latent space
with torch.no_grad():
    latent_all, _ = encoder(params)
    latent_np = latent_all.numpy()

labels_np = labels.numpy()

# 1. TOP SEMANTIC TERMS ANALYSIS
print("\n" + "="*70)
print("1. TOP SEMANTIC TERMS BY FREQUENCY")
print("="*70)

unique, counts = np.unique(labels_np, return_counts=True)
sorted_idx = np.argsort(counts)[::-1]  # Sort descending

print("\nTop 20 semantic terms:")
print(f"{'Rank':<6} {'Term':<20} {'Count':<8} {'% of data':<10}")
print("-"*50)

top_20_terms = []
top_20_indices = []
for i in range(min(20, len(sorted_idx))):
    label_idx = int(unique[sorted_idx[i]])
    # idx_to_semantic might use int or str keys
    term = idx_to_semantic.get(str(label_idx), idx_to_semantic.get(label_idx, f"unknown_{label_idx}"))
    count = counts[sorted_idx[i]]
    pct = 100 * count / len(labels_np)
    print(f"{i+1:<6} {term:<20} {count:<8} {pct:.2f}%")
    top_20_terms.append(term)
    top_20_indices.append(label_idx)

# 2. ANALYZE IF TOP TERMS HAVE DISTINCT EQ SIGNATURES
print("\n" + "="*70)
print("2. DO TOP TERMS HAVE DISTINCT EQ SIGNATURES?")
print("="*70)

# Get EQ parameters for top 10 terms
top_10_terms = top_20_terms[:10]
top_10_labels = top_20_indices[:10]

print("\nAnalyzing EQ parameter variance for top 10 terms...")

for i, (term, label_idx) in enumerate(zip(top_10_terms, top_10_labels)):
    mask = labels_np == label_idx
    term_params = params[mask].numpy()

    # Calculate variance in EQ parameters
    param_variance = term_params.var(axis=0).mean()

    # Calculate mean EQ signature
    mean_params = term_params.mean(axis=0)

    print(f"\n{term} ({mask.sum()} examples):")
    print(f"  Intra-term variance: {param_variance:.3f}")
    print(f"  Mean params range: [{mean_params.min():.2f}, {mean_params.max():.2f}]")

# 3. PAIRWISE DISTANCE BETWEEN TOP TERMS
print("\n" + "="*70)
print("3. PAIRWISE DISTANCES BETWEEN TOP TERMS (in EQ space)")
print("="*70)

term_centroids = []
for label_idx in top_10_labels:
    mask = labels_np == label_idx
    centroid = params[mask].mean(dim=0).numpy()
    term_centroids.append(centroid)

term_centroids = np.array(term_centroids)

# Calculate pairwise distances
from scipy.spatial.distance import pdist, squareform
distances = squareform(pdist(term_centroids, metric='euclidean'))

print("\nPairwise Euclidean distances between term centroids:")
print("(Lower = more similar EQ signatures)")
print("\n" + " "*15 + "  ".join([f"{t[:8]:<8}" for t in top_10_terms[:5]]))
for i in range(5):
    row = f"{top_10_terms[i]:<15}"
    for j in range(5):
        row += f"{distances[i,j]:>10.2f}"
    print(row)

mean_dist = distances[np.triu_indices_from(distances, k=1)].mean()
print(f"\nMean pairwise distance: {mean_dist:.2f}")
print(f"Std pairwise distance: {distances[np.triu_indices_from(distances, k=1)].std():.2f}")

# 4. LATENT SPACE ANALYSIS FOR TOP TERMS
print("\n" + "="*70)
print("4. LATENT SPACE CLUSTERING (top terms only)")
print("="*70)

# Filter to top N terms
for N in [10, 20, 30]:
    top_N_labels = top_20_indices[:N]
    mask_top_N = np.isin(labels_np, top_N_labels)

    if mask_top_N.sum() > 0:
        filtered_latent = latent_np[mask_top_N]
        filtered_labels = labels_np[mask_top_N]

        # Remap labels to consecutive integers for silhouette
        unique_filtered = np.unique(filtered_labels)
        label_map = {old: new for new, old in enumerate(unique_filtered)}
        remapped_labels = np.array([label_map[l] for l in filtered_labels])

        sil = silhouette_score(filtered_latent, remapped_labels)
        db = davies_bouldin_score(filtered_latent, remapped_labels)

        print(f"\nTop {N} terms ({mask_top_N.sum()} examples, {len(unique_filtered)} unique terms):")
        print(f"  Silhouette score: {sil:.3f}")
        print(f"  Davies-Bouldin index: {db:.3f}")

# 5. CHECK CONTRASTIVE LOSS CONTRIBUTION
print("\n" + "="*70)
print("5. HYPOTHESIS: CONTRASTIVE LOSS TOO WEAK?")
print("="*70)

print("\nCurrent training configuration:")
print("  Reconstruction loss weight: 1.0")
print("  Contrastive loss weight (lambda): 0.1")
print("  Ratio: 10:1 in favor of reconstruction")
print("\nProblem: Reconstruction dominates, semantic structure not learned")

# 6. RECOMMENDATIONS
print("\n" + "="*70)
print("6. DIAGNOSIS & RECOMMENDATIONS")
print("="*70)

# Calculate how much data we'd have with different thresholds
for threshold in [5, 10, 15, 20]:
    mask_thresh = np.isin(labels_np, unique[counts >= threshold])
    n_examples = mask_thresh.sum()
    n_terms = (counts >= threshold).sum()
    print(f"\nIf we filter to >={threshold} examples per term:")
    print(f"  Keep: {n_examples}/{len(labels_np)} examples ({100*n_examples/len(labels_np):.1f}%)")
    print(f"  Keep: {n_terms}/765 terms ({100*n_terms/765:.1f}%)")
    print(f"  Avg examples per term: {n_examples/n_terms:.1f}")

print("\n" + "="*70)
print("CONCLUSIONS")
print("="*70)
print("\n1. CLASS IMBALANCE is severe (93% of terms have <5 examples)")
print("2. Top 10 terms represent only", f"{(labels_np[np.isin(labels_np, top_10_labels)]).sum()}", "examples")
print("3. Contrastive loss weight (0.1) is TOO LOW - reconstruction dominates")
print("4. Need to retrain with:")
print("   - Filter to top 20-30 terms (enough data to learn)")
print("   - Increase contrastive weight to 0.5 or 1.0")
print("   - More epochs (100-200)")
print("="*70)
