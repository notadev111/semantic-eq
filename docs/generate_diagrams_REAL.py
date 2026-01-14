"""
Generate REAL Technical Diagrams from Trained Model
====================================================

Uses actual trained model data - NO SIMULATION!
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.neural_eq_morphing import NeuralResidualEQEncoder, NeuralResidualEQDecoder


def load_real_model_data():
    """Load actual trained model"""
    model_path = Path(__file__).parent.parent / "neural_eq_model.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}")

    print(f"Loading trained model from {model_path}...")
    checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')

    # Initialize encoder
    encoder = NeuralResidualEQEncoder(input_dim=40, latent_dim=32)
    encoder.load_state_dict(checkpoint['encoder'])
    encoder.eval()

    return encoder, checkpoint


def create_real_latent_space_diagram():
    """Create latent space visualization from REAL trained model"""

    encoder, checkpoint = load_real_model_data()

    # Get real training data
    params = checkpoint['training_data']['params']
    labels = checkpoint['training_data']['labels']
    idx_to_semantic = checkpoint['idx_to_semantic']

    print(f"Loaded {len(params)} real training examples")
    print(f"Computing latent representations...")

    # Encode all training examples to latent space
    with torch.no_grad():
        latent_all, _ = encoder(params)
        latent_np = latent_all.numpy()

    # Project to 2D using PCA (faster than t-SNE for this many points)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    latent_2d = pca.fit_transform(latent_np)

    print(f"PCA variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")

    # Select key semantic terms to visualize
    key_terms = ['warm', 'bright', 'punchy', 'smooth', 'dark', 'harsh']
    semantic_to_idx = checkpoint['semantic_to_idx']

    # Filter to only terms that exist
    key_terms = [t for t in key_terms if t in semantic_to_idx]

    if not key_terms:
        print("Warning: None of the key terms found, using first 6 terms")
        key_terms = list(semantic_to_idx.keys())[:6]

    print(f"Visualizing terms: {key_terms}")

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot all points in light gray
    ax.scatter(latent_2d[:, 0], latent_2d[:, 1],
              c='lightgray', s=20, alpha=0.3, label='Other terms')

    # Color scheme
    colors = ['orangered', 'gold', 'red', 'lightblue', 'darkslateblue', 'darkred']

    # Plot key semantic terms
    for term, color in zip(key_terms, colors):
        if term not in semantic_to_idx:
            continue

        label_idx = semantic_to_idx[term]
        mask = (labels == label_idx).numpy()

        if mask.sum() == 0:
            continue

        # Points for this term
        term_points = latent_2d[mask]

        # Plot points
        ax.scatter(term_points[:, 0], term_points[:, 1],
                  c=color, s=50, alpha=0.6, edgecolors='black',
                  linewidth=0.5, label=term)

        # Compute and plot centroid
        centroid = term_points.mean(axis=0)
        ax.scatter(centroid[0], centroid[1], c=color, s=300, marker='*',
                  edgecolors='black', linewidth=2, zorder=10)

        # Label
        ax.text(centroid[0], centroid[1] + 0.15, term.upper(),
               ha='center', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Draw interpolation path between warm and bright (if both exist)
    if 'warm' in key_terms and 'bright' in key_terms:
        warm_idx = semantic_to_idx['warm']
        bright_idx = semantic_to_idx['bright']

        warm_mask = (labels == warm_idx).numpy()
        bright_mask = (labels == bright_idx).numpy()

        if warm_mask.sum() > 0 and bright_mask.sum() > 0:
            warm_center = latent_2d[warm_mask].mean(axis=0)
            bright_center = latent_2d[bright_mask].mean(axis=0)

            # Interpolation line
            ax.plot([warm_center[0], bright_center[0]],
                   [warm_center[1], bright_center[1]],
                   'k--', linewidth=3, alpha=0.7, label='Interpolation Path')

            # Sample points along interpolation
            for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
                x_interp = (1-alpha) * warm_center[0] + alpha * bright_center[0]
                y_interp = (1-alpha) * warm_center[1] + alpha * bright_center[1]

                ax.scatter(x_interp, y_interp, c='green', s=150, marker='D',
                          edgecolors='black', linewidth=2, zorder=11)

                # Only label a few points to avoid clutter
                if alpha in [0.0, 0.5, 1.0]:
                    ax.text(x_interp, y_interp - 0.2, f'α={alpha:.1f}',
                           ha='center', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    ax.set_xlabel('Principal Component 1 (PCA projection)', fontsize=12)
    ax.set_ylabel('Principal Component 2 (PCA projection)', fontsize=12)
    ax.set_title(f'Learned Latent Space Structure\\n(Real trained model with {len(params)} examples)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    return fig


def create_architecture_diagram():
    """System architecture - conceptual, not data-driven"""
    # Import from original
    from generate_diagrams import create_architecture_diagram as orig
    return orig()


def create_interpolation_flow_diagram():
    """Interpolation flow - conceptual, not data-driven"""
    from generate_diagrams import create_interpolation_flow_diagram as orig
    return orig()


def main():
    """Generate diagrams using REAL data where possible"""

    print("="*70)
    print("GENERATING DIAGRAMS FROM REAL TRAINED MODEL")
    print("="*70)

    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_dir = project_root / "outputs" / "plots" / "technical_diagrams"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")

    # Generate diagrams
    print("\n" + "-"*70)
    print("Generating diagrams...")
    print("-"*70)

    diagrams = {}

    # 1. Architecture (conceptual)
    print("\n[1/3] Architecture diagram (conceptual)...")
    diagrams['architecture'] = create_architecture_diagram()

    # 2. Latent space (REAL DATA)
    print("\n[2/3] Latent space clustering (REAL trained model)...")
    diagrams['latent_space'] = create_real_latent_space_diagram()

    # 3. Interpolation flow (conceptual)
    print("\n[3/3] Interpolation flow (conceptual)...")
    diagrams['interpolation_flow'] = create_interpolation_flow_diagram()

    # Save all diagrams
    print("\n" + "-"*70)
    print("Saving diagrams...")
    print("-"*70)

    for name, fig in diagrams.items():
        output_path = output_dir / f"{name}.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        file_size = output_path.stat().st_size / 1024  # KB
        print(f"[OK] Saved: {output_path.name} ({file_size:.0f} KB)")
        plt.close(fig)  # Free memory

    print("\n" + "="*70)
    print("COMPLETED - All diagrams use REAL data where applicable")
    print("="*70)
    print("\nGenerated diagrams:")
    print("  1. architecture.png - System overview (conceptual)")
    print("  2. latent_space.png - REAL latent space from trained model")
    print("  3. interpolation_flow.png - Inference pipeline (conceptual)")
    print("\nNote: Training loss curves NOT generated (no loss history saved)")
    print("      Re-run training with modified script to save loss history")


if __name__ == '__main__':
    main()
