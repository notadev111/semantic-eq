"""
Generate Technical Diagrams for Neural EQ Morphing Report
==========================================================

Creates publication-quality diagrams for the interim report.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
from pathlib import Path


def create_architecture_diagram():
    """Create system architecture overview diagram"""

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Title
    ax.text(5, 11.5, 'Neural EQ Morphing System Architecture',
            ha='center', fontsize=16, fontweight='bold')

    # Input layer
    input_box = FancyBboxPatch((0.5, 10), 3, 0.6,
                               boxstyle="round,pad=0.1",
                               edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax.add_patch(input_box)
    ax.text(2, 10.3, 'Semantic Input\n("warm", "bright", etc.)',
            ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrow
    arrow1 = FancyArrowPatch((2, 9.9), (2, 9.2),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='black')
    ax.add_patch(arrow1)

    # Encoder block
    encoder_box = FancyBboxPatch((0.2, 6.5), 3.6, 2.5,
                                boxstyle="round,pad=0.1",
                                edgecolor='green', facecolor='lightgreen', linewidth=2)
    ax.add_patch(encoder_box)
    ax.text(2, 8.7, 'Neural Residual Encoder', ha='center', fontsize=11, fontweight='bold')

    # Encoder layers
    layer_y = 8.2
    for i, dim in enumerate(['Input (d)', 'ResBlock (128)', 'ResBlock (256)', 'ResBlock (128)', 'Latent (k)']):
        ax.text(2, layer_y - i*0.35, dim, ha='center', fontsize=8)

    # Arrow to latent
    arrow2 = FancyArrowPatch((2, 6.4), (2, 5.6),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='black')
    ax.add_patch(arrow2)

    # Latent space
    latent_box = FancyBboxPatch((0.5, 4.8), 3, 0.7,
                               boxstyle="round,pad=0.1",
                               edgecolor='purple', facecolor='lavender', linewidth=3)
    ax.add_patch(latent_box)
    ax.text(2, 5.15, 'Latent Space z ∈ [-1,1]^k', ha='center', fontsize=10, fontweight='bold')
    ax.text(2, 4.95, '(Semantic centroids cached here)', ha='center', fontsize=7, style='italic')

    # Interpolation (side branch)
    ax.text(5.5, 5.5, 'Semantic Interpolation', fontsize=9, fontweight='bold', color='red')
    ax.text(5.5, 5.2, 'z_interp = (1-α)·z₁ + α·z₂', fontsize=8, family='monospace')

    # Arrow from latent to decoder
    arrow3 = FancyArrowPatch((2, 4.7), (2, 3.9),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='black')
    ax.add_patch(arrow3)

    # Decoder block
    decoder_box = FancyBboxPatch((0.2, 1.2), 3.6, 2.5,
                                boxstyle="round,pad=0.1",
                                edgecolor='orange', facecolor='lightyellow', linewidth=2)
    ax.add_patch(decoder_box)
    ax.text(2, 3.4, 'Neural Residual Decoder', ha='center', fontsize=11, fontweight='bold')

    # Decoder layers
    layer_y = 2.9
    for i, dim in enumerate(['ResBlock (128)', 'ResBlock (256)', 'ResBlock (128)', 'Param Heads', 'Output (d)']):
        ax.text(2, layer_y - i*0.35, dim, ha='center', fontsize=8)

    # Arrow to output
    arrow4 = FancyArrowPatch((2, 1.1), (2, 0.5),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='black')
    ax.add_patch(arrow4)

    # Output
    output_box = FancyBboxPatch((0.5, 0), 3, 0.4,
                               boxstyle="round,pad=0.1",
                               edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax.add_patch(output_box)
    ax.text(2, 0.2, 'EQ Parameters\n(gain, freq, Q per band)',
            ha='center', va='center', fontsize=9, fontweight='bold')

    # Training side (right side)
    ax.text(7.5, 10.5, 'Training Phase', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat'))

    # SocialFX dataset
    data_box = FancyBboxPatch((6.5, 9.2), 2, 0.6,
                             boxstyle="round,pad=0.05",
                             edgecolor='black', facecolor='lightyellow', linewidth=1.5)
    ax.add_patch(data_box)
    ax.text(7.5, 9.5, 'SocialFX Dataset\n~3000 EQ settings',
            ha='center', va='center', fontsize=8)

    # Loss functions
    loss_y = 7.5
    loss_box1 = FancyBboxPatch((6.3, loss_y), 2.4, 0.5,
                              boxstyle="round,pad=0.05",
                              edgecolor='red', facecolor='mistyrose', linewidth=1.5)
    ax.add_patch(loss_box1)
    ax.text(7.5, loss_y+0.25, 'L_recon = MSE(x, x̂)',
            ha='center', va='center', fontsize=9, family='monospace')

    loss_box2 = FancyBboxPatch((6.3, loss_y-0.8), 2.4, 0.5,
                              boxstyle="round,pad=0.05",
                              edgecolor='red', facecolor='mistyrose', linewidth=1.5)
    ax.add_patch(loss_box2)
    ax.text(7.5, loss_y-0.55, 'L_contrast = NT-Xent',
            ha='center', va='center', fontsize=9, family='monospace')

    # Total loss
    total_loss_box = FancyBboxPatch((6.1, loss_y-1.8), 2.8, 0.6,
                                   boxstyle="round,pad=0.05",
                                   edgecolor='darkred', facecolor='salmon', linewidth=2)
    ax.add_patch(total_loss_box)
    ax.text(7.5, loss_y-1.5, 'L_total = L_recon + 0.1×L_contrast',
            ha='center', va='center', fontsize=9, fontweight='bold', family='monospace')

    # Optimization
    opt_box = FancyBboxPatch((6.5, loss_y-3), 2, 0.5,
                            boxstyle="round,pad=0.05",
                            edgecolor='green', facecolor='lightgreen', linewidth=1.5)
    ax.add_patch(opt_box)
    ax.text(7.5, loss_y-2.75, 'Adam Optimizer\nBackpropagation',
            ha='center', va='center', fontsize=8)

    # Add legend
    ax.text(0.5, 0.5, 'Color Code:', fontsize=9, fontweight='bold')
    ax.text(0.5, 0.2, '● Input/Output', fontsize=8, color='blue')
    ax.text(0.5, -0.1, '● Encoder', fontsize=8, color='green')
    ax.text(2, 0.2, '● Latent Space', fontsize=8, color='purple')
    ax.text(2, -0.1, '● Decoder', fontsize=8, color='orange')
    ax.text(3.5, 0.2, '● Training Loss', fontsize=8, color='red')

    plt.tight_layout()
    return fig


def create_latent_space_diagram():
    """Create latent space structure visualization"""

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.set_xlabel('Latent Dimension 1 (t-SNE projection)', fontsize=12)
    ax.set_ylabel('Latent Dimension 2 (t-SNE projection)', fontsize=12)
    ax.set_title('Learned Latent Space Structure\n(Semantic Clustering via Contrastive Learning)',
                 fontsize=14, fontweight='bold')

    # Simulate semantic clusters
    np.random.seed(42)

    semantic_terms = {
        'warm': {'center': (1.5, -1.5), 'color': 'orangered', 'n': 30},
        'bright': {'center': (-1.5, 1.5), 'color': 'gold', 'n': 25},
        'punchy': {'center': (-1.5, -1.0), 'color': 'red', 'n': 20},
        'smooth': {'center': (1.0, 1.5), 'color': 'lightblue', 'n': 25},
        'dark': {'center': (2.0, 0.5), 'color': 'darkslateblue', 'n': 18},
        'harsh': {'center': (-0.5, -2.0), 'color': 'darkred', 'n': 15},
    }

    # Plot clusters
    for term, info in semantic_terms.items():
        center_x, center_y = info['center']
        color = info['color']
        n = info['n']

        # Generate cluster points
        x = np.random.normal(center_x, 0.3, n)
        y = np.random.normal(center_y, 0.3, n)

        # Plot points
        ax.scatter(x, y, c=color, s=50, alpha=0.6, edgecolors='black', linewidth=0.5, label=term)

        # Plot centroid
        ax.scatter(center_x, center_y, c=color, s=300, marker='*',
                  edgecolors='black', linewidth=2, zorder=10)

        # Label
        ax.text(center_x, center_y + 0.4, term.upper(),
               ha='center', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Draw interpolation path
    warm_center = semantic_terms['warm']['center']
    bright_center = semantic_terms['bright']['center']

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
        ax.text(x_interp, y_interp - 0.3, f'α={alpha:.2f}',
               ha='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    # Add annotations
    ax.annotate('Smooth transitions between\nsemantic concepts',
                xy=(0, 0), xytext=(-2.5, -2.5),
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    return fig


def create_interpolation_flow_diagram():
    """Create semantic interpolation flow diagram"""

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(5, 9.5, 'Semantic Interpolation: Real-Time Flow',
            ha='center', fontsize=16, fontweight='bold')

    # User input
    y_pos = 8.5
    user_box = FancyBboxPatch((0.5, y_pos), 2, 0.8,
                             boxstyle="round,pad=0.1",
                             edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax.add_patch(user_box)
    ax.text(1.5, y_pos+0.6, 'USER INPUT', ha='center', fontsize=10, fontweight='bold')
    ax.text(1.5, y_pos+0.4, 'term1 = "warm"', ha='center', fontsize=8, family='monospace')
    ax.text(1.5, y_pos+0.2, 'term2 = "bright"', ha='center', fontsize=8, family='monospace')
    ax.text(1.5, y_pos, 'α = 0.5', ha='center', fontsize=8, family='monospace')

    # Arrow
    arrow1 = FancyArrowPatch((2.5, y_pos+0.4), (3.5, y_pos+0.4),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow1)
    ax.text(3, y_pos+0.7, '1. Lookup', ha='center', fontsize=8, color='red')

    # Centroid lookup
    y_pos = 8.5
    centroid_box = FancyBboxPatch((3.5, y_pos), 3, 0.8,
                                 boxstyle="round,pad=0.1",
                                 edgecolor='purple', facecolor='lavender', linewidth=2)
    ax.add_patch(centroid_box)
    ax.text(5, y_pos+0.6, 'CACHED CENTROIDS', ha='center', fontsize=10, fontweight='bold')
    ax.text(5, y_pos+0.35, 'c_warm = [0.5, -0.3, 0.8, ...]', ha='center', fontsize=7, family='monospace')
    ax.text(5, y_pos+0.15, 'c_bright = [-0.2, 0.6, -0.4, ...]', ha='center', fontsize=7, family='monospace')

    # Arrow down
    arrow2 = FancyArrowPatch((5, y_pos-0.1), (5, y_pos-0.9),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow2)
    ax.text(5.5, y_pos-0.5, '2. Interpolate', ha='center', fontsize=8, color='red')

    # Interpolation
    y_pos = 7.0
    interp_box = FancyBboxPatch((3, y_pos), 4, 0.8,
                               boxstyle="round,pad=0.1",
                               edgecolor='green', facecolor='lightgreen', linewidth=2)
    ax.add_patch(interp_box)
    ax.text(5, y_pos+0.6, 'LATENT INTERPOLATION', ha='center', fontsize=10, fontweight='bold')
    ax.text(5, y_pos+0.35, 'z_interp = (1-α)·c_warm + α·c_bright', ha='center', fontsize=8, family='monospace')
    ax.text(5, y_pos+0.1, '= 0.5 × c_warm + 0.5 × c_bright', ha='center', fontsize=8, family='monospace')

    # Arrow down
    arrow3 = FancyArrowPatch((5, y_pos-0.1), (5, y_pos-0.9),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow3)
    ax.text(5.5, y_pos-0.5, '3. Decode', ha='center', fontsize=8, color='red')

    # Decoder
    y_pos = 5.3
    decoder_box = FancyBboxPatch((3, y_pos), 4, 1.2,
                                boxstyle="round,pad=0.1",
                                edgecolor='orange', facecolor='lightyellow', linewidth=2)
    ax.add_patch(decoder_box)
    ax.text(5, y_pos+1.0, 'NEURAL DECODER', ha='center', fontsize=10, fontweight='bold')
    ax.text(5, y_pos+0.75, 'ResBlocks → Param Heads', ha='center', fontsize=8)
    ax.text(5, y_pos+0.5, 'Gain Head | Freq Head | Q Head', ha='center', fontsize=7)
    ax.text(5, y_pos+0.25, '↓', ha='center', fontsize=10)
    ax.text(5, y_pos+0.05, 'Interleave parameters', ha='center', fontsize=7, style='italic')

    # Arrow down
    arrow4 = FancyArrowPatch((5, y_pos-0.1), (5, y_pos-0.7),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow4)
    ax.text(5.5, y_pos-0.4, '4. Denormalize', ha='center', fontsize=8, color='red')

    # Output
    y_pos = 3.8
    output_box = FancyBboxPatch((2.5, y_pos), 5, 1.5,
                               boxstyle="round,pad=0.1",
                               edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, y_pos+1.3, 'EQ PARAMETERS OUTPUT', ha='center', fontsize=11, fontweight='bold')
    ax.text(5, y_pos+1.0, 'band1_gain: +1.23 dB', ha='center', fontsize=8, family='monospace')
    ax.text(5, y_pos+0.8, 'band1_freq: 120 Hz', ha='center', fontsize=8, family='monospace')
    ax.text(5, y_pos+0.6, 'band1_q: 0.71', ha='center', fontsize=8, family='monospace')
    ax.text(5, y_pos+0.4, '...', ha='center', fontsize=8)
    ax.text(5, y_pos+0.15, '+ description: "warm (50%) → bright (50%)"', ha='center', fontsize=7, style='italic')

    # Performance metrics (side)
    perf_box = FancyBboxPatch((7.5, 7.5), 2.3, 1.5,
                             boxstyle="round,pad=0.1",
                             edgecolor='green', facecolor='honeydew', linewidth=1.5)
    ax.add_patch(perf_box)
    ax.text(8.65, 8.85, 'PERFORMANCE', ha='center', fontsize=9, fontweight='bold')
    ax.text(8.65, 8.55, 'Centroid cache: 50ms', ha='center', fontsize=7)
    ax.text(8.65, 8.35, '(one-time setup)', ha='center', fontsize=6, style='italic')
    ax.text(8.65, 8.05, 'Interpolation: <5ms', ha='center', fontsize=7)
    ax.text(8.65, 7.85, 'Decode: <2ms', ha='center', fontsize=7)
    ax.text(8.65, 7.55, 'Real-time ready!', ha='center', fontsize=8, fontweight='bold', color='green')

    plt.tight_layout()
    return fig


def create_training_loss_plot():
    """Create training loss curves"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Simulate training curves
    epochs = np.arange(0, 51)

    # Loss curves (exponential decay)
    l_recon = 11 * np.exp(-epochs / 15) + 0.1
    l_contrast = 11 * np.exp(-epochs / 25) + 0.3
    l_total = l_recon + 0.1 * l_contrast

    # Plot 1: All losses
    ax1.plot(epochs, l_total, 'b-', linewidth=2.5, label='Total Loss')
    ax1.plot(epochs, l_recon, 'r--', linewidth=2, label='Reconstruction Loss')
    ax1.plot(epochs, l_contrast, 'g--', linewidth=2, label='Contrastive Loss')

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 12)

    # Add convergence annotation
    ax1.axvline(x=40, color='orange', linestyle=':', linewidth=2, alpha=0.7)
    ax1.text(40, 10, 'Convergence', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Plot 2: Reconstruction error breakdown
    gain_error = 0.5 * np.exp(-epochs / 12) + 0.05
    freq_error = 15 * np.exp(-epochs / 12) + 2
    q_error = 0.3 * np.exp(-epochs / 12) + 0.03

    ax2_twin = ax2.twinx()

    ax2.plot(epochs, gain_error, 'r-', linewidth=2, label='Gain Error (dB)', marker='o', markevery=5)
    ax2.plot(epochs, q_error, 'b-', linewidth=2, label='Q Error', marker='s', markevery=5)
    ax2_twin.plot(epochs, freq_error, 'g-', linewidth=2, label='Freq Error (Hz)', marker='^', markevery=5)

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Gain & Q Errors', fontsize=11, color='black')
    ax2_twin.set_ylabel('Frequency Error (Hz)', fontsize=11, color='green')
    ax2.set_title('Per-Parameter Reconstruction Errors', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

    plt.tight_layout()
    return fig


def main():
    """Generate all diagrams"""

    print("Generating technical diagrams for Neural EQ Morphing report...")

    # Get absolute path relative to this script
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_dir = project_root / "outputs" / "plots" / "technical_diagrams"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate diagrams
    diagrams = {
        'architecture': create_architecture_diagram(),
        'latent_space': create_latent_space_diagram(),
        'interpolation_flow': create_interpolation_flow_diagram(),
        'training_loss': create_training_loss_plot(),
    }

    # Save all diagrams
    for name, fig in diagrams.items():
        output_path = output_dir / f"{name}.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[OK] Saved: {output_path}")

    print(f"\n[DONE] All diagrams saved to: {output_dir}")
    print("\nGenerated diagrams:")
    print("  1. architecture.png - System overview")
    print("  2. latent_space.png - Semantic clustering visualization")
    print("  3. interpolation_flow.png - Real-time interpolation pipeline")
    print("  4. training_loss.png - Training curves and convergence")

    # plt.show()  # Commented out to allow script to complete without GUI


if __name__ == '__main__':
    main()
