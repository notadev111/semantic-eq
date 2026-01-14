"""
Semantic Interpolation Demo
============================

Interactive demonstration of the new Semantic Interpolation Mode.
This allows users to quickly search through EQ configurations by
blending between two semantic terms with a single slider.

Inspired by the "Semantic Mode" from FlowEQ, where users can:
- Select two semantic descriptors (e.g., "warm" and "bright")
- Use a single interpolation slider to blend between them
- Get real-time EQ parameter updates

Usage:
    python semantic_interpolation_demo.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from core.neural_eq_morphing import NeuralEQMorphingSystem, SocialFXDatasetLoader


def demo_semantic_interpolation():
    """
    Demonstrate semantic interpolation between EQ terms
    """

    print("="*70)
    print("SEMANTIC INTERPOLATION MODE - Interactive EQ Search")
    print("="*70)
    print("\nThis demo shows how to quickly explore EQ configurations by")
    print("blending between two semantic terms with a single slider.\n")

    # Load dataset
    print("Step 1: Loading EQ dataset...")
    loader = SocialFXDatasetLoader()
    eq_settings = loader.load_socialfx_dataset()

    # Initialize system
    print("\nStep 2: Initializing neural system...")
    system = NeuralEQMorphingSystem(latent_dim=32, device='cpu')

    if not system.load_dataset(eq_settings):
        print("Failed to load dataset")
        return

    # Train (quick training for demo)
    print("\nStep 3: Training neural networks...")
    print("(Using quick training for demo - 30 epochs)")
    system.train(epochs=30, batch_size=16, learning_rate=0.001)

    print("\n" + "="*70)
    print("SEMANTIC INTERPOLATION: WARM ↔ BRIGHT")
    print("="*70)

    # Check if terms are available
    available_terms = list(system.semantic_to_idx.keys())
    print(f"\nAvailable semantic terms: {available_terms}")

    # Use first two available terms for demo
    if len(available_terms) < 2:
        print("Not enough semantic terms for interpolation demo")
        return

    # Try to use warm and bright if available, otherwise use first two
    term1 = 'warm' if 'warm' in available_terms else available_terms[0]
    term2 = 'bright' if 'bright' in available_terms else available_terms[1]

    print(f"\n🎚️  INTERPOLATION SLIDER: {term1.upper()} ↔ {term2.upper()}")
    print("="*70)

    # Simulate slider movement from 0.0 to 1.0
    slider_positions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Collect results for visualization
    all_results = []

    for alpha in slider_positions:
        result = system.interpolate_semantic_terms(term1, term2, alpha)

        if result:
            all_results.append(result)

            # Display result
            blend = result['blend_percentage']
            print(f"\nSlider Position: {alpha:.1f}")
            print(f"  Blend: {blend[term1]} {term1} + {blend[term2]} {term2}")
            print(f"  Description: {result['description']}")

            # Show a few key EQ parameters
            params = result['parameters']
            param_names = list(params.keys())

            # Try to show low and high band gains (common EQ parameters)
            display_params = []
            for pname in param_names:
                if 'band1_gain' in pname or 'band5_gain' in pname:
                    display_params.append(pname)

            # If not found, show first 3 parameters
            if not display_params:
                display_params = param_names[:3]

            print(f"  EQ Parameters (sample):")
            for pname in display_params[:3]:
                print(f"    {pname}: {params[pname]:+.2f}")

    # Visualize interpolation path
    print("\n" + "="*70)
    print("VISUALIZATION: EQ Parameter Interpolation Path")
    print("="*70)

    visualize_interpolation(all_results, term1, term2)

    # Interactive simulation
    print("\n" + "="*70)
    print("INTERACTIVE SIMULATION")
    print("="*70)
    print("\n💡 In a real application, you would:")
    print("   1. Have two dropdown menus to select semantic terms")
    print("   2. Have a slider (0-100%) to control interpolation")
    print("   3. Get real-time EQ parameters as the slider moves")
    print("   4. Apply the EQ parameters to your audio in real-time")

    print(f"\nExample code:")
    print(f"  # User selects: term_a='{term1}', term_b='{term2}', slider=0.5")
    print(f"  result = system.interpolate_semantic_terms('{term1}', '{term2}', 0.5)")
    print(f"  eq_params = result['parameters']")
    print(f"  # Apply eq_params to your audio processing chain")

    print("\n✅ Demo complete! The semantic interpolation system is ready to use.")


def visualize_interpolation(results, term1, term2):
    """Visualize how parameters change during interpolation"""

    if not results:
        print("No results to visualize")
        return

    try:
        # Extract data
        alphas = [r['interpolation'] for r in results]

        # Get parameter names
        param_names = list(results[0]['parameters'].keys())

        # Choose interesting parameters to plot (gains if available)
        plot_params = []
        for pname in param_names:
            if 'gain' in pname.lower():
                plot_params.append(pname)

        # If no gains found, use first 4 parameters
        if not plot_params:
            plot_params = param_names[:4]
        else:
            plot_params = plot_params[:4]  # Limit to 4 for clarity

        # Create plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Plot 1: Parameter evolution
        ax1 = axes[0]
        for pname in plot_params:
            values = [r['parameters'][pname] for r in results]
            ax1.plot(alphas, values, 'o-', label=pname, linewidth=2, markersize=6)

        ax1.set_xlabel(f'Interpolation (0={term1}, 1={term2})', fontsize=12)
        ax1.set_ylabel('Parameter Value', fontsize=12)
        ax1.set_title(f'EQ Parameter Interpolation: {term1.upper()} → {term2.upper()}',
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='50/50 blend')

        # Plot 2: All parameters heatmap
        ax2 = axes[1]

        # Create matrix of all parameters
        all_param_names = list(results[0]['parameters'].keys())
        param_matrix = np.zeros((len(all_param_names), len(results)))

        for i, pname in enumerate(all_param_names):
            for j, r in enumerate(results):
                param_matrix[i, j] = r['parameters'][pname]

        # Normalize for visualization
        param_matrix_norm = (param_matrix - param_matrix.mean(axis=1, keepdims=True)) / (param_matrix.std(axis=1, keepdims=True) + 1e-6)

        im = ax2.imshow(param_matrix_norm, aspect='auto', cmap='RdBu_r',
                       interpolation='nearest', vmin=-2, vmax=2)

        ax2.set_xlabel(f'Interpolation (0={term1}, 1={term2})', fontsize=12)
        ax2.set_ylabel('EQ Parameters', fontsize=12)
        ax2.set_title('All EQ Parameters (normalized)', fontsize=14, fontweight='bold')
        ax2.set_xticks(np.arange(len(results))[::2])
        ax2.set_xticklabels([f'{alphas[i]:.1f}' for i in range(len(alphas))][::2])

        # Reduce y-tick labels if too many parameters
        if len(all_param_names) > 20:
            step = len(all_param_names) // 10
            ax2.set_yticks(np.arange(0, len(all_param_names), step))
            ax2.set_yticklabels([all_param_names[i] for i in range(0, len(all_param_names), step)],
                              fontsize=8)
        else:
            ax2.set_yticks(np.arange(len(all_param_names)))
            ax2.set_yticklabels(all_param_names, fontsize=8)

        plt.colorbar(im, ax=ax2, label='Normalized Value')

        plt.tight_layout()

        # Save plot
        output_dir = Path("./outputs/plots")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"semantic_interpolation_{term1}_{term2}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

        print(f"\n✅ Visualization saved: {output_path}")
        plt.show()

    except Exception as e:
        print(f"Visualization error: {e}")
        print("Continuing without visualization...")


if __name__ == '__main__':
    print("\nSemantic Interpolation Demo")
    print("This demonstrates the new 'Semantic Mode' feature\n")

    demo_semantic_interpolation()

    print("\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("✓ Semantic interpolation allows quick EQ exploration")
    print("✓ Single slider controls blend between two semantic terms")
    print("✓ Real-time parameter updates as slider moves")
    print("✓ Based on learned latent space representations")
    print("✓ Perfect for interactive UI applications")
    print("\nThis is similar to the 'Semantic Mode' in FlowEQ,")
    print("where users can seamlessly move between semantic descriptors!")
