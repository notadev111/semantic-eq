"""
Visualize Latent Space & Generate EQ Profile Plots
===================================================

Creates visualizations for academic report:
1. 2D latent space map (t-SNE/PCA projection)
2. EQ frequency response curves for each semantic term
3. Spectral analysis of what each term "sounds like"
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from core.neural_eq_morphing_safedb_v2 import NeuralEQMorphingSAFEDBV2

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def compute_eq_frequency_response(eq_params_13, sample_rate=44100, num_points=1000):
    """
    Compute frequency response curve from SAFE-DB EQ parameters

    SAFE-DB format (13 params):
      Band 1 (Low Shelf):  [Gain, Freq]
      Band 2 (Bell):       [Gain, Freq, Q]
      Band 3 (Bell):       [Gain, Freq, Q]
      Band 4 (Bell):       [Gain, Freq, Q]
      Band 5 (High Shelf): [Gain, Freq]

    Returns:
        frequencies: Array of frequencies (Hz)
        magnitude_db: Frequency response in dB
    """
    # Frequency range for evaluation (20 Hz - 20 kHz, log scale)
    frequencies = np.logspace(np.log10(20), np.log10(20000), num_points)

    # Initialize flat response (0 dB)
    magnitude_db = np.zeros_like(frequencies)

    # Extract parameters
    # Band 1: Low Shelf
    gain1 = eq_params_13[0]
    freq1 = eq_params_13[1]
    q1 = 0.7  # Default Q for shelf

    # Band 2: Bell
    gain2 = eq_params_13[2]
    freq2 = eq_params_13[3]
    q2 = eq_params_13[4]

    # Band 3: Bell
    gain3 = eq_params_13[5]
    freq3 = eq_params_13[6]
    q3 = eq_params_13[7]

    # Band 4: Bell
    gain4 = eq_params_13[8]
    freq4 = eq_params_13[9]
    q4 = eq_params_13[10]

    # Band 5: High Shelf
    gain5 = eq_params_13[11]
    freq5 = eq_params_13[12]
    q5 = 0.7  # Default Q for shelf

    # Compute shelf filter responses (simplified)
    def low_shelf(f, fc, gain, q):
        """Low shelf filter approximation"""
        w = f / fc
        g = 10 ** (gain / 20)
        # Simplified shelf response
        slope = 1 / (1 + (w / q) ** 2)
        return 20 * np.log10(1 + (g - 1) * slope)

    def high_shelf(f, fc, gain, q):
        """High shelf filter approximation"""
        w = fc / f
        g = 10 ** (gain / 20)
        # Simplified shelf response
        slope = 1 / (1 + (w / q) ** 2)
        return 20 * np.log10(1 + (g - 1) * slope)

    def bell(f, fc, gain, q):
        """Bell (peaking) filter"""
        w = f / fc
        bw = 1 / q
        # Simplified bell response
        response = gain / (1 + ((w - 1/w) / bw) ** 2)
        return response

    # Apply each band
    magnitude_db += low_shelf(frequencies, freq1, gain1, q1)
    magnitude_db += bell(frequencies, freq2, gain2, q2)
    magnitude_db += bell(frequencies, freq3, gain3, q3)
    magnitude_db += bell(frequencies, freq4, gain4, q4)
    magnitude_db += high_shelf(frequencies, freq5, gain5, q5)

    return frequencies, magnitude_db


def plot_2d_latent_space(system, output_dir="./figures"):
    """
    Create 2D visualization of semantic term latent space

    Uses t-SNE and PCA to project 32D latent vectors to 2D
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("\n" + "="*70)
    print("GENERATING 2D LATENT SPACE VISUALIZATION")
    print("="*70)

    # Get all semantic terms
    terms = list(system.term_to_idx.keys())

    # Encode each term to latent space
    latent_vectors = []
    for term in terms:
        # Encode using the encoder
        term_idx = system.term_to_idx[term]
        # Get a sample EQ setting with this term
        sample_settings = [s for s in system.eq_settings if s.semantic_label == term]
        if sample_settings:
            eq_params = torch.FloatTensor(sample_settings[0].eq_params_normalized).unsqueeze(0).to(system.device)
            with torch.no_grad():
                z, _ = system.encoder(eq_params)
            latent_vectors.append(z.detach().cpu().numpy().flatten())
        else:
            # Fallback: zero vector
            latent_vectors.append(np.zeros(system.latent_dim))

    latent_vectors = np.array(latent_vectors)

    print(f"\nLatent vectors shape: {latent_vectors.shape}")
    print(f"Terms: {terms}")

    # t-SNE projection
    print("\nComputing t-SNE projection...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(terms)-1))
    latent_2d_tsne = tsne.fit_transform(latent_vectors)

    # PCA projection
    print("Computing PCA projection...")
    pca = PCA(n_components=2)
    latent_2d_pca = pca.fit_transform(latent_vectors)

    print(f"PCA explained variance: {pca.explained_variance_ratio_}")

    # Create figure with both projections
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot t-SNE
    ax1 = axes[0]
    scatter1 = ax1.scatter(latent_2d_tsne[:, 0], latent_2d_tsne[:, 1],
                           s=200, c=range(len(terms)), cmap='tab20',
                           alpha=0.7, edgecolors='black', linewidth=1.5)

    for i, term in enumerate(terms):
        ax1.annotate(term, (latent_2d_tsne[i, 0], latent_2d_tsne[i, 1]),
                    fontsize=11, ha='center', va='bottom',
                    fontweight='bold')

    ax1.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax1.set_title('Semantic Terms in Latent Space (t-SNE)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot PCA
    ax2 = axes[1]
    scatter2 = ax2.scatter(latent_2d_pca[:, 0], latent_2d_pca[:, 1],
                          s=200, c=range(len(terms)), cmap='tab20',
                          alpha=0.7, edgecolors='black', linewidth=1.5)

    for i, term in enumerate(terms):
        ax2.annotate(term, (latent_2d_pca[i, 0], latent_2d_pca[i, 1]),
                    fontsize=11, ha='center', va='bottom',
                    fontweight='bold')

    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax2.set_title('Semantic Terms in Latent Space (PCA)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / "latent_space_2d_map.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")

    plt.close()

    # Create interactive version with warm-bright axis
    fig, ax = plt.subplots(figsize=(12, 10))

    scatter = ax.scatter(latent_2d_pca[:, 0], latent_2d_pca[:, 1],
                        s=300, c=range(len(terms)), cmap='tab20',
                        alpha=0.7, edgecolors='black', linewidth=2)

    # Annotate with larger text
    for i, term in enumerate(terms):
        ax.annotate(term, (latent_2d_pca[i, 0], latent_2d_pca[i, 1]),
                   fontsize=13, ha='center', va='bottom',
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # Draw warm-bright axis if both exist
    if 'warm' in terms and 'bright' in terms:
        warm_idx = terms.index('warm')
        bright_idx = terms.index('bright')
        ax.plot([latent_2d_pca[warm_idx, 0], latent_2d_pca[bright_idx, 0]],
               [latent_2d_pca[warm_idx, 1], latent_2d_pca[bright_idx, 1]],
               'r--', linewidth=2, alpha=0.5, label='Warm ↔ Bright axis')
        ax.legend(fontsize=12)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=14)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=14)
    ax.set_title('Semantic Audio Terms - Latent Space Representation',
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)

    output_path = output_dir / "latent_space_annotated.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")

    plt.close()

    return latent_2d_tsne, latent_2d_pca


def plot_eq_curves(system, output_dir="./figures"):
    """
    Generate EQ frequency response curves for each semantic term
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("\n" + "="*70)
    print("GENERATING EQ FREQUENCY RESPONSE CURVES")
    print("="*70)

    terms = list(system.term_to_idx.keys())

    # Generate EQ for each term
    eq_curves = {}
    for term in terms:
        print(f"\nProcessing: {term}")
        eq_params = system.generate_eq_from_term(term)
        frequencies, magnitude_db = compute_eq_frequency_response(eq_params)
        eq_curves[term] = (frequencies, magnitude_db)

    # Plot all curves together
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = plt.cm.tab20(np.linspace(0, 1, len(terms)))

    for i, term in enumerate(terms):
        freqs, mag = eq_curves[term]
        ax.semilogx(freqs, mag, label=term, linewidth=2, alpha=0.8, color=colors[i])

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel('Frequency (Hz)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Magnitude (dB)', fontsize=14, fontweight='bold')
    ax.set_title('EQ Frequency Response Curves - All Semantic Terms',
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper left', fontsize=10, ncol=2)
    ax.set_xlim(20, 20000)
    ax.set_ylim(-15, 15)

    # Add frequency markers
    freq_labels = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    ax.set_xticks(freq_labels)
    ax.set_xticklabels([f'{f}' if f < 1000 else f'{f//1000}k' for f in freq_labels])

    output_path = output_dir / "eq_curves_all_terms.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")

    plt.close()

    # Plot key terms individually (warm, bright, muddy, clear)
    key_terms = ['warm', 'bright', 'muddy', 'clear']
    available_key_terms = [t for t in key_terms if t in terms]

    if available_key_terms:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for i, term in enumerate(available_key_terms[:4]):
            if i >= len(axes):
                break

            ax = axes[i]
            freqs, mag = eq_curves[term]

            ax.semilogx(freqs, mag, linewidth=3, color='steelblue', label=f'{term.upper()} EQ')
            ax.fill_between(freqs, 0, mag, alpha=0.3, color='steelblue')
            ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)

            ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Magnitude (dB)', fontsize=12, fontweight='bold')
            ax.set_title(f'{term.upper()} - EQ Frequency Response',
                        fontsize=14, fontweight='bold')
            ax.grid(True, which='both', alpha=0.3)
            ax.set_xlim(20, 20000)
            ax.set_ylim(-15, 15)
            ax.set_xticks(freq_labels)
            ax.set_xticklabels([f'{f}' if f < 1000 else f'{f//1000}k' for f in freq_labels])

            # Add band annotations
            eq_params = system.generate_eq_from_term(term)
            y_pos = 12
            ax.text(eq_params[1], y_pos, f'B1: {eq_params[0]:.1f}dB',
                   fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.text(eq_params[3], y_pos, f'B2: {eq_params[2]:.1f}dB',
                   fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.text(eq_params[6], y_pos, f'B3: {eq_params[5]:.1f}dB',
                   fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.text(eq_params[9], y_pos, f'B4: {eq_params[8]:.1f}dB',
                   fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.text(eq_params[12], y_pos, f'B5: {eq_params[11]:.1f}dB',
                   fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        output_path = output_dir / "eq_curves_key_terms.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

        plt.close()

    return eq_curves


def plot_spectral_profiles(system, output_dir="./figures"):
    """
    Show spectral energy distribution for each semantic term
    (What frequency bands are emphasized)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("\n" + "="*70)
    print("GENERATING SPECTRAL ENERGY PROFILES")
    print("="*70)

    terms = list(system.term_to_idx.keys())

    # Define frequency bands (octave-based)
    bands = {
        'Sub\n20-60': (20, 60),
        'Bass\n60-250': (60, 250),
        'Low Mid\n250-500': (250, 500),
        'Mid\n500-2k': (500, 2000),
        'Hi Mid\n2k-6k': (2000, 6000),
        'High\n6k-12k': (6000, 12000),
        'Air\n12k-20k': (12000, 20000)
    }

    band_names = list(bands.keys())

    # Compute energy in each band for each term
    profiles = {}

    for term in terms:
        eq_params = system.generate_eq_from_term(term)
        freqs, mag_db = compute_eq_frequency_response(eq_params)

        # Convert dB to linear magnitude
        mag_linear = 10 ** (mag_db / 20)

        # Integrate energy in each band
        band_energies = []
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs <= high)
            energy = np.mean(mag_linear[mask])
            band_energies.append(energy)

        profiles[term] = band_energies

    # Plot as heatmap
    profile_matrix = np.array([profiles[term] for term in terms])

    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(profile_matrix, cmap='RdYlGn', aspect='auto', interpolation='nearest')

    ax.set_xticks(range(len(band_names)))
    ax.set_xticklabels(band_names, fontsize=11)
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels(terms, fontsize=11)

    ax.set_xlabel('Frequency Band', fontsize=14, fontweight='bold')
    ax.set_ylabel('Semantic Term', fontsize=14, fontweight='bold')
    ax.set_title('Spectral Energy Profile by Semantic Term\n(Green = Boost, Red = Cut)',
                fontsize=16, fontweight='bold', pad=20)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Relative Energy (Linear)', fontsize=12)

    # Add text annotations
    for i in range(len(terms)):
        for j in range(len(band_names)):
            text = ax.text(j, i, f'{profile_matrix[i, j]:.2f}',
                         ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    output_path = output_dir / "spectral_profiles_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")

    plt.close()

    # Plot warm vs bright comparison
    if 'warm' in terms and 'bright' in terms:
        fig, ax = plt.subplots(figsize=(12, 7))

        x = np.arange(len(band_names))
        width = 0.35

        warm_energies = profiles['warm']
        bright_energies = profiles['bright']

        bars1 = ax.bar(x - width/2, warm_energies, width, label='Warm', color='orangered', alpha=0.8)
        bars2 = ax.bar(x + width/2, bright_energies, width, label='Bright', color='skyblue', alpha=0.8)

        ax.axhline(1.0, color='black', linewidth=1, linestyle='--', alpha=0.5, label='Flat (no change)')

        ax.set_xlabel('Frequency Band', fontsize=14, fontweight='bold')
        ax.set_ylabel('Relative Energy (Linear)', fontsize=14, fontweight='bold')
        ax.set_title('Spectral Profile Comparison: Warm vs Bright',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(band_names, fontsize=11)
        ax.legend(fontsize=12)
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = output_dir / "spectral_comparison_warm_bright.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

        plt.close()


def main():
    print("="*70)
    print("LATENT SPACE & EQ PROFILE VISUALIZATION")
    print("="*70)

    # Load trained V2 model
    print("\nLoading V2 model...")
    system = NeuralEQMorphingSAFEDBV2()
    system.load_model("neural_eq_safedb_v2.pt")
    system.load_dataset(min_examples=5)

    print(f"\nLoaded {len(system.term_to_idx)} semantic terms")

    # Create output directory
    output_dir = Path("./figures")
    output_dir.mkdir(exist_ok=True)

    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING ALL VISUALIZATIONS")
    print("="*70)

    # 1. Latent space 2D map
    plot_2d_latent_space(system, output_dir)

    # 2. EQ frequency response curves
    plot_eq_curves(system, output_dir)

    # 3. Spectral energy profiles
    plot_spectral_profiles(system, output_dir)

    print("\n" + "="*70)
    print("ALL VISUALIZATIONS COMPLETE!")
    print("="*70)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print("\nGenerated files:")
    for file in sorted(output_dir.glob("*.png")):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()
