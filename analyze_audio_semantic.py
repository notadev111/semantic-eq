"""
Semantic Audio Analysis with Temporal Evolution
================================================

Research-grade analysis of audio semantic profiles using the E2E-DDSP encoder.
Performs windowed analysis to track how semantic characteristics evolve over time.

Features:
- Temporal semantic profile (how "bright", "warm" etc. change over time)
- Top semantic matches with confidence scores
- Latent space visualization with morphing trajectories
- EQ suggestions for target semantics (single or blended)

Usage:
    python analyze_audio_semantic.py --audio song.wav
    python analyze_audio_semantic.py --audio song.wav --target warm --intensity 0.7
    python analyze_audio_semantic.py --audio song.wav --blend warm:0.5 punchy:0.3
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Local imports
from core.audio_encoder import FastAudioEncoder, AudioEncoderConfig
from core.neural_eq_morphing_safedb_v2 import NeuralEQMorphingSAFEDBV2


@dataclass
class SemanticAnalysisResult:
    """Container for semantic analysis results."""
    audio_path: str
    duration_sec: float
    window_size_sec: float
    hop_size_sec: float

    # Per-window results: [n_windows, n_terms]
    temporal_scores: np.ndarray
    timestamps: np.ndarray  # Center time of each window
    term_names: List[str]

    # Aggregated results
    mean_scores: Dict[str, float]
    std_scores: Dict[str, float]  # Variability over time

    # Latent representations
    latent_trajectory: np.ndarray  # [n_windows, latent_dim]


def load_audio_full(audio_path: str, sample_rate: int = 44100) -> Tuple[torch.Tensor, float]:
    """Load full audio file, return tensor and duration."""
    import soundfile as sf
    import librosa

    audio_np, sr = sf.read(audio_path)

    # Convert to mono by averaging channels
    # Note: This preserves tonal characteristics while simplifying the analysis
    # Stereo information (panning, width) is not captured by semantic EQ descriptors
    if len(audio_np.shape) > 1:
        audio_np = audio_np.mean(axis=1)

    # Resample if needed
    if sr != sample_rate:
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=sample_rate)

    duration = len(audio_np) / sample_rate
    waveform = torch.from_numpy(audio_np).float()

    return waveform, duration


def load_encoder(encoder_path: str, device: str) -> FastAudioEncoder:
    """Load audio encoder checkpoint."""
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
    else:
        encoder.load_state_dict(checkpoint)

    encoder.to(device)
    encoder.eval()
    return encoder


def load_v2_model(v2_model_path: str, device: str) -> NeuralEQMorphingSAFEDBV2:
    """Load V2 semantic EQ model with dataset."""
    model = NeuralEQMorphingSAFEDBV2(latent_dim=32, device=device)
    model.load_dataset(min_examples=3)
    model.load_model(v2_model_path)
    return model


def compute_semantic_centroids(v2_model, device: str) -> Dict[str, torch.Tensor]:
    """Precompute semantic centroids for all terms."""
    centroids = {}

    for term in v2_model.term_to_idx.keys():
        examples = [s for s in v2_model.eq_settings if s.semantic_label == term]
        if not examples:
            continue

        eq_params_list = [torch.from_numpy(ex.eq_params_normalized).float()
                         for ex in examples[:20]]  # Use more examples for stable centroids
        eq_batch = torch.stack(eq_params_list).to(device)

        with torch.no_grad():
            z_semantic, _ = v2_model.encoder(eq_batch)
            centroids[term] = z_semantic.mean(dim=0)

    return centroids


def analyze_window(
    audio_window: torch.Tensor,
    encoder: FastAudioEncoder,
    centroids: Dict[str, torch.Tensor],
    device: str
) -> Tuple[Dict[str, float], torch.Tensor]:
    """Analyze a single audio window, return scores and latent."""
    with torch.no_grad():
        audio = audio_window.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, samples]
        z_audio = encoder(audio)

        scores = {}
        for term, z_centroid in centroids.items():
            sim = torch.nn.functional.cosine_similarity(
                z_audio, z_centroid.unsqueeze(0)
            )
            scores[term] = sim.item()

        return scores, z_audio.squeeze(0)


def compute_window_energy(window: torch.Tensor) -> float:
    """Compute RMS energy of audio window."""
    return torch.sqrt(torch.mean(window ** 2)).item()


def compute_perceptual_weight(window: torch.Tensor, sample_rate: int = 44100) -> float:
    """
    Compute perceptual importance weight based on:
    - RMS energy (louder = more important)
    - Spectral flux (more change = more interesting)
    - Mid-frequency energy (where human hearing is most sensitive)
    """
    # RMS energy
    rms = torch.sqrt(torch.mean(window ** 2)).item()

    # Simple spectral centroid proxy (higher = brighter, weight mid-range more)
    # This is a simplified perceptual weighting
    fft = torch.fft.rfft(window)
    magnitudes = torch.abs(fft)
    freqs = torch.fft.rfftfreq(len(window), 1/sample_rate)

    # Weight frequencies by perceptual importance (A-weighting approximation)
    # Peak sensitivity around 2-4kHz
    freq_weights = torch.exp(-0.5 * ((freqs - 3000) / 2000) ** 2)
    weighted_energy = (magnitudes * freq_weights).sum().item()

    # Combine: energy * spectral relevance
    # Normalize to reasonable range
    perceptual_weight = rms * (1 + 0.1 * weighted_energy / (magnitudes.sum().item() + 1e-8))

    return perceptual_weight


def analyze_audio_temporal(
    audio_path: str,
    encoder: FastAudioEncoder,
    v2_model: NeuralEQMorphingSAFEDBV2,
    device: str,
    window_size_sec: float = 3.0,
    hop_size_sec: float = 1.0,
    sample_rate: int = 44100,
    use_energy_weighting: bool = True
) -> SemanticAnalysisResult:
    """
    Perform temporal semantic analysis with overlapping windows.

    Args:
        audio_path: Path to audio file
        encoder: Trained audio encoder
        v2_model: V2 semantic EQ model
        device: torch device
        window_size_sec: Analysis window size in seconds
        hop_size_sec: Hop between windows in seconds
        sample_rate: Audio sample rate
        use_energy_weighting: Weight windows by perceptual importance

    Returns:
        SemanticAnalysisResult with temporal and aggregated scores
    """
    print(f"\nAnalyzing: {audio_path}")
    print(f"Window: {window_size_sec}s, Hop: {hop_size_sec}s")
    print(f"Energy weighting: {'enabled' if use_energy_weighting else 'disabled'}")

    # Load audio
    waveform, duration = load_audio_full(audio_path, sample_rate)
    print(f"Duration: {duration:.1f}s")

    # Precompute centroids
    print("Computing semantic centroids...")
    centroids = compute_semantic_centroids(v2_model, device)
    term_names = list(centroids.keys())
    print(f"Analyzing {len(term_names)} semantic terms")

    # Windowed analysis
    window_samples = int(window_size_sec * sample_rate)
    hop_samples = int(hop_size_sec * sample_rate)

    n_windows = max(1, (len(waveform) - window_samples) // hop_samples + 1)

    temporal_scores = []
    latent_trajectory = []
    timestamps = []
    window_weights = []

    print(f"Processing {n_windows} windows...")
    for i in range(n_windows):
        start = i * hop_samples
        end = start + window_samples

        if end > len(waveform):
            # Pad last window if needed
            window = torch.nn.functional.pad(
                waveform[start:], (0, end - len(waveform))
            )
        else:
            window = waveform[start:end]

        scores, z = analyze_window(window, encoder, centroids, device)

        # Compute perceptual weight for this window
        if use_energy_weighting:
            weight = compute_perceptual_weight(window, sample_rate)
        else:
            weight = 1.0

        temporal_scores.append([scores[t] for t in term_names])
        latent_trajectory.append(z.cpu().numpy())
        timestamps.append((start + end) / 2 / sample_rate)
        window_weights.append(weight)

    temporal_scores = np.array(temporal_scores)
    latent_trajectory = np.array(latent_trajectory)
    timestamps = np.array(timestamps)
    window_weights = np.array(window_weights)

    # Normalize weights
    window_weights = window_weights / (window_weights.sum() + 1e-8)

    # Compute aggregated stats (weighted vs unweighted)
    if use_energy_weighting:
        # Weighted mean: louder/more perceptually important sections contribute more
        mean_scores = {t: np.average(temporal_scores[:, i], weights=window_weights)
                      for i, t in enumerate(term_names)}
        # Weighted std
        std_scores = {t: np.sqrt(np.average((temporal_scores[:, i] - mean_scores[t])**2,
                                            weights=window_weights))
                     for i, t in enumerate(term_names)}
    else:
        mean_scores = {t: temporal_scores[:, i].mean() for i, t in enumerate(term_names)}
        std_scores = {t: temporal_scores[:, i].std() for i, t in enumerate(term_names)}

    return SemanticAnalysisResult(
        audio_path=audio_path,
        duration_sec=duration,
        window_size_sec=window_size_sec,
        hop_size_sec=hop_size_sec,
        temporal_scores=temporal_scores,
        timestamps=timestamps,
        term_names=term_names,
        mean_scores=mean_scores,
        std_scores=std_scores,
        latent_trajectory=latent_trajectory
    )


def compute_blended_eq(
    v2_model: NeuralEQMorphingSAFEDBV2,
    z_audio: torch.Tensor,
    blend_targets: Dict[str, float],
    centroids: Dict[str, torch.Tensor],
    device: str
) -> torch.Tensor:
    """
    Compute EQ for blended semantic targets.

    Args:
        v2_model: V2 model with decoder
        z_audio: Current audio latent [latent_dim]
        blend_targets: Dict of {term: intensity} e.g. {"warm": 0.5, "punchy": 0.3}
        centroids: Precomputed semantic centroids
        device: torch device

    Returns:
        EQ parameters tensor
    """
    with torch.no_grad():
        # Start from audio latent
        z_morphed = z_audio.clone()

        # Apply each target's contribution
        for term, intensity in blend_targets.items():
            if term not in centroids:
                print(f"Warning: Unknown term '{term}', skipping")
                continue

            z_target = centroids[term]
            # Interpolate towards target
            z_morphed = z_morphed + intensity * (z_target - z_audio)

        # Decode to EQ params
        eq_params = v2_model.decoder(z_morphed.unsqueeze(0))

        return eq_params.squeeze(0)


def plot_temporal_analysis(
    result: SemanticAnalysisResult,
    output_path: str,
    top_k: int = 6
):
    """Create temporal semantic analysis visualization."""
    fig = plt.figure(figsize=(16, 12))

    # Get top terms by mean score
    sorted_terms = sorted(result.mean_scores.items(), key=lambda x: -x[1])
    top_terms = [t[0] for t in sorted_terms[:top_k]]
    top_indices = [result.term_names.index(t) for t in top_terms]

    # 1. Temporal evolution of top terms
    ax1 = fig.add_subplot(2, 2, 1)
    for idx, term in zip(top_indices, top_terms):
        ax1.plot(result.timestamps, result.temporal_scores[:, idx],
                label=term, linewidth=2, alpha=0.8)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Semantic Score')
    ax1.set_title('Temporal Semantic Evolution')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # 2. Mean scores with variability
    ax2 = fig.add_subplot(2, 2, 2)
    means = [result.mean_scores[t] for t in top_terms]
    stds = [result.std_scores[t] for t in top_terms]
    x = np.arange(len(top_terms))

    bars = ax2.bar(x, means, yerr=stds, capsize=5, color='steelblue', alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(top_terms, rotation=45, ha='right')
    ax2.set_ylabel('Mean Score ± Std')
    ax2.set_title('Semantic Profile (mean ± temporal variability)')
    ax2.set_ylim(0, 1)

    # 3. Latent trajectory (PCA projection)
    ax3 = fig.add_subplot(2, 2, 3)
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    trajectory_2d = pca.fit_transform(result.latent_trajectory)

    # Color by time
    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectory_2d)))
    ax3.scatter(trajectory_2d[:, 0], trajectory_2d[:, 1], c=colors, s=50, alpha=0.7)
    ax3.plot(trajectory_2d[:, 0], trajectory_2d[:, 1], 'k-', alpha=0.3, linewidth=1)

    # Mark start and end
    ax3.scatter(trajectory_2d[0, 0], trajectory_2d[0, 1],
               c='green', s=150, marker='o', label='Start', zorder=5)
    ax3.scatter(trajectory_2d[-1, 0], trajectory_2d[-1, 1],
               c='red', s=150, marker='s', label='End', zorder=5)

    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax3.set_title('Latent Space Trajectory (colored by time)')
    ax3.legend()

    # 4. Heatmap of all terms over time
    ax4 = fig.add_subplot(2, 2, 4)

    # Sort terms by mean score for better visualization
    sorted_indices = [result.term_names.index(t) for t, _ in sorted_terms[:15]]
    sorted_names = [t for t, _ in sorted_terms[:15]]

    heatmap_data = result.temporal_scores[:, sorted_indices].T

    im = ax4.imshow(heatmap_data, aspect='auto', cmap='RdYlBu_r',
                    extent=[result.timestamps[0], result.timestamps[-1],
                           len(sorted_names)-0.5, -0.5])
    ax4.set_yticks(range(len(sorted_names)))
    ax4.set_yticklabels(sorted_names)
    ax4.set_xlabel('Time (s)')
    ax4.set_title('Semantic Heatmap (top 15 terms)')
    plt.colorbar(im, ax=ax4, label='Score')

    plt.suptitle(f'Semantic Analysis: {Path(result.audio_path).name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved visualization to: {output_path}")


def print_analysis_summary(
    result: SemanticAnalysisResult,
    target: Optional[str] = None,
    blend_targets: Optional[Dict[str, float]] = None,
    eq_params: Optional[torch.Tensor] = None
):
    """Print formatted analysis summary."""
    print("\n" + "="*70)
    print("SEMANTIC AUDIO ANALYSIS RESULTS")
    print("="*70)

    print(f"\nAudio: {result.audio_path}")
    print(f"Duration: {result.duration_sec:.1f}s")
    print(f"Windows analyzed: {len(result.timestamps)}")

    # Top semantic matches
    print("\n" + "-"*70)
    print("TOP SEMANTIC MATCHES (mean ± std over time)")
    print("-"*70)

    sorted_terms = sorted(result.mean_scores.items(), key=lambda x: -x[1])
    for i, (term, mean) in enumerate(sorted_terms[:10], 1):
        std = result.std_scores[term]
        variability = "stable" if std < 0.05 else "variable" if std < 0.1 else "highly variable"
        print(f"  {i:2}. {term:<15} {mean:.4f} ± {std:.4f}  ({variability})")

    # Temporal insights
    print("\n" + "-"*70)
    print("TEMPORAL INSIGHTS")
    print("-"*70)

    # Find most variable terms
    sorted_by_var = sorted(result.std_scores.items(), key=lambda x: -x[1])
    print("Most temporally variable characteristics:")
    for term, std in sorted_by_var[:5]:
        print(f"  - {term}: std={std:.4f}")

    # EQ suggestion
    if eq_params is not None:
        print("\n" + "-"*70)
        if blend_targets:
            blend_str = " + ".join([f"{t}({i:.0%})" for t, i in blend_targets.items()])
            print(f"EQ SUGGESTION FOR: {blend_str}")
        elif target:
            print(f"EQ SUGGESTION FOR: '{target}'")
        print("-"*70)

        freqs = [60, 150, 400, 1000, 2500, 6000, 12000]
        gains = eq_params[:7].cpu().numpy()
        gains_db = (gains - 0.5) * 24

        print(f"{'Frequency':<12} {'Gain (dB)':>10}")
        print("-"*22)
        for f, g in zip(freqs, gains_db):
            print(f"{f:>6} Hz    {g:>+8.2f}")

    print("\n" + "="*70)


def parse_blend_string(blend_str: str) -> Dict[str, float]:
    """Parse blend string like 'warm:0.5,punchy:0.3' into dict."""
    blend = {}
    for part in blend_str.split(','):
        if ':' in part:
            term, intensity = part.strip().split(':')
            blend[term.strip()] = float(intensity)
    return blend


def main():
    parser = argparse.ArgumentParser(description='Semantic audio analysis with temporal evolution')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    parser.add_argument('--encoder', type=str, default='audio_encoder_e2e.pt',
                       help='Audio encoder checkpoint')
    parser.add_argument('--v2-model', type=str, default='neural_eq_safedb_v2.pt',
                       help='V2 semantic EQ model')
    parser.add_argument('--window', type=float, default=3.0,
                       help='Analysis window size in seconds')
    parser.add_argument('--hop', type=float, default=1.0,
                       help='Hop size between windows in seconds')
    parser.add_argument('--target', type=str, default=None,
                       help='Target semantic for EQ suggestion (e.g., "warm")')
    parser.add_argument('--blend', type=str, default=None,
                       help='Blend multiple targets (e.g., "warm:0.5,punchy:0.3")')
    parser.add_argument('--intensity', type=float, default=0.5,
                       help='EQ intensity for single target')
    parser.add_argument('--output', type=str, default='semantic_analysis.png',
                       help='Output visualization path')
    parser.add_argument('--no-energy-weight', action='store_true',
                       help='Disable energy-based weighting (use uniform weights)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Check files
    if not Path(args.audio).exists():
        print(f"Error: Audio file not found: {args.audio}")
        return
    if not Path(args.encoder).exists():
        print(f"Error: Encoder not found: {args.encoder}")
        return
    if not Path(args.v2_model).exists():
        print(f"Error: V2 model not found: {args.v2_model}")
        return

    # Load models
    print("\nLoading models...")
    v2_model = load_v2_model(args.v2_model, device)
    encoder = load_encoder(args.encoder, device)

    # Run temporal analysis
    result = analyze_audio_temporal(
        args.audio, encoder, v2_model, device,
        window_size_sec=args.window,
        hop_size_sec=args.hop,
        use_energy_weighting=not args.no_energy_weight
    )

    # Compute EQ if target specified
    eq_params = None
    blend_targets = None

    if args.blend:
        blend_targets = parse_blend_string(args.blend)
        centroids = compute_semantic_centroids(v2_model, device)
        # Use mean latent as starting point
        z_mean = torch.from_numpy(result.latent_trajectory.mean(axis=0)).float().to(device)
        eq_params = compute_blended_eq(v2_model, z_mean, blend_targets, centroids, device)
    elif args.target:
        centroids = compute_semantic_centroids(v2_model, device)
        z_mean = torch.from_numpy(result.latent_trajectory.mean(axis=0)).float().to(device)
        blend_targets = {args.target: args.intensity}
        eq_params = compute_blended_eq(v2_model, z_mean, blend_targets, centroids, device)

    # Print summary
    print_analysis_summary(result, args.target, blend_targets, eq_params)

    # Create visualization
    plot_temporal_analysis(result, args.output)

    print(f"\nAnalysis complete!")


if __name__ == '__main__':
    main()
