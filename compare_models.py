"""
Compare Old vs New Audio Encoder Models
========================================

Compares the embedding-matching model (audio_encoder_best.pt)
with the E2E differentiable model (audio_encoder_e2e.pt) on real audio.

Usage:
    python compare_models.py --audio path/to/audio.wav
    python compare_models.py --audio path/to/audio.wav --target "warm"
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Local imports
from core.audio_encoder import FastAudioEncoder, AudioEncoderConfig
from core.neural_eq_morphing_safedb_v2 import NeuralEQMorphingSAFEDBV2


def load_audio(audio_path: str, sample_rate: int = 44100, duration: float = 5.0):
    """Load audio file and return tensor."""
    import soundfile as sf
    import librosa

    # Load with soundfile
    audio_np, sr = sf.read(audio_path)

    # Convert to mono if stereo
    if len(audio_np.shape) > 1:
        audio_np = audio_np.mean(axis=1)

    # Resample if needed
    if sr != sample_rate:
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=sample_rate)

    # Convert to tensor
    waveform = torch.from_numpy(audio_np).float().unsqueeze(0)  # [1, samples]

    # Trim/pad to duration
    n_samples = int(sample_rate * duration)
    if waveform.shape[1] > n_samples:
        waveform = waveform[:, :n_samples]
    elif waveform.shape[1] < n_samples:
        waveform = torch.nn.functional.pad(waveform, (0, n_samples - waveform.shape[1]))

    return waveform


def load_encoder(encoder_path: str, device: str):
    """Load an audio encoder checkpoint."""
    checkpoint = torch.load(encoder_path, map_location=device, weights_only=False)

    if 'config' in checkpoint:
        config = AudioEncoderConfig(**checkpoint['config'])
        encoder = FastAudioEncoder(config)
    else:
        encoder = FastAudioEncoder()

    # Handle different checkpoint formats
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


def load_v2_model(v2_model_path: str, device: str):
    """Load the V2 semantic EQ model."""
    model = NeuralEQMorphingSAFEDBV2(latent_dim=32, device=device)
    # Load dataset first (needed for eq_settings)
    model.load_dataset(min_examples=3)
    # Then load trained weights
    model.load_model(v2_model_path)
    return model


def get_semantic_profile(encoder, v2_model, audio: torch.Tensor, device: str):
    """Get semantic descriptor scores for audio."""
    with torch.no_grad():
        audio = audio.to(device)
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)  # Add batch dim

        # Encode audio
        z_audio = encoder(audio)

        # Get scores for each semantic descriptor
        scores = {}
        for term in v2_model.term_to_idx.keys():
            # Get centroid embedding for this term
            examples = [s for s in v2_model.eq_settings if s.semantic_label == term]
            if not examples:
                continue

            # Get average latent for this term
            eq_params_list = []
            for ex in examples[:10]:  # Use up to 10 examples
                eq_params_list.append(ex.to_tensor())

            eq_batch = torch.stack(eq_params_list).to(device)
            z_semantic, _ = v2_model.encoder(eq_batch)
            z_centroid = z_semantic.mean(dim=0)

            # Cosine similarity
            sim = torch.nn.functional.cosine_similarity(z_audio, z_centroid.unsqueeze(0))
            scores[term] = sim.item()

        return scores, z_audio


def get_eq_prediction(v2_model, z_audio: torch.Tensor, target_semantic: str, intensity: float = 0.5, device: str = 'cpu'):
    """Get EQ prediction for a target semantic using latent traversal."""
    with torch.no_grad():
        # Get centroid for target semantic
        examples = [s for s in v2_model.eq_settings if s.semantic_label == target_semantic]
        if not examples:
            print(f"Warning: No examples for '{target_semantic}'")
            return torch.zeros(7)

        eq_params_list = [ex.to_tensor() for ex in examples[:10]]
        eq_batch = torch.stack(eq_params_list).to(device)
        z_target, _ = v2_model.encoder(eq_batch)
        z_centroid = z_target.mean(dim=0, keepdim=True)

        # Latent traversal
        z_morphed = z_audio + intensity * (z_centroid - z_audio)

        # Decode to EQ params
        eq_params = v2_model.decoder(z_morphed)

        return eq_params


def plot_comparison(old_scores, new_scores, old_eq, new_eq, target_semantic, output_path):
    """Create comparison visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Get common descriptors
    common_descriptors = sorted(set(old_scores.keys()) & set(new_scores.keys()))[:15]  # Limit to 15

    # 1. Semantic profile comparison
    ax1 = axes[0, 0]
    x = np.arange(len(common_descriptors))
    width = 0.35

    old_vals = [old_scores.get(d, 0) for d in common_descriptors]
    new_vals = [new_scores.get(d, 0) for d in common_descriptors]

    ax1.bar(x - width/2, old_vals, width, label='Old Model', color='steelblue', alpha=0.7)
    ax1.bar(x + width/2, new_vals, width, label='New E2E Model', color='coral', alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(common_descriptors, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Cosine Similarity')
    ax1.set_title('Semantic Profile Comparison')
    ax1.legend()
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # 2. Score difference
    ax2 = axes[0, 1]
    diff = [new_scores.get(d, 0) - old_scores.get(d, 0) for d in common_descriptors]
    colors = ['green' if d > 0 else 'red' for d in diff]
    ax2.bar(x, diff, color=colors, alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(common_descriptors, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Score Difference (New - Old)')
    ax2.set_title('Model Difference (green = new higher)')
    ax2.axhline(y=0, color='gray', linestyle='--')

    # 3. EQ curves comparison
    ax3 = axes[1, 0]
    freqs = [60, 150, 400, 1000, 2500, 6000, 12000]

    old_gains = old_eq.squeeze().cpu().numpy()
    new_gains = new_eq.squeeze().cpu().numpy()

    # Only use first 7 values (gains)
    old_gains = old_gains[:7] if len(old_gains) >= 7 else old_gains
    new_gains = new_gains[:7] if len(new_gains) >= 7 else new_gains

    # Convert from normalized [0,1] to dB [-12, +12]
    old_gains_db = (old_gains - 0.5) * 24
    new_gains_db = (new_gains - 0.5) * 24

    ax3.semilogx(freqs[:len(old_gains_db)], old_gains_db, 'o-', label='Old Model', color='steelblue', linewidth=2, markersize=8)
    ax3.semilogx(freqs[:len(new_gains_db)], new_gains_db, 's-', label='New E2E Model', color='coral', linewidth=2, markersize=8)
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Gain (dB)')
    ax3.set_title(f'EQ Prediction for "{target_semantic}"')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-15, 15)
    ax3.axhline(y=0, color='gray', linestyle='--')

    # 4. EQ difference
    ax4 = axes[1, 1]
    min_len = min(len(old_gains_db), len(new_gains_db))
    eq_diff = new_gains_db[:min_len] - old_gains_db[:min_len]
    colors = ['green' if d > 0 else 'red' for d in eq_diff]
    ax4.bar(range(min_len), eq_diff, color=colors, alpha=0.7)
    ax4.set_xticks(range(min_len))
    ax4.set_xticklabels([f'{f}Hz' for f in freqs[:min_len]])
    ax4.set_ylabel('Gain Difference (dB)')
    ax4.set_title('EQ Difference (New - Old)')
    ax4.axhline(y=0, color='gray', linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare old vs new audio encoder models')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    parser.add_argument('--target', type=str, default='warm', help='Target semantic for EQ prediction')
    parser.add_argument('--intensity', type=float, default=0.5, help='EQ intensity')
    parser.add_argument('--old-model', type=str, default='audio_encoder_best.pt', help='Old model path')
    parser.add_argument('--new-model', type=str, default='audio_encoder_e2e.pt', help='New E2E model path')
    parser.add_argument('--v2-model', type=str, default='neural_eq_safedb_v2.pt', help='V2 base model path')
    parser.add_argument('--output', type=str, default='model_comparison.png', help='Output plot path')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Check files exist
    if not Path(args.audio).exists():
        print(f"Error: Audio file not found: {args.audio}")
        return
    if not Path(args.old_model).exists():
        print(f"Error: Old model not found: {args.old_model}")
        return
    if not Path(args.new_model).exists():
        print(f"Error: New model not found: {args.new_model}")
        print("Did you pull it from the cluster? Run:")
        print(f'  scp zceeddu@athens:~/semantic-eq/audio_encoder_best.pt "{args.new_model}"')
        return
    if not Path(args.v2_model).exists():
        print(f"Error: V2 model not found: {args.v2_model}")
        return

    # Load audio
    print(f"Loading audio: {args.audio}")
    audio = load_audio(args.audio)

    # Load V2 model (shared between both encoders)
    print(f"Loading V2 model: {args.v2_model}")
    v2_model = load_v2_model(args.v2_model, device)

    # Load old encoder
    print(f"Loading old encoder: {args.old_model}")
    old_encoder = load_encoder(args.old_model, device)

    # Load new encoder
    print(f"Loading new E2E encoder: {args.new_model}")
    new_encoder = load_encoder(args.new_model, device)

    # Get semantic profiles
    print("Computing semantic profiles...")
    old_scores, old_z = get_semantic_profile(old_encoder, v2_model, audio, device)
    new_scores, new_z = get_semantic_profile(new_encoder, v2_model, audio, device)

    # Get EQ predictions
    print(f"Computing EQ predictions for '{args.target}'...")
    old_eq = get_eq_prediction(v2_model, old_z, args.target, args.intensity, device)
    new_eq = get_eq_prediction(v2_model, new_z, args.target, args.intensity, device)

    # Print summary
    print("\n" + "="*60)
    print("SEMANTIC PROFILE COMPARISON")
    print("="*60)

    common_terms = sorted(set(old_scores.keys()) & set(new_scores.keys()))
    print(f"{'Descriptor':<15} {'Old':>10} {'New':>10} {'Diff':>10}")
    print("-"*45)
    for desc in common_terms[:15]:
        diff = new_scores[desc] - old_scores[desc]
        print(f"{desc:<15} {old_scores[desc]:>10.4f} {new_scores[desc]:>10.4f} {diff:>+10.4f}")

    print("\n" + "="*60)
    print(f"EQ PREDICTION FOR '{args.target.upper()}' (intensity={args.intensity})")
    print("="*60)
    freqs = [60, 150, 400, 1000, 2500, 6000, 12000]

    old_gains = old_eq.squeeze().cpu().numpy()[:7]
    new_gains = new_eq.squeeze().cpu().numpy()[:7]
    old_db = (old_gains - 0.5) * 24
    new_db = (new_gains - 0.5) * 24

    print(f"{'Freq':<10} {'Old (dB)':>10} {'New (dB)':>10} {'Diff':>10}")
    print("-"*42)
    for i, f in enumerate(freqs[:len(old_db)]):
        diff = new_db[i] - old_db[i]
        print(f"{f:<10} {old_db[i]:>+10.2f} {new_db[i]:>+10.2f} {diff:>+10.2f}")

    # Plot comparison
    plot_comparison(old_scores, new_scores, old_eq, new_eq, args.target, args.output)

    print(f"\nComparison complete! Plot saved to: {args.output}")


if __name__ == '__main__':
    main()
