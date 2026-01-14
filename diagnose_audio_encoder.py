"""
Diagnose Audio Encoder Training Issues
======================================

Check if the Audio Encoder actually learned to distinguish different audio.
"""

import torch
import numpy as np
from pathlib import Path

print("="*70)
print("DIAGNOSING AUDIO ENCODER")
print("="*70)

from core.adaptive_eq_generator import AdaptiveEQGenerator

# Load system
generator = AdaptiveEQGenerator(
    v2_model_path='neural_eq_safedb_v2.pt',
    audio_encoder_path='audio_encoder_best.pt'
)

print("\n" + "="*70)
print("TEST 1: Random Noise Variations")
print("="*70)

# Test with very different synthetic signals
test_cases = [
    ("Random noise", torch.randn(1, 1, 88200)),
    ("Low frequency sine", torch.sin(torch.linspace(0, 100*2*np.pi, 88200)).unsqueeze(0).unsqueeze(0)),
    ("High frequency sine", torch.sin(torch.linspace(0, 5000*2*np.pi, 88200)).unsqueeze(0).unsqueeze(0)),
    ("Silence", torch.zeros(1, 1, 88200)),
    ("Loud noise", torch.randn(1, 1, 88200) * 5.0),
]

print("\nTesting with 5 very different audio signals:")

# Get device from audio encoder
device = next(generator.audio_encoder.parameters()).device
print(f"Running on device: {device}")

results = []
for name, audio in test_cases:
    # Move audio to same device as model
    audio = audio.to(device)

    # Get latent vector
    z_audio = generator.audio_encoder(audio)

    # Get semantic profile
    profile = generator.get_semantic_profile(audio, top_k=5)

    print(f"\n{name}:")
    print(f"  Latent vector stats: mean={z_audio.mean():.3f}, std={z_audio.std():.3f}")
    print(f"  Top 3 semantics:")
    for term, sim in profile[:3]:
        print(f"    - {term:15s}: {sim:.3f}")

    results.append({
        'name': name,
        'latent': z_audio.detach().cpu().numpy(),
        'profile': profile
    })

# Check if latent vectors are different
print("\n" + "="*70)
print("LATENT VECTOR ANALYSIS")
print("="*70)

print("\nComputing pairwise distances between latent vectors:")
for i in range(len(results)):
    for j in range(i+1, len(results)):
        dist = np.linalg.norm(results[i]['latent'] - results[j]['latent'])
        print(f"  {results[i]['name']:20s} vs {results[j]['name']:20s}: {dist:.3f}")

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

# Check if all latents are too similar
latents = np.array([r['latent'].flatten() for r in results])
avg_distance = np.mean([np.linalg.norm(latents[i] - latents[j])
                        for i in range(len(latents))
                        for j in range(i+1, len(latents))])

print(f"\nAverage latent distance: {avg_distance:.3f}")

if avg_distance < 0.5:
    print("\n[PROBLEM] Latent vectors are TOO SIMILAR!")
    print("This suggests the Audio Encoder did not learn properly.")
    print("\nPossible causes:")
    print("  1. Training loss didn't converge")
    print("  2. Learning rate too high/low")
    print("  3. Not enough training epochs")
    print("  4. Audio Encoder architecture too simple")
    print("\nSolution: Re-train with more epochs or check training loss curve")
else:
    print("\n[OK] Latent vectors are sufficiently different")

# Check semantic profile diversity
print("\n" + "="*70)
print("SEMANTIC PROFILE DIVERSITY")
print("="*70)

all_top_terms = [r['profile'][0][0] for r in results]
unique_top_terms = len(set(all_top_terms))

print(f"\nUnique top-1 terms across 5 different signals: {unique_top_terms}/5")

if unique_top_terms == 1:
    print("\n[PROBLEM] All audio gets the same top semantic term!")
    print("This confirms the Audio Encoder is not working properly.")
elif unique_top_terms < 3:
    print("\n[WARNING] Low diversity in semantic classification")
else:
    print("\n[OK] Good semantic diversity")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

if avg_distance < 0.5 or unique_top_terms < 3:
    print("\nThe Audio Encoder needs re-training!")
    print("\nSteps to fix:")
    print("  1. Check training logs - did loss decrease?")
    print("  2. Re-train with more epochs: python train_audio_encoder.py --epochs 100")
    print("  3. Check if training data was synthesized correctly")
    print("  4. Verify contrastive loss was working")
else:
    print("\nAudio Encoder seems to be working correctly!")
    print("The issue might be with the specific audio files you're testing.")
