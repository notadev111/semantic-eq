"""
Quick Sanity Check - Test Audio Encoder is Working
===================================================

This script does a quick test to verify the trained Audio Encoder
is working correctly before you test with real audio.

Usage:
    python quick_sanity_check.py
"""

import torch
import numpy as np
from pathlib import Path

print("="*70)
print("QUICK SANITY CHECK - AUDIO ENCODER")
print("="*70)

# Check if models exist
v2_path = Path('neural_eq_safedb_v2.pt')
audio_encoder_path = Path('audio_encoder_best.pt')

if not v2_path.exists():
    print(f"\nERROR: V2 model not found: {v2_path}")
    exit(1)

if not audio_encoder_path.exists():
    print(f"\nERROR: Audio Encoder not found: {audio_encoder_path}")
    print("Please train it first: python train_audio_encoder.py")
    exit(1)

print("\n[OK] Models found")
print(f"  - V2 Model: {v2_path}")
print(f"  - Audio Encoder: {audio_encoder_path}")

# Load the system
print("\nLoading Adaptive EQ Generator...")
from core.adaptive_eq_generator import AdaptiveEQGenerator

generator = AdaptiveEQGenerator(
    v2_model_path=str(v2_path),
    audio_encoder_path=str(audio_encoder_path)
)

print("\n[OK] System loaded successfully!")
print(f"  - Available semantic terms: {len(generator.semantic_embeddings)}")
print(f"  - Terms: {', '.join(list(generator.semantic_embeddings.keys())[:5])}...")

# Test 1: Generate random audio and analyze
print("\n" + "="*70)
print("TEST 1: Analyzing Synthetic Audio")
print("="*70)

synthetic_audio = torch.randn(1, 1, 88200)  # 2 seconds random noise
print("\nGenerated synthetic audio (random noise)")

# Get semantic profile
profile = generator.get_semantic_profile(synthetic_audio, top_k=5)

print("\nTop 5 semantic characteristics:")
for i, (term, similarity) in enumerate(profile, 1):
    bar = "=" * int(similarity * 40)
    print(f"  {i}. {term:15s} [{similarity:.3f}] {bar}")

# Test 2: Generate adaptive EQ
print("\n" + "="*70)
print("TEST 2: Generating Adaptive EQ")
print("="*70)

target = 'warm'
eq_params, similarity = generator.generate_adaptive_eq(
    synthetic_audio,
    semantic_target=target,
    intensity=0.7,
    return_similarity=True
)

print(f"\nTarget: '{target}' with intensity 0.7")
print(f"Current similarity: {similarity:.3f}")
print(f"\nGenerated EQ parameters:")
print(f"  Band 1: {eq_params[0]:+6.2f}dB @ {eq_params[1]:7.1f}Hz")
print(f"  Band 2: {eq_params[2]:+6.2f}dB @ {eq_params[3]:7.1f}Hz")
print(f"  Band 3: {eq_params[5]:+6.2f}dB @ {eq_params[6]:7.1f}Hz")
print(f"  Band 4: {eq_params[8]:+6.2f}dB @ {eq_params[9]:7.1f}Hz")
print(f"  Band 5: {eq_params[11]:+6.2f}dB @ {eq_params[12]:7.1f}Hz")

# Test 3: Verify different audio produces different results
print("\n" + "="*70)
print("TEST 3: Testing Adaptive Behavior")
print("="*70)

# Create two different synthetic signals
audio1 = torch.randn(1, 1, 88200) * 0.5  # Lower amplitude
audio2 = torch.randn(1, 1, 88200) * 2.0  # Higher amplitude

eq1, sim1 = generator.generate_adaptive_eq(audio1, 'warm', 0.7, return_similarity=True)
eq2, sim2 = generator.generate_adaptive_eq(audio2, 'warm', 0.7, return_similarity=True)

print("\nAudio 1 (quiet):")
print(f"  Similarity to 'warm': {sim1:.3f}")
print(f"  Band 1 gain: {eq1[0]:+.2f}dB")

print("\nAudio 2 (loud):")
print(f"  Similarity to 'warm': {sim2:.3f}")
print(f"  Band 1 gain: {eq2[0]:+.2f}dB")

eq_diff = np.abs(eq1 - eq2).mean()
print(f"\nAverage EQ parameter difference: {eq_diff:.3f}")

if eq_diff > 0.01:
    print("[OK] GOOD: Different audio produces different EQ (adaptive!)")
else:
    print("[WARN] WARNING: EQ parameters are too similar (might not be adaptive)")

# Summary
print("\n" + "="*70)
print("SANITY CHECK COMPLETE")
print("="*70)

print("\n[OK] Audio Encoder is loaded and working!")
print("[OK] System can analyze audio semantic profiles")
print("[OK] System can generate adaptive EQ parameters")

print("\n" + "-"*70)
print("READY FOR REAL AUDIO TESTING!")
print("-"*70)

print("\nNext steps:")
print("  1. Get some real audio files (different genres/styles)")
print("  2. Run: python test_with_real_audio.py --input your_song.wav")
print("  3. Check the visualizations in ./analysis_results/")

print("\nYou can now skip synthetic testing and go straight to real audio!")
