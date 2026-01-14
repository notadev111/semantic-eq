"""
Test Adaptive System Components
================================

Quick test to verify all modules are working correctly.

Usage:
    python test_adaptive_system.py
"""

import torch
import numpy as np
from pathlib import Path


def test_audio_encoder():
    """Test Audio Encoder"""
    print("\n" + "="*70)
    print("TEST 1: Audio Encoder")
    print("="*70)

    from core.audio_encoder import FastAudioEncoder, AudioEncoderConfig

    encoder = FastAudioEncoder(**AudioEncoderConfig.STANDARD)
    print(f"[OK] Model created: {encoder.get_num_parameters():,} parameters")

    # Test inference
    audio = torch.randn(2, 1, 88200)  # 2 samples, 2 seconds each
    z_audio = encoder(audio)

    print(f"[OK] Input: {audio.shape}")
    print(f"[OK] Output: {z_audio.shape}")
    print(f"[OK] Range: [{z_audio.min():.3f}, {z_audio.max():.3f}]")

    return True


def test_training_synthesis():
    """Test Training Data Synthesis"""
    print("\n" + "="*70)
    print("TEST 2: Training Data Synthesis")
    print("="*70)

    from core.training_data_synthesis import PinkNoiseGenerator, BiquadEQFilter

    # Test pink noise
    generator = PinkNoiseGenerator()
    pink_noise = generator.generate(44100 * 2)
    print(f"[OK] Pink noise: {pink_noise.shape}")

    # Test EQ filter
    eq_filter = BiquadEQFilter()
    eq_params = np.array([
        2.0, 200.0,           # Band 1
        1.5, 1000.0, 1.5,     # Band 2
        -1.0, 3000.0, 2.0,    # Band 3
        0.5, 8000.0, 1.0,     # Band 4
        1.0, 12000.0          # Band 5
    ])

    processed = eq_filter.apply_eq(pink_noise, eq_params)
    print(f"[OK] EQ applied: {processed.shape}")
    print(f"[OK] Range: [{processed.min():.3f}, {processed.max():.3f}]")

    return True


def test_adaptive_generator():
    """Test Adaptive EQ Generator (requires trained models)"""
    print("\n" + "="*70)
    print("TEST 3: Adaptive EQ Generator")
    print("="*70)

    v2_path = Path('neural_eq_safedb_v2.pt')
    audio_encoder_path = Path('audio_encoder_best.pt')

    if not v2_path.exists():
        print("[WARN] V2 model not found, skipping test")
        print(f"   Expected: {v2_path}")
        return False

    if not audio_encoder_path.exists():
        print("[WARN] Audio Encoder not found, skipping test")
        print(f"   Expected: {audio_encoder_path}")
        print("   Train with: python train_audio_encoder.py")
        return False

    from core.adaptive_eq_generator import AdaptiveEQGenerator

    generator = AdaptiveEQGenerator(
        v2_model_path=str(v2_path),
        audio_encoder_path=str(audio_encoder_path)
    )

    print(f"[OK] Generator loaded")
    print(f"[OK] Semantic terms: {len(generator.semantic_embeddings)}")

    # Test with synthetic audio
    audio = torch.randn(1, 1, 88200)

    # Test adaptive EQ generation
    eq_params, similarity = generator.generate_adaptive_eq(
        audio,
        semantic_target='warm',
        intensity=0.7,
        return_similarity=True
    )

    print(f"[OK] Generated EQ: {eq_params.shape}")
    print(f"[OK] Similarity to 'warm': {similarity:.3f}")
    print(f"[OK] Band 1 gain: {eq_params[0]:+.2f}dB @ {eq_params[1]:.1f}Hz")

    # Test semantic profile
    profile = generator.get_semantic_profile(audio, top_k=5)
    print(f"[OK] Semantic profile (top 5):")
    for term, sim in profile[:3]:
        print(f"   - {term}: {sim:.3f}")

    return True


def test_streaming_processor():
    """Test Streaming Processor (requires trained models)"""
    print("\n" + "="*70)
    print("TEST 4: Streaming Processor")
    print("="*70)

    v2_path = Path('neural_eq_safedb_v2.pt')
    audio_encoder_path = Path('audio_encoder_best.pt')

    if not v2_path.exists() or not audio_encoder_path.exists():
        print("⚠️  Models not found, skipping test")
        return False

    from core.streaming_adaptive_eq import StreamingAdaptiveEQ

    processor = StreamingAdaptiveEQ(
        v2_model_path=str(v2_path),
        audio_encoder_path=str(audio_encoder_path),
        frame_size=512,
        update_interval=4
    )

    print(f"✅ Processor created")
    print(f"✅ Estimated latency: {processor.get_latency_ms():.2f}ms")

    # Set target
    processor.set_target('warm', intensity=0.7)
    print(f"✅ Target set: 'warm' @ 0.7")

    # Simulate streaming
    n_frames = 10
    for i in range(n_frames):
        audio_frame = np.random.randn(2, 512).astype(np.float32) * 0.1
        processed_frame = processor.process_frame(audio_frame)

        if i == 0:
            print(f"✅ Processed frame: {processed_frame.shape}")

    print(f"✅ Processed {n_frames} frames successfully")

    return True


def main():
    print("="*70)
    print("ADAPTIVE SYSTEM - COMPONENT TESTS")
    print("="*70)

    results = {}

    # Test 1: Audio Encoder
    try:
        results['audio_encoder'] = test_audio_encoder()
    except Exception as e:
        print(f"[FAIL] Audio Encoder test failed: {e}")
        results['audio_encoder'] = False

    # Test 2: Training Synthesis
    try:
        results['training_synthesis'] = test_training_synthesis()
    except Exception as e:
        print(f"[FAIL] Training Synthesis test failed: {e}")
        results['training_synthesis'] = False

    # Test 3: Adaptive Generator
    try:
        results['adaptive_generator'] = test_adaptive_generator()
    except Exception as e:
        print(f"[FAIL] Adaptive Generator test failed: {e}")
        results['adaptive_generator'] = False

    # Test 4: Streaming Processor
    try:
        results['streaming_processor'] = test_streaming_processor()
    except Exception as e:
        print(f"[FAIL] Streaming Processor test failed: {e}")
        results['streaming_processor'] = False

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for component, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{component:25s}: {status}")

    # Overall
    all_core_passed = results['audio_encoder'] and results['training_synthesis']
    print("\n" + "-"*70)

    if all_core_passed:
        print("[PASS] CORE COMPONENTS: All tests passed!")
        print("\nNext steps:")
        print("  1. Train Audio Encoder: python train_audio_encoder.py")
        print("  2. Test with demo: python demo_adaptive_eq.py --input mix.wav --analyze")
    else:
        print("[FAIL] CORE COMPONENTS: Some tests failed")
        print("\nPlease check the errors above")

    if not results.get('adaptive_generator', False):
        print("\n[WARN] MODELS NOT TRAINED")
        print("   Adaptive Generator requires trained models:")
        print("   - V2 model: python train_neural_eq_v2.py")
        print("   - Audio Encoder: python train_audio_encoder.py")


if __name__ == "__main__":
    main()
