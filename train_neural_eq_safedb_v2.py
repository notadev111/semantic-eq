"""
Train Neural EQ Morphing System V2 on SAFE-DB
==============================================

IMPROVED VERSION with proper normalization.

Usage:
    python train_neural_eq_safedb_v2.py
    python train_neural_eq_safedb_v2.py --epochs 150 --batch-size 64
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from core.neural_eq_morphing_safedb_v2 import NeuralEQMorphingSAFEDBV2


def main():
    parser = argparse.ArgumentParser(description='Train Neural EQ Morphing V2 on SAFE-DB')

    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of training epochs (default: 150)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (default: 64, larger for better contrastive learning)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--latent-dim', type=int, default=32,
                       help='Latent space dimension (default: 32)')
    parser.add_argument('--contrastive-start', type=float, default=0.1,
                       help='Initial contrastive weight (default: 0.1)')
    parser.add_argument('--contrastive-end', type=float, default=0.5,
                       help='Final contrastive weight (default: 0.5)')
    parser.add_argument('--min-examples', type=int, default=5,
                       help='Minimum examples per term (default: 5)')
    parser.add_argument('--output', type=str, default='neural_eq_safedb_v2.pt',
                       help='Output model path')
    parser.add_argument('--save-every', type=int, default=30,
                       help='Save checkpoint every N epochs')

    args = parser.parse_args()

    print("="*70)
    print("TRAIN NEURAL EQ MORPHING SYSTEM V2 - SAFE-DB")
    print("="*70)
    print("\nKEY IMPROVEMENTS FROM V1:")
    print("  - LOG-SCALE normalization for frequencies")
    print("  - Proper decoder with sigmoid activation")
    print("  - Annealed contrastive weight (0.1 -> 0.5)")
    print("  - Larger batch size (64 vs 32)")
    print("  - Learning rate scheduling")
    print(f"\nConfiguration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Latent dimension: {args.latent_dim}")
    print(f"  Contrastive weight: {args.contrastive_start} → {args.contrastive_end}")
    print(f"  Min examples per term: {args.min_examples}")
    print(f"  Output model: {args.output}")
    print()

    # Initialize
    print("Initializing neural EQ morphing system V2...")
    system = NeuralEQMorphingSAFEDBV2(latent_dim=args.latent_dim)

    # Load dataset
    print("\nLoading SAFE-DB dataset...")
    system.load_dataset(min_examples=args.min_examples)

    # Train
    print("\nStarting training...")
    system.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        contrastive_weight_start=args.contrastive_start,
        contrastive_weight_end=args.contrastive_end,
        save_every=args.save_every
    )

    # Save final model
    print(f"\nSaving final model to: {args.output}")
    system.save_model(args.output)

    # Test generation
    print("\n" + "="*70)
    print("TESTING MODEL - GENERATING EQ PARAMETERS")
    print("="*70)

    test_terms = ['warm', 'bright', 'clear']

    for term in test_terms:
        try:
            params = system.generate_eq_from_term(term)
            print(f"\n'{term}':")
            print(f"  Band 1: Gain={params[0]:6.2f}dB, Freq={params[1]:7.1f}Hz")
            print(f"  Band 2: Gain={params[2]:6.2f}dB, Freq={params[3]:7.1f}Hz, Q={params[4]:5.2f}")
            print(f"  Band 3: Gain={params[5]:6.2f}dB, Freq={params[6]:7.1f}Hz, Q={params[7]:5.2f}")
            print(f"  Band 4: Gain={params[8]:6.2f}dB, Freq={params[9]:7.1f}Hz, Q={params[10]:5.2f}")
            print(f"  Band 5: Gain={params[11]:6.2f}dB, Freq={params[12]:7.1f}Hz")

            # Validate
            gains = [params[0], params[2], params[5], params[8], params[11]]
            freqs = [params[1], params[3], params[6], params[9], params[12]]
            qs = [params[4], params[7], params[10]]

            valid_gains = all(-13 <= g <= 13 for g in gains)
            valid_freqs = all(20 <= f <= 22000 for f in freqs)
            valid_qs = all(0.09 <= q <= 11 for q in qs)

            if valid_gains and valid_freqs and valid_qs:
                print(f"  STATUS: VALID (all parameters in range!)")
            else:
                print(f"  STATUS: INVALID")
                if not valid_gains:
                    print(f"    - Gains out of range: {[g for g in gains if not (-13 <= g <= 13)]}")
                if not valid_freqs:
                    print(f"    - Freqs out of range: {[f for f in freqs if not (20 <= f <= 22000)]}")
                if not valid_qs:
                    print(f"    - Qs out of range: {[q for q in qs if not (0.09 <= q <= 11)]}")

        except ValueError:
            print(f"\n'{term}': Not found in dataset")

    # Test interpolation
    print("\n" + "="*70)
    print("TESTING SEMANTIC INTERPOLATION")
    print("="*70)

    try:
        print("\nInterpolating 'warm' -> 'bright':")
        for alpha in [0.0, 0.5, 1.0]:
            params = system.interpolate_terms('warm', 'bright', alpha)
            print(f"  alpha={alpha:.1f}: Band1_Gain={params[0]:6.2f}dB, Band1_Freq={params[1]:7.1f}Hz")
    except ValueError as e:
        print(f"\nInterpolation test skipped: {e}")

    print("\n" + "="*70)
    print("TRAINING COMPLETE - V2!")
    print("="*70)
    print(f"\nModel saved to: {args.output}")
    print("\nComparison with V1:")
    print("  V1: Frequencies 10-80x too high, Silhouette: -0.50")
    print("  V2: Should have valid frequencies, Silhouette: >0.3")
    print("\nNext steps:")
    print("  1. Check generated EQ parameters above")
    print("  2. Review clustering metrics")
    print("  3. Compare V1 vs V2 in your report")


if __name__ == "__main__":
    main()
