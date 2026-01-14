"""
Train Neural EQ Morphing System on SAFE-DB
===========================================

Simple training script for the SAFE-DB version of neural EQ morphing.

Usage:
    python train_neural_eq_safedb.py
    python train_neural_eq_safedb.py --epochs 100 --batch-size 32
    python train_neural_eq_safedb.py --min-examples 15 --contrastive-weight 0.7
"""

import argparse
from pathlib import Path
import sys

# Add core to path
sys.path.insert(0, str(Path(__file__).parent))

from core.neural_eq_morphing_safedb import NeuralEQMorphingSAFEDB


def main():
    parser = argparse.ArgumentParser(description='Train Neural EQ Morphing System on SAFE-DB')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--latent-dim', type=int, default=32,
                       help='Latent space dimension (default: 32)')
    parser.add_argument('--contrastive-weight', type=float, default=0.5,
                       help='Weight for contrastive loss (default: 0.5, higher = better clustering)')

    # Dataset parameters
    parser.add_argument('--min-examples', type=int, default=10,
                       help='Minimum examples per semantic term (default: 10)')
    parser.add_argument('--include-audio', action='store_true',
                       help='Load audio features (slower but more complete)')

    # Output
    parser.add_argument('--output', type=str, default='neural_eq_safedb.pt',
                       help='Output model path (default: neural_eq_safedb.pt)')
    parser.add_argument('--save-every', type=int, default=20,
                       help='Save checkpoint every N epochs (default: 20)')

    args = parser.parse_args()

    print("="*70)
    print("TRAIN NEURAL EQ MORPHING SYSTEM - SAFE-DB")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Latent dimension: {args.latent_dim}")
    print(f"  Contrastive weight: {args.contrastive_weight}")
    print(f"  Min examples per term: {args.min_examples}")
    print(f"  Include audio features: {args.include_audio}")
    print(f"  Output model: {args.output}")
    print()

    # Initialize system
    print("Initializing neural EQ morphing system...")
    system = NeuralEQMorphingSAFEDB(latent_dim=args.latent_dim)

    # Load dataset
    print("\nLoading SAFE-DB dataset...")
    system.load_dataset(
        min_examples=args.min_examples,
        include_audio=args.include_audio
    )

    # Train
    print("\nStarting training...")
    system.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        contrastive_weight=args.contrastive_weight,
        save_every=args.save_every
    )

    # Save final model
    print(f"\nSaving final model to: {args.output}")
    system.save_model(args.output)

    # Test generation
    print("\n" + "="*70)
    print("TESTING MODEL")
    print("="*70)

    print("\nGenerating EQ parameters for common terms:")
    test_terms = ['warm', 'bright', 'punchy', 'smooth']

    for term in test_terms:
        try:
            params = system.generate_eq_from_term(term)
            print(f"\n'{term}':")
            print(f"  Band 1: Gain={params[0]:6.2f}dB, Freq={params[1]:7.1f}Hz")
            print(f"  Band 2: Gain={params[2]:6.2f}dB, Freq={params[3]:7.1f}Hz, Q={params[4]:5.2f}")
            print(f"  Band 3: Gain={params[5]:6.2f}dB, Freq={params[6]:7.1f}Hz, Q={params[7]:5.2f}")
            print(f"  Band 4: Gain={params[8]:6.2f}dB, Freq={params[9]:7.1f}Hz, Q={params[10]:5.2f}")
            print(f"  Band 5: Gain={params[11]:6.2f}dB, Freq={params[12]:7.1f}Hz")
        except ValueError:
            print(f"\n'{term}': Not found in dataset")

    # Test interpolation
    print("\n" + "="*70)
    print("TESTING SEMANTIC INTERPOLATION")
    print("="*70)

    try:
        print("\nInterpolating 'warm' -> 'bright':")
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            params = system.interpolate_terms('warm', 'bright', alpha)
            print(f"\n  alpha={alpha:.2f}: Band1_Gain={params[0]:6.2f}dB, Band5_Gain={params[11]:6.2f}dB")
    except ValueError as e:
        print(f"\nInterpolation test skipped: {e}")

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModel saved to: {args.output}")
    print("\nNext steps:")
    print("  1. Review clustering metrics above")
    print("  2. Test on real audio with semantic_mastering_safedb.py")
    print("  3. Compare with SocialFX version")
    print("  4. Generate visualizations for your report")


if __name__ == "__main__":
    main()
