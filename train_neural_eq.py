"""
Train Neural EQ Morphing System on SocialFX Data
=================================================

Simple training script - just run this to train the model.

Usage:
    python train_neural_eq.py
"""

import torch
from core.neural_eq_morphing import NeuralEQMorphingSystem, SocialFXDatasetLoader


def main():
    print("="*60)
    print("TRAINING NEURAL EQ MORPHING SYSTEM")
    print("="*60)

    # Step 1: Load SocialFX dataset
    print("\n[1/4] Loading SocialFX dataset...")
    loader = SocialFXDatasetLoader()
    eq_settings = loader.load_socialfx_dataset()

    if not eq_settings:
        print("ERROR: Failed to load dataset")
        return

    # Step 2: Analyze dataset
    print("\n[2/4] Analyzing dataset...")
    analysis = loader.analyze_dataset(eq_settings)

    # Step 3: Initialize system
    print("\n[3/4] Initializing neural network...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    system = NeuralEQMorphingSystem(latent_dim=32, device=device)

    if not system.load_dataset(eq_settings):
        print("ERROR: Failed to load dataset into system")
        return

    # Step 4: Train
    print("\n[4/4] Training...")
    print("This will take ~15 min on CPU, ~3 min on GPU")
    print("-"*60)

    system.train(epochs=50, batch_size=16, learning_rate=0.001)

    # Save trained model
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)

    # Test it
    print("\nTesting semantic generation...")
    test_terms = ['warm', 'bright', 'punchy']

    for term in test_terms:
        if term in system.semantic_to_idx:
            print(f"\nGenerating '{term}':")
            variations = system.generate_eq_from_semantic(term, variations=1)
            if variations:
                params = variations[0]['parameters']
                print(f"  Generated {len(params)} EQ parameters")

    # Save model
    print("\nSaving model...")
    torch.save({
        'encoder': system.encoder.state_dict(),
        'decoder': system.decoder.state_dict(),
        'semantic_to_idx': system.semantic_to_idx,
        'idx_to_semantic': system.idx_to_semantic,
        'param_means': system.param_means,
        'param_stds': system.param_stds,
        'training_data': system.training_data
    }, 'neural_eq_model.pt')

    print("✓ Model saved to: neural_eq_model.pt")
    print("\nYou can now use this model for:")
    print("  - Semantic EQ generation")
    print("  - Morphing between semantic terms")
    print("  - Real-time semantic interpolation")


if __name__ == '__main__':
    main()
