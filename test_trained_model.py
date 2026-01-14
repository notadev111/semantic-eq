"""
Test the Trained Neural EQ Model
=================================

This script loads the trained model and demonstrates its capabilities.
"""

import torch
import sys
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent / 'core'))

def main():
    print("="*60)
    print("TESTING TRAINED NEURAL EQ MODEL")
    print("="*60)

    # Load the trained model
    print("\n[1/3] Loading trained model...")
    try:
        checkpoint = torch.load('neural_eq_model.pt', map_location='cpu', weights_only=False)
        print("   Model loaded successfully!")
    except Exception as e:
        print(f"   Error loading model: {e}")
        return

    # Check what's in the checkpoint
    print("\n[2/3] Model checkpoint contains:")
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], dict):
            print(f"   - {key}: {len(checkpoint[key])} entries")
        elif isinstance(checkpoint[key], torch.nn.Module):
            total = sum(p.numel() for p in checkpoint[key].parameters())
            print(f"   - {key}: Neural network ({total:,} parameters)")
        elif hasattr(checkpoint[key], 'shape'):
            print(f"   - {key}: Tensor {list(checkpoint[key].shape)}")
        else:
            print(f"   - {key}: {type(checkpoint[key]).__name__}")

    # Inspect encoder architecture
    print("\n[3/3] Model details:")
    encoder_state = checkpoint['encoder']
    decoder_state = checkpoint['decoder']

    enc_params = sum(p.numel() for p in encoder_state.values())
    dec_params = sum(p.numel() for p in decoder_state.values())

    print(f"   Encoder: {enc_params:,} parameters")
    print(f"   Decoder: {dec_params:,} parameters")
    print(f"   Total: {enc_params + dec_params:,} parameters")

    # Semantic terms
    semantic_terms = list(checkpoint['semantic_to_idx'].keys())
    print(f"\n   Semantic terms trained: {len(semantic_terms)}")
    print(f"   Example terms: {', '.join(semantic_terms[:15])}")

    # Training data info
    training_data = checkpoint['training_data']
    print(f"\n   Training examples: {training_data['params'].shape[0]}")
    print(f"   EQ parameters per example: {training_data['n_params']}")

    # Summary
    print("\n" + "="*60)
    print("MODEL ANALYSIS COMPLETE!")
    print("="*60)
    print("\nThe model is trained and ready to use.")
    print("\nTo test it on audio:")
    print("  1. Load the system: system = NeuralEQMorphingSystem(...)")
    print("  2. Load dataset: system.load_dataset(...)")
    print("  3. Load trained weights from checkpoint")
    print("  4. Generate EQ: system.generate_eq_from_semantic('warm')")
    print("\nNext recommended steps:")
    print("  - Run semantic_term_analysis.py for deeper insights")
    print("  - Test on real audio files")
    print("  - Visualize latent space")

if __name__ == '__main__':
    main()
