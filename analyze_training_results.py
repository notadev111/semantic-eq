"""
Analyze SAFE-DB Training Results
"""
import torch
import numpy as np
from pathlib import Path

print("="*70)
print("SAFE-DB NEURAL EQ TRAINING RESULTS ANALYSIS")
print("="*70)

# Load the final model
model_path = "neural_eq_safedb.pt"
checkpoint = torch.load(model_path, map_location='cpu')

print(f"\nModel: {model_path}")
print(f"Latent dimension: {checkpoint['latent_dim']}")
print(f"Number of semantic terms: {len(checkpoint['term_to_idx'])}")

# Show semantic terms
print(f"\n{'='*70}")
print("SEMANTIC TERMS IN MODEL")
print("="*70)
terms = list(checkpoint['term_to_idx'].keys())
print(f"Total terms: {len(terms)}\n")
for i, term in enumerate(terms, 1):
    print(f"  {i:2d}. {term}")

# Training history
print(f"\n{'='*70}")
print("TRAINING HISTORY")
print("="*70)

if 'history' in checkpoint:
    hist = checkpoint['history']
    n_epochs = len(hist['total_loss'])

    print(f"\nTotal epochs trained: {n_epochs}")

    print(f"\nFirst epoch:")
    print(f"  Reconstruction loss: {hist['reconstruction_loss'][0]:.4f}")
    print(f"  Contrastive loss: {hist['contrastive_loss'][0]:.4f}")
    print(f"  Total loss: {hist['total_loss'][0]:.4f}")

    print(f"\nFinal epoch:")
    print(f"  Reconstruction loss: {hist['reconstruction_loss'][-1]:.4f}")
    print(f"  Contrastive loss: {hist['contrastive_loss'][-1]:.4f}")
    print(f"  Total loss: {hist['total_loss'][-1]:.4f}")

    print(f"\nLast 10 epochs:")
    print(f"  Reconstruction: {[f'{x:.4f}' for x in hist['reconstruction_loss'][-10:]]}")
    print(f"  Contrastive:    {[f'{x:.4f}' for x in hist['contrastive_loss'][-10:]]}")
    print(f"  Total:          {[f'{x:.4f}' for x in hist['total_loss'][-10:]]}")

    # Check convergence
    final_recon = hist['reconstruction_loss'][-1]
    final_contr = hist['contrastive_loss'][-1]

    print(f"\n{'='*70}")
    print("CONVERGENCE ANALYSIS")
    print("="*70)

    if final_recon < 0.2:
        print("✓ Reconstruction loss converged well (< 0.2)")
    elif final_recon < 0.5:
        print("~ Reconstruction loss moderate (0.2-0.5)")
    else:
        print("✗ Reconstruction loss high (> 0.5) - may need more epochs")

    if final_contr < 0.5:
        print("✓ Contrastive loss converged well (< 0.5)")
    elif final_contr < 1.0:
        print("~ Contrastive loss moderate (0.5-1.0)")
    else:
        print("✗ Contrastive loss high (> 1.0) - clustering may be poor")

# Normalization stats
print(f"\n{'='*70}")
print("PARAMETER NORMALIZATION")
print("="*70)
print(f"\nMean: {checkpoint['param_mean'].numpy()}")
print(f"Std:  {checkpoint['param_std'].numpy()}")

print(f"\n{'='*70}")
print("RECOMMENDATIONS")
print("="*70)

if 'history' in checkpoint:
    final_recon = hist['reconstruction_loss'][-1]
    final_contr = hist['contrastive_loss'][-1]

    if final_recon > 0.5 or final_contr > 1.0:
        print("\n⚠ Losses are high. Recommendations:")
        print("  - Train for more epochs (150-200)")
        print("  - Increase contrastive weight (0.7 or 1.0)")
        print("  - Check if model is learning (loss should decrease)")
    elif final_recon < 0.2 and final_contr < 0.5:
        print("\n✓ Model converged well!")
        print("  - Reconstruction and contrastive losses are good")
        print("  - Ready for evaluation and testing")
        print("  - Next: Evaluate clustering metrics")
    else:
        print("\n~ Model partially converged")
        print("  - Consider training a bit longer for better results")

print("\n" + "="*70)
