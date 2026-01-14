"""
Check Audio Encoder Training History
=====================================
"""

import torch
from pathlib import Path

print("="*70)
print("CHECKING AUDIO ENCODER TRAINING HISTORY")
print("="*70)

checkpoint_path = Path('audio_encoder_best.pt')

if not checkpoint_path.exists():
    print(f"\nERROR: {checkpoint_path} not found!")
    exit(1)

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print(f"\nCheckpoint contents:")
for key in checkpoint.keys():
    print(f"  - {key}")

# Check training history
if 'train_losses' in checkpoint:
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']

    print(f"\n" + "="*70)
    print("TRAINING HISTORY")
    print("="*70)

    print(f"\nTotal epochs trained: {len(train_losses)}")
    print(f"Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")

    print(f"\nTraining loss progression:")
    print("-" * 50)

    # Show every 5 epochs
    for i in range(0, len(train_losses), 5):
        epoch = i + 1
        train_loss = train_losses[i]
        val_loss = val_losses[i]
        print(f"  Epoch {epoch:3d}: Train={train_loss:.4f}, Val={val_loss:.4f}")

    # Show last epoch
    if len(train_losses) > 0:
        print(f"  Epoch {len(train_losses):3d}: Train={train_losses[-1]:.4f}, Val={val_losses[-1]:.4f}")

    # Check if training worked
    print(f"\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)

    initial_loss = train_losses[0]
    final_loss = train_losses[-1]
    loss_reduction = initial_loss - final_loss

    print(f"\nInitial loss: {initial_loss:.4f}")
    print(f"Final loss:   {final_loss:.4f}")
    print(f"Reduction:    {loss_reduction:.4f} ({loss_reduction/initial_loss*100:.1f}%)")

    if loss_reduction < 0.01:
        print("\n[PROBLEM] Loss barely decreased!")
        print("The model did NOT learn properly.")
        print("\nLikely causes:")
        print("  - Learning rate too low")
        print("  - Model architecture issue")
        print("  - Training data problem")
        print("\nSolution: Re-train with adjusted settings")
    elif loss_reduction < 0.1:
        print("\n[WARNING] Loss decreased but not enough")
        print("Model learned something but probably not well")
        print("\nSolution: Train for more epochs")
    else:
        print("\n[OK] Loss decreased significantly")
        print("Training appeared to work")
        print("\nIf audio analysis still fails, the issue might be:")
        print("  - Model needs MORE epochs")
        print("  - Contrastive loss weight needs tuning")
else:
    print("\nNo training history found in checkpoint")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

print("\nRe-train the Audio Encoder with MORE epochs:")
print("  python train_audio_encoder.py --epochs 100 --batch-size 32")
print("\nThis will take ~4-6 hours but should work better")
