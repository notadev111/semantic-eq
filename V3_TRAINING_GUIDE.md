# Neural EQ Morphing V3 - Training Guide

## What is V3?

V3 fixes a critical limitation in V2: **limited frequency range**.

- **V2 problem**: Only generated frequencies 83-7,901 Hz (dataset goes up to 20,000 Hz!)
- **Root cause**: Data-dependent normalization learned average range, not full range
- **V3 solution**: Fixed normalization bounds (20-20,000 Hz) for full spectrum

## Quick Start

### 1. Activate Virtual Environment

```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Train V3 Model

```bash
# Default settings (recommended)
python train_neural_eq_safedb_v3.py

# Custom settings
python train_neural_eq_safedb_v3.py --epochs 200 --batch-size 128
```

### 3. Expected Output

```
======================================================================
TRAIN NEURAL EQ MORPHING SYSTEM V3 - SAFE-DB
======================================================================

KEY IMPROVEMENT FROM V2:
  - FIXED normalization bounds: 20-20,000 Hz (not data-dependent)
  - V2 only generated 83-7,901 Hz (missed high frequencies!)
  - V3 can utilize FULL audible spectrum

Configuration:
  Epochs: 150
  Batch size: 64
  Learning rate: 0.001
  ...

Training...
Epoch 1/150: Recon Loss=0.452, Contrast Loss=2.31, Total=0.683
...
Epoch 150/150: Recon Loss=0.148, Contrast Loss=1.42, Total=0.859

Model saved to neural_eq_safedb_v3.pt
```

## Training Parameters

| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `--epochs` | 150 | 150-200 | More epochs = better convergence |
| `--batch-size` | 64 | 64-128 | Larger = better contrastive learning |
| `--learning-rate` | 0.001 | 0.001 | Standard Adam LR |
| `--latent-dim` | 32 | 32-64 | Latent space dimension |
| `--contrastive-start` | 0.1 | 0.1 | Initial contrastive weight |
| `--contrastive-end` | 0.5 | 0.5 | Final contrastive weight |
| `--min-examples` | 5 | 5 | Min examples per semantic term |
| `--output` | neural_eq_safedb_v3.pt | - | Output model path |
| `--save-every` | 30 | 30 | Checkpoint interval |

## Expected Results

### Frequency Range

**V2 (limited)**:
- Minimum: 83 Hz
- Maximum: 7,901 Hz
- **Problem**: Missing 8-20 kHz (critical for "bright" sounds!)

**V3 (full spectrum)**:
- Minimum: ~20 Hz (theoretical lower bound)
- Maximum: ~20,000 Hz (theoretical upper bound)
- **Benefit**: Can generate proper high-frequency boosts for bright/airy sounds

### Example Outputs

**"bright" - V2 vs V3**:
```
V2 (limited):
  Band 5 (High Shelf): +3.49dB @ 5,785 Hz  ✓ OK but low

V3 (expected):
  Band 5 (High Shelf): +3.49dB @ 12,000 Hz  ✓✓ BETTER (proper high shelf!)
```

**"warm" - V2 vs V3**:
```
V2:
  Band 2 (Bell): +5.46dB @ 287 Hz  ✓ Good

V3 (expected):
  Band 2 (Bell): +5.46dB @ 287 Hz  ✓ Similar (warm doesn't need high freqs)
```

### Clustering Metrics

Target performance (similar to V2):
- **Silhouette Score**: 0.05-0.15 (moderate clustering)
- **Davies-Bouldin Index**: 3-5 (lower is better)
- **Reconstruction Loss**: <0.2 (final)

## Training Time

- **CPU**: ~30-40 minutes (150 epochs)
- **GPU**: ~8-12 minutes (150 epochs)

## What to Check After Training

### 1. Loss Curves

```
Reconstruction loss should:
  - Decrease steadily
  - Converge to <0.2
  - Not plateau early

Contrastive loss should:
  - Decrease initially
  - May increase slightly (normal with annealing)
  - Should stabilize
```

### 2. Generated Parameters

Test the model (see next section) and verify:
- ✅ All frequencies in 20-20,000 Hz range
- ✅ "Bright" sounds have high-shelf >10 kHz
- ✅ "Warm" sounds boost mid-bass ~200-500 Hz
- ✅ Smooth interpolation between semantic terms

### 3. Clustering Quality

Run test script to check:
- Silhouette score >0 (positive = distinct clusters)
- Davies-Bouldin index <10 (lower = better separation)

## Testing the Trained Model

Create `test_safedb_model_v3.py` (copy from V2 version and update imports):

```bash
python test_safedb_model_v3.py
```

Expected output:
```
Testing Neural EQ Morphing V3
==============================

Term: 'bright'
  Band 1 (Low Shelf):  -3.2dB @ 145 Hz
  Band 5 (High Shelf): +4.1dB @ 14500 Hz  <- HIGH FREQUENCY! ✅

Clustering Metrics:
  Silhouette: 0.12 (GOOD)
  Davies-Bouldin: 4.2 (GOOD)
```

## Troubleshooting

### Issue 1: High frequencies still limited

**Symptoms**: V3 generates max ~8-10 kHz instead of ~20 kHz

**Cause**: Dataset bias - most examples use lower frequencies

**Solutions**:
1. Train longer (200-300 epochs)
2. Increase batch size (128)
3. Check if "bright" examples in dataset actually use high frequencies
4. Consider data augmentation (shift frequencies up)

### Issue 2: Reconstruction loss doesn't converge

**Symptoms**: Loss stays >0.5 after 150 epochs

**Possible causes**:
- Learning rate too high/low
- Contrastive weight too high (>0.5)
- Batch size too small (<32)

**Solutions**:
- Reduce learning rate to 0.0005
- Reduce contrastive_end to 0.3
- Increase batch size to 128

### Issue 3: Clustering worse than V2

**Symptoms**: Silhouette <0, Davies-Bouldin >10

**Cause**: Fixed bounds may be too wide, making clustering harder

**Solution**: This is a trade-off - V3 prioritizes frequency range over clustering quality

## Files Created

After training, you'll have:
- `neural_eq_safedb_v3.pt` - Trained model
- `training_history_v3.png` - Loss curves (if plotting enabled)
- Checkpoints: `neural_eq_safedb_v3_epoch_30.pt`, etc.

## Next Steps

1. **Test the model**: Verify frequency ranges with test script
2. **Compare to V2**: Check if high frequencies improved
3. **Apply to audio**: Use `semantic_mastering_safedb_v3.py` (create from V2)
4. **A/B testing**: Compare V2 vs V3 on "bright" audio samples

## Summary

V3 addresses V2's limited frequency range by using fixed normalization bounds. This allows the model to generate frequencies across the full audible spectrum (20-20,000 Hz), which is critical for semantic terms like "bright", "airy", and "sparkle" that require high-frequency content.

**Key benefit**: Proper high-shelf EQ for bright sounds (12-20 kHz) instead of limited to <8 kHz.
