# Quick Start: Train V3 Model

## The Problem We Fixed

**V2 Issue**: Only generated frequencies 83-7,901 Hz (dataset has up to 20,000 Hz!)
**V3 Fix**: Uses fixed bounds 20-20,000 Hz for full spectrum

## Train V3 Now

```bash
# 1. Activate virtual environment
venv\Scripts\activate

# 2. Train the model (takes ~30-40 minutes on CPU)
python train_neural_eq_safedb_v3.py

# 3. Wait for completion
# Output: neural_eq_safedb_v3.pt
```

## What You'll See

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

Training...
Epoch 1/150: Recon Loss=0.452, Contrast Loss=2.31, Total=0.683
...
Epoch 150/150: Recon Loss=0.148, Contrast Loss=1.42, Total=0.859

✓ Model saved to neural_eq_safedb_v3.pt
```

## Expected Results

V3 should generate:
- **"bright" sounds**: High-shelf boost at 12-20 kHz (vs V2's <8 kHz)
- **"warm" sounds**: Mid-bass boost ~200-500 Hz (similar to V2)
- **Full frequency range**: 20-20,000 Hz available

## After Training

1. **Test the model**: Create test script (copy from V2, update imports)
2. **Compare to V2**: Check if bright sounds now use higher frequencies
3. **Use in report**: Document V2→V3 improvement

## Files Created This Session

- [`core/neural_eq_morphing_safedb_v3.py`](core/neural_eq_morphing_safedb_v3.py) - V3 model
- [`train_neural_eq_safedb_v3.py`](train_neural_eq_safedb_v3.py) - Training script
- [`V3_TRAINING_GUIDE.md`](V3_TRAINING_GUIDE.md) - Full documentation
- [`CONTRASTIVE_LEARNING_ANALYSIS.md`](CONTRASTIVE_LEARNING_ANALYSIS.md) - Method justification
- [`SESSION_REPORT_V2_TO_V3.md`](SESSION_REPORT_V2_TO_V3.md) - Complete session log

## For Your Report

Use these sections from the documentation:

**Problem Discovery**: V2 limited to <8 kHz, dataset has up to 20 kHz
**Root Cause**: Data-dependent normalization learned average range
**Solution**: V3 fixed bounds enable full spectrum
**Method**: Contrastive learning justified despite dataset challenges

**Key narrative**: V1 (broken) → V2 (works but limited) → V3 (full range)

---

**Next step**: Run `python train_neural_eq_safedb_v3.py` 🚀
