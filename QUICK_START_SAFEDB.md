# Quick Start: SAFE-DB Neural EQ Morphing

## TL;DR - Run This Now

```bash
cd "c:\Users\danie\Documents\!ELEC0030 Project\semantic_mastering_system"
python train_neural_eq_safedb.py
```

Training takes ~15-20 minutes on CPU. You'll get a trained model: `neural_eq_safedb.pt`

## What You Just Built

A neural network that:
- Learns from **1,700 real EQ settings** (SAFE-DB dataset)
- Maps semantic terms like "warm", "bright" to **13 EQ parameters (5 bands)**
- Uses **ResNet + Contrastive Learning** for better clustering
- Enables **semantic interpolation** between terms

## Key Points

### 13 Parameters ≠ 13 Bands!
The 13 parameters represent **5 EQ bands**:
- **Band 1**: Low shelf (gain, freq)
- **Bands 2-4**: Parametric bells (gain, freq, Q)
- **Band 5**: High shelf (gain, freq)

### Why SAFE-DB > SocialFX?

| Metric | SocialFX | SAFE-DB |
|--------|----------|---------|
| "warm" examples | ~60 | **457** |
| "bright" examples | ~80 | **421** |
| Parameters | 40 (complex) | **13 (simpler)** |
| Expected clustering | Moderate | **Better** |

More examples per term = better semantic clustering!

## Files Created

1. **[core/neural_eq_morphing_safedb.py](core/neural_eq_morphing_safedb.py)** - Main implementation
2. **[train_neural_eq_safedb.py](train_neural_eq_safedb.py)** - Training script
3. **[SAFE_DB_IMPLEMENTATION.md](SAFE_DB_IMPLEMENTATION.md)** - Full documentation
4. **[SAFE_DB_ANALYSIS_SUMMARY.md](SAFE_DB_ANALYSIS_SUMMARY.md)** - Dataset analysis

## Training Options

### Default (Recommended)
```bash
python train_neural_eq_safedb.py
```

### Custom Settings
```bash
python train_neural_eq_safedb.py \
    --epochs 150 \
    --contrastive-weight 0.7 \
    --min-examples 15
```

### Key Parameters
- `--epochs`: More = better convergence (try 100-200)
- `--contrastive-weight`: Higher = better clustering (try 0.5-1.0)
- `--min-examples`: Filter rare terms (10-20 recommended)

## After Training

### Check Clustering Quality
Look for these metrics in the output:
- **Silhouette Score > 0.5**: Good clustering!
- **Davies-Bouldin < 1.0**: Well-separated clusters

### Compare with SocialFX
You now have TWO neural models:
1. `neural_eq_model.pt` (SocialFX, 40 params)
2. `neural_eq_safedb.pt` (SAFE-DB, 13 params)

Compare them in your report!

## What's Next?

1. **Train the model** (you can do this now)
   ```bash
   python train_neural_eq_safedb.py
   ```

2. **Create semantic_mastering_safedb.py** (to apply EQ to audio)
   - Similar to existing `semantic_mastering.py`
   - But uses SAFE-DB dataset and 13-param structure

3. **Compare both approaches** in your report
   - Which clusters better?
   - Which is more practical?
   - Trade-offs?

## Expected Results

After training, you should see:
```
Clustering Metrics:
  Silhouette Score: 0.6-0.7 (GOOD!)
  Davies-Bouldin Index: 0.8-1.2 (GOOD!)
```

If clustering is poor (silhouette < 0.3):
- Increase `--contrastive-weight` to 0.7 or 1.0
- Increase `--min-examples` to 15 or 20
- Train longer (150-200 epochs)

## Need Help?

See the full documentation:
- [SAFE_DB_IMPLEMENTATION.md](SAFE_DB_IMPLEMENTATION.md) - Complete guide
- [SAFE_DB_ANALYSIS_SUMMARY.md](SAFE_DB_ANALYSIS_SUMMARY.md) - Dataset details

Or just ask me!
