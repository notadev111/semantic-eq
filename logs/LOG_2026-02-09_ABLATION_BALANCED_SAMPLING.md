# Ablation Study: Balanced Class Sampling
## Date: 2026-02-09

---

## Problem Identified: Class Imbalance in SAFE-DB Training Data

### Training Data Distribution
The SAFE-DB dataset has severe class imbalance:

| Term | Examples | % of Total |
|------|----------|------------|
| warm | 532 | 38.9% |
| bright | 504 | 36.8% |
| test | 181 | 13.2% |
| clear | 8 | 0.6% |
| thin | 7 | 0.5% |
| boomy | 7 | 0.5% |
| re27 | 7 | 0.5% |
| vocals | 6 | 0.4% |
| airy | 6 | 0.4% |
| tinny | 5 | 0.4% |
| muddy | 5 | 0.4% |
| deep | 5 | 0.4% |
| brighter | 5 | 0.4% |
| full | 5 | 0.4% |
| **Total** | **1369** | |

**Key issue**: bright + warm = 76% of all training examples.

### Impact on Model Predictions (BEFORE fix)

Ran the model (`audio_encoder_e2e.pt`, trained with uniform sampling) on 12 FMA listening test clips with diverse spectral characteristics.

**Spectral diversity of test clips:**
- Spectral centroid range: 966 - 3082 Hz
- Bass/treble ratio range: 4.73 - 39.55
- All clips are bass-heavy (B/T > 3.0), but there IS genuine tonal variation

**Model predictions showed strong "bright" bias:**

| Clip | Genre | Spectral Centroid | B/T Ratio | Model Top 3 |
|------|-------|-------------------|-----------|-------------|
| 001544 | Folk (quiet) | 1610 Hz | 4.73 | bright(0.91), airy(0.85), tinny(0.78) |
| 006367 | International | 1754 Hz | 27.93 | bright(0.90), airy(0.83), thin(0.78) |
| 014542 | Hip-Hop | 1900 Hz | 15.47 | bright(0.90), airy(0.82), thin(0.79) |
| 014570 | Instrumental | 966 Hz | 39.55 | bright(0.89), airy(0.81), tinny(0.78) |
| 028802 | Pop | 3082 Hz | 8.84 | bright(0.89), airy(0.80), clear(0.77) |
| 030486 | Pop | 1809 Hz | 23.54 | bright(0.87), airy(0.82), clear(0.76) |
| **040242** | **Folk** | **1495 Hz** | **29.13** | **bright(0.71), warm(0.66), full(0.65)** |
| 048293 | Electronic | 2917 Hz | 9.37 | bright(0.89), airy(0.82), thin(0.78) |
| 071508 | Electronic | 2453 Hz | 15.14 | bright(0.90), airy(0.81), tinny(0.79) |
| 089846 | Hip-Hop | 3008 Hz | 10.69 | bright(0.91), airy(0.82), tinny(0.80) |
| 115761 | Rock | 3045 Hz | 6.65 | bright(0.89), airy(0.82), thin(0.80) |
| 115765 | Rock | 3042 Hz | 6.87 | bright(0.89), airy(0.81), tinny(0.74) |

**Key observations:**
- "bright" is predicted as #1 for ALL 12 clips
- Only Clip 040242 (Folk Instrumental, B/T=29.1) shows differentiation: warm and full emerge
- Track 014570 has the lowest centroid (966 Hz, B/T=39.5) but model still predicts bright(0.89) - clearly wrong
- Track 028802 has the highest centroid (3082 Hz) and gets bright(0.89) - same score as the darkest clip

**Diagnosis**: The audio encoder's semantic consistency loss is dominated by bright/warm centroids during training because uniform sampling draws ~76% bright/warm examples per batch. Minority terms (boomy, deep, muddy, full, airy, thin, tinny, clear) rarely appear in training batches, so the encoder never learns to distinguish them.

---

## Fix Applied: Balanced Class Sampling

### Code Change
File: `train_audio_encoder_e2e.py`, method `create_training_batch()`

**Before (biased uniform sampling):**
```python
setting = np.random.choice(self.v2_system.eq_settings)
```
In a batch of 32: ~12 bright, ~12 warm, ~4 test, ~4 everything else.

**After (balanced sampling):**
```python
term = np.random.choice(self._term_list)  # Equal probability per term
setting = np.random.choice(self._term_settings_index[term])
```
In a batch of 32: ~2-3 per term across all 14 terms.

### Expected Impact
- Each term gets equal representation in every batch
- Minority terms (boomy, deep, muddy etc.) will now appear as frequently as bright/warm
- The model should learn to differentiate across ALL semantic terms
- Trade-off: fewer examples per minority term means noisier gradients for those terms, but the balanced exposure should outweigh this

### Concern: Low-count terms
Terms with only 5 examples (tinny, muddy, deep, brighter, full) will be heavily oversampled. The same 5 EQ settings will repeat many times during training. This could lead to:
- Overfitting to those specific EQ curves
- Less robust centroids
This is a known limitation. Future work: data augmentation or collecting more labeled EQ settings.

---

## Fix 2: FMA Audio in Training (Domain Generalization)

### Problem: Pink Noise Only Training
**Discovery**: All previous training runs (glad-cherry-4, twilight-snowflake-5) used **pink noise only** as source audio. The `--fma-path` argument existed in argparse but was never wired to the trainer constructor. The `FMALoader` in `core/multi_source_dataset.py` was fully implemented but unused.

### Impact
- Model learned EQ signatures on a perfectly neutral (flat spectrum) signal
- At inference time on real music: spectral content is dominated by musical content (vocals, drums, bass), not EQ
- The model can't separate "this sounds bright because of EQ" from "this has hi-hats/cymbals"
- Domain gap between training (pink noise) and evaluation (real music) degrades generalization

### Code Change
File: `train_audio_encoder_e2e.py`

**Constructor**: Added `fma_path` and `fma_ratio` parameters, initializes `FMALoader` from `core/multi_source_dataset.py`

**`create_training_batch()`**: Now randomly selects source audio per sample:
```python
use_fma = (self.fma_loader is not None and np.random.random() < self.fma_ratio)
if use_fma:
    source_audio = self.fma_loader.load_random_clip()  # random 2s crop from FMA
else:
    source_audio = self.pink_generator.generate(...)    # fresh pink noise
```

**`main()`**: Now passes `fma_path` and `fma_ratio` from args to trainer.

### Ratio: 50% FMA / 50% Pink Noise
- Pink noise (50%): clean EQ learning signal, no musical content confounds
- FMA real music (50%): domain adaptation, learn to recognize EQ on real music
- FMA uses 8000 tracks from fma_small, random 2s crops with resampling

**Scientific justification**: Mixed synthetic/real training is a standard domain adaptation technique (Tobin et al. 2017 "Domain Randomization"). Pink noise provides a clean, unconfounded EQ learning signal (flat spectrum → any spectral change = pure EQ effect), while FMA provides real musical content for domain generalization. Equal weighting (50/50) is the unbiased default when no prior evidence favours either source. The ratio itself could be an additional ablation variable (30/70, 70/30) in future work.

---

## Experiment Plan

### Ablation Conditions
| Run | Sampling | Audio Source | Notes |
|-----|----------|-------------|-------|
| Run 1 (glad-cherry-4) | Uniform | 100% pink noise | Baseline (biased) |
| Run 2 (twilight-snowflake-5) | Uniform | 100% pink noise | Same as Run 1 |
| **Run 3** | **Balanced** | **50% FMA + 50% pink noise** | **Both fixes** |

### Run 3: Balanced Sampling + FMA Audio
- **Date**: 2026-02-09 (push fix, start tonight/tomorrow)
- **Config**: balanced sampling + 50% FMA audio
- **Command**: `python train_audio_encoder_e2e.py --epochs 100 --device cuda --fma-path ./data/fma/fma_small --fma-ratio 0.5`
- **Epochs**: 100
- **Device**: UCL GPU cluster (athens.ee.ucl.ac.uk)
- **W&B project**: semantic-eq-e2e

### Evaluation Plan
After retraining, compare:
1. Training loss curves (Run 3 vs Run 1)
2. Re-run on same 12 listening test clips
3. Compare model predictions before/after
4. Report as ablation study in paper

### Success Criteria
- Model should predict different top descriptors for spectrally different clips
- "warm"/"full"/"deep" should appear for bass-heavy clips (014570, 040242, 001544)
- "bright"/"airy"/"thin" should appear for treble-heavy clips (028802, 071508, 115761)
- Overall: more variance in predictions across clips
- FMA training should improve generalization (lower semantic bias on real music)

---

## Files Modified
- `train_audio_encoder_e2e.py`:
  - Added `fma_path` and `fma_ratio` to constructor
  - Initialized `FMALoader` from `core/multi_source_dataset.py`
  - Mixed FMA/pink noise source audio in `create_training_batch()`
  - Wired args through `main()` → trainer
  - Added balanced class sampling (previous fix)
  - Added training source metadata to W&B config

## Related Files
- `core/multi_source_dataset.py`: `FMALoader` class (already existed, now used)
- `results/listening_test/model_predictions.json`: Pre-fix model predictions
- `analyze_clip_spectra.py`: Spectral analysis of test clips
- `run_listening_test_analysis.py`: Batch analysis script
