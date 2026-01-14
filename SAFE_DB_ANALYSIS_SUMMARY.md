# SAFE-DB Dataset Analysis Summary

## Overview

You have discovered the **SAFE-DB** (Semantic Audio Feature Extraction Database) which is different from the SocialFX dataset you've been using.

## Dataset Comparison

### SAFE-DB
- **Source**: research/data/SAFEEqualiserUserData.csv + SAFEEqualiserAudioFeatureData.csv
- **Size**: 1,700 EQ settings
- **Unique semantic terms**: 368
- **Top terms**:
  - warm: 457 examples
  - bright: 421 examples
  - test: 181 examples
- **Parameters**: 13 EQ parameters per setting
- **Audio features**: Paired before/after processing (3,398 rows total)

### SocialFX (current system)
- **Source**: HuggingFace `seungheondoh/socialfx-original`
- **Size**: 1,595 EQ settings
- **Unique semantic terms**: 765
- **Top terms**: More diverse, fewer examples each
- **Parameters**: 40-parameter graphic EQ format

## Key Differences

| Aspect | SAFE-DB | SocialFX |
|--------|---------|----------|
| Examples/term | **HIGHER** (warm: 457) | LOWER (spread across 765 terms) |
| Data quality | **Better for ML** (more concentrated) | Sparser distribution |
| Audio features | **YES** (before/after pairs) | NO |
| Parameters | 13 params (simpler) | 40 params (complex) |
| Best for | **Clustering & ML** | Diversity |

## SAFE-DB EQ Parameter Structure

Based on parameter range analysis, the 13 parameters appear to follow this pattern:

```
Param  0 (col 5):  GAIN (dB)       [-12, 12]
Param  1 (col 6):  FREQUENCY (Hz)  [22, 1000]
Param  2 (col 7):  GAIN (dB)       [-12, 12]
Param  3 (col 8):  FREQUENCY (Hz)  [82, 3900]
Param  4 (col 9):  Q-FACTOR        [0.1, 10]
Param  5 (col 10): GAIN (dB)       [-12, 12]
Param  6 (col 11): FREQUENCY (Hz)  [180, 4700]
Param  7 (col 12): Q-FACTOR        [0.1, 10]
Param  8 (col 13): GAIN (dB)       [-12, 12]
Param  9 (col 14): FREQUENCY (Hz)  [220, 10000]
Param 10 (col 15): Q-FACTOR        [0.1, 10]
Param 11 (col 16): GAIN (dB)       [-12, 12]
Param 12 (col 17): FREQUENCY (Hz)  [580, 20000]
```

### Likely Structure: 4-5 Parametric Bands

Looking at the pattern, this appears to be **4 parametric EQ bands** plus possibly fixed shelves:

- **Band 1** (Low shelf?): Gain, Frequency
- **Band 2**: Gain, Frequency, Q
- **Band 3**: Gain, Frequency, Q
- **Band 4**: Gain, Frequency, Q
- **Band 5** (High shelf?): Gain, Frequency

**Note**: Columns 3-4 (values 1-2) might be filter type selectors (bell/shelf/etc)

## UserData vs AudioFeatureData

### SAFEEqualiserUserData.csv
- **Row count**: 1,700
- **Column 0**: ID
- **Column 1**: Semantic term (e.g., "warm", "bright")
- **Column 2**: IP address (user tracking)
- **Columns 3-4**: Possibly filter types
- **Columns 5-17**: EQ parameters (13 values)
- **Column 24**: Hash/UUID

### SAFEEqualiserAudioFeatureData.csv
- **Row count**: 3,398 (2x UserData)
- **Column 0**: ID (matches UserData)
- **Column 1**: "processed" or "unprocessed"
- **Columns 2-81**: 80 audio features (MFCCs, spectral features, etc.)

**Relationship**: Each user EQ setting (UserData row) has 2 corresponding AudioFeatureData rows - one for the audio before processing, one after.

## Advantages of SAFE-DB for Your Project

### 1. Better ML Training
- **More examples per term**: warm (457) vs SocialFX's sparse distribution
- **Better clustering**: Contrastive learning needs multiple examples per class
- **Reduced class imbalance**: 368 terms vs 765 means less sparsity

### 2. Evaluation Capabilities
- **Audio features included**: Can evaluate how well your system works
- **Before/after pairs**: Can measure perceptual impact
- **Objective metrics possible**: Spectral analysis, MFCCs, etc.

### 3. Simpler Parameter Space
- **13 params vs 40**: Easier to learn, faster training
- **Standard parametric EQ**: More familiar format
- **Better for 13-band system**: Already close to your target!

## Recommendations for Your New Implementation

### Option 1: Use SAFE-DB Directly (RECOMMENDED)
Build new versions using SAFE-DB:

1. **semantic_mastering_safe.py**
   - Load SAFEEqualiserUserData.csv
   - Parse 13-parameter EQ format
   - Average parameters by semantic term
   - Apply EQ with dasp-pytorch

2. **neural_eq_morphing_safe.py**
   - Train on SAFE-DB (1,700 examples)
   - Input: 13 parameters (simpler than 40)
   - Better clustering expected (warm: 457 examples!)
   - Same architecture: ResNet + Contrastive Learning

### Option 2: Expand to 13-Band Custom Format
If you want exactly 13 bands:

- Convert SAFE's ~4-5 bands to your 13-band structure
- Design your own parameter mapping
- More work, but more control

### Option 3: Hybrid Approach
- Use SAFE-DB's rich semantic labels
- Convert to your desired 13-band format
- Best of both worlds

## ML Model Considerations

### Can you use the same neural architecture?

**YES, with minor changes:**

Current model (SocialFX):
- Input: 40 parameters
- Latent: 32 dimensions
- Output: 40 parameters

New model (SAFE-DB):
- Input: 13 parameters (CHANGE)
- Latent: 16-32 dimensions (can keep same or reduce)
- Output: 13 parameters (CHANGE)

**Changes needed:**
1. Update input layer: 40 → 13 dimensions
2. Update output layer: 40 → 13 dimensions
3. Update output heads for (gain, freq, Q) structure
4. *Keep same architecture*: ResNet + Contrastive Learning

**Expected improvement:**
- Faster training (fewer parameters)
- Better clustering (more examples per term)
- Silhouette score should improve significantly!

## Next Steps

1. ✅ **Analyzed SAFE-DB structure**
2. ⏳ **Decide on approach** (Option 1, 2, or 3)
3. ⏳ **Create new semantic_mastering_safe.py**
4. ⏳ **Create new neural_eq_morphing_safe.py**
5. ⏳ **Train and evaluate**
6. ⏳ **Compare with SocialFX version**

## Questions to Answer

Before proceeding:

1. **Do you want to use SAFE-DB as-is (13 params) or design your own 13-band structure?**
2. **Should we keep both implementations (SocialFX + SAFE) for comparison?**
3. **Do you want to use the audio features for evaluation?**

## Code Files

Analysis scripts created:
- `analyze_safe_dataset.py` - Dataset structure analysis
- `decode_safe_eq_params.py` - EQ parameter decoder

Next to create:
- `core/semantic_mastering_safe.py` - SAFE-DB semantic mastering
- `core/neural_eq_morphing_safe.py` - SAFE-DB neural approach
- `train_neural_eq_safe.py` - Training script

## References

- [SAFE Project Paper](http://www.eecs.qmul.ac.uk/~gyorgyf/files/papers/stables2014safe.pdf)
- [SAFE-DB Dataset](http://www.semanticaudio.co.uk/datasets/data/)
- [Semantic Audio Labs](https://semanticaudio.co.uk/)

---

**Bottom Line**: SAFE-DB is likely BETTER than SocialFX for your neural EQ project because:
- ✅ More examples per semantic term (better clustering)
- ✅ Simpler parameter space (13 vs 40)
- ✅ Audio features included (better evaluation)
- ✅ Already close to your 13-band goal

**Recommendation**: Build the SAFE-DB version and compare results with your current SocialFX implementation!
