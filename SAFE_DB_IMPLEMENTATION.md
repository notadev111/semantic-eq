# SAFE-DB Neural EQ Morphing Implementation

## Overview

New implementation of neural EQ morphing using the SAFE-DB dataset instead of SocialFX.

## Key Differences from SocialFX Version

| Aspect | SocialFX | SAFE-DB |
|--------|----------|---------|
| **File** | `core/neural_eq_morphing.py` | `core/neural_eq_morphing_safedb.py` |
| **Dataset** | HuggingFace (1,595 examples) | Local CSV (1,700 examples) |
| **Unique terms** | 765 (sparse) | 368 (concentrated) |
| **Top term examples** | Distributed | warm: 457, bright: 421 |
| **Parameters** | 40 (complex) | **13 (simpler)** |
| **EQ bands** | Variable | **5 bands** (2 shelves + 3 bells) |
| **Input dimension** | 40 | **13** |
| **Expected clustering** | Poor (class imbalance) | **Better** (more examples/term) |

## Dataset Structure

### SAFE-DB 13 Parameters = 5 EQ Bands

**IMPORTANT**: It's NOT 13 bands! It's 5 bands with 13 total parameters:

```
Band 1 (Low Shelf):     Params 0-1:   Gain, Frequency
Band 2 (Bell):          Params 2-4:   Gain, Frequency, Q
Band 3 (Bell):          Params 5-7:   Gain, Frequency, Q
Band 4 (Bell):          Params 8-10:  Gain, Frequency, Q
Band 5 (High Shelf):    Params 11-12: Gain, Frequency
```

### Parameter Ranges

```python
# From analysis of SAFE-DB dataset
Band 1: Gain [-12, 12] dB,  Freq [22, 1000] Hz
Band 2: Gain [-12, 12] dB,  Freq [82, 3900] Hz,    Q [0.1, 10]
Band 3: Gain [-12, 12] dB,  Freq [180, 4700] Hz,   Q [0.1, 10]
Band 4: Gain [-12, 12] dB,  Freq [220, 10000] Hz,  Q [0.1, 10]
Band 5: Gain [-12, 12] dB,  Freq [580, 20000] Hz
```

## Files Created

### 1. Core Implementation
**[core/neural_eq_morphing_safedb.py](core/neural_eq_morphing_safedb.py)**

Complete neural EQ morphing system for SAFE-DB:
- `SAFEDBDatasetLoader`: Loads UserData.csv + AudioFeatureData.csv
- `NeuralEQEncoder`: 13 params → 32-dim latent space
- `NeuralEQDecoder`: 32-dim latent → 13 params
- `ContrastiveLoss`: Semantic clustering loss
- `NeuralEQMorphingSAFEDB`: Main training system

### 2. Training Script
**[train_neural_eq_safedb.py](train_neural_eq_safedb.py)**

Simple CLI for training:
```bash
python train_neural_eq_safedb.py
python train_neural_eq_safedb.py --epochs 100 --contrastive-weight 0.5
```

### 3. Analysis Scripts
**[analyze_safe_dataset.py](analyze_safe_dataset.py)** - Dataset structure analysis
**[decode_safe_eq_params.py](decode_safe_eq_params.py)** - Parameter decoder
**[SAFE_DB_ANALYSIS_SUMMARY.md](SAFE_DB_ANALYSIS_SUMMARY.md)** - Complete analysis

## Architecture

### Encoder
```
Input: [batch, 13] EQ parameters
  ↓
ResidualBlock(13 → 64)
ResidualBlock(64 → 128)
ResidualBlock(128 → 64)
  ↓
Linear(64 → 64) → Linear(64 → 32) → Tanh
  ↓
Latent: [batch, 32]
  ↓
Linear(32 → 128) for contrastive learning
  ↓
Semantic Embedding: [batch, 128]
```

### Decoder
```
Input: [batch, 32] latent
  ↓
ResidualBlock(32 → 64)
ResidualBlock(64 → 128)
ResidualBlock(128 → 64)
  ↓
Specialized heads:
  - Gain head: [64 → 32 → 6] × 12 dB
  - Freq head: [64 → 32 → 5] → map to Hz ranges
  - Q head:    [64 → 32 → 3] × 9.9 + 0.1
  ↓
Reconstruct 13 params: [batch, 13]
```

### Loss Function
```python
L_total = L_reconstruction + λ × L_contrastive

L_reconstruction = MSE(params_original, params_reconstructed)

L_contrastive = -log(exp(sim(z_i, z_pos)/τ) / Σ_j exp(sim(z_i, z_j)/τ))
```

**Default**: λ = 0.5 (higher than SocialFX's 0.1 for better clustering)

## Usage

### Basic Training
```python
from core.neural_eq_morphing_safedb import NeuralEQMorphingSAFEDB

# Initialize
system = NeuralEQMorphingSAFEDB(latent_dim=32)

# Load dataset (filter to terms with ≥10 examples)
system.load_dataset(min_examples=10, include_audio=False)

# Train
system.train(
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    contrastive_weight=0.5
)

# Save
system.save_model("neural_eq_safedb.pt")
```

### Generate EQ from Semantic Term
```python
# Load trained model
system = NeuralEQMorphingSAFEDB()
system.load_model("neural_eq_safedb.pt")

# Generate EQ
params = system.generate_eq_from_term("warm")
# Returns: [13] numpy array with EQ parameters
```

### Semantic Interpolation
```python
# Interpolate between two terms
params = system.interpolate_terms("warm", "bright", alpha=0.5)
# alpha=0.0 → "warm"
# alpha=0.5 → halfway between
# alpha=1.0 → "bright"
```

## Training the Model

### Option 1: CLI
```bash
# Basic training (100 epochs, default settings)
python train_neural_eq_safedb.py

# Custom settings
python train_neural_eq_safedb.py \
    --epochs 150 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --latent-dim 32 \
    --contrastive-weight 0.7 \
    --min-examples 15 \
    --output neural_eq_safedb_custom.pt
```

### Option 2: Python Script
```python
from core.neural_eq_morphing_safedb import NeuralEQMorphingSAFEDB

system = NeuralEQMorphingSAFEDB(latent_dim=32)
system.load_dataset(min_examples=10)
system.train(epochs=100, contrastive_weight=0.5)
system.save_model("neural_eq_safedb.pt")
```

### Expected Training Time
- **CPU**: ~15-20 minutes (100 epochs, 1700 examples)
- **GPU**: ~3-5 minutes

## Expected Results

### Clustering Metrics (SAFE-DB vs SocialFX)

**SocialFX (filtered, 458 examples, 24 terms)**:
- Silhouette: 0.3-0.5 (moderate clustering)
- Davies-Bouldin: <1.5

**SAFE-DB (1700 examples, ~50 terms with ≥10 examples)**:
- Silhouette: **0.5-0.7** (BETTER - more examples per term!)
- Davies-Bouldin: **<1.0**

### Why SAFE-DB Should Cluster Better

1. **Better term distribution**:
   - warm: 457 examples (vs SocialFX's max ~60)
   - bright: 421 examples
   - More examples → better semantic embeddings

2. **Simpler parameter space**:
   - 13 params vs 40 (easier to learn)
   - Less noise in latent space

3. **Concentrated vocabulary**:
   - 368 terms vs 765 (less sparsity)
   - Better contrastive learning

## Comparison with SocialFX

### Keep Both Versions!

Your project now has:
1. **SocialFX version** ([neural_eq_morphing.py](core/neural_eq_morphing.py))
   - 40 parameters, 1595 examples
   - Good for: diversity, comprehensive evaluation

2. **SAFE-DB version** ([neural_eq_morphing_safedb.py](core/neural_eq_morphing_safedb.py))
   - 13 parameters, 1700 examples
   - Good for: clustering, simplicity, common terms

### For Your Report

Compare both approaches:
- **Dataset comparison**: SocialFX vs SAFE-DB characteristics
- **Clustering metrics**: Silhouette scores, latent space visualizations
- **Parameter complexity**: 40 vs 13 dimensions
- **Practical usage**: Which works better for common terms?

This demonstrates:
✓ Critical analysis
✓ Multiple approaches
✓ Evidence-based comparison
✓ Scientific rigor

## Audio Feature Data

The SAFE-DB dataset includes audio features (before/after processing):
- 80 features per sample
- MFCCs, spectral features, etc.
- Can be used for evaluation

To load with audio features:
```python
system.load_dataset(min_examples=10, include_audio=True)
```

**Note**: This is slower but provides more complete data.

## Next Steps

1. ✅ **Train the SAFE-DB model**
   ```bash
   python train_neural_eq_safedb.py --epochs 100
   ```

2. ⏳ **Evaluate clustering**
   - Check silhouette scores
   - Generate t-SNE visualization
   - Compare with SocialFX version

3. ⏳ **Test on audio**
   - Create `semantic_mastering_safedb.py` (to apply EQ to audio)
   - Test with common terms (warm, bright, etc.)
   - Compare A/B with SocialFX version

4. ⏳ **Generate report figures**
   - Latent space visualization
   - Clustering metrics comparison
   - Training curves
   - Parameter distribution analysis

5. ⏳ **Write up findings**
   - Which dataset works better?
   - Why does SAFE-DB cluster better?
   - Trade-offs between approaches

## Troubleshooting

### Import Errors
If you get import errors, make sure you're in the project directory:
```bash
cd "c:\Users\danie\Documents\!ELEC0030 Project\semantic_mastering_system"
python train_neural_eq_safedb.py
```

### Dataset Not Found
Make sure the CSV files are in the correct location:
```
semantic_mastering_system/
  research/
    data/
      SAFEEqualiserUserData.csv
      SAFEEqualiserAudioFeatureData.csv
```

### Poor Clustering
If clustering is poor (silhouette < 0.3):
- Increase `--contrastive-weight` (try 0.7 or 1.0)
- Increase `--min-examples` (try 15 or 20)
- Train for more epochs (150-200)

## Summary

You now have a complete SAFE-DB implementation that:
- ✅ Loads both UserData and AudioFeatureData
- ✅ Uses 13 parameters (5 EQ bands)
- ✅ Trains with ResNet + Contrastive Learning
- ✅ Should achieve better clustering than SocialFX
- ✅ Provides semantic interpolation
- ✅ Includes comprehensive analysis tools

**Recommendation**: Train both versions and compare results in your report!
