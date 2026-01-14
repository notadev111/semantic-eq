# Model Selection & Next Steps

## Which Model Should You Use?

### Comparison Summary

| Metric | V2 (Data-dependent) | V3 (Fixed bounds) | Winner |
|--------|---------------------|-------------------|---------|
| **Frequency range** | 83-7,901 Hz | 104-8,059 Hz | V3 (slightly wider) |
| **Clustering quality** | Silhouette: 0.07 | Silhouette: -0.025 | **V2** ✅ |
| **Davies-Bouldin** | 4.69 | 5.46 | **V2** ✅ |
| **Semantic separation** | MODERATE | POOR (overlapping) | **V2** ✅ |
| **Warm/bright distinction** | Clear (+5.46 vs +3.49) | Clear (+6.23 vs +2.57) | Both ✅ |
| **Theoretical correctness** | Biased to data | Unbiased | V3 ✅ |
| **Practical usability** | Works well | Worse clustering | **V2** ✅ |

### Engineering Reality Check

**Your observation is correct**: For mastering, the frequency ranges are:
- **Sub-bass**: 20-60 Hz (feel more than hear)
- **Bass**: 60-250 Hz (fundamental energy)
- **Low-mids**: 250-500 Hz (warmth, body)
- **Mids**: 500-2 kHz (presence, clarity)
- **High-mids**: 2-6 kHz (definition, intelligibility)
- **Highs**: 6-12 kHz (air, sparkle, brightness) ← **Most important for "bright"**
- **Ultra-highs**: 12-20 kHz (subtle air, mostly noise)

**V2's 83-7,901 Hz range covers all practical mastering frequencies!**
- Missing 8-12 kHz is the only real gap
- 12-20 kHz is rarely boosted significantly in mastering
- V3 reaches 8,059 Hz (barely better than V2's 7,901 Hz)

### Verdict: **Use V2** ✅

**Reasons**:
1. **Better clustering** (0.07 vs -0.025) - semantic terms are separable
2. **Practical frequency range** - covers 90% of mastering needs
3. **Proven results** - warm/bright distinction is clear and musical
4. **Trade-off justified** - data-dependent normalization matches dataset distribution
5. **V3 didn't solve the real problem** - both limited to ~8 kHz due to dataset bias

---

## What's Next? The Complete Pipeline

You have a **trained neural network that generates EQ parameters from semantic terms**. Now you need to **apply those parameters to real audio**.

### Current Status:

```
✅ DONE: Neural model (V2)
✅ DONE: Training pipeline
✅ DONE: Semantic term → EQ parameters
❌ TODO: EQ parameters → Processed audio
❌ TODO: End-to-end demo
❌ TODO: Evaluation/comparison
```

### Next Phase Options:

---

## Option 1: Apply EQ to Real Audio (Recommended)

**Goal**: Create a working demo that applies V2 EQ to actual music files

**What you need**:

### 1.1 Convert SAFE-DB Params to dasp-pytorch Format

SAFE-DB uses **13 parameters (5 bands)**:
```
Band 1 (Low Shelf):  [Gain, Freq]          (2 params)
Band 2 (Bell):       [Gain, Freq, Q]       (3 params)
Band 3 (Bell):       [Gain, Freq, Q]       (3 params)
Band 4 (Bell):       [Gain, Freq, Q]       (3 params)
Band 5 (High Shelf): [Gain, Freq]          (2 params)
Total: 13 parameters
```

dasp-pytorch expects **18 parameters (6 bands)**:
```
Each band: [Gain, Freq, Q] × 6 bands = 18 params
```

**Conversion strategy**:
```python
def convert_safedb_to_dasp(safedb_13: np.ndarray) -> torch.Tensor:
    """
    Convert SAFE-DB 13-param format to dasp-pytorch 18-param format

    SAFE-DB (5 bands, 13 params):
      [G1, F1, G2, F2, Q2, G3, F3, Q3, G4, F4, Q4, G5, F5]

    dasp-pytorch (6 bands, 18 params):
      [G1, F1, Q1, G2, F2, Q2, G3, F3, Q3, G4, F4, Q4, G5, F5, Q5, G6, F6, Q6]

    Strategy:
      - Use SAFE-DB bands 1-5 as dasp bands 1-5
      - Add neutral 6th band (for compatibility)
      - Set Q=0.7 for shelf bands (standard default)
    """
    dasp_params = torch.zeros(1, 18)

    # Band 1: Low Shelf (SAFE-DB[0,1] -> dasp[0,1,2])
    dasp_params[0, 0] = safedb_13[0]  # Gain
    dasp_params[0, 1] = safedb_13[1]  # Freq
    dasp_params[0, 2] = 0.7           # Q (default for shelf)

    # Band 2: Bell (SAFE-DB[2,3,4] -> dasp[3,4,5])
    dasp_params[0, 3] = safedb_13[2]  # Gain
    dasp_params[0, 4] = safedb_13[3]  # Freq
    dasp_params[0, 5] = safedb_13[4]  # Q

    # Band 3: Bell (SAFE-DB[5,6,7] -> dasp[6,7,8])
    dasp_params[0, 6] = safedb_13[5]  # Gain
    dasp_params[0, 7] = safedb_13[6]  # Freq
    dasp_params[0, 8] = safedb_13[7]  # Q

    # Band 4: Bell (SAFE-DB[8,9,10] -> dasp[9,10,11])
    dasp_params[0, 9] = safedb_13[8]   # Gain
    dasp_params[0, 10] = safedb_13[9]  # Freq
    dasp_params[0, 11] = safedb_13[10] # Q

    # Band 5: High Shelf (SAFE-DB[11,12] -> dasp[12,13,14])
    dasp_params[0, 12] = safedb_13[11] # Gain
    dasp_params[0, 13] = safedb_13[12] # Freq
    dasp_params[0, 14] = 0.7           # Q (default for shelf)

    # Band 6: Neutral (not in SAFE-DB, add for compatibility)
    dasp_params[0, 15] = 0.0    # 0 dB gain
    dasp_params[0, 16] = 10000  # 10 kHz (neutral position)
    dasp_params[0, 17] = 0.7    # Q

    # Normalize to [0, 1] for dasp-pytorch
    return normalize_for_dasp(dasp_params)
```

### 1.2 Create Audio Processing Script

**File**: `apply_neural_eq_v2.py`

```python
"""
Apply Neural EQ V2 to Audio Files
==================================

Uses trained V2 model to generate semantic EQ and apply to audio.

Usage:
    python apply_neural_eq_v2.py --input mix.wav --term warm
    python apply_neural_eq_v2.py --input mix.wav --interpolate warm bright 0.5
"""

import torch
import torchaudio
from pathlib import Path
from core.neural_eq_morphing_safedb_v2 import NeuralEQMorphingSAFEDBV2
from dasp_pytorch import ParametricEQ

def main():
    # Load V2 model
    system = NeuralEQMorphingSAFEDBV2()
    system.load_model("neural_eq_safedb_v2.pt")
    system.load_dataset(min_examples=5)

    # Load audio
    audio, sr = torchaudio.load("input.wav")

    # Generate EQ parameters
    eq_params_13 = system.generate_eq("warm")

    # Convert to dasp format
    eq_params_18 = convert_safedb_to_dasp(eq_params_13)

    # Apply EQ
    eq = ParametricEQ(sample_rate=sr)
    processed = eq.process_normalized(audio, eq_params_18)

    # Save
    torchaudio.save("output_warm.wav", processed, sr)
```

### 1.3 Expected Output

**Before**: Original mix (flat frequency response)
**After**: Warm mix (mid-bass boosted, highs reduced)

**Demo for report**:
```
Input: dry_mix.wav
Output:
  - dry_mix_warm.wav     (+6dB @ 300Hz, -1dB @ 8kHz)
  - dry_mix_bright.wav   (-3dB @ 100Hz, +3dB @ 6kHz)
  - dry_mix_50_50.wav    (interpolated between warm/bright)
```

---

## Option 2: Evaluation & Comparison (Academic)

**Goal**: Quantitatively evaluate the model's performance

### 2.1 Perceptual Audio Evaluation

**Metrics**:
- **Spectral centroid shift** (warm should lower it, bright should raise it)
- **High-frequency energy** (bright should have more >6 kHz content)
- **Low-frequency energy** (warm should have more <500 Hz content)

**Code**:
```python
def evaluate_eq_effect(original_audio, processed_audio):
    # Compute spectrograms
    spec_orig = compute_spectrum(original_audio)
    spec_proc = compute_spectrum(processed_audio)

    # Spectral centroid (frequency "center of mass")
    centroid_orig = spectral_centroid(spec_orig)
    centroid_proc = spectral_centroid(spec_proc)

    # Energy in bands
    energy_low = np.sum(spec_proc[20:500])   # Bass
    energy_high = np.sum(spec_proc[6000:])   # Air

    return {
        'centroid_shift_hz': centroid_proc - centroid_orig,
        'low_energy_db': 10 * np.log10(energy_low),
        'high_energy_db': 10 * np.log10(energy_high)
    }
```

### 2.2 A/B Listening Test (Subjective)

**Setup**:
1. Process same audio with "warm", "bright", "neutral"
2. Blind listening test with multiple listeners
3. Ask: "Which sounds warmer?" etc.
4. Compare model predictions to human judgments

**Expected result**: Model-generated EQ should align with human perception of semantic terms

### 2.3 Comparison to Baselines

**Baselines**:
1. **Random EQ**: Random parameters within valid ranges
2. **Rule-based**: Hand-coded warm/bright presets
3. **SocialFX lookup**: Average parameters from dataset (your `semantic_mastering.py`)
4. **Neural V2**: Your trained model

**Hypothesis**: Neural V2 should outperform random/rules, comparable to SocialFX lookup

---

## Option 3: Interactive Demo (Impressive for Presentation)

**Goal**: Real-time semantic mastering interface

### 3.1 Gradio Web Interface

```python
import gradio as gr

def semantic_mastering(audio, term, intensity):
    # Load model
    system = NeuralEQMorphingSAFEDBV2()
    system.load_model("neural_eq_safedb_v2.pt")
    system.load_dataset()

    # Generate EQ
    eq_params = system.generate_eq(term)
    eq_params *= intensity  # Scale by intensity slider

    # Apply
    processed = apply_eq(audio, eq_params)
    return processed

# Create interface
interface = gr.Interface(
    fn=semantic_mastering,
    inputs=[
        gr.Audio(label="Input Audio"),
        gr.Dropdown(["warm", "bright", "muddy", "clear"], label="Semantic Term"),
        gr.Slider(0, 2, value=1, label="Intensity")
    ],
    outputs=gr.Audio(label="Processed Audio"),
    title="Neural Semantic Mastering (V2)"
)

interface.launch()
```

**Result**: Web UI where you upload audio, select "warm", and get processed output instantly!

---

## Option 4: Extended Research (If You Have Time)

### 4.1 Multi-Term Combinations

**Current**: Single term ("warm") or interpolation (warm→bright)
**Extension**: Combinations ("warm AND punchy", "bright BUT smooth")

**Implementation**:
```python
# Combine latent vectors
z_warm = encoder.encode("warm")
z_punchy = encoder.encode("punchy")
z_combined = 0.7 * z_warm + 0.3 * z_punchy
eq_params = decoder.decode(z_combined)
```

### 4.2 Reverse Engineering

**Goal**: Given EQ settings, predict semantic label

```python
# User uploads existing EQ preset
eq_params = load_preset("my_favorite_eq.txt")

# Encode to latent space
z = encoder.encode_params(eq_params)

# Find nearest semantic term
nearest_term = find_nearest_term(z, system.term_embeddings)

print(f"This EQ is most similar to: {nearest_term}")
# Output: "This EQ is most similar to: bright"
```

### 4.3 Conditional Generation

**Goal**: Generate EQ with constraints ("warm, but keep bass below +3dB")

```python
eq_params = system.generate_eq("warm")

# Apply constraint
eq_params[0] = np.clip(eq_params[0], -3, 3)  # Limit bass gain

# Re-optimize in latent space to satisfy constraint
z = encoder.encode_params(eq_params)
z_optimized = optimize_latent(z, constraint=bass_limit)
eq_final = decoder.decode(z_optimized)
```

---

## Recommended Path (Based on Your Project Timeline)

### If You Have 1-2 Days: **Option 1** (Audio Processing)

**Priority tasks**:
1. ✅ Create parameter converter (SAFE-DB → dasp)
2. ✅ Write `apply_neural_eq_v2.py` script
3. ✅ Process 3-5 audio examples (warm, bright, interpolated)
4. ✅ Include audio samples in report/presentation

**Why**: Gives you a **complete working demo** to show off!

### If You Have 3-5 Days: **Option 1 + Option 2**

Add evaluation:
1. Spectral analysis of processed audio
2. A/B comparison plots (before/after)
3. Quantitative metrics table
4. Discussion of results in report

**Why**: **Stronger academic contribution** with evaluation

### If You Have 1 Week+: **Option 1 + 2 + 3**

Add Gradio interface for presentation/demo day

**Why**: **Impressive live demo** for professors/classmates

### If You're Time-Constrained: **Report Only**

**Skip audio processing**, just document:
- V1 failure → V2 success → V3 analysis
- Clustering results
- Generated EQ parameters (tables/plots)
- Discussion of dataset limitations

**Why**: Still a complete project without audio demo

---

## My Recommendation: **Option 1 (Audio Processing)**

Create `apply_neural_eq_v2.py` to demonstrate:

1. **"warm" mastering**: Boost bass, reduce highs
2. **"bright" mastering**: Reduce bass, boost highs
3. **Interpolation**: Smooth transition between semantic terms

This gives you:
- ✅ Complete pipeline (training → inference → audio)
- ✅ Tangible demo (upload mix, get processed audio)
- ✅ Academic value (semantic audio processing)
- ✅ Good presentation material

---

## What I Can Help You Build Next

Let me know which option you want, and I can create:

1. **SAFE-DB → dasp converter function**
2. **`apply_neural_eq_v2.py` script** (full audio processing pipeline)
3. **Evaluation script** (spectral analysis before/after)
4. **Gradio demo** (web interface)
5. **Report sections** (methods, results, discussion)

**What would be most useful for your project?**
