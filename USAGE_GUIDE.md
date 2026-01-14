# Usage Guide: Visualizations & Audio Processing

## Quick Start

### 1. Generate Visualizations for Report

```bash
# Activate environment
venv\Scripts\activate

# Generate all visualizations
python visualize_latent_space.py
```

**Output** (in `./figures/` directory):
- `latent_space_2d_map.png` - t-SNE and PCA projections
- `latent_space_annotated.png` - Annotated with warm↔bright axis
- `eq_curves_all_terms.png` - All EQ frequency responses overlaid
- `eq_curves_key_terms.png` - Individual plots for warm/bright/muddy/clear
- `spectral_profiles_heatmap.png` - Energy distribution across frequency bands
- `spectral_comparison_warm_bright.png` - Direct warm vs bright comparison

**Time**: ~2 minutes

---

### 2. Apply EQ to Audio Files

```bash
# Single semantic term
python apply_neural_eq_v2.py --input your_mix.wav --term warm

# Interpolation (blend two terms)
python apply_neural_eq_v2.py --input your_mix.wav --interpolate warm bright 0.5

# With intensity control (make it stronger/weaker)
python apply_neural_eq_v2.py --input your_mix.wav --term warm --intensity 1.5

# Custom output path
python apply_neural_eq_v2.py --input your_mix.wav --term bright --output bright_result.wav
```

**Output**: Processed audio files in `./processed/` directory (or custom path)

**Time**: ~5-30 seconds per file (depending on length)

---

## Visualization Details

### 1. Latent Space 2D Map

**What it shows**: Where each semantic term lives in the learned 32D latent space, projected to 2D

**Interpretations**:
- **Close terms** = similar EQ characteristics (e.g., warm ≈ full)
- **Far terms** = opposite characteristics (e.g., warm ↔ bright)
- **Clusters** = related semantic concepts

**Use in report**:
- "Figure X shows the learned semantic structure..."
- "Terms cluster by tonal characteristics (warm/muddy vs bright/clear)"
- "The warm↔bright axis represents the primary dimension of variation"

---

### 2. EQ Frequency Response Curves

**What it shows**: The actual EQ curve (gain vs frequency) for each semantic term

**Key features**:
- **X-axis**: Frequency (20 Hz - 20 kHz, log scale)
- **Y-axis**: Gain (dB)
- **0 dB line**: No change (flat)
- **Above 0**: Boost
- **Below 0**: Cut

**Reading the curves**:
- **Warm**: Boost ~300 Hz (bass/low-mid), cut ~8 kHz (highs)
- **Bright**: Cut ~100 Hz (bass), boost ~6 kHz (highs)
- **Muddy**: Boost ~300 Hz, boost ~800 Hz (thick low-mids)
- **Clear**: Cut ~300 Hz, boost ~3 kHz (clarity/presence)

**Use in report**:
- "Figure X shows the generated EQ curves for each semantic term"
- "The model learned perceptually meaningful frequency adjustments"
- "Warm vs bright exhibit complementary spectral shaping"

---

### 3. Spectral Energy Profiles

**What it shows**: How much energy is in each frequency band after EQ

**Frequency bands**:
- **Sub (20-60 Hz)**: Subwoofer, felt bass
- **Bass (60-250 Hz)**: Fundamental bass notes
- **Low-Mid (250-500 Hz)**: Body, warmth
- **Mid (500-2000 Hz)**: Vocals, presence
- **Hi-Mid (2-6 kHz)**: Definition, clarity
- **High (6-12 kHz)**: Brightness, air
- **Ultra (12-20 kHz)**: Subtle shimmer

**Color coding**:
- **Green**: Energy added (boost)
- **Red**: Energy reduced (cut)
- **Yellow**: Minimal change

**Use in report**:
- "Heatmap reveals semantic terms modify distinct frequency regions"
- "Warm emphasizes low-mid energy while reducing highs"
- "Model learned spectral balance strategies typical of audio engineering"

---

## Audio Processing Examples

### Example 1: Make a Thin Mix Warmer

```bash
python apply_neural_eq_v2.py --input thin_mix.wav --term warm
```

**Result**: `processed/thin_mix_warm.wav`
- +6 dB @ 300 Hz (adds body)
- -1 dB @ 8 kHz (reduces harshness)

---

### Example 2: Brighten a Muddy Mix

```bash
python apply_neural_eq_v2.py --input muddy_mix.wav --term bright
```

**Result**: `processed/muddy_mix_bright.wav`
- -3 dB @ 100 Hz (cleans up bass)
- +3 dB @ 6 kHz (adds clarity)

---

### Example 3: Create a Balanced Sound

```bash
# 50/50 blend of warm and bright
python apply_neural_eq_v2.py --input mix.wav --interpolate warm bright 0.5
```

**Result**: `processed/mix_warm_bright_0.50.wav`
- Balanced EQ between extremes
- Smooth interpolation in latent space

---

### Example 4: Subtle Warm Enhancement

```bash
# Apply warm at 60% intensity
python apply_neural_eq_v2.py --input mix.wav --term warm --intensity 0.6
```

**Result**: Gentler warmth (3.6 dB @ 300 Hz instead of 6 dB)

---

## Understanding the Model (Important!)

### Current Behavior (V2):

The model applies **FIXED EQ** based on semantic term:
- "warm" → Same EQ for ALL audio files
- Does NOT analyze input audio
- Does NOT adapt based on what the audio needs

**Analogy**: It's like having EQ presets labeled "warm", "bright", etc.

---

### What This Means:

✅ **Good for**:
- Creative intent ("I WANT warm sound")
- Consistent style across tracks
- Learning what semantic terms mean
- Starting point for manual adjustment

⚠️ **Limitations**:
- Might over-process already-warm audio
- Same EQ regardless of input characteristics
- Not "intelligent" mastering (yet!)

---

### Auto-Mastering (Next Feature):

Will analyze input audio FIRST, then suggest appropriate EQ:
```
Input: Already bright mix
Analysis: "This is 75% bright"
Suggestion: "Apply 60% warm to balance"
Result: Adaptive, context-aware processing
```

**This is what we're building next!**

---

## Troubleshooting

### Visualization Script Issues

**Error**: `No module named 'sklearn'`
```bash
pip install scikit-learn
```

**Error**: `No module named 'seaborn'`
```bash
pip install seaborn
```

**Error**: `Model file not found`
- Make sure `neural_eq_safedb_v2.pt` is in the current directory
- Or specify path: Modify `system.load_model("path/to/model.pt")`

---

### Audio Processing Issues

**Error**: `No module named 'dasp_pytorch'`
```bash
pip install git+https://github.com/csteinmetz1/dasp-pytorch.git
```

**Error**: `Unknown term 'xyz'`
- Check available terms: They're printed when the model loads
- Use exact term names (case-sensitive): 'warm' not 'Warm'

**Error**: `Input file not found`
- Use full path or make sure audio file is in current directory
- Supported formats: WAV, MP3, FLAC (anything torchaudio supports)

**Output sounds distorted/clipped**:
- Limiter is applied by default to prevent clipping
- To disable: `--no-limiter`
- Or reduce intensity: `--intensity 0.7`

---

## For Your Report

### Recommended Figures:

1. **Figure 1**: Latent space 2D map (annotated version)
   - Caption: "Learned semantic structure in latent space. Terms cluster by tonal characteristics, with warm-bright representing the primary axis of variation."

2. **Figure 2**: EQ curves for warm/bright/muddy/clear (4-panel)
   - Caption: "Generated EQ frequency response curves for key semantic terms. Model learned perceptually meaningful spectral shaping strategies."

3. **Figure 3**: Spectral energy heatmap (all terms)
   - Caption: "Spectral energy distribution by semantic term. Color indicates relative energy boost (green) or cut (red) in each frequency band."

4. **Figure 4**: Warm vs Bright comparison (bar chart)
   - Caption: "Direct comparison of spectral profiles. Warm emphasizes low-mid frequencies while bright boosts high frequencies, demonstrating complementary characteristics."

### Recommended Audio Examples:

Process 2-3 audio files with different terms:
1. Original mix → Warm version
2. Original mix → Bright version
3. Original mix → Interpolated (50/50) version

**Include in report**:
- "Audio examples demonstrate the model's ability to apply semantically meaningful EQ adjustments to real music"
- Link to online repository or supplementary materials

---

## Next Steps

After generating visualizations and processing audio:

1. ✅ **Document results** in report
2. ✅ **Analyze outputs** (do they sound right?)
3. ⏳ **Build auto-mastering** (analyze input audio)
4. ⏳ **Create web interface** (interactive demo)
5. ⏳ **User evaluation** (A/B listening tests)

---

## Summary of Commands

```bash
# Generate all visualizations
python visualize_latent_space.py

# Process audio with warm EQ
python apply_neural_eq_v2.py --input mix.wav --term warm

# Interpolate warm→bright
python apply_neural_eq_v2.py --input mix.wav --interpolate warm bright 0.5

# List available terms
python apply_neural_eq_v2.py --help
```

**All outputs saved automatically!**
