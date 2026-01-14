# How to Use Your Trained Neural EQ Model

## What You Have

The trained model file [neural_eq_model.pt](neural_eq_model.pt) contains:
- **514,071 trained parameters** (encoder + decoder)
- **765 semantic terms** learned from real audio engineers
- **1,595 training examples** from SocialFX dataset
- Normalization parameters and training data references

## What Can You Do With It?

### 1. Generate EQ from Semantic Terms

Load the model and generate EQ settings from descriptive words:

```python
import torch
import sys
sys.path.append('core')
from neural_eq_morphing import NeuralEQMorphingSystem, SocialFXDatasetLoader

# Initialize system
system = NeuralEQMorphingSystem(latent_dim=32, device='cpu')

# Load SocialFX dataset
loader = SocialFXDatasetLoader()
eq_settings = loader.load_socialfx_dataset()
system.load_dataset(eq_settings)

# Load trained weights
checkpoint = torch.load('neural_eq_model.pt', map_location='cpu', weights_only=False)
system.encoder.load_state_dict(checkpoint['encoder'])
system.decoder.load_state_dict(checkpoint['decoder'])
system.encoder.eval()
system.decoder.eval()

# Generate EQ for "warm"
eq_curves = system.generate_eq_from_semantic('warm', variations=5)
```

### 2. Morph Between Semantic Terms

Create smooth transitions between different sonic characteristics:

```python
# Morph from "warm" to "bright" in 10 steps
morphed_curves = system.morph_between_terms('warm', 'bright', steps=10)

# Step 0: 100% warm
# Step 5: 50% warm, 50% bright
# Step 10: 100% bright
```

### 3. Apply to Real Audio

Use the EQ settings on actual audio files:

```python
import soundfile as sf
import numpy as np

# Load audio
audio, sr = sf.read('input_track.wav')

# Generate EQ for "punchy"
eq_params = system.generate_eq_from_semantic('punchy', variations=1)[0]

# Apply EQ (you'll need to implement EQ filter based on parameters)
# The eq_params contain 40 values representing 10 bands of EQ
# Each band has: [gain, frequency, Q, type]
```

### 4. Visualize Latent Space

See how semantic terms cluster in the learned 32D embedding space:

```python
# Extract latent representations
latent_vectors = []
term_labels = []

for term in ['warm', 'bright', 'punchy', 'smooth', 'aggressive']:
    examples = [ex for ex in system.dataset if term in ex['semantic_terms']]
    for ex in examples[:10]:
        params = torch.FloatTensor(ex['normalized_params']).unsqueeze(0)
        latent, _ = system.encoder(params)
        latent_vectors.append(latent.detach().numpy()[0])
        term_labels.append(term)

# Use PCA or t-SNE to visualize in 2D
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
latent_2d = pca.fit_transform(latent_vectors)

# Plot
import matplotlib.pyplot as plt
plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=term_labels)
plt.title('Semantic Term Latent Space')
plt.show()
```

## What To Do With It For Your Report

### Methodology Section
- Explain Neural Residual Networks + Contrastive Learning
- Compare to VAE approaches (FlowEQ) - your approach avoids posterior collapse
- Architecture: 40D input → 32D latent → 40D output
- Training: 50 epochs, 1595 examples, 765 semantic terms

### Results Section

#### 1. Semantic Generation Examples
Show EQ curves generated for different terms:
- "warm" → Bass boost, treble cut
- "bright" → Treble boost
- "punchy" → Mid-range emphasis

#### 2. Latent Space Analysis
- Visualize how semantic terms cluster
- Show that similar terms (warm/smooth) are nearby
- Show that opposite terms (warm/bright) are far apart

#### 3. Morphing Demonstrations
- Show smooth transitions between semantic terms
- Demonstrate real-time interpolation capability

#### 4. Comparison with Base Approaches
Compare 3 approaches:
1. **Base Semantic Mastering** - Simple averaging
2. **Adaptive Semantic Mastering** - Audio-aware selection
3. **Neural Semantic Mastering** - Your trained model

Show pros/cons:
- Base: Fast, interpretable, but rigid
- Adaptive: Context-aware, but still averaging
- Neural: Flexible, learned representations, but needs training

### Evaluation Section

#### Quantitative Metrics
- Reconstruction loss: How well can the model recreate training examples?
- Latent space clustering: Silhouette score for semantic term groupings
- Consistency: Do generated EQ curves for same term look similar?

#### Qualitative Evaluation (if you have time)
- Listening tests with real people
- A/B comparisons between approaches
- Survey results on perceived quality

## Practical Examples

### Example 1: Quick Test
```python
# Test the model works
checkpoint = torch.load('neural_eq_model.pt', weights_only=False)
print(f"Trained on {len(checkpoint['semantic_to_idx'])} semantic terms")
print(f"Example terms: {list(checkpoint['semantic_to_idx'].keys())[:10]}")
```

### Example 2: Generate & Visualize
```python
# Generate EQ for multiple terms and compare
import matplotlib.pyplot as plt

terms = ['warm', 'bright', 'punchy', 'smooth']
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for term, ax in zip(terms, axes.flat):
    eq_params = system.generate_eq_from_semantic(term, variations=5)

    # Plot all variations
    for params in eq_params:
        # Simplified frequency response (assume 10 bands)
        freqs = np.linspace(20, 20000, 10)
        gains = params[1::4] * 24 - 12  # Denormalize gain
        ax.plot(freqs, gains, alpha=0.5)

    ax.set_title(f'"{term}"')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Gain (dB)')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('neural_eq_comparison.png', dpi=150)
```

### Example 3: Real-time Interpolation
```python
# Interactive morphing for live audio applications
def get_interpolated_eq(term1, term2, alpha):
    """
    Get EQ that blends between two semantic terms

    Args:
        term1: First semantic term (e.g., "warm")
        term2: Second semantic term (e.g., "bright")
        alpha: Blend amount 0.0-1.0 (0=term1, 1=term2)
    """
    return system.interpolate_semantic_terms(term1, term2, alpha)

# Example: 50% warm, 50% bright
eq_params = get_interpolated_eq('warm', 'bright', alpha=0.5)
```

## Next Steps

1. **Run semantic_term_analysis.py** (happening now via quick_start.py)
   - Understand what each semantic term actually means
   - Get visualizations for your report

2. **Test on Real Audio**
   - Apply the generated EQ to actual tracks
   - Listen and evaluate the results
   - Compare with base/adaptive approaches

3. **Visualize Latent Space**
   - Create 2D projections of the 32D latent space
   - Show clustering of semantic terms

4. **Write Up Results**
   - Document your findings
   - Include visualizations
   - Discuss limitations and future work

## Key Insights for Discussion

Your neural approach has several advantages:

1. **Learned Representations**: Captures complex patterns beyond simple averaging
2. **Smooth Interpolation**: Can blend between semantic terms in latent space
3. **Contrastive Learning**: Groups similar semantic terms together
4. **Real Engineer Data**: Trained on 1,595 actual examples from professionals

But also limitations:

1. **Needs Training**: Requires 15 min training time vs instant for base approach
2. **Black Box**: Less interpretable than simple averaging
3. **Data Dependent**: Quality depends on training data coverage
4. **Computational Cost**: More expensive than base approaches

These trade-offs are perfect discussion material for your report!

---

**Bottom Line**: Your trained model is ready to use. While the semantic analysis runs, you have a complete neural EQ system that can generate, morph, and interpolate semantic EQ settings. The analysis will show you what each term means, and then you can use the neural model to generate those EQ curves intelligently.
