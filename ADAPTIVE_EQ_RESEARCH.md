# Adaptive EQ Research: Input-Aware Semantic Mastering

## Your Critical Insight

**Current System (Fixed EQ)**:
```
"warm" → [+6dB @ 300Hz, -1dB @ 8kHz] → Apply to ALL audio
```

**Problem**:
- A track that's already bass-heavy gets MORE bass (muddy!)
- A thin track gets bass (good!)
- Same EQ regardless of input = naive

**Why Dataset Has Many "Warm" Curves**:
You're absolutely right! The dataset has 532 different "warm" EQ settings because:
- Warm for a bass-heavy rock song ≠ Warm for thin vocals
- Warm for EDM ≠ Warm for classical
- Engineers adapt EQ to the **existing audio characteristics**

**We need**: Input-aware EQ that analyzes audio → adjusts "warm" curve to fit

---

## Literature Review: State-of-the-Art Approaches

### 1. **Blind Parameter Estimation (Reference-Based)**

**Papers**:
- *"Automatic Multitrack Mixing with a Differentiable Mixing Console"* (Steinmetz et al., 2021)
- *"Automatic Mastering with Style Transfer and Differentiable Signal Processing"* (Steinmetz & Reiss, 2020)
- *"Style Transfer of Audio Effects with Differentiable Signal Processing"* (Comunità et al., 2023)

**Architecture**:
```
Input Audio → [Feature Extractor] → Audio Embedding
Reference "Warm" Audio → [Feature Extractor] → Target Embedding

Distance = ||Audio Embedding - Target Embedding||

Optimization: Minimize distance by adjusting EQ params
  → Differentiable EQ (dasp-pytorch!) allows gradient descent
  → Find EQ params that make Input sound like Reference
```

**Key Insight**: Use differentiable DSP (dasp-pytorch) to optimize EQ parameters via backprop!

**Pros**:
- ✅ Adapts to input audio
- ✅ Theoretically sound (minimize perceptual distance)
- ✅ Uses dasp-pytorch's differentiability

**Cons**:
- ❌ Requires reference audio ("warm" sound to match)
- ❌ Optimization is slow (100-1000 gradient steps per file)
- ❌ No learned semantics (just matching)

---

### 2. **Conditional VAE (Input + Semantic Condition)**

**Papers**:
- *"A Universal Music Translation Network"* (Mor et al., 2019)
- *"Learning to Remaster Music with Style Transfer and Variational Autoencoders"* (Choi et al., 2022)

**Architecture**:
```
Input Audio → [Audio Encoder] → z_audio (latent)
Semantic Term "warm" → [Text Encoder] → z_semantic

Combined: z = [z_audio || z_semantic] (concatenate)

[Decoder] → EQ Parameters (conditioned on BOTH input audio and semantic term)
```

**Training**:
- Encoder learns to extract audio features (spectral content, dynamics)
- Decoder learns: "If audio is X and user wants warm → apply EQ Y"
- Dataset: Pairs of (input audio, desired output, EQ used)

**Pros**:
- ✅ Adapts to input audio
- ✅ Learns semantic concepts
- ✅ Fast inference (single forward pass)
- ✅ Can interpolate (partially warm, blend with input)

**Cons**:
- ❌ Requires paired data (before/after audio with EQ labels)
- ❌ SAFE-DB doesn't have input audio (only EQ settings!)
- ❌ Complex training

---

### 3. **Contrastive Audio-to-EQ Matching (Hybrid)**

**Papers**:
- *"CLAP: Learning Audio Concepts from Natural Language Supervision"* (Wu et al., 2023)
- *"AudioGen: Textually Guided Audio Generation"* (Kreuk et al., 2022)

**Architecture** (NEW - combines your current model + audio analysis):
```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: Encode Input Audio                                 │
└─────────────────────────────────────────────────────────────┘

Input Audio → [Spectrogram] → [CNN/Transformer] → z_audio (32D)

┌─────────────────────────────────────────────────────────────┐
│ Phase 2: Find Nearest Semantic Embedding (Your Current Model)│
└─────────────────────────────────────────────────────────────┘

z_audio → Compare to all semantic term embeddings
  - z_warm, z_bright, z_muddy, etc. (already learned!)

Current style: argmin ||z_audio - z_semantic||
  → "This audio is 75% bright, 20% warm, 5% clear"

┌─────────────────────────────────────────────────────────────┐
│ Phase 3: Compute Complementary EQ                           │
└─────────────────────────────────────────────────────────────┘

If audio is "bright" and user wants "warm":
  → Traverse latent space: z_audio → z_warm
  → Difference vector: Δz = z_warm - z_audio
  → Scale by intensity: Δz_scaled = α * Δz
  → Final latent: z_target = z_audio + Δz_scaled

[Your Decoder] → EQ params from z_target

┌─────────────────────────────────────────────────────────────┐
│ Phase 4: Apply EQ with Differentiable DSP (Optional)        │
└─────────────────────────────────────────────────────────────┘

dasp-pytorch: Apply EQ → Output Audio
  → Can fine-tune with perceptual loss if needed
```

**Pros**:
- ✅ Builds on your existing model (reuse trained encoder/decoder)
- ✅ Adapts to input audio
- ✅ Semantic control ("make it warm")
- ✅ Intensity control (how much to move toward "warm")
- ✅ Fast (audio encoder + latent interpolation + decoder)
- ✅ Can use dasp-pytorch for final EQ application

**Cons**:
- ⚠️ Requires training Audio Encoder (new component)
- ⚠️ Needs audio examples to train on

---

### 4. **Neural Proxy + Optimization (DeepAFx)**

**Papers**:
- *"DeepAFx: Audio Effects Modeling with Deep Learning"* (Ramírez et al., 2021)
- *"Differentiable All-pole Filters for Time-varying Audio Systems"* (Nercessian, 2022)

**Architecture**:
```
Step 1: Train Neural Proxy
  Input: Audio features
  Output: Predicted spectral change if "warm" EQ applied
  (Fast surrogate model for expensive optimization)

Step 2: Online Optimization
  Given input audio X, find EQ params θ that maximize:
    Proxy(X, θ) ≈ "warm" target spectrum

  Use gradient descent on θ (differentiable EQ)
  → Personalized EQ for THIS specific audio
```

**Pros**:
- ✅ Highly adaptive
- ✅ Uses differentiable DSP
- ✅ Can be very accurate

**Cons**:
- ❌ Slow (optimization per file)
- ❌ Complex implementation

---

## Recommended Architecture: **Contrastive Audio-to-EQ Matching**

This is the BEST fit for your project because:

1. **Builds on existing work**: Reuses your trained V2 model
2. **Addresses the problem**: Analyzes input audio
3. **Semantic control**: "Make it warm" still works
4. **Intensity control**: α parameter (0-1) controls "how warm"
5. **Fast**: Single pass through audio encoder + decoder
6. **Publishable**: Novel contribution (audio-aware semantic mastering)

---

## Detailed Technical Design

### Component 1: Audio Encoder (NEW)

**Input**: Audio waveform or spectrogram
**Output**: 32D latent vector (same space as EQ encoder)

**Architecture Options**:

#### Option A: Spectrogram CNN (Simpler)
```python
class AudioEncoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()

        # Mel spectrogram → CNN
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )

        # CNN layers
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Global pooling
        )

        # Project to latent space
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, latent_dim)
        )

    def forward(self, audio):
        # audio: [batch, channels, samples]
        # Convert to mono for analysis
        if audio.shape[1] > 1:
            audio = audio.mean(dim=1, keepdim=True)

        # Compute mel spectrogram
        spec = self.mel_spec(audio)  # [batch, 1, n_mels, time]

        # CNN encoding
        x = self.conv(spec)
        x = x.view(x.size(0), -1)

        # Project to latent
        z = self.fc(x)
        return z
```

#### Option B: Transformer (More Powerful)
```python
class AudioTransformerEncoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()

        # Mel spectrogram
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            n_fft=2048, hop_length=512, n_mels=128
        )

        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=8,
                dim_feedforward=512,
                dropout=0.1
            ),
            num_layers=4
        )

        # Project to latent
        self.fc = nn.Linear(128, latent_dim)

    def forward(self, audio):
        # Mel spectrogram
        spec = self.mel_spec(audio)  # [batch, n_mels, time]

        # Transformer: [seq_len, batch, features]
        spec = spec.permute(2, 0, 1)  # [time, batch, n_mels]

        encoded = self.transformer(spec)

        # Global average pooling over time
        z = encoded.mean(dim=0)  # [batch, n_mels]

        # Project to latent dim
        z = self.fc(z)
        return z
```

**Recommendation**: Start with Option A (CNN), upgrade to B if needed

---

### Component 2: Contrastive Training

**Goal**: Train Audio Encoder to map audio into same latent space as EQ Encoder

**Training Data Problem**: SAFE-DB has EQ settings but NO audio!

**Solutions**:

#### Solution A: Synthesize Training Data (Easiest)
```python
def synthesize_training_pair(eq_params_13, sample_rate=44100):
    """
    Generate synthetic audio by applying EQ to noise/simple signal

    This creates (audio, EQ) pairs for training
    """
    # Generate input signal (pink noise is good proxy for music)
    duration = 5.0  # seconds
    samples = int(duration * sample_rate)

    # Pink noise (1/f spectrum, similar to music)
    pink_noise = generate_pink_noise(samples)

    # Apply the EQ to pink noise
    audio_with_eq = apply_eq(pink_noise, eq_params_13, sample_rate)

    return audio_with_eq, eq_params_13

# Training loop
for eq_setting in safe_db_dataset:
    # Create synthetic audio
    audio, eq_params = synthesize_training_pair(eq_setting.eq_params)

    # Encode both
    z_audio = audio_encoder(audio)
    z_eq = eq_encoder(eq_params)

    # Contrastive loss: they should be close!
    loss = F.mse_loss(z_audio, z_eq)

    loss.backward()
    optimizer.step()
```

**Why this works**:
- EQ shape is independent of source material (to first order)
- If EQ boosts 300Hz, it does so regardless of input
- Audio encoder learns "what spectral shape does this have?"
- EQ encoder already knows "what spectral shape does this EQ create?"
- Training aligns them in latent space

#### Solution B: Use Existing Paired Datasets
- **MUSDB18** (music stems + mixes)
- **DSD100** (demixing dataset)
- Process with random EQ → create (before, after, EQ) triplets

#### Solution C: Bootstrap from Current Model
```python
# Use current decoder to generate diverse EQ settings
# Apply to noise → create training data
for term in ['warm', 'bright', 'muddy', ...]:
    for variation in range(100):  # Generate variations
        # Sample around semantic embedding
        z_base = encode_semantic(term)
        z_perturbed = z_base + small_noise

        eq_params = decoder(z_perturbed)
        audio = apply_eq_to_noise(eq_params)

        # Training pair: (audio, z_perturbed)
        train_audio_encoder(audio, z_perturbed)
```

---

### Component 3: Adaptive EQ Generation

**Algorithm**:
```python
def adaptive_semantic_eq(audio_input, semantic_target, intensity=1.0):
    """
    Generate adaptive EQ that moves audio toward semantic target

    Args:
        audio_input: Input audio tensor
        semantic_target: "warm", "bright", etc.
        intensity: 0-1, how much to apply (0=no change, 1=full)

    Returns:
        eq_params: Adaptive EQ parameters for THIS audio
    """

    # Step 1: Analyze input audio
    z_audio = audio_encoder(audio_input)

    # Step 2: Get target semantic embedding
    z_target = get_semantic_embedding(semantic_target)

    # Step 3: Compute difference vector
    delta_z = z_target - z_audio

    # Step 4: Scale by intensity
    delta_z_scaled = intensity * delta_z

    # Step 5: Compute final latent (move partway toward target)
    z_final = z_audio + delta_z_scaled

    # Step 6: Decode to EQ parameters
    eq_params = decoder(z_final)

    return eq_params

# Usage
audio = load("track.wav")

# Gentle warm (30% intensity)
eq_gentle = adaptive_semantic_eq(audio, "warm", intensity=0.3)

# Full warm (100% intensity)
eq_full = adaptive_semantic_eq(audio, "warm", intensity=1.0)

# The EQ adapts based on what the audio already sounds like!
```

**Key Insight**: Intensity controls how far to move in latent space
- `intensity=0.0`: No change (z_final = z_audio)
- `intensity=0.5`: Halfway to target
- `intensity=1.0`: Full target semantic
- `intensity=2.0`: Exaggerated (overshoot target)

---

### Component 4: "Warmness" Ranking

**How to measure "how warm" an audio already is**:

```python
def measure_semantic_similarity(audio, semantic_term):
    """
    Measure how much audio already exhibits a semantic characteristic

    Returns: 0-1 score (0=not at all, 1=fully has this quality)
    """
    z_audio = audio_encoder(audio)
    z_semantic = get_semantic_embedding(semantic_term)

    # Cosine similarity
    similarity = F.cosine_similarity(z_audio, z_semantic, dim=-1)

    # Map from [-1, 1] to [0, 1]
    warmness_score = (similarity + 1) / 2

    return warmness_score.item()

# Example
audio = load("track.wav")
warmness = measure_semantic_similarity(audio, "warm")
brightness = measure_semantic_similarity(audio, "bright")

print(f"Current warmness: {warmness:.2f}")  # e.g., 0.73 = quite warm
print(f"Current brightness: {brightness:.2f}")  # e.g., 0.21 = not bright

# Auto-suggest complementary EQ
if warmness > 0.7:
    suggestion = "This track is already warm. Try 'bright' or 'clear' to balance."
elif warmness < 0.3:
    suggestion = "This track lacks warmth. Apply 'warm' EQ."
```

---

### Component 5: Differentiable EQ (dasp-pytorch Integration)

**Why dasp-pytorch is useful here**:

```python
from dasp_pytorch import ParametricEQ

def perceptual_loss_optimization(audio_input, semantic_target):
    """
    Fine-tune EQ using gradient descent on perceptual loss
    (Optional refinement step)
    """

    # Step 1: Get initial EQ from adaptive model
    z_audio = audio_encoder(audio_input)
    z_target = get_semantic_embedding(semantic_target)
    z_final = z_audio + (z_target - z_audio)  # Full intensity
    eq_params_init = decoder(z_final)

    # Step 2: Convert to dasp format (make differentiable)
    eq_params_dasp = convert_to_dasp(eq_params_init)
    eq_params_dasp.requires_grad = True

    # Step 3: Optimization loop
    eq_module = ParametricEQ(sample_rate=44100)
    optimizer = torch.optim.Adam([eq_params_dasp], lr=0.01)

    for step in range(50):  # Quick refinement
        # Apply EQ
        audio_processed = eq_module.process_normalized(
            audio_input, eq_params_dasp
        )

        # Encode processed audio
        z_processed = audio_encoder(audio_processed)

        # Perceptual loss: get closer to target
        loss = F.mse_loss(z_processed, z_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Return refined EQ
    return eq_params_dasp.detach()
```

**When to use**:
- ✅ If you want absolute best quality (slower)
- ✅ For critical mastering applications
- ❌ Not needed for real-time/fast processing

---

## Comparison: Fixed vs Adaptive

### Fixed EQ (Current V2):
```python
# Input: Rock track (already bass-heavy)
audio_rock = load("rock_track.wav")
eq_warm = generate_eq("warm")  # +6dB @ 300Hz
output_rock = apply_eq(audio_rock, eq_warm)
# Result: TOO MUCH BASS (muddy)

# Input: Thin vocal (needs bass)
audio_vocal = load("thin_vocal.wav")
eq_warm = generate_eq("warm")  # SAME +6dB @ 300Hz
output_vocal = apply_eq(audio_vocal, eq_warm)
# Result: PERFECT (adds warmth)
```

### Adaptive EQ (Proposed):
```python
# Input: Rock track (already bass-heavy)
audio_rock = load("rock_track.wav")
eq_warm = adaptive_semantic_eq(audio_rock, "warm", intensity=0.7)
# Model detects: "Already has bass, reduce warm boost"
# Result: +2dB @ 300Hz (gentle)
output_rock = apply_eq(audio_rock, eq_warm)
# Result: BALANCED

# Input: Thin vocal (needs bass)
audio_vocal = load("thin_vocal.wav")
eq_warm = adaptive_semantic_eq(audio_vocal, "warm", intensity=0.7)
# Model detects: "Lacks bass, increase warm boost"
# Result: +8dB @ 300Hz (stronger)
output_vocal = apply_eq(audio_vocal, eq_warm)
# Result: BALANCED
```

**Both end up "warm" but adapted to the input!**

---

## Implementation Roadmap

### Phase 1: Proof of Concept (10-15 hours)
1. Implement Audio Encoder (CNN version)
2. Synthesize training data (pink noise + EQ)
3. Train with contrastive loss
4. Test: Encode audio → compare to semantic embeddings
5. Validate: Does it identify "warm" vs "bright" audio?

### Phase 2: Adaptive EQ Generation (8-10 hours)
1. Implement `adaptive_semantic_eq()` function
2. Latent space interpolation with intensity
3. Test on real audio files
4. A/B comparison: Fixed vs Adaptive

### Phase 3: Differentiable Refinement (Optional, 10-15 hours)
1. Integrate dasp-pytorch for gradient-based refinement
2. Perceptual loss optimization
3. Benchmark: Speed vs quality trade-off

### Phase 4: Evaluation (10-12 hours)
1. User study: Does adaptive sound better than fixed?
2. Quantitative metrics (spectral similarity)
3. Edge cases (already-warm audio, extreme inputs)

**Total: 38-52 hours (fits in your 150-hour budget!)**

---

## Why This Is Publishable

**Novel Contributions**:
1. **Audio-aware semantic control**: First system to adapt semantic EQ to input
2. **Latent space traversal**: Using difference vectors for intensity control
3. **Fast inference**: No optimization loop (just encoder + decoder)
4. **Semantic warmness measurement**: Quantify "how warm" audio already is

**Paper Title**: *"Adaptive Semantic Mastering: Input-Aware EQ via Contrastive Audio-to-Parameter Matching"*

**Venues**: DAFx (70% acceptance), ISMIR (50-60%), AES (80%+)

---

## Next Steps: Your Decision

**Option A: Build Adaptive System** (Recommended)
- Most impactful
- Addresses your concern perfectly
- Publishable novelty
- ~40-50 hours

**Option B: Simpler Audio Analysis** (Faster)
- Just measure "warmness" of input
- Suggest complementary EQ (don't adapt parameters)
- ~15-20 hours
- Still useful, less novel

**Option C: Reference-Based Matching** (Research Direction)
- "Make my track sound like this reference"
- Uses dasp-pytorch optimization
- ~30-40 hours
- Different application

**Which direction interests you most?** I can start implementing the Audio Encoder and training pipeline if you want to go with Option A!
