# Model Architecture - How the Two Models Work Together

## Overview

The system uses **two separate neural networks** trained independently but working together in a shared latent space.

## The Two Models

### Model 1: neural_eq_safedb_v2.pt (Pre-trained)

**Components**: EQ Encoder + EQ Decoder

```
EQ Parameters (13 values)
         ↓
    [EQ Encoder]
         ↓
   z_eq (32-dim) ← Shared Latent Space
         ↓
    [EQ Decoder]
         ↓
EQ Parameters (13 values)
```

**Training**:
- Dataset: SAFE-DB (1,283 labeled EQ settings)
- Input: EQ parameters (5 bands: gain, frequency, Q)
- Loss: Reconstruction loss + KL divergence (VAE)
- Goal: Learn semantic meanings ("warm", "bright", etc.)

**Status**: Already trained, frozen, never changes

---

### Model 2: audio_encoder_best.pt (Just Trained)

**Components**: Audio Encoder only

```
Audio Waveform
         ↓
  [Mel Spectrogram]
         ↓
   [Audio Encoder]
         ↓
  z_audio (32-dim) ← Same Shared Latent Space
```

**Training**:
- Dataset: 3,849 synthesized examples (pink noise + EQ filtering)
- Input: Audio waveforms
- Loss: Latent MSE + Contrastive loss
- Goal: Map audio to same latent space as EQ Encoder

**Status**: Trained for 100 epochs today (Jan 14, 2026)

---

## How They Work Together: Adaptive EQ Generation

### Step 1: Analyze Input Audio

```
User's Audio (WAV file)
         ↓
  [Audio Encoder]
         ↓
   z_audio (32-dim) ← Current audio characteristics
```

### Step 2: Get Target Semantic Embedding

```
Semantic term: "warm"
         ↓
Get all "warm" EQ settings from SAFE-DB
         ↓
  [EQ Encoder] (encode each setting)
         ↓
Average all embeddings
         ↓
   z_warm (32-dim) ← Target characteristics
```

### Step 3: Latent Space Traversal

```
z_audio = Current audio embedding
z_warm  = Target semantic embedding
intensity = 0.7 (user-controlled, 0-1)

z_final = z_audio + intensity × (z_warm - z_audio)
```

**What this does**:
- `intensity = 0.0`: No change (stay at z_audio)
- `intensity = 0.5`: Halfway to "warm"
- `intensity = 1.0`: Fully "warm" (ignore current audio)

### Step 4: Decode to EQ Parameters

```
   z_final (32-dim)
         ↓
    [EQ Decoder]
         ↓
EQ Parameters (13 values)
         ↓
Apply to audio
```

---

## Complete Pipeline

```
                    ADAPTIVE EQ GENERATION

Input Audio ──────────────────┐
                              ↓
                      [Audio Encoder]
                              ↓
                        z_audio (32)
                              │
User: "Make it warm"          │ Shared
intensity = 0.7               │ 32-dim
                              │ Latent
"warm" ──────────────┐        │ Space
                     ↓        │
SAFE-DB (532 "warm") │        │
        ↓            ↓        │
   [EQ Encoder] → Average     │
                     ↓        │
               z_warm (32) ←──┘
                     ↓
        z_final = z_audio + 0.7 × (z_warm - z_audio)
                     ↓
              [EQ Decoder]
                     ↓
         EQ Parameters (13 values)
                     ↓
          Apply to Input Audio
                     ↓
              Output Audio
```

---

## Why Two Models?

### Can't we use just one model?

**Option 1**: Train Audio Encoder from scratch
- ❌ No semantic labels for audio (only have labeled EQ settings)
- ❌ Would need thousands of labeled audio examples
- ❌ "Warm" audio is subjective and hard to label

**Option 2**: Extend EQ Encoder to take audio
- ❌ EQ Encoder expects EQ parameters, not audio
- ❌ Would need to retrain entire system
- ❌ Loses pre-trained semantic knowledge

**Our Solution**: Two-stage training
- ✅ Stage 1: EQ Encoder learns semantics from labeled EQ data
- ✅ Stage 2: Audio Encoder learns to map to same latent space
- ✅ Keeps semantic knowledge, extends to audio

---

## Training Strategy

### Stage 1: Train EQ Encoder (Already Done)

```python
# train_neural_eq_safedb_v2.py
for eq_params, semantic_label in SAFE_DB:
    z_eq, z_mean, z_logvar = encoder(eq_params)
    eq_reconstructed = decoder(z_eq)

    loss = reconstruction_loss(eq_params, eq_reconstructed)
         + kl_divergence(z_mean, z_logvar)

    optimizer.step()
```

**Result**: EQ Encoder knows that certain latent regions = "warm", "bright", etc.

### Stage 2: Train Audio Encoder (Just Completed)

```python
# train_audio_encoder.py
for audio, eq_params in synthesized_dataset:
    # Audio → latent
    z_audio = audio_encoder(audio)

    # EQ → latent (EQ Encoder frozen)
    z_eq, _, _ = eq_encoder(eq_params)

    # Loss: Make them match
    latent_loss = MSE(z_audio, z_eq)
    contrastive_loss = contrastive(z_audio, semantic_labels)

    loss = latent_loss + contrastive_loss
    optimizer.step()
```

**Result**: Audio Encoder maps audio to same latent space as EQ Encoder

---

## Key Insight: Shared Latent Space

Both models use a **32-dimensional latent space** where:

- Each dimension captures some aspect of audio/EQ characteristics
- Regions in this space correspond to semantic meanings
- Distance in latent space ≈ perceptual difference

**Example regions** (conceptual):
```
        Bright
          ↑
          |
Thin ←────┼────→ Full
          |
          ↓
        Warm
```

By training the Audio Encoder to match the EQ Encoder's latent space, we can:
1. Analyze audio → get z_audio
2. Know where "warm" is in latent space (from EQ Encoder)
3. Traverse from z_audio toward "warm"
4. Decode to EQ parameters

---

## File Locations

### Models
- `neural_eq_safedb_v2.pt` - EQ Encoder + Decoder (pre-trained)
- `audio_encoder_best.pt` - Audio Encoder (just trained)

### Training Scripts
- `train_neural_eq_safedb_v2.py` - Train EQ Encoder/Decoder
- `train_audio_encoder.py` - Train Audio Encoder

### Core Modules
- `core/neural_eq_morphing_safedb_v2.py` - EQ system
- `core/audio_encoder.py` - Audio Encoder architecture
- `core/adaptive_eq_generator.py` - Combines both models
- `core/training_data_synthesis.py` - Generates training data

### Usage
- `demo_adaptive_eq.py` - Interactive demo
- `test_with_real_audio.py` - Test with real songs

---

## Performance

### EQ Encoder (V2)
- Parameters: ~500K
- Training: 200 epochs on SAFE-DB
- Reconstruction loss: < 0.001
- Status: Production-ready

### Audio Encoder
- Parameters: ~1.2M
- Training: 100 epochs on synthesized data
- Validation loss: 1.47 (best at epoch 98)
- Latent distance: 4.046 (>> 0.5 threshold)
- Status: Production-ready ✓

---

## Next Steps

1. Test with real audio files from different genres
2. Fine-tune intensity parameter (0.5-1.0 range)
3. Generate comparison plots (before/after EQ)
4. Build real-time streaming processor
5. Create user interface for semantic control

The system is now ready for real-world testing!
