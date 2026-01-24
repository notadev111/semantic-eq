# Training Log: 2026-01-19 to 2026-01-20

## Summary

Implemented and trained an **End-to-End Differentiable Semantic EQ System** using differentiable signal processing (DDSP). This represents a significant technical advancement over the previous embedding-matching approach.

---

## Day 1: 2026-01-19

### Work Completed

1. **Created Differentiable EQ Module** (`core/differentiable_eq.py`)
   - Wrapper around `dasp-pytorch` (Steinmetz et al., 2022)
   - 7-band parametric EQ with differentiable gain, frequency, Q parameters
   - Enables gradient flow through actual signal processing operations

2. **Created Multi-Source Dataset Module** (`core/multi_source_dataset.py`)
   - Support for FMA (Free Music Archive) dataset
   - Support for MUSDB18 stems dataset
   - Configurable mixing of real audio sources

3. **Created E2E Training Script** (`train_audio_encoder_e2e.py`)
   - Novel "Semantic Consistency Loss": after applying predicted EQ, re-encode the output and verify it's closer to target semantic
   - Combined loss: semantic consistency + parameter reconstruction + audio quality (multi-resolution STFT)
   - Curriculum learning with intensity annealing (0.3 → 1.0 over training)

4. **Trained E2E Model on UCL Cluster**
   - 50 epochs, ~8.5 hours total
   - Semantic loss: 0.32 → 0.057 (82% reduction)
   - Best checkpoint at epoch 48
   - Model saved as `audio_encoder_e2e.pt`

### Why This Approach?

**Problem with previous approach (embedding-matching)**:
- Audio encoder trained to match precomputed V2 embeddings
- No guarantee that predicted EQ actually produces the desired semantic effect
- "Open-loop" training - never verified the EQ worked

**E2E Differentiable solution**:
- "Closed-loop" training with actual signal processing in the loop
- Apply predicted EQ differentiably → re-encode output → verify semantic improvement
- Gradients flow through the entire pipeline: Audio → Encoder → EQ Params → DSP → Output Audio → Encoder → Loss

---

## Day 2: 2026-01-20

### Work Completed

1. **Model Comparison Script** (`compare_models.py`)
   - Side-by-side comparison of old (embedding-matching) vs new (E2E-DDSP) encoders
   - Loads SAFE-DB dataset for semantic centroids
   - Computes semantic profiles and EQ predictions for both models
   - Generates visualization plots

2. **Comparison Results**
   - Models produce very similar outputs (differences < 0.05)
   - This is expected: both trained to align with same V2 latent space
   - Key difference is *how* they generalize to unseen audio

3. **Temporal Semantic Analysis Script** (`analyze_audio_semantic.py`)
   - Windowed analysis (configurable window/hop size)
   - Tracks semantic evolution over time throughout a song
   - Energy-weighted aggregation (perceptually important sections weighted higher)
   - Latent space trajectory visualization (PCA projection)
   - Blended semantic morphing (e.g., "warm:0.5 + punchy:0.3")

4. **Research Alignment Check**
   - Verified current work aligns with ELEC0030 project goals
   - On track for 150-hour budget
   - Core contributions intact: semantic-to-EQ mapping, real-time inference, latent interpolation

---

## Technical Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    E2E Differentiable Training                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Input Audio ──→ Audio Encoder ──→ z_audio (latent)            │
│                                          │                       │
│                                          ▼                       │
│   Target Semantic ──→ V2 Encoder ──→ z_target                   │
│                                          │                       │
│                                          ▼                       │
│                              Latent Traversal                    │
│                       z_morphed = z_audio + α(z_target - z_audio)│
│                                          │                       │
│                                          ▼                       │
│                              V2 Decoder ──→ EQ Params            │
│                                          │                       │
│                                          ▼                       │
│                         ┌────────────────────────────┐           │
│                         │  Differentiable EQ (dasp)  │           │
│                         │  - 7-band parametric       │           │
│                         │  - Gradients flow through  │           │
│                         └────────────────────────────┘           │
│                                          │                       │
│                                          ▼                       │
│                              Output Audio                        │
│                                          │                       │
│                                          ▼                       │
│                         Audio Encoder ──→ z_output               │
│                                          │                       │
│                                          ▼                       │
│                         Semantic Consistency Loss:               │
│                         L = ||z_output - z_target||²             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Metrics

| Metric | Start | End | Improvement |
|--------|-------|-----|-------------|
| Semantic Loss | 0.32 | 0.057 | 82% ↓ |
| Param Loss | 0.025 | 0.023 | 8% ↓ |
| Quality Loss | 0.25 | 0.26 | stable |
| Training Time | - | 8.5 hrs | 50 epochs |

---

## Files Created/Modified

### New Files
- `core/differentiable_eq.py` - DDSP wrapper module
- `core/multi_source_dataset.py` - FMA/MUSDB dataset support
- `train_audio_encoder_e2e.py` - E2E training script
- `compare_models.py` - Model comparison tool
- `analyze_audio_semantic.py` - Temporal semantic analysis
- `audio_encoder_e2e.pt` - Trained E2E model checkpoint

### Key Dependencies Added
- `dasp-pytorch` - Differentiable audio signal processing
- `auraloss` - Multi-resolution STFT loss

---

## Next Steps

1. **Test on diverse audio** - Verify E2E model generalizes better than embedding-matching
2. **FMA training** - Train on real music instead of synthetic pink noise
3. **CLAP integration** - Natural language interface ("make it warmer")
4. **Evaluation suite** - Quantitative comparison against FlowEQ baseline
5. **AES paper preparation** - Focus on semantic consistency loss novelty

---

## Notes for AES Publication

### Novel Contributions
1. **Semantic Consistency Loss** - Novel self-supervised signal: verify EQ output moves toward target semantic in latent space
2. **End-to-end differentiable semantic EQ** - First to combine DDSP with semantic audio descriptors
3. **Temporal semantic analysis** - Track how semantic characteristics evolve through a song
4. **Latent space morphing** - Blend multiple semantic targets via interpolation

### Scope (Narrow & Focused)
- Single effect type: Parametric EQ (not compression, reverb, etc.)
- Single dataset: SAFE-DB semantic descriptors
- Single architecture: ResNet encoder + contrastive learning
- Clear ablation: embedding-matching vs E2E-DDSP

### Related Work Positioning
- **FlowEQ** (Steinmetz et al., 2021): Normalizing flows for EQ, but limited semantic control
- **LLM2FX** (2024): Text-to-effect but no differentiable signal processing
- **DDSP** (Engel et al., 2020): Differentiable synthesis, we extend to semantic EQ
- **Ours**: Combines semantic understanding with differentiable EQ processing
