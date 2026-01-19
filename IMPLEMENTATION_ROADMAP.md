# Implementation Roadmap for AES Publication

## Overview

This document outlines the implementation strategy for making the Semantic EQ system publication-ready for AES (Audio Engineering Society).

## Current Status

### Completed ✅
1. **Original Audio Encoder** - Trained 100 epochs on pink noise + EQ
   - Validation loss: 1.4747
   - Latent distance: 4.046 (well above 0.5 threshold)
   - Successfully distinguishes different audio types

2. **Differentiable EQ Module** (`core/differentiable_eq.py`)
   - Wrapper around dasp-pytorch
   - Converts SAFE-DB 13-param format to dasp-pytorch 18-param format
   - Includes SemanticEQLoss with three components

3. **Multi-Source Dataset** (`core/multi_source_dataset.py`)
   - Supports pink noise, FMA, and MUSDB18
   - Configurable ratios for each source
   - Handles loading, resampling, and augmentation

4. **E2E Training Script** (`train_audio_encoder_e2e.py`)
   - End-to-end differentiable training
   - "Semantic consistency loss" - novel self-supervised signal
   - W&B integration for experiment tracking

---

## Phase 1: Differentiable EQ (Current Focus)

### Goal
Replace non-differentiable EQ application with dasp-pytorch for end-to-end training.

### Novel Contribution
**"Semantic Consistency Loss"**: After applying EQ, re-encode output and verify it's closer to target semantic in latent space.

```
Audio → AudioEncoder → z_audio → Traverse → Decoder → params → dasp_EQ → output
                                                                    ↓
                                          z_output = AudioEncoder(output)
                                                                    ↓
                                          Loss: MSE(z_output, z_target)
```

### Files Created
- `core/differentiable_eq.py` - Differentiable EQ wrapper
- `train_audio_encoder_e2e.py` - E2E training script

### Installation
```bash
pip install dasp-pytorch auraloss
```

### Training Commands
```bash
# Pink noise only (quick test)
python train_audio_encoder_e2e.py --epochs 50 --device cuda --no-wandb

# Fine-tune from pretrained
python train_audio_encoder_e2e.py --epochs 50 --device cuda \
    --pretrained audio_encoder_best.pt
```

---

## Phase 2: Real Music Training Data

### Goal
Train on real music for better generalization (address domain gap from pink noise).

### Datasets

| Dataset | Size | Download | License |
|---------|------|----------|---------|
| FMA Small | 7.2 GB (8,000 tracks × 30s) | [GitHub](https://github.com/mdeff/fma) | CC |
| FMA Medium | 22 GB (25,000 tracks) | [GitHub](https://github.com/mdeff/fma) | CC |
| MUSDB18-HQ | ~10 GB (150 tracks + stems) | [Zenodo](https://zenodo.org/records/3338373) | Academic |

### Download Instructions

**FMA Small (Recommended for testing)**:
```bash
mkdir -p data
cd data
wget https://os.unil.cloud.switch.ch/fma/fma_small.zip
unzip fma_small.zip
```

**MUSDB18-HQ**:
- Request access at https://zenodo.org/records/3338373
- Download and extract to `data/musdb18`

### Training with Multi-Source Data
```bash
# Recommended ratio: 30% pink noise, 50% FMA, 20% MUSDB
python train_audio_encoder_e2e.py --epochs 100 --device cuda \
    --fma-path ./data/fma_small \
    --musdb-path ./data/musdb18 \
    --pink-ratio 0.3 --fma-ratio 0.5 --musdb-ratio 0.2
```

### Expected Improvements
- Better generalization to real music (not just pink noise)
- Diverse spectral content (drums, vocals, instruments)
- More robust semantic classification

---

## Phase 3: CLAP Integration

### Goal
Replace custom Audio Encoder with pretrained CLAP embeddings for zero-shot semantic understanding.

### Why CLAP?
- Trained on 630K audio-text pairs
- Already understands "warm", "bright", "muddy" from language
- Better generalization to unseen audio

### Installation
```bash
pip install laion-clap
```

### Architecture Change
```
# Current:
Audio → AudioEncoder (1.2M params, trained on pink noise) → z_audio (32-dim)

# With CLAP:
Audio → CLAP (frozen, pretrained) → z_clap (512-dim) → Projection (trainable) → z_audio (32-dim)
```

### Implementation (TODO)
```python
import laion_clap

# Load CLAP
clap_model = laion_clap.CLAP_Module(enable_fusion=False)
clap_model.load_ckpt()

# Freeze CLAP
for param in clap_model.parameters():
    param.requires_grad = False

# Trainable projection
projection = nn.Sequential(
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Linear(128, 32)
)

# Forward pass
z_clap = clap_model.get_audio_embedding_from_data(audio)
z_audio = projection(z_clap)
```

### Expected Improvements
- Zero-shot: Works on unseen audio without retraining
- Language-guided: Could accept natural language descriptions
- Better semantic understanding from large-scale pretraining

---

## Phase 4: User Study

### Goal
Perceptual validation that semantic EQ actually achieves target semantics.

### Study Design

**Task**: "Rate how [warm/bright/etc.] this audio sounds" (1-7 scale)

**Conditions**:
1. Original audio (baseline)
2. Your system (Adaptive Semantic EQ)
3. flowEQ (Steinmetz, AES 2019) if available
4. Static EQ (fixed curve for each semantic)

**Participants**: 10-15 audio professionals or trained listeners

**Stimuli**:
- 10 audio clips × 4 conditions × 3 semantic targets = 120 trials
- Randomized presentation order
- Include attention checks

### Analysis
- Mean ratings per condition
- Statistical tests (ANOVA, post-hoc comparisons)
- Effect size (Cohen's d)

### Expected Outcome
Your adaptive system should:
- Achieve higher semantic ratings than static EQ
- Show content-adaptive behavior (different EQ for different audio)
- Maintain audio quality (no artifacts)

---

## Paper Structure for AES

### Title Options
1. "Adaptive Semantic Equalization via Latent Space Traversal with Differentiable Signal Processing"
2. "End-to-End Differentiable Semantic EQ: Fitting Audio to Professional EQ Curves"
3. "Learning Semantic EQ from Labeled Parameters: A Two-Stage Latent Space Approach"

### Abstract (Draft)
> We present a novel approach to semantic audio equalization that generates adaptive EQ parameters based on input audio characteristics and semantic descriptors. Unlike traditional systems that apply fixed EQ curves, our method analyzes input audio to determine appropriate EQ adjustments for achieving semantic targets like "warm" or "bright". The system uses a two-stage training approach: (1) a VAE learns semantic embeddings from the SAFE-DB dataset of professionally-labeled EQ settings, and (2) an audio encoder learns to map audio to this shared latent space using synthesized training data. We introduce a "semantic consistency loss" that enables end-to-end training with differentiable signal processing. Experiments show our system produces content-adaptive EQ that outperforms static semantic EQ baselines in perceptual evaluations.

### Key Contributions
1. **Novel training strategy**: Pink noise + EQ synthesis to learn audio-to-semantic mapping without labeled audio
2. **Semantic consistency loss**: Self-supervised signal for end-to-end differentiable training
3. **Adaptive EQ generation**: Content-aware EQ that varies based on input audio
4. **SAFE-DB curve fitting**: Leveraging professional EQ settings for semantic grounding

### Sections
1. Introduction
2. Related Work (flowEQ, DeepAFx-ST, CLAP, SAFE-DB)
3. Method
   - 3.1 SAFE-DB and Semantic Embeddings
   - 3.2 Audio Encoder Training
   - 3.3 Differentiable EQ and Semantic Consistency Loss
   - 3.4 Latent Space Traversal for Adaptive EQ
4. Experiments
   - 4.1 Training Setup
   - 4.2 Ablation: Pink Noise vs Multi-Source Training
   - 4.3 Comparison with Static EQ and flowEQ
   - 4.4 User Study
5. Results
6. Discussion and Future Work
7. Conclusion

---

## Timeline for AES March Submission

### Week 1-2: Differentiable EQ
- [ ] Install dasp-pytorch on cluster
- [ ] Run E2E training with pink noise
- [ ] Verify gradient flow and convergence
- [ ] Compare E2E vs original model

### Week 3: Multi-Source Training
- [ ] Download FMA small dataset
- [ ] Train with mixed data
- [ ] Evaluate generalization to real music

### Week 4: CLAP Integration (Optional)
- [ ] Implement CLAP projection layer
- [ ] Compare CLAP vs custom encoder
- [ ] Ablation study

### Week 5-6: User Study
- [ ] Design listening test
- [ ] Recruit participants
- [ ] Run study and analyze results

### Week 7-8: Paper Writing
- [ ] Write methods section
- [ ] Generate figures and tables
- [ ] Write results and discussion
- [ ] Submit to AES

---

## References

1. Steinmetz, C.J. (2019). "flowEQ: β-VAE for intelligent control of parametric EQ". AES 147th Convention.

2. Steinmetz, C.J., Bryan, N.J., & Reiss, J.D. (2022). "Style Transfer of Audio Effects with Differentiable Signal Processing". JAES.

3. Stables, R., et al. (2014). "SAFE: A system for the extraction and retrieval of semantic audio descriptors". ISMIR.

4. Elizalde, B., et al. (2023). "CLAP: Learning Audio Concepts from Natural Language Supervision". ICASSP.

5. Defferrard, M., et al. (2017). "FMA: A Dataset For Music Analysis". ISMIR.

---

## Quick Start

```bash
# 1. Install dependencies
pip install dasp-pytorch auraloss wandb

# 2. Test differentiable EQ locally
python -c "from core.differentiable_eq import test_differentiable_eq; test_differentiable_eq()"

# 3. Run E2E training
python train_audio_encoder_e2e.py --epochs 100 --device cuda

# 4. (Optional) Download FMA and train with real music
mkdir -p data && cd data
wget https://os.unil.cloud.switch.ch/fma/fma_small.zip
unzip fma_small.zip
cd ..
python train_audio_encoder_e2e.py --epochs 100 --device cuda --fma-path ./data/fma_small
```
