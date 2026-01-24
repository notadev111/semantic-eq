# Related Work: Semantic Audio Equalization & Differentiable DSP

*Research survey compiled January 2026*

---

## 1. Semantic Audio Datasets

### SAFE-DB (Stables et al., ISMIR 2014)
- **Paper**: [SAFE: A System for the Extraction and Retrieval of Semantic Audio Descriptors](https://www.researchgate.net/publication/273573906_SAFE_A_system_for_the_extraction_and_retrieval_of_semantic_audio_descriptors)
- **Website**: [semanticaudio.co.uk](http://www.semanticaudio.co.uk/datasets/data/)
- **Contribution**: First crowdsourced semantic audio dataset from DAW plugins (EQ, compressor, reverb, distortion)
- **Scale**: ~1,700 EQ settings with semantic labels ("warm", "bright", etc.)
- **Limitation**: Simple averaging loses variance; small scale
- **Our use**: Primary training dataset for V2 model

### SocialFX (Doh et al., ISMIR 2023)
- **Paper**: [LLM2FX: Leveraging Language Models for Audio Effects](https://arxiv.org/html/2505.20770v1)
- **Contribution**: Larger dataset (~3K examples) mined from audio production forums
- **Advantage**: Natural language descriptions from real practitioners
- **Limitation**: No differentiable processing pipeline
- **Our use**: Referenced as alternative dataset

---

## 2. Neural Semantic EQ

### FlowEQ (Steinmetz, AES 2019)
- **Website**: [csteinmetz1.github.io/flowEQ](https://csteinmetz1.github.io/flowEQ/)
- **Paper**: AES 147th Convention, Gold Award winner
- **Approach**: β-VAE (disentangled variational autoencoder) on SAFE-DB
- **Latent space**: 1-3 dimensions only
- **Modes**: "Traverse" (latent exploration) and "Semantic" (descriptor-based)
- **Limitations**:
  - KL divergence collapse restricts latent capacity
  - No verification that EQ achieves semantic goal
  - Open-loop: predicts params but doesn't verify output
- **Our advantage**: 32D latent space, closed-loop verification via semantic consistency loss

### Automatic Audio Equalization with Semantic Embeddings (Moliner et al., AES 2025)
- **Paper**: [AES Conference on AI/ML for Audio, London 2025](https://www.researchgate.net/publication/395658936_Automatic_Audio_Equalization_with_Semantic_Embeddings)
- **Authors**: Moliner, Välimäki, Drossos, Hämäläinen (Aalto University)
- **Approach**: Predict log-mel features → derive inverse filter
- **Goal**: Blind equalization (match target spectrum), NOT semantic control
- **Key difference from our work**:
  - They do automatic correction, we do user-controlled semantic transformation
  - They use semantic embeddings as backbone features, we use them as targets
  - No user intent/semantic descriptor input in their system
- **Our advantage**: User specifies desired semantic ("warm"), system verifies it achieved

### LLM2FX (Doh et al., 2023)
- **Paper**: [Can Large Language Models Predict Audio Effects Parameters?](https://arxiv.org/html/2505.20770v1)
- **Approach**: LLM predicts effect parameters from natural language
- **Contribution**: Text-to-effect parameter mapping
- **Limitation**: No differentiable signal processing, no closed-loop verification
- **Our advantage**: End-to-end differentiable with semantic consistency loss

---

## 3. Differentiable Audio Effects

### DDSP (Engel et al., ICLR 2020)
- **Paper**: Differentiable Digital Signal Processing
- **Contribution**: First to implement DSP (filters, reverbs, synthesizers) in auto-diff framework
- **Impact**: Enabled neural networks to incorporate DSP as differentiable layers
- **Limitation**: Focused on synthesis, not audio effects or semantic control
- **Our use**: Foundational concept for differentiable EQ

### dasp-pytorch (Steinmetz, JAES 2022)
- **Paper**: [Style Transfer of Audio Effects with Differentiable Signal Processing](https://www.christiansteinmetz.com/)
- **GitHub**: Differentiable audio signal processors in PyTorch
- **Effects**: Parametric EQ, compressor, reverb, distortion
- **Contribution**: Production-ready differentiable effects library
- **Limitation**: No semantic control interface
- **Our use**: Core DSP backend for differentiable EQ

### DeepAFx (Adobe Research, ICASSP 2021)
- **GitHub**: [adobe-research/DeepAFx](https://github.com/adobe-research/DeepAFx)
- **Paper**: [Differentiable Signal Processing With Black-Box Audio Effects](https://arxiv.org/abs/2105.04752)
- **Approach**: Third-party effects as differentiable NN layers
- **Applications**: Amp emulation, breath removal, automatic mastering
- **Limitation**: No semantic control, black-box approximation
- **Our advantage**: White-box differentiable EQ with semantic targets

### BE-AFX: Blind Estimation of Audio Effects (ICASSP 2024)
- **Paper**: [Blind estimation of audio effects using auto-encoder and DDSP](https://hal.science/hal-04539329)
- **Venue**: ICASSP 2024, Seoul, South Korea
- **Approach**: Auto-encoder optimizes audio quality metric
- **Contribution**: Parameter-free blind effect estimation
- **Limitation**: Parameter-focused, not semantic
- **Our advantage**: Semantic descriptor targets, not just parameter matching

---

## 4. Style Transfer & Intelligent Mixing

### ST-ITO: Style Transfer with Inference-Time Optimization (ISMIR 2024)
- **Venue**: ISMIR 2024, San Francisco
- **Authors**: Steinmetz, Singh et al.
- **Approach**: Gradient-free optimization at inference time
- **Contribution**: Works with non-differentiable effects
- **Limitation**: No semantic descriptors, reference-based only
- **Our advantage**: Semantic targets without reference audio

### Music Mixing Style Transfer (2023)
- **Paper**: [Contrastive Learning Approach to Disentangle Audio Effects](https://www.researchgate.net/publication/371287503_Music_Mixing_Style_Transfer_A_Contrastive_Learning_Approach_to_Disentangle_Audio_Effects)
- **Approach**: Contrastive learning for effect disentanglement
- **Goal**: Transfer mixing style from reference
- **Limitation**: Requires reference recording
- **Our advantage**: Semantic targets without reference

### GRAFX (DAFx 2024)
- **Paper**: [DAFx24 Proceedings](https://www.dafx.de/paper-archive/2024/papers/DAFx24_paper_77.pdf)
- **Authors**: Seoul National University & Sony
- **Contribution**: Open-source library for audio processing graphs in PyTorch
- **Focus**: Infrastructure for batched GPU processing
- **Limitation**: Library, not application
- **Our use**: Potential future integration

### Deep Learning for Intelligent Audio Mixing
- **Survey**: [Deep Learning and Intelligent Audio Mixing](https://www.researchgate.net/publication/330967800_Deep_Learning_and_Intelligent_Audio_Mixing)
- **Key insight**: Automatic mixing is part of Intelligent Music Production (IMP)
- **Approaches**: DNN for dynamic range compression, CNN/DRN for EQ
- **Our position**: Semantic control layer on top of intelligent mixing

---

## 5. Language-Audio Models

### CLAP (LAION-AI / Microsoft, 2022-2024)
- **GitHub**: [LAION-AI/CLAP](https://github.com/LAION-AI/CLAP), [microsoft/CLAP](https://github.com/microsoft/CLAP)
- **Paper**: [Learning Audio Concepts From Natural Language Supervision](https://arxiv.org/abs/2206.04769)
- **Approach**: Contrastive language-audio pretraining (like CLIP for audio)
- **Scale**: 4.6M audio-text pairs (Microsoft version)
- **Applications**: Zero-shot classification, retrieval, captioning
- **Limitation**: General audio understanding, not production-specific
- **Our future work**: CLAP integration for text-to-EQ ("make it warmer")

### T-CLAP: Temporal-Enhanced CLAP (2024)
- **Paper**: [Temporal-Enhanced Contrastive Language-Audio Pretraining](https://arxiv.org/abs/2404.17806)
- **Contribution**: Addresses temporal information capture
- **Improvement**: ~30pp improvement in temporal-retrieval accuracy
- **Our relevance**: Could improve temporal semantic analysis

### Contrastive Audio-Language Learning for Music (ISMIR 2022)
- **Paper**: [ISMIR 2022 Proceedings](https://archives.ismir.net/ismir2022/paper/000077.pdf)
- **Author**: Ilaria Manco
- **Contribution**: Music-specific contrastive learning
- **Our future work**: Music-aware semantic embeddings

---

## 6. Commercial & Applied Systems

### RoEx
- **Website**: [roexaudio.com](https://www.roexaudio.com/)
- **Product**: AI mixing and mastering services
- **Approach**: Upload stems, get mixed/mastered output

### LANDR
- **Product**: Automatic mastering platform
- **Features**: Reference tracks, style selection, loudness control

### iZotope (Neoverb, Ozone)
- **Products**: AI-powered mixing/mastering plugins
- **Features**: Reverb Assistant, Master Assistant

### Sound Doctor
- **Claim**: "Cutting-edge AI mixing and mastering software of 2024"

---

## 7. Our Novel Contribution

### Gap Analysis

| Existing Work | Has Semantic Control | Has Differentiable DSP | Has Closed-Loop Verification |
|--------------|---------------------|----------------------|------------------------------|
| FlowEQ | ✓ | ✗ | ✗ |
| DeepAFx/dasp | ✗ | ✓ | ✗ |
| LLM2FX | ✓ | ✗ | ✗ |
| Moliner 2025 | ✗ (blind EQ) | Partial | ✗ |
| ST-ITO | ✗ (reference) | ✗ | ✗ |
| **Ours** | **✓** | **✓** | **✓** |

### Our Contributions

1. **Semantic Consistency Loss**: Novel self-supervised signal that verifies EQ output moves toward target semantic in latent space

2. **End-to-End Differentiable Semantic EQ**: First system combining:
   - Differentiable parametric EQ (via dasp)
   - Semantic descriptor targets (from SAFE-DB)
   - Closed-loop verification

3. **Temporal Semantic Analysis**: Track how semantic characteristics evolve through a song with energy-weighted aggregation

4. **Latent Space Morphing**: Blend multiple semantic targets via interpolation

### Suggested Paper Titles

1. "Closed-Loop Semantic Audio Equalization via Differentiable Signal Processing"
2. "Semantic EQ with End-to-End Differentiable Verification"
3. "Learning to Verify Timbral Transformations: Semantic Consistency for Neural EQ"

---

## 8. Key References (BibTeX)

```bibtex
@inproceedings{stables2014safe,
  title={SAFE: A system for the extraction and retrieval of semantic audio descriptors},
  author={Stables, Ryan and Enderby, Sean and De Man, Brecht and Fazekas, Gyorgy and Reiss, Joshua D},
  booktitle={ISMIR},
  year={2014}
}

@misc{steinmetz2019floweq,
  title={flowEQ: Automatic Equalization with Learned Latent Spaces},
  author={Steinmetz, Christian J},
  howpublished={AES 147th Convention},
  year={2019}
}

@article{steinmetz2022style,
  title={Style transfer of audio effects with differentiable signal processing},
  author={Steinmetz, Christian J and Bryan, Nicholas J and Reiss, Joshua D},
  journal={Journal of the Audio Engineering Society},
  volume={70},
  number={9},
  pages={708--721},
  year={2022}
}

@inproceedings{engel2020ddsp,
  title={DDSP: Differentiable digital signal processing},
  author={Engel, Jesse and Hantrakul, Lamtharn and Gu, Chenjie and Roberts, Adam},
  booktitle={ICLR},
  year={2020}
}

@inproceedings{doh2023llm2fx,
  title={LLM2FX: Leveraging Language Models for Audio Effects Learning},
  author={Doh, Seungheon and others},
  booktitle={ISMIR},
  year={2023}
}

@inproceedings{moliner2025automatic,
  title={Automatic Audio Equalization with Semantic Embeddings},
  author={Moliner, Eloi and V{\"a}lim{\"a}ki, Vesa and Drossos, Konstantinos and H{\"a}m{\"a}l{\"a}inen, Matti},
  booktitle={AES International Conference on AI and Machine Learning for Audio},
  year={2025}
}

@article{wu2022clap,
  title={Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation},
  author={Wu, Yusong and others},
  journal={arXiv preprint arXiv:2206.04769},
  year={2022}
}
```

---

## 9. Conference Venues

| Venue | Focus | Deadline (typical) |
|-------|-------|-------------------|
| **AES Convention** | Audio engineering | Rolling |
| **DAFx** | Digital audio effects | March |
| **ISMIR** | Music information retrieval | April |
| **ICASSP** | Signal processing | October |
| **WASPAA** | Audio & acoustics | March |

---

*Last updated: January 20, 2026*
