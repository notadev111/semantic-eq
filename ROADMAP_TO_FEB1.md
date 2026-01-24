# Roadmap to AES Convention Paper Submission (Feb 1, 2026)

## Target: AES Category 1 - Peer-Reviewed Paper

**Submission Type**: Full paper (6-10 pages)
**Deadline**: February 1, 2026
**Days Remaining**: ~8 days (from Jan 24)

---

## Strategy: Use What We Have

The E2E differentiable semantic EQ system is **already complete and trained**. The focus now is 100% on **writing the paper** using existing results. We will NOT attempt new features before the deadline.

### What We Have (Ready to Use)
- Trained E2E model (`audio_encoder_e2e.pt`) - 82% semantic loss reduction
- V2 semantic EQ model (`neural_eq_safedb_v2.pt`)
- Temporal semantic analysis script with energy weighting
- Model comparison infrastructure
- Comprehensive related work survey (58+ citations)
- SAFE-DB dataset integration
- Differentiable EQ via dasp-pytorch

### What We're Cutting (Future Work Section)
- CLAP integration for natural language interface
- FMA dataset retraining
- Formal listening tests with statistical analysis
- Real-time plugin implementation

---

## Paper Structure (6-8 pages target)

### 1. Abstract (150-200 words)
- Problem: Semantic descriptors → EQ without verification
- Solution: End-to-end differentiable semantic EQ with closed-loop verification
- Contribution: Semantic Consistency Loss
- Results: 82% semantic loss reduction on SAFE-DB

### 2. Introduction (0.75-1 page)
- Motivation: Bridging semantic intent and parametric EQ
- Problem statement: Existing approaches are open-loop
- Contribution summary (3-4 bullet points)
- Paper organization

### 3. Related Work (1-1.5 pages)
- Semantic audio datasets (SAFE-DB, SocialFX)
- Neural EQ (FlowEQ, LLM2FX, Moliner 2025)
- Differentiable DSP (DDSP, dasp-pytorch, DeepAFx)
- Position our work in the gap

### 4. Method (2-2.5 pages)
- **4.1 System Overview** (architecture diagram)
- **4.2 V2 Semantic Encoder** (EQ params ↔ latent space)
- **4.3 Audio Encoder** (waveform → latent alignment)
- **4.4 Differentiable EQ** (dasp-pytorch integration)
- **4.5 Semantic Consistency Loss** (the novel contribution)
  - Mathematical formulation
  - Why it matters (closed-loop verification)

### 5. Experiments (1.5-2 pages)
- **5.1 Dataset**: SAFE-DB statistics
- **5.2 Training Setup**: Hyperparameters, curriculum learning
- **5.3 Results**:
  - Semantic loss curves (0.32 → 0.057)
  - Ablation: E2E vs embedding-matching
  - Qualitative: semantic profile comparisons
- **5.4 Temporal Analysis**: Energy-weighted aggregation demo

### 6. Discussion (0.5-0.75 page)
- Limitations (single effect type, dataset size)
- Why closed-loop matters for production use
- Computational cost considerations

### 7. Conclusion & Future Work (0.5 page)
- Summary of contributions
- Future: CLAP integration, multi-effect chains, listening tests

### 8. References (~0.5 page)
- 15-25 key citations from RELATED_WORK_2025.md

---

## Daily Schedule

### Day 1 (Jan 24 - Today)
**Goal**: Paper skeleton + figures pipeline

- [ ] Create LaTeX template (AES format)
- [ ] Write Section 1 (Introduction) draft
- [ ] Run `analyze_audio_semantic.py` on 2-3 test tracks
- [ ] Generate Figure 1: System architecture diagram
- [ ] Generate Figure 2: Training loss curves

### Day 2 (Jan 25)
**Goal**: Method section complete

- [ ] Write Section 4.1-4.3 (System, V2 Encoder, Audio Encoder)
- [ ] Write Section 4.4 (Differentiable EQ)
- [ ] Write Section 4.5 (Semantic Consistency Loss) - THE KEY SECTION
- [ ] Generate Figure 3: Latent space visualization (PCA/t-SNE)

### Day 3 (Jan 26)
**Goal**: Experiments section complete

- [ ] Write Section 5.1-5.2 (Dataset, Training)
- [ ] Write Section 5.3 (Results) with all numbers
- [ ] Write Section 5.4 (Temporal Analysis)
- [ ] Generate Figure 4: Temporal semantic evolution plot
- [ ] Generate Figure 5: Semantic profile comparison (old vs E2E)

### Day 4 (Jan 27)
**Goal**: Related work + scaffolding

- [ ] Write Section 3 (Related Work) - use RELATED_WORK_2025.md
- [ ] Create gap analysis table (already in related work doc)
- [ ] Write Section 2 (Abstract) - easier after method is done
- [ ] First complete draft assembled

### Day 5 (Jan 28)
**Goal**: Discussion + conclusion

- [ ] Write Section 6 (Discussion)
- [ ] Write Section 7 (Conclusion & Future Work)
- [ ] Compile full paper, check page count
- [ ] Fix any formatting issues

### Day 6 (Jan 29)
**Goal**: First revision pass

- [ ] Read entire paper for flow
- [ ] Strengthen contribution claims
- [ ] Check all figures have proper captions
- [ ] Verify all citations are correct
- [ ] Check math notation consistency

### Day 7 (Jan 30)
**Goal**: Polish + audio examples

- [ ] Generate 3-4 audio examples (before/after EQ)
- [ ] Create supplementary materials page if needed
- [ ] Proofread for grammar/typos
- [ ] Get feedback if possible (supervisor/peer)

### Day 8 (Jan 31)
**Goal**: Final review + submission prep

- [ ] Address any feedback
- [ ] Final proofread
- [ ] Prepare submission materials
- [ ] Upload audio examples to hosting
- [ ] Review AES submission requirements

### Day 9 (Feb 1) - DEADLINE
**Goal**: Submit

- [ ] Final check of all materials
- [ ] Submit to AES before deadline
- [ ] Backup all submission files

---

## Key Figures Needed

| Figure | Description | Script/Source |
|--------|-------------|---------------|
| Fig 1 | System architecture | Manual (draw.io or LaTeX) |
| Fig 2 | Training loss curves | Extract from training logs |
| Fig 3 | Latent space PCA | `analyze_audio_semantic.py` |
| Fig 4 | Temporal semantic evolution | `analyze_audio_semantic.py` |
| Fig 5 | E2E vs embedding-matching comparison | `compare_models.py` |
| Fig 6 | EQ frequency response example | Generate from params |

---

## Writing Tips for This Paper

1. **Lead with the novel contribution**: Semantic Consistency Loss should be highlighted early and often

2. **Don't oversell**: We're not claiming state-of-the-art on everything, just a novel approach to closed-loop semantic verification

3. **Be honest about limitations**: Single effect type, synthetic training data, no formal listening tests (yet)

4. **Math should be clear**: The loss function is the key equation, make it prominent

5. **Figures tell the story**: Training curves + architecture + qualitative examples = compelling

---

## Backup Plan

If paper isn't ready by Jan 31:
- Switch to Category 2 (Express) at last minute
- 250-word abstract + 750-word summary is much faster
- Still get presentation slot, just no proceedings publication

---

## Files to Create/Update

- [ ] `paper/main.tex` - LaTeX source
- [ ] `paper/figures/` - All figure files
- [ ] `paper/refs.bib` - BibTeX references
- [ ] `supplementary/` - Audio examples + code link

---

## Resources

- AES Convention Paper Guidelines: [aes.org](https://www.aes.org)
- LaTeX template: AES provides official template
- Related work: `docs/RELATED_WORK_2025.md`
- Training results: `TRAINING_LOG_2026-01-20.md`
- Project documentation: `docs/INTERIM_REPORT.md`

---

*Created: January 24, 2026*
*Target: AES 160th Convention, Category 1 Peer-Reviewed Paper*
