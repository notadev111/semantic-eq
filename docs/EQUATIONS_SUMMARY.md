# Mathematical Equations for Interim Report

## Quick Reference: Essential Equations for EE Audience

### 1. Core Architecture Equations

#### Residual Block (Foundation)
```
h_{l+1} = σ(W_l h_l + b_l) + h_l
```
Where:
- `σ = ReLU` activation function
- `+ h_l` is the skip/residual connection
- Enables gradient flow in deep networks

#### Full Residual Block Implementation
```
ResBlock(x, d_out):
  h₁ = LayerNorm(σ(W₁x + b₁))
  h₂ = Dropout(h₁, p=0.1)
  h₃ = LayerNorm(W₂h₂ + b₂)

  skip = W_skip·x  if dim(x) ≠ d_out
         x         otherwise

  return σ(h₃ + skip)
```

---

### 2. Encoder Equations

#### Latent Space Projection
```
z = tanh(W₂·σ(W₁·h + b₁) + b₂)  ∈ [-1, 1]^32
```

Purpose: Maps high-dimensional EQ parameters to bounded latent space

#### Semantic Embedding (for contrastive learning)
```
e_semantic = W_sem·z  ∈ ℝ^128
```

---

### 3. Decoder Equations

#### Specialized Parameter Heads
```
Base features: b = σ(W₂·σ(W₁·z + b₁) + b₂)

Gain head:      g_i = 12·tanh(W_g·b)           ∈ [-12, +12] dB
Frequency head: f_i = sigmoid(W_f·b)            ∈ [0, 1]
Q-factor head:  q_i = 0.1 + 9.9·sigmoid(W_q·b) ∈ [0.1, 10.0]
```

Purpose: Enforce physical constraints on EQ parameters

---

### 4. Loss Functions

#### Reconstruction Loss (Mean Squared Error)
```
L_recon = (1/N) Σᵢ₌₁ᴺ ||xᵢ - x̂ᵢ||₂²

where x̂ᵢ = D(E(xᵢ))
```

Measures: How accurately can we reconstruct input EQ settings?

#### Contrastive Loss (Normalized Temperature-scaled Cross Entropy)
```
L_contrast = -(1/N) Σᵢ₌₁ᴺ log( exp(sim(eᵢ, eₚ)/τ) / Σⱼ≠ᵢ exp(sim(eᵢ, eⱼ)/τ) )

where:
  sim(eᵢ, eⱼ) = (eᵢ · eⱼ)/(||eᵢ|| ||eⱼ||)  (cosine similarity)
  p = positive pair (same semantic label as i)
  τ = 0.1 (temperature parameter)
```

Intuition:
- **Numerator**: Pull together embeddings with same label
- **Denominator**: Push apart embeddings with different labels

#### Total Loss
```
L_total = L_recon + λ·L_contrast

where λ = 0.1 (contrastive weight)
```

Balances: Accurate reconstruction + semantic clustering

---

### 5. Data Normalization

#### Z-Score Normalization (Training)
```
x_norm,j = (xⱼ - μⱼ)/σⱼ

where:
  μⱼ = (1/N) Σᵢ₌₁ᴺ xᵢ,ⱼ        (mean of parameter j)
  σⱼ = sqrt((1/N) Σᵢ(xᵢ,ⱼ-μⱼ)²) (std of parameter j)
```

#### Denormalization (Inference)
```
xⱼ = x_norm,j · σⱼ + μⱼ
```

Purpose: Stabilize training, ensure parameters on similar scales

---

### 6. Semantic Interpolation

#### Centroid Computation (Pre-processing)
```
For semantic term s:
  Z_s = {E(xᵢ) : yᵢ = s}  (all latent vectors with label s)
  c_s = (1/|Z_s|) Σ_{z∈Z_s} z
```

#### Linear Interpolation (Real-time)
```
z_interp(α) = (1-α)·c₁ + α·c₂

where:
  c₁ = centroid[term₁]
  c₂ = centroid[term₂]
  α ∈ [0,1] is blend factor
```

#### Full Interpolation Pipeline
```
1. Lookup:      c₁, c₂ = centroids[term₁], centroids[term₂]
2. Interpolate: z = (1-α)·c₁ + α·c₂
3. Decode:      p_norm = D(z)
4. Denormalize: p = p_norm · σ + μ
```

**Special cases**:
- α = 0   → 100% term₁
- α = 0.5 → 50/50 blend
- α = 1   → 100% term₂

---

### 7. Performance Metrics

#### Reconstruction Error
```
MSE = (1/N_test) Σᵢ ||xᵢ - D(E(xᵢ))||²
MAE = (1/N_test) Σᵢ |xᵢ - D(E(xᵢ))|
```

#### Clustering Quality (Silhouette Score)
```
s = (1/N) Σᵢ (bᵢ - aᵢ)/max(aᵢ, bᵢ)

where:
  aᵢ = avg distance to points in same cluster
  bᵢ = avg distance to points in nearest other cluster
```

Range: [-1, 1]
- s > 0.5: Strong clustering
- s ≈ 0: Overlapping clusters
- s < 0: Poor clustering

---

## How to Present These in Your Report

### Section 1: Architecture
Include:
- Residual block equation
- Encoder latent projection
- Decoder parameter heads

### Section 2: Training
Include:
- Total loss function
- Reconstruction loss
- Contrastive loss (with intuition)

### Section 3: Inference
Include:
- Centroid computation
- Linear interpolation
- Full pipeline

### Section 4: Results
Include:
- Reconstruction error metrics
- Clustering quality (Silhouette score)

---

## Signal Processing Context (For EE Audience)

### Classical vs Neural Approach

**Classical (Hand-designed)**:
```
Audio → FFT → Feature extraction → Rule-based EQ → IFFT → Output
```

**Neural (Data-driven)**:
```
Semantic term → Encoder → Latent space → Decoder → EQ parameters
                  ↑                           ↑
            (Learned from                (Learned from
             real data)                   real data)
```

### Analogy to Transform Coding

Think of the neural system as:
1. **Encoder** = Forward transform (like DCT)
2. **Latent space** = Compressed representation (like DCT coefficients)
3. **Decoder** = Inverse transform (like IDCT)

**Key difference**: Transform is learned from data, not analytically derived

---

## Computational Complexity

### Training
```
Forward pass:  O(d·h₁ + h₁·h₂ + ... + hₙ·k)
               ≈ O(10⁶) operations per sample

Backward pass: O(10⁶) operations per sample

Total training: O(N·E·10⁶)
              = 1,595 × 50 × 10⁶
              ≈ 8×10¹⁰ operations
              ≈ 4 minutes on CPU
```

### Inference
```
Centroid lookup:  O(1) hash table access
Interpolation:    O(k) = O(32) vector operations
Decode:          O(k·h + h·d) ≈ O(10⁴) operations
Total:           <5ms
```

**Real-time constraint**: Must complete in <10ms (typical audio buffer)
**Our system**: <5ms ✓

---

## Summary: Equations You MUST Include

### Minimal Set (if space limited):
1. Residual block: `h_{l+1} = σ(Wh + b) + h`
2. Total loss: `L = L_recon + λ·L_contrast`
3. Interpolation: `z(α) = (1-α)·c₁ + α·c₂`

### Extended Set (recommended):
Add:
4. Reconstruction loss
5. Contrastive loss (full form)
6. Parameter heads (gain/freq/Q)

### Full Set (if detailed report):
Include all equations from this document.

---

**Remember**: For EE audience, emphasize:
- Signal processing analogies
- Computational complexity
- Real-time constraints
- Physical parameter bounds
- Numerical stability (normalization, bounded activations)
