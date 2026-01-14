# Mathematical Equations Reference for Neural EQ Morphing

## LaTeX-Ready Equations for Interim Report

---

## 1. Problem Formulation

**Objective**: Learn mapping from semantic space to parameter space

```latex
\text{Learn } f: \mathcal{S} \rightarrow \mathcal{P}
```

Where:
```latex
\mathcal{S} = \{\text{warm}, \text{bright}, \text{punchy}, \ldots\} \quad \text{(semantic labels)}
\mathcal{P} \subset \mathbb{R}^d \quad \text{(EQ parameter space)}
```

**Via Latent Space Decomposition**:

```latex
f = D \circ E
```

Where:
```latex
E: \mathcal{P} \rightarrow \mathcal{Z} \quad \text{(Encoder)}
D: \mathcal{Z} \rightarrow \mathcal{P} \quad \text{(Decoder)}
\mathcal{Z} \subset [-1,1]^k \quad \text{(Latent space, } k=32 \text{ or } 64\text{)}
```

---

## 2. Encoder Architecture

**Encoder Mapping**:

```latex
E(x) = (z, e_{\text{sem}})
```

Where:
```latex
x \in \mathbb{R}^d \quad \text{(input EQ parameters)}
z \in \mathbb{R}^k \quad \text{(latent representation)}
e_{\text{sem}} \in \mathbb{R}^{128} \quad \text{(semantic embedding)}
```

**Residual Block**:

```latex
\text{ResBlock}(x) = \text{ReLU}(h + \text{skip}(x))
```

Where:
```latex
h = \text{LayerNorm}(\text{Linear}_2(\text{Dropout}(\text{ReLU}(\text{LayerNorm}(\text{Linear}_1(x))))))
\text{skip}(x) = \begin{cases}
    \text{Linear}_{\text{skip}}(x) & \text{if } \dim(x) \neq \dim(\text{out}) \\
    x & \text{otherwise}
\end{cases}
```

**Latent Projection**:

```latex
z = \tanh(\text{Linear}_{k}(\text{ReLU}(\text{Linear}_{2k}(h))))
```

**Semantic Embedding**:

```latex
e_{\text{sem}} = \text{Linear}_{128}(z)
```

---

## 3. Decoder Architecture

**Decoder Mapping**:

```latex
D(z) = \hat{x}
```

Where:
```latex
z \in \mathbb{R}^k \quad \text{(latent input)}
\hat{x} \in \mathbb{R}^d \quad \text{(reconstructed parameters)}
```

**Parameter-Specific Heads**:

```latex
\text{base} = \text{Linear}_d(\text{ReLU}(\text{Linear}_{2d}(h)))
```

```latex
\text{gains} = \tanh(\text{Linear}_{d/3}(\text{base})) \times 12.0 \quad [\text{dB}]
\text{freqs} = \sigma(\text{Linear}_{d/3}(\text{base})) \quad [0,1]
\text{qs} = \sigma(\text{Linear}_{d/3}(\text{base})) \times 9.9 + 0.1 \quad [0.1, 10]
```

**Interleaved Output**:

```latex
\hat{x}[3i] = \text{gains}[i], \quad \hat{x}[3i+1] = \text{freqs}[i], \quad \hat{x}[3i+2] = \text{qs}[i]
```

For $i = 0, 1, \ldots, n_{\text{bands}}-1$

---

## 4. Loss Functions

### 4.1 Reconstruction Loss

```latex
\mathcal{L}_{\text{recon}} = \frac{1}{N} \sum_{i=1}^{N} \|x_i - D(E(x_i))\|_2^2
```

### 4.2 Contrastive Loss (NT-Xent)

**For a batch** $\mathcal{B} = \{(x_i, y_i)\}_{i=1}^{N}$ where $y_i$ are semantic labels:

```latex
z_i, e_i = E(x_i) \quad \forall i \in \mathcal{B}
```

**Normalized embeddings**:

```latex
\bar{e}_i = \frac{e_i}{\|e_i\|_2}
```

**Similarity function**:

```latex
\text{sim}(i,j) = \frac{\bar{e}_i \cdot \bar{e}_j}{\tau}
```

Where $\tau = 0.1$ is the temperature parameter.

**Positive pairs**:

```latex
\text{pos}(i) = \{j \in \mathcal{B} : y_j = y_i, j \neq i\}
```

**Contrastive loss per sample**:

```latex
\mathcal{L}_i = -\log \frac{\sum_{j \in \text{pos}(i)} \exp(\text{sim}(i,j))}{\sum_{j \in \mathcal{B}, j \neq i} \exp(\text{sim}(i,j))}
```

**Total contrastive loss**:

```latex
\mathcal{L}_{\text{contrast}} = \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}_i
```

### 4.3 Combined Loss

```latex
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \lambda \mathcal{L}_{\text{contrast}}
```

Where $\lambda = 0.1$ is the contrastive loss weight.

---

## 5. Data Normalization

**Z-score normalization**:

```latex
\mu_j = \frac{1}{N} \sum_{i=1}^{N} x_{i,j}
```

```latex
\sigma_j = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_{i,j} - \mu_j)^2 + \epsilon}
```

```latex
\tilde{x}_{i,j} = \frac{x_{i,j} - \mu_j}{\sigma_j}
```

Where $\epsilon = 10^{-6}$ prevents division by zero.

**Denormalization**:

```latex
x_{i,j} = \tilde{x}_{i,j} \sigma_j + \mu_j
```

---

## 6. Semantic Interpolation

### 6.1 Centroid Computation

For semantic term $s$, let $\mathcal{E}_s = \{x_i : y_i = s\}$ be all examples with label $s$.

**Encode all examples**:

```latex
\mathcal{Z}_s = \{E(x_i) : x_i \in \mathcal{E}_s\}
```

**Compute centroid**:

```latex
c_s = \frac{1}{|\mathcal{Z}_s|} \sum_{z \in \mathcal{Z}_s} z
```

### 6.2 Linear Interpolation

Given semantic terms $s_1, s_2$ and interpolation factor $\alpha \in [0,1]$:

**Retrieve centroids**:

```latex
c_1 = c_{s_1}, \quad c_2 = c_{s_2}
```

**Interpolate in latent space**:

```latex
z_{\text{interp}}(\alpha) = (1-\alpha) c_1 + \alpha c_2
```

**Alternative form**:

```latex
z_{\text{interp}}(\alpha) = c_1 + \alpha(c_2 - c_1)
```

**Decode to parameters**:

```latex
p_{\text{interp}} = D(z_{\text{interp}}(\alpha))
```

### 6.3 Interpolation Properties

**Boundary conditions**:

```latex
z_{\text{interp}}(0) = c_1, \quad z_{\text{interp}}(1) = c_2
```

**Midpoint**:

```latex
z_{\text{interp}}(0.5) = \frac{c_1 + c_2}{2}
```

**Distance along path**:

```latex
d(\alpha) = \|z_{\text{interp}}(\alpha) - c_1\| = \alpha \|c_2 - c_1\|
```

**Total path length**:

```latex
L = \|c_2 - c_1\|_2
```

---

## 7. Training Optimization

**Optimizer**: Adam

```latex
\theta_{t+1} = \theta_t - \eta \frac{m_t}{\sqrt{v_t} + \epsilon}
```

Where:
```latex
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(first moment)}
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(second moment)}
g_t = \nabla_\theta \mathcal{L}_{\text{total}} \quad \text{(gradient)}
```

**Hyperparameters**:
```latex
\eta = 0.001 \quad \text{(learning rate)}
\beta_1 = 0.9, \quad \beta_2 = 0.999
\epsilon = 10^{-8}
```

---

## 8. Evaluation Metrics

### 8.1 Reconstruction Error

**Mean Squared Error**:

```latex
\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} \|x_i - \hat{x}_i\|_2^2
```

**Mean Absolute Error**:

```latex
\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} \|x_i - \hat{x}_i\|_1
```

### 8.2 Clustering Quality

**Silhouette Score**:

```latex
s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}
```

Where:
```latex
a(i) = \text{avg distance to same cluster}
b(i) = \text{avg distance to nearest other cluster}
```

**Davies-Bouldin Index**:

```latex
\text{DB} = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} \frac{\sigma_i + \sigma_j}{d(c_i, c_j)}
```

Where $\sigma_i$ is cluster scatter and $d(c_i, c_j)$ is centroid distance.

### 8.3 Interpolation Smoothness

**Average step size**:

```latex
\Delta_{\text{avg}} = \frac{1}{n-1} \sum_{i=1}^{n-1} \|p(\alpha_{i+1}) - p(\alpha_i)\|
```

For $\alpha_i \in \{0, \frac{1}{n-1}, \frac{2}{n-1}, \ldots, 1\}$

---

## 9. Information-Theoretic Perspective

### 9.1 Mutual Information (Conceptual)

**Goal**: Maximize mutual information between semantic labels and latent representations

```latex
\max_{E} I(Y; Z) = H(Z) - H(Z|Y)
```

Where:
```latex
I(Y;Z) \quad \text{is mutual information}
H(Z) \quad \text{is entropy of latent space}
H(Z|Y) \quad \text{is conditional entropy}
```

**Approximated by contrastive loss**, which encourages:
- High $I(Y;Z)$: same label → similar latent codes
- Low $H(Z|Y)$: latent codes predictive of labels

---

## 10. Computational Complexity

### 10.1 Forward Pass

**Encoder**:

```latex
\mathcal{O}_{\text{encoder}} = \mathcal{O}(d \cdot h_1 + h_1 \cdot h_2 + h_2 \cdot h_3 + h_3 \cdot k)
```

For hidden dimensions $h_1=128, h_2=256, h_3=128$:

```latex
\mathcal{O}_{\text{encoder}} \approx \mathcal{O}(256d + 49k)
```

**Decoder**: Same complexity

**Total**: $\mathcal{O}(512d + 98k)$ per sample

### 10.2 Training

**Per epoch**:

```latex
\mathcal{O}_{\text{epoch}} = \mathcal{O}\left(\frac{N}{B} \cdot (512d + 98k)\right)
```

Where $N$ = dataset size, $B$ = batch size

### 10.3 Interpolation

**Centroid computation** (one-time):

```latex
\mathcal{O}_{\text{cache}} = \mathcal{O}(N \cdot (512d + 98k))
```

**Single interpolation** (after caching):

```latex
\mathcal{O}_{\text{interp}} = \mathcal{O}(k + 256d)
```

Dominated by decoder forward pass.

---

## 11. Probability Distributions (Implicit)

Although our model is deterministic, the learned latent space induces distributions:

**Empirical distribution of latent codes for term** $s$:

```latex
p(z|s) \approx \mathcal{N}(c_s, \Sigma_s)
```

Where:
```latex
c_s = \mathbb{E}[z|y=s] \quad \text{(centroid)}
\Sigma_s = \text{Cov}(z|y=s) \quad \text{(covariance)}
```

**Interpolated distribution** (mixture):

```latex
p(z|\alpha, s_1, s_2) \approx (1-\alpha) \mathcal{N}(c_1, \Sigma_1) + \alpha \mathcal{N}(c_2, \Sigma_2)
```

---

## 12. Comparison with VAE Formulation

### VAE (FlowEQ)

**Evidence Lower Bound (ELBO)**:

```latex
\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta \cdot D_{KL}(q_\phi(z|x) \| p(z))
```

Problems:
- KL term can cause posterior collapse
- $\beta$ balancing is tricky
- Small latent dims due to KL pressure

### Our Approach

**Direct reconstruction + semantic structure**:

```latex
\mathcal{L}_{\text{ours}} = \|x - \hat{x}\|^2 + \lambda \mathcal{L}_{\text{contrast}}
```

Advantages:
- No posterior collapse
- Explicit semantic clustering
- Scalable latent dimensions

---

## Quick Copy-Paste LaTeX

### Encoder
```latex
E: \mathbb{R}^d \rightarrow \mathbb{R}^k, \quad
z = \tanh(\text{ResBlocks}(x))
```

### Decoder
```latex
D: \mathbb{R}^k \rightarrow \mathbb{R}^d, \quad
\hat{x} = \text{Heads}(\text{ResBlocks}(z))
```

### Loss
```latex
\mathcal{L} = \|x - D(E(x))\|^2 - \lambda \log \frac{\sum_{y_j=y_i} \exp(\text{sim}(i,j))}{\sum_{j \neq i} \exp(\text{sim}(i,j))}
```

### Interpolation
```latex
z(\alpha) = (1-\alpha) c_1 + \alpha c_2, \quad
p(\alpha) = D(z(\alpha))
```

---

**For use in**: ELEC0030 Interim Report
**Last updated**: November 2024
