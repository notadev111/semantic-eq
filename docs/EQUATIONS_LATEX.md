# Mathematical Equations in LaTeX Format

## Copy-paste these directly into your LaTeX report

---

## 1. Core Architecture

### Residual Block (Basic Form)

```latex
h_{l+1} = \sigma(W_l h_l + b_l) + h_l
```

Where $\sigma$ is the ReLU activation and $+h_l$ is the skip connection.

### Full Residual Block Implementation

```latex
\text{ResBlock}(x, d_{\text{out}}):
\begin{aligned}
h_1 &= \text{LayerNorm}(\sigma(W_1 x + b_1)) \\
h_2 &= \text{Dropout}(h_1, p=0.1) \\
h_3 &= \text{LayerNorm}(W_2 h_2 + b_2) \\
\text{skip} &= \begin{cases}
W_{\text{skip}} \cdot x & \text{if } \dim(x) \neq d_{\text{out}} \\
x & \text{otherwise}
\end{cases} \\
\text{output} &= \sigma(h_3 + \text{skip})
\end{aligned}
```

---

## 2. Encoder Architecture

### Latent Space Projection

```latex
z = \tanh(W_2 \cdot \sigma(W_1 \cdot h + b_1) + b_2), \quad z \in [-1,1]^{32}
```

### Semantic Embedding

```latex
e_{\text{semantic}} = W_{\text{sem}} \cdot z, \quad e_{\text{semantic}} \in \mathbb{R}^{128}
```

### Full Encoder Forward Pass

```latex
\begin{aligned}
h_1 &= \text{ResBlock}(x, 128) \\
h_2 &= \text{ResBlock}(h_1, 256) \\
h_3 &= \text{ResBlock}(h_2, 128) \\
z &= \tanh(W_{\text{proj}} \cdot h_3 + b_{\text{proj}}) \\
e_{\text{sem}} &= W_{\text{sem}} \cdot z
\end{aligned}
```

Where $x \in \mathbb{R}^{40}$ are the normalized EQ parameters.

---

## 3. Decoder Architecture

### Parameter Heads

```latex
\begin{aligned}
b &= \sigma(W_2 \cdot \sigma(W_1 \cdot z + b_1) + b_2) \\
g_i &= 12 \cdot \tanh(W_g \cdot b), \quad g_i \in [-12, 12] \text{ dB} \\
f_i &= \text{sigmoid}(W_f \cdot b), \quad f_i \in [0, 1] \\
q_i &= 0.1 + 9.9 \cdot \text{sigmoid}(W_q \cdot b), \quad q_i \in [0.1, 10.0]
\end{aligned}
```

### Full Decoder Forward Pass

```latex
\begin{aligned}
h_1 &= \text{ResBlock}(z, 128) \\
h_2 &= \text{ResBlock}(h_1, 256) \\
h_3 &= \text{ResBlock}(h_2, 128) \\
b &= \sigma(W_2 \cdot \sigma(W_1 \cdot h_3)) \\
\hat{x} &= [g_1, f_1, q_1, g_2, f_2, q_2, \ldots, g_{10}, f_{10}, q_{10}]
\end{aligned}
```

Where $\hat{x} \in \mathbb{R}^{40}$ are the reconstructed EQ parameters.

---

## 4. Loss Functions

### Reconstruction Loss (MSE)

```latex
\mathcal{L}_{\text{recon}} = \frac{1}{N} \sum_{i=1}^{N} \|x_i - \hat{x}_i\|_2^2
```

Where:
```latex
\hat{x}_i = D(E(x_i))
```

$E$ is the encoder, $D$ is the decoder.

### Contrastive Loss (NT-Xent)

```latex
\mathcal{L}_{\text{contrast}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(e_i, e_p)/\tau)}{\sum_{j \neq i} \exp(\text{sim}(e_i, e_j)/\tau)}
```

Where:
```latex
\begin{aligned}
\text{sim}(e_i, e_j) &= \frac{e_i \cdot e_j}{\|e_i\| \|e_j\|} \quad \text{(cosine similarity)} \\
p &= \text{positive pair (same label as } i \text{)} \\
\tau &= 0.1 \quad \text{(temperature parameter)}
\end{aligned}
```

### Total Loss

```latex
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \lambda \cdot \mathcal{L}_{\text{contrast}}
```

Where $\lambda = 0.1$ is the contrastive loss weight.

---

## 5. Data Normalization

### Z-Score Normalization (Forward)

```latex
x_{\text{norm},j} = \frac{x_j - \mu_j}{\sigma_j}
```

Where:
```latex
\begin{aligned}
\mu_j &= \frac{1}{N} \sum_{i=1}^{N} x_{i,j} \quad \text{(mean)} \\
\sigma_j &= \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_{i,j} - \mu_j)^2} \quad \text{(standard deviation)}
\end{aligned}
```

### Denormalization (Inverse)

```latex
x_j = x_{\text{norm},j} \cdot \sigma_j + \mu_j
```

---

## 6. Semantic Interpolation

### Centroid Computation

```latex
\text{For semantic term } s:
\begin{aligned}
\mathcal{Z}_s &= \{E(x_i) : y_i = s\} \quad \text{(all latent vectors with label } s \text{)} \\
c_s &= \frac{1}{|\mathcal{Z}_s|} \sum_{z \in \mathcal{Z}_s} z
\end{aligned}
```

### Linear Interpolation

```latex
z_{\text{interp}}(\alpha) = (1-\alpha) \cdot c_1 + \alpha \cdot c_2
```

Where:
- $c_1 = \text{centroid}[\text{term}_1]$
- $c_2 = \text{centroid}[\text{term}_2]$
- $\alpha \in [0,1]$ is the blend factor

### Alternative Form

```latex
z_{\text{interp}}(\alpha) = c_1 + \alpha \cdot (c_2 - c_1)
```

### Full Interpolation Pipeline

```latex
\begin{aligned}
\text{1. Lookup:} \quad & c_1, c_2 = \text{centroids}[\text{term}_1], \text{centroids}[\text{term}_2] \\
\text{2. Interpolate:} \quad & z = (1-\alpha) \cdot c_1 + \alpha \cdot c_2 \\
\text{3. Decode:} \quad & p_{\text{norm}} = D(z) \\
\text{4. Denormalize:} \quad & p = p_{\text{norm}} \cdot \sigma + \mu
\end{aligned}
```

---

## 7. Performance Metrics

### Mean Squared Error (MSE)

```latex
\text{MSE} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \|x_i - D(E(x_i))\|_2^2
```

### Mean Absolute Error (MAE)

```latex
\text{MAE} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} |x_i - D(E(x_i))|
```

### Silhouette Score (Clustering Quality)

```latex
s = \frac{1}{N} \sum_{i=1}^{N} \frac{b_i - a_i}{\max(a_i, b_i)}
```

Where:
```latex
\begin{aligned}
a_i &= \text{avg distance to points in same cluster} \\
b_i &= \text{avg distance to points in nearest other cluster}
\end{aligned}
```

Score range: $s \in [-1, 1]$
- $s > 0.5$: Strong clustering
- $s \approx 0$: Overlapping clusters
- $s < 0$: Poor clustering

---

## 8. Training Algorithm

### Gradient Descent Update

```latex
\begin{aligned}
\mathcal{L}_{\text{total}} &= \mathcal{L}_{\text{recon}} + \lambda \cdot \mathcal{L}_{\text{contrast}} \\
\nabla_{\theta} \mathcal{L}_{\text{total}} &= \frac{\partial \mathcal{L}_{\text{total}}}{\partial \theta} \\
\theta_{t+1} &= \theta_t - \eta \cdot \nabla_{\theta} \mathcal{L}_{\text{total}}
\end{aligned}
```

Where:
- $\theta$ = model parameters (encoder + decoder)
- $\eta = 0.001$ = learning rate
- $t$ = training iteration

### Adam Optimizer (Used in Practice)

```latex
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) \nabla_{\theta} \mathcal{L} \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) (\nabla_{\theta} \mathcal{L})^2 \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t} \\
\hat{v}_t &= \frac{v_t}{1-\beta_2^t} \\
\theta_{t+1} &= \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
```

Default parameters: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

---

## 9. Computational Complexity

### Forward Pass Complexity

```latex
\text{Encoder: } O\left(\sum_{l=1}^{L} d_l \cdot d_{l+1}\right) \approx O(10^5) \text{ operations}
```

Where $d_l$ is the dimension at layer $l$.

### Training Complexity

```latex
\text{Total training: } O(N \cdot E \cdot C_{\text{forward}})
```

Where:
- $N = 1595$ = number of training examples
- $E = 50$ = number of epochs
- $C_{\text{forward}} \approx 10^6$ = operations per forward pass

### Inference Complexity

```latex
\begin{aligned}
\text{Centroid lookup:} \quad & O(1) \quad \text{(hash table)} \\
\text{Interpolation:} \quad & O(k) = O(32) \quad \text{(vector operations)} \\
\text{Decode:} \quad & O(k \cdot h + h \cdot d) \approx O(10^4) \\
\text{Total inference:} \quad & O(10^4) < 5\text{ms}
\end{aligned}
```

---

## 10. Signal Processing Equations

### Parametric EQ Transfer Function

For context, each EQ band implements:

```latex
H(z) = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{a_0 + a_1 z^{-1} + a_2 z^{-2}}
```

Where coefficients depend on:
- $g$ = gain (dB)
- $f_c$ = center frequency (Hz)
- $Q$ = quality factor (bandwidth)

Our neural network outputs $(g, f_c, Q)$ for each band.

### Frequency Response

```latex
|H(e^{j\omega})| = \left|\frac{b_0 + b_1 e^{-j\omega} + b_2 e^{-2j\omega}}{a_0 + a_1 e^{-j\omega} + a_2 e^{-2j\omega}}\right|
```

Where $\omega = 2\pi f / f_s$ (normalized frequency).

---

## 11. Key Theorems & Properties

### Universal Approximation Theorem

Neural networks with sufficient width can approximate any continuous function:

```latex
\forall f \in C([0,1]^n), \, \epsilon > 0, \, \exists \text{ network } N:
\sup_{x \in [0,1]^n} |f(x) - N(x)| < \epsilon
```

Justifies using neural networks for EQ parameter mapping.

### Lipschitz Continuity (Interpolation Smoothness)

```latex
\|D(z_1) - D(z_2)\| \leq L \cdot \|z_1 - z_2\|
```

Where $L$ is the Lipschitz constant of the decoder.

Ensures smooth parameter transitions during interpolation.

---

## 12. Summary: Essential Equations for Report

### Minimal Set (Space-limited):

```latex
\begin{aligned}
\text{Residual:} \quad & h_{l+1} = \sigma(W_l h_l + b_l) + h_l \\
\text{Total loss:} \quad & \mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda \mathcal{L}_{\text{contrast}} \\
\text{Interpolation:} \quad & z(\alpha) = (1-\alpha) c_1 + \alpha c_2
\end{aligned}
```

### Extended Set (Recommended):

Add:
```latex
\begin{aligned}
\mathcal{L}_{\text{recon}} &= \frac{1}{N} \sum_{i=1}^{N} \|x_i - \hat{x}_i\|_2^2 \\
\mathcal{L}_{\text{contrast}} &= -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(s_{ip}/\tau)}{\sum_j \exp(s_{ij}/\tau)} \\
g_i &= 12 \tanh(W_g b), \quad f_i = \sigma(W_f b), \quad q_i = 0.1 + 9.9\sigma(W_q b)
\end{aligned}
```

---

## LaTeX Packages Required

Include these in your document preamble:

```latex
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{bm}  % for bold math symbols
```

For aligned equations:
```latex
\usepackage{align}
```

---

## Example LaTeX Document Structure

```latex
\documentclass{article}
\usepackage{amsmath, amssymb, bm}

\begin{document}

\section{Mathematical Framework}

\subsection{Residual Block Architecture}

The core building block is the residual connection:
\begin{equation}
h_{l+1} = \sigma(W_l h_l + b_l) + h_l
\end{equation}
where $\sigma$ denotes the ReLU activation function and $+h_l$ is the skip connection enabling gradient flow.

\subsection{Loss Function}

The total loss combines reconstruction and contrastive objectives:
\begin{equation}
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \lambda \cdot \mathcal{L}_{\text{contrast}}
\end{equation}

The reconstruction loss is:
\begin{equation}
\mathcal{L}_{\text{recon}} = \frac{1}{N} \sum_{i=1}^{N} \|x_i - D(E(x_i))\|_2^2
\end{equation}

The contrastive loss (NT-Xent) is:
\begin{equation}
\mathcal{L}_{\text{contrast}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(e_i, e_p)/\tau)}{\sum_{j \neq i} \exp(\text{sim}(e_i, e_j)/\tau)}
\end{equation}

\subsection{Semantic Interpolation}

Real-time interpolation between semantic terms:
\begin{equation}
z_{\text{interp}}(\alpha) = (1-\alpha) \cdot c_1 + \alpha \cdot c_2, \quad \alpha \in [0,1]
\end{equation}

\end{document}
```

---

**Note**: All equations are ready to copy-paste into your LaTeX document. Just ensure you have the required packages imported.
