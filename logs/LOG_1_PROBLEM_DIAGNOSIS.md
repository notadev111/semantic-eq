# Log 1: Problem Diagnosis & New Approach

## Problems Identified in Current Model

### 1. **Feature Normalization Issue** (CRITICAL)
**Problem**: Z-score normalization doesn't work for mixed-scale parameters
- Gains: -12 to +12 dB (range: 24)
- Frequencies: 22 to 20,000 Hz (range: 19,978)
- Q: 0.1 to 10 (range: 9.9)

**Why z-score fails**:
```python
freq_normalized = (freq - mean) / std
# freq=20,000, mean=7000, std=2800
# normalized = (20000 - 7000) / 2800 = 4.64
# HUGE values in "normalized" space!
```

**Solution**: Use **log-scale for frequencies** + **min-max normalization**

### 2. **Decoder Output Constraints** (CRITICAL)
**Problem**: Decoder uses sigmoid → linear mapping but values escape bounds
```python
freqs = sigmoid(x)  # Should be [0,1]
freq_hz = freqs * (max - min) + min
# But we're getting 1.6 MHz instead of 20 kHz!
```

**Why it fails**: The linear layers BEFORE sigmoid produce huge values that saturate sigmoid

**Solution**: Use **tanh** for gains, **log-space for frequencies**, **softplus for Q**

### 3. **Loss Function Imbalance** (CRITICAL)
**Problem**: MSE treats all parameters equally
- Error in gain: 1 dB² = 1 loss
- Error in freq: 1000 Hz² = 1,000,000 loss!
- Model ignores gains/Q completely

**Solution**: **Per-parameter weighted loss** or **log-scale everything**

### 4. **Contrastive Learning Failure** (CRITICAL)
**Problem**: Contrastive loss stayed at 3.25 (no improvement)
- Model not learning semantic structure
- All terms have similar embeddings
- Temperature might be wrong

**Solution**: Check temperature, batch composition, and embedding normalization

## Why VAE Works for Others (FlowEQ Analysis)

FlowEQ paper uses **β-VAE** successfully. Key differences:

### 1. **Log-Scale Frequencies**
FlowEQ explicitly mentions:
- "Frequency parameters are converted to log-scale before encoding"
- This is STANDARD in audio ML
- We missed this!

### 2. **Separate Decoders per Parameter Type**
FlowEQ uses:
- Gain decoder: tanh activation × gain_range
- Frequency decoder: **log-space** prediction
- Q decoder: softplus activation

### 3. **Perceptual Loss**
FlowEQ uses:
- Frequency-weighted MSE (more weight on audible frequencies)
- Not just raw parameter MSE

### 4. **KL Annealing**
VAEs use:
- Gradual increase of KL weight during training
- Prevents posterior collapse
- We used fixed contrastive weight (0.5) from start

## Root Causes Summary

| Issue | Current Model | Should Be |
|-------|---------------|-----------|
| Freq normalization | Linear z-score | **Log-scale** |
| Freq decoder | Sigmoid → linear | **Log-space prediction** |
| Loss weighting | Equal MSE | **Weighted by parameter type** |
| Contrastive learning | Fixed weight | **Annealed/balanced** |
| Output constraints | Soft (sigmoid) | **Hard clipping + better activation** |

## New Model Requirements

### Data Preprocessing
1. **Log-scale frequencies**: `freq_log = log10(freq)`
2. **Min-max normalize gains**: `gain_norm = (gain + 12) / 24` → [0, 1]
3. **Log-scale Q values**: `q_log = log10(q)`
4. **Separate normalization per parameter type**

### Model Architecture
1. **Encoder**: Same ResNet structure (working fine)
2. **Latent space**: Keep 32-dim (good size)
3. **Decoder**: Separate heads with proper activations:
   - Gain head: Tanh → scale to [-12, 12]
   - Freq head: Linear → **interpret as log10(freq)** → convert back
   - Q head: Softplus → [0.1, 10]

### Loss Function
```python
# Separate losses
loss_gain = MSE(gains_normalized, gains_recon_normalized)
loss_freq = MSE(freqs_log_normalized, freqs_log_recon_normalized)
loss_q = MSE(qs_log_normalized, qs_log_recon_normalized)

# Weighted combination
loss_recon = loss_gain + loss_freq + loss_q  # All on same scale now!
loss_total = loss_recon + λ * loss_contrastive
```

### Training Strategy
1. **Contrastive weight annealing**: Start low (0.1), increase to 0.5
2. **Larger batches**: 64 instead of 32 (better contrastive learning)
3. **More epochs**: 150-200 for better convergence
4. **Learning rate scheduling**: Reduce on plateau

## Why This Will Work

### 1. Log-Scale Normalization
```python
# Frequency example
freq = 100 Hz        → log10(100) = 2.0
freq = 1000 Hz       → log10(1000) = 3.0
freq = 10000 Hz      → log10(10000) = 4.0

# Now all in range [~1.3, 4.3] - similar to gains!
# Normalization works properly
```

### 2. Proper Decoder Activations
```python
# Gains: tanh gives [-1, 1], scale to [-12, 12]
gains = torch.tanh(gain_head) * 12.0  # Guaranteed bounds

# Freqs: predict in log-space, convert back
freq_log = freq_head  # Unbounded, but normalized
freq_hz = 10 ** freq_log  # Convert to Hz

# Q: softplus gives [0, ∞), then scale
q = torch.nn.functional.softplus(q_head) * 0.99 + 0.1
```

### 3. Balanced Loss
All parameters on similar scales:
- Gain normalized: [0, 1]
- Freq log normalized: [0, 1]
- Q log normalized: [0, 1]

MSE errors all comparable → balanced learning!

## Implementation Plan

### Phase 1: Data Preprocessing (New)
Create `SAFEDBDatasetLoaderV2` with:
- Log-scale frequency transformation
- Min-max normalization per parameter type
- Proper denormalization for output

### Phase 2: Model Architecture (Modified)
Create `neural_eq_morphing_safedb_v2.py` with:
- Same encoder (ResNet)
- New decoder with proper activations
- Separate output heads
- Hard clipping as safety

### Phase 3: Training (Improved)
- Weighted loss function
- Contrastive weight annealing
- Batch size 64
- 150 epochs
- Learning rate scheduling

### Phase 4: Validation
- Check all output parameters in valid ranges
- Verify clustering improvement
- Compare with Log 1 results

## Expected Results

With these fixes:
- ✅ Frequencies in [20, 20000] Hz
- ✅ Gains in [-12, 12] dB
- ✅ Q in [0.1, 10]
- ✅ Silhouette score > 0.3 (vs -0.50 in Log 1)
- ✅ Contrastive loss decreasing (vs flat in Log 1)

## Next Steps

1. ✅ Archive Log 1 results
2. ⏳ Implement v2 preprocessing
3. ⏳ Implement v2 model
4. ⏳ Train and evaluate
5. ⏳ Compare Log 1 vs Log 2

---

**Key Lesson**: Always use log-scale for frequency parameters in audio ML!
