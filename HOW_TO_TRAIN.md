# How to Train the Neural EQ System

## What Changed

✅ **Removed**: Synthetic fallback dataset generation
✅ **Now uses**: Only real SocialFX dataset (1,595 engineer examples)
✅ **Cleaned**: Removed verbose comments

## Training the Model

### Option 1: Simple (Recommended)

```bash
python train_neural_eq.py
```

That's it! The script will:
1. Load SocialFX dataset from HuggingFace
2. Analyze the data
3. Initialize neural networks (Encoder + Decoder)
4. Train for 50 epochs (~15 min on CPU, ~3 min on GPU)
5. Save the trained model to `neural_eq_model.pt`

### Option 2: Run the main script directly

```bash
cd core
python neural_eq_morphing.py
```

This runs the full demo with training + examples.

## What You Get

The trained model can:

1. **Generate EQ from semantic terms**
   ```python
   system.generate_eq_from_semantic('warm', variations=5)
   ```

2. **Morph between terms**
   ```python
   system.morph_between_terms('warm', 'bright', steps=10)
   ```

3. **Real-time interpolation** (NEW!)
   ```python
   system.interpolate_semantic_terms('warm', 'bright', alpha=0.5)
   # alpha=0.0 → 100% warm
   # alpha=0.5 → 50/50 blend
   # alpha=1.0 → 100% bright
   ```

## Dataset Info

**SocialFX** contains:
- ~1,595 EQ parameter sets
- From real audio engineers
- 40-parameter EQ format
- 765 unique semantic terms

The training script will automatically:
- Filter terms with ≥8 examples
- Normalize parameters
- Split train/val
- Handle all preprocessing

## Training Parameters

Default settings (good for undergrad project):
- **Latent dimension**: 32
- **Epochs**: 50
- **Batch size**: 16
- **Learning rate**: 0.001

You can modify these in [train_neural_eq.py](train_neural_eq.py) if needed.

## Troubleshooting

**"Error loading SocialFX dataset"**
- Make sure you're connected to internet
- Run: `huggingface-cli login` if needed
- Check you have pandas and pyarrow installed

**Training is slow**
- Normal on CPU: ~15 minutes
- Use Google Colab with GPU if you want faster training
- Can reduce epochs to 30 if you're in a hurry

**CUDA out of memory**
- Reduce batch_size to 8
- Or train on CPU (just set `device='cpu'`)

## Next Steps After Training

1. **Test on real audio**: Use the trained model to process actual tracks
2. **Visualize latent space**: See how semantic terms cluster
3. **Compare approaches**: Neural vs Base vs Adaptive semantic mastering
4. **For your report**: Include training curves, latent space visualization, examples

## Model Architecture

- **Encoder**: Neural Residual Network (input → 32D latent)
- **Decoder**: Neural Residual Network (32D latent → output)
- **Loss**: MSE Reconstruction + Contrastive Learning
- **Innovation**: Contrastive loss pulls same semantic terms together in latent space

This is the key difference from VAE approaches (FlowEQ) - more stable training, no posterior collapse.

---

**Estimated time investment**:
- First run: 20 min (15 min training + 5 min exploration)
- Analysis for report: 2-3 hours
- Total: Reasonable for 150-hour project scope
