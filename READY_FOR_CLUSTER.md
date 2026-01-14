# Ready for Cluster Training - Summary

## What We Fixed

1. **Added W&B Integration** to [train_audio_encoder.py](train_audio_encoder.py)
   - Logs training/validation losses in real-time
   - Tracks best model performance
   - Monitors learning rate, gradients, and model parameters
   - Project name: `semantic-eq-audio-encoder`

2. **Created cluster requirements** in [requirements_cluster.txt](requirements_cluster.txt)
   - Minimal dependencies for training on cluster
   - Includes `wandb>=0.15.0`

3. **Setup guide** in [CLUSTER_SETUP.md](CLUSTER_SETUP.md)
   - Step-by-step instructions for athens.ee.ucl.ac.uk
   - No job submission script needed (direct execution)
   - Background training with `nohup` or `screen`

4. **Updated .gitignore** to exclude:
   - `wandb/` directory (training logs)
   - `*.log` files
   - Virtual environments

## Files to Commit (You'll do this)

```bash
git add .
git commit -m "Add W&B integration and cluster setup"
git push
```

Key files:
- `train_audio_encoder.py` (W&B integrated)
- `requirements_cluster.txt` (cluster dependencies)
- `CLUSTER_SETUP.md` (setup guide)
- `.gitignore` (updated)
- All core modules and scripts

## On the Cluster - Quick Start

### 1. SSH and Clone

```bash
ssh <your_username>@athens.ee.ucl.ac.uk
cd ~
git clone https://github.com/<your_username>/<repo>.git
cd <repo>
```

### 2. Load Modules

```bash
module load cuda/11.8
module load python3/3.9
# Check if pytorch module exists:
module avail pytorch
# If yes: module load pytorch/2.0-cuda11.8
# If no: create venv and pip install torch
```

### 3. Install Dependencies

If using venv:
```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements_cluster.txt
```

### 4. Set Up W&B

```bash
pip install wandb
wandb login
# Enter your API key from https://wandb.ai/authorize
```

### 5. Run Training (Background)

```bash
# Run in background with nohup
nohup python train_audio_encoder.py --epochs 100 --device cuda --batch-size 32 > train.log 2>&1 &

# Monitor progress
tail -f train.log

# Watch on W&B: https://wandb.ai
```

### 6. Expected Output

Training should show:
```
======================================================================
TRAINING AUDIO ENCODER FOR ADAPTIVE SEMANTIC EQ
======================================================================

Loading pre-trained V2 model: neural_eq_safedb_v2.pt
  Loaded 1283 EQ settings
  Semantic terms: 25

Audio Encoder parameters: 1,234,567

======================================================================
SYNTHESIZING TRAINING DATA
======================================================================

Dataset created: 3,849 examples

Dataset split:
  Train: 3,464
  Val: 385

======================================================================
STARTING TRAINING
======================================================================
  W&B logging enabled: sunny-dawn-42

Epoch 1/100
----------------------------------------------------------------------
Epoch 1: 100%|████████| 109/109 [00:45<00:00, 2.41it/s, loss=1.8234, latent=1.7821, contr=0.0826]

  Train Loss: 1.8234 (Latent: 1.7821, Contrastive: 0.0826)
  Val Loss: 1.6543 (Latent: 1.6112, Contrastive: 0.0862)
  [BEST] Best model saved (val_loss: 1.6543)
```

## After Training Completes (~30-60 min)

### 1. Check Results

```bash
# Still on cluster
python diagnose_audio_encoder.py
```

Should show:
```
Average latent distance: 2.456  # > 0.5 = GOOD
Unique top-1 terms: 5/5         # = 5 = GOOD
```

### 2. Copy Model Back

```bash
# On your laptop
scp <username>@athens.ee.ucl.ac.uk:~/<repo>/audio_encoder_best.pt .
```

### 3. Test Locally

```bash
# On your laptop
python test_with_real_audio.py --input your_song.wav
```

Now different audio should give **different** semantic profiles!

## W&B Dashboard

While training, visit https://wandb.ai to see:

- **Real-time loss curves** (train vs val)
- **Latent loss vs Contrastive loss** breakdown
- **Learning rate schedule** (cosine annealing)
- **GPU utilization and memory**
- **Model gradients and parameters**
- **System metrics** (CPU, RAM, GPU temp)

## Training Parameters

```python
{
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'scheduler': 'CosineAnnealingLR',
    'optimizer': 'AdamW',
    'dataset_size': 3849,  # 1283 EQ settings × 3 augmentations
    'model_params': ~1.2M
}
```

## If Training Fails

### Out of Memory:
```bash
python train_audio_encoder.py --epochs 100 --device cuda --batch-size 16
```

### W&B Issues:
```bash
python train_audio_encoder.py --epochs 100 --device cuda --no-wandb
```

### Module Not Found:
```bash
pip install -r requirements_cluster.txt
```

## Success Criteria

Training successful if:
- [x] Loss decreases from ~1.8 to < 1.0
- [x] Validation loss doesn't diverge from training loss
- [x] `diagnose_audio_encoder.py` shows latent distance > 0.5
- [x] Different audio gives different semantic profiles
- [x] W&B shows smooth loss curves

## Next Steps After Successful Training

1. Test with real audio on different genres
2. Fine-tune intensity parameter (0.5-1.0)
3. Generate comparison plots
4. Build streaming processor for real-time EQ
5. Create demo/UI for semantic EQ control

---

**You're all set!** Just push to GitHub and follow CLUSTER_SETUP.md
