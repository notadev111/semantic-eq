# UCL Cluster Training Setup Guide

Quick guide for training the Audio Encoder on UCL's athens.ee.ucl.ac.uk cluster.

## Step 1: Push to GitHub (You do this)

```bash
git add .
git commit -m "Ready for cluster training with W&B"
git push
```

## Step 2: SSH to Cluster

```bash
ssh <your_username>@athens.ee.ucl.ac.uk
```

## Step 3: Clone Repository

```bash
# Navigate to your home or scratch space
cd ~
# or if scratch is available:
# cd /scratch/<your_username>

# Clone your repo
git clone https://github.com/<your_username>/<repo_name>.git
cd <repo_name>
```

## Step 4: Set Up Environment

### Option A: Use existing modules (recommended)

```bash
# Check available modules
module avail cuda
module avail python
module avail pytorch

# Load modules (adjust versions to what's available)
module load cuda/11.8
module load python3/3.9
module load pytorch/2.0-cuda11.8  # if available
```

### Option B: Create virtual environment

```bash
# Load CUDA and Python
module load cuda/11.8
module load python3/3.9

# Create venv
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements_cluster.txt
```

## Step 5: Set Up Weights & Biases

```bash
# Install wandb (if not already in requirements_cluster.txt)
pip install wandb

# Login to W&B (you'll need your API key from wandb.ai)
wandb login
# It will prompt for your API key - get it from https://wandb.ai/authorize
```

## Step 6: Run Training

### Interactive session (for testing - 30 min limit usually):

```bash
# Test that everything works
python check_hardware.py

# Start training
python train_audio_encoder.py --epochs 100 --device cuda --batch-size 32
```

### Background session (recommended for 100 epochs):

```bash
# Use nohup to run in background
nohup python train_audio_encoder.py --epochs 100 --device cuda --batch-size 32 > train.log 2>&1 &

# Get the process ID
echo $!

# Monitor progress
tail -f train.log

# Or detach and come back later - training will continue!
```

### Using screen (alternative for background):

```bash
# Start screen session
screen -S training

# Run training
python train_audio_encoder.py --epochs 100 --device cuda --batch-size 32

# Detach: Press Ctrl+A then D
# Reattach later: screen -r training
```

## Step 7: Monitor Training

### On Cluster:

```bash
# Watch log file
tail -f train.log

# Check GPU usage
nvidia-smi

# Check if process is running
ps aux | grep train_audio_encoder
```

### On Weights & Biases:

1. Go to https://wandb.ai
2. Click on your project "semantic-eq-audio-encoder"
3. Watch live training metrics:
   - Train/Val loss curves
   - Latent loss vs Contrastive loss
   - Learning rate schedule
   - GPU usage

## Step 8: Copy Model Back to Laptop

After training completes (~30-60 minutes):

```bash
# On your laptop, run:
scp <your_username>@athens.ee.ucl.ac.uk:~/<repo_name>/audio_encoder_best.pt .
```

## Expected Training Time

- **100 epochs**: 30-60 minutes on A100/V100 GPU
- **Check progress**: Every ~5-10 epochs should show decreasing loss
- **Final loss**: Should be < 1.0 if training worked properly

## Troubleshooting

### Can't find CUDA:

```bash
module load cuda
nvidia-smi  # Check GPU is available
```

### Out of memory:

```bash
# Reduce batch size
python train_audio_encoder.py --epochs 100 --device cuda --batch-size 16
```

### W&B not logging:

```bash
# Disable W&B if having issues
python train_audio_encoder.py --epochs 100 --device cuda --no-wandb
```

## After Training

1. Check diagnostic:
```bash
python diagnose_audio_encoder.py
```

2. Test with real audio:
```bash
python test_with_real_audio.py --input your_audio.wav
```

3. Copy results back to laptop:
```bash
scp <username>@athens.ee.ucl.ac.uk:~/<repo>/audio_encoder_best.pt .
scp -r <username>@athens.ee.ucl.ac.uk:~/<repo>/analysis_results ./
```

## Quick Commands Reference

```bash
# SSH
ssh <username>@athens.ee.ucl.ac.uk

# Check GPU
nvidia-smi

# Start training in background
nohup python train_audio_encoder.py --epochs 100 --device cuda > train.log 2>&1 &

# Monitor
tail -f train.log

# Copy model back
scp <username>@athens.ee.ucl.ac.uk:~/<repo>/audio_encoder_best.pt .
```
