"""
Check Available Hardware for Training
======================================
"""

import torch
import platform

print("="*70)
print("HARDWARE CHECK")
print("="*70)

# System info
print(f"\nPlatform: {platform.system()} {platform.release()}")
print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")

# CUDA availability
print(f"\n" + "="*70)
print("GPU AVAILABILITY")
print("="*70)

cuda_available = torch.cuda.is_available()
print(f"\nCUDA Available: {cuda_available}")

if cuda_available:
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
        print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")

    print(f"\nRecommendation: TRAIN ON GPU!")
    print(f"Command: python train_audio_encoder.py --epochs 100 --device cuda")
else:
    print(f"\nNo GPU detected. Training will use CPU.")
    print(f"\nRecommendation: Train on UCL cluster if available!")

# Estimate training time
print(f"\n" + "="*70)
print("ESTIMATED TRAINING TIME (100 epochs)")
print("="*70)

dataset_size = 1283 * 3  # 1283 EQ settings × 3 augmentations = ~3,849 examples
batch_size = 32
steps_per_epoch = dataset_size // batch_size

print(f"\nDataset: ~{dataset_size:,} training examples")
print(f"Batch size: {batch_size}")
print(f"Steps per epoch: {steps_per_epoch}")

if cuda_available:
    gpu_name = torch.cuda.get_device_name(0).lower()

    if 'rtx 40' in gpu_name or 'rtx 50' in gpu_name:
        time_estimate = "1-2 hours"
        speed = "~30-40 sec/epoch"
    elif 'rtx 30' in gpu_name or 'rtx 20' in gpu_name:
        time_estimate = "2-3 hours"
        speed = "~60-90 sec/epoch"
    elif 'gtx' in gpu_name or 'mx' in gpu_name:
        time_estimate = "4-6 hours"
        speed = "~2-3 min/epoch"
    else:
        time_estimate = "2-4 hours"
        speed = "~60-120 sec/epoch"

    print(f"\nGPU ({torch.cuda.get_device_name(0)})")
    print(f"  Speed: {speed}")
    print(f"  Total time: {time_estimate}")
    print(f"  [OK] Recommended for training!")
else:
    print(f"\nCPU (No GPU)")
    print(f"  Speed: ~5-10 min/epoch")
    print(f"  Total time: 8-16 hours")
    print(f"  [WARN] Very slow - use GPU if possible!")

# UCL cluster recommendation
print(f"\n" + "="*70)
print("UCL CLUSTER OPTION")
print("="*70)

print(f"\nIf you have access to UCL's HPC cluster:")
print(f"  1. SSH to cluster: ssh <username>@myriad.rc.ucl.ac.uk")
print(f"  2. Request GPU node: qsub -l gpu=1 -l h_rt=4:00:00")
print(f"  3. Load modules: module load python pytorch")
print(f"  4. Run training: python train_audio_encoder.py --epochs 100 --device cuda")
print(f"\nCluster advantages:")
print(f"  - Powerful GPUs (A100/V100)")
print(f"  - Training time: ~30-60 minutes")
print(f"  - Run in background")
print(f"  - Free up your laptop!")

print(f"\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

if cuda_available:
    print(f"\n[OK] You have a GPU! Train on your laptop:")
    print(f"  python train_audio_encoder.py --epochs 100 --device cuda --batch-size 32")
    print(f"\nExpected time: {time_estimate}")
else:
    print(f"\n[WARN] No GPU detected on laptop")
    print(f"\nOptions:")
    print(f"  1. Train on UCL cluster (RECOMMENDED - 30-60 min)")
    print(f"  2. Train on laptop CPU (8-16 hours - can run overnight)")
    print(f"\nFor CPU training:")
    print(f"  python train_audio_encoder.py --epochs 100 --batch-size 16")
    print(f"  (Reduce batch size to 16 for CPU)")
