#!/bin/bash -l

# Job name
#$ -N audio_encoder_train

# Request GPU
#$ -l gpu=1

# Request runtime (4 hours should be enough)
#$ -l h_rt=4:0:0

# Request memory (16GB should be plenty)
#$ -l mem=16G

# Set working directory
#$ -wd /home/<your_username>/semantic_mastering_system

# Output and error logs
#$ -o train_output.log
#$ -e train_error.log

# Email notification (optional - change to your email)
#$ -M your.email@ucl.ac.uk
#$ -m be  # email at beginning and end

# Load modules
module load cuda/11.8
module load python3/3.9

# Activate virtual environment (if you created one)
source venv/bin/activate

# Or load PyTorch module directly
# module load pytorch/2.0-cuda11.8

# Print GPU info
echo "========================================"
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "GPU info:"
nvidia-smi
echo "========================================"

# Run training
python train_audio_encoder.py --epochs 100 --device cuda --batch-size 32

# Print completion
echo "========================================"
echo "Job finished at: $(date)"
echo "========================================"
