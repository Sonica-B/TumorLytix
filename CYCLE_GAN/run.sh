#!/bin/bash
#SBATCH --job-name=RA_model_training
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -C A100|V100|H100
#SBATCH --mail-type=ALL            # Email notifications for ALL events (BEGIN, END, FAIL, REQUEUE, etc.)
#SBATCH --mail-user=dtbhangale@wpi.edu  # Replace with your email address

# Log the node and GPU information
echo "Running on $(hostname)" >> output.log
nvidia-smi >> output.log

# Activate the virtual environment
# source TumorLytix/bin/activate

# Run the Python training script
python3 CYCLE_GAN/train.py
