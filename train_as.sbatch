#!/bin/bash
#SBATCH --job-name=arg_struct_train
#SBATCH --output=logs/arg_struct_train.out
#SBATCH --error=logs/arg_struct_train.err
#SBATCH --time=05:00:00             # adjust as needed
#SBATCH --gres=gpu:1
#SBATCH --mem=32G                   # system RAM (not GPU VRAM)
#SBATCH --cpus-per-task=4
#SBATCH --partition=electronic       
#SBATCH --mail-user=nlp.labat@gmail.com   # Replace with your email
#SBATCH --mail-type=BEGIN,END,FAIL             # Notify on job start, end, and failure     

# Activate your environment
source /home/labat/miniforge3/bin/activate /home/labat/micromamba/envs/segmenter

# Print CUDA info (optional for debugging)
echo "Running on $(hostname)"
nvidia-smi

cd /home/labat/granddebat/as_extraction

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run your training script
python main.py /home/labat/granddebat/as_extraction/config.yml