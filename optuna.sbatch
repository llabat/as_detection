#!/bin/bash
#SBATCH --job-name=arg_struct_optuna
#SBATCH --output=logs/optuna.out
#SBATCH --error=logs/optuna.err
#SBATCH --time=10:00:00             
#SBATCH --gres=gpu:1
#SBATCH --mem=32G                   
#SBATCH --cpus-per-task=4
#SBATCH --partition=electronic       
#SBATCH --mail-user=nlp.labat@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL     

source /home/labat/miniforge3/bin/activate /home/labat/micromamba/envs/segmenter

echo "Running on $(hostname)"
nvidia-smi

cd /home/labat/granddebat/as_extraction

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline

# Run Optuna tuning (new file)
python optuna_main.py /home/labat/granddebat/as_extraction/config.yml