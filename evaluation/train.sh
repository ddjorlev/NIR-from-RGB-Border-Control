#!/bin/bash
#SBATCH --job-name=nir_rgb_train
#SBATCH --output=output/logs/%x_%j.out
#SBATCH --error=output/logs/%x_%j.err
#SBATCH --partition=frida
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00



# Set up Python venv in workspace if not present
if [ ! -d "../.venv" ]; then
	python3 -m venv ../.venv
	source ../.venv/bin/activate
	pip install --upgrade pip
	pip install -r ../requirements.txt
else
	source ../.venv/bin/activate
fi


# Set wandb environment variables
export WANDB_PROJECT=nir-from-rgb-diffusion
export WANDB_ENTITY=your_wandb_entity  # <-- Set your wandb entity if needed
export WANDB_MODE=online

# Optional: Set deterministic behavior
export PYTHONHASHSEED=42

cd $(dirname "$0")/..

echo "Starting training on $(hostname) at $(date)"

# Run the training script
python src/train.py

echo "Training finished at $(date)"
