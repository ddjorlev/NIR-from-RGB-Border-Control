#!/bin/bash
#SBATCH --job-name=pid_valmetrics_v1
#SBATCH --account=fri-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/d/hpc/home/dd25660/ibb/NIR-from-RGB-Border-Control/slurm/logs/valmetrics_v1_%j.out
#SBATCH --error=/d/hpc/home/dd25660/ibb/NIR-from-RGB-Border-Control/slurm/logs/valmetrics_v1_%j.err

set -euo pipefail
cd /d/hpc/home/dd25660/ibb/NIR-from-RGB-Border-Control
source /d/hpc/home/dd25660/myenv/bin/activate

python analysis/compute_validation_metrics.py --checkpoint checkpoints_tufts/checkpoint_best.pt --tag v1
