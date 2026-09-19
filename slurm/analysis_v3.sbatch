#!/bin/bash
#SBATCH --job-name=pid_analysis_v3
#SBATCH --account=fri-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/d/hpc/home/dd25660/ibb/NIR-from-RGB-Border-Control/slurm/logs/analysis_v3_%j.out
#SBATCH --error=/d/hpc/home/dd25660/ibb/NIR-from-RGB-Border-Control/slurm/logs/analysis_v3_%j.err

set -euo pipefail

cd /d/hpc/home/dd25660/ibb/NIR-from-RGB-Border-Control
source /d/hpc/home/dd25660/myenv/bin/activate

python analysis/make_sample_grid.py \
    --checkpoint checkpoints_tufts_v3/checkpoint_best.pt \
    --out ../archive_extract/dlord_rgbnir_verification/analysis/figures/qualitative_v3.png

python analysis/compute_validation_metrics.py \
    --checkpoint checkpoints_tufts_v3/checkpoint_best.pt \
    --tag v3

echo "ANALYSIS_V3_DONE"
