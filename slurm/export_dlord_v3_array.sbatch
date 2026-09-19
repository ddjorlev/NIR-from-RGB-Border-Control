#!/bin/bash
#SBATCH --job-name=pid_expo_v3
#SBATCH --account=fri-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=/d/hpc/home/dd25660/ibb/NIR-from-RGB-Border-Control/slurm/logs/export_dlord_v3_%A_%a.out
#SBATCH --error=/d/hpc/home/dd25660/ibb/NIR-from-RGB-Border-Control/slurm/logs/export_dlord_v3_%A_%a.err

set -euo pipefail

cd /d/hpc/home/dd25660/ibb/NIR-from-RGB-Border-Control
source /d/hpc/home/dd25660/myenv/bin/activate

python -m src.export_dlord_synthetic_nir \
    --checkpoint ./checkpoints_tufts_v3/checkpoint_best.pt \
    --input-root ../archive_extract/dlord_rgbnir_verification/DLORD_rgb2nir \
    --output-root ../archive_extract/dlord_rgbnir_verification/DLORD_synthetic_nir_PID_v3 \
    --output-size 112 \
    --num-steps 25 \
    --batch-size 64 \
    --shard-index ${SLURM_ARRAY_TASK_ID} \
    --num-shards 4
