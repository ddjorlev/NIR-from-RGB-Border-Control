#!/bin/bash
#SBATCH --job-name=pid_samplegrid_v1
#SBATCH --account=fri-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/d/hpc/home/dd25660/ibb/NIR-from-RGB-Border-Control/slurm/logs/samplegrid_v1_%j.out
#SBATCH --error=/d/hpc/home/dd25660/ibb/NIR-from-RGB-Border-Control/slurm/logs/samplegrid_v1_%j.err

set -euo pipefail
cd /d/hpc/home/dd25660/ibb/NIR-from-RGB-Border-Control
source /d/hpc/home/dd25660/myenv/bin/activate

python analysis/make_sample_grid.py \
    --checkpoint checkpoints_tufts/checkpoint_best.pt \
    --out ../archive_extract/dlord_rgbnir_verification/analysis/figures/qualitative_v1_best.png \
    --num-steps 50
