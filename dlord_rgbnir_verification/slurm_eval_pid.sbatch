#!/bin/bash
#SBATCH --job-name=pid_eval_dlord
#SBATCH --account=fri-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/d/hpc/home/dd25660/ibb/archive_extract/dlord_rgbnir_verification/slurm_logs/eval_pid_%j.out
#SBATCH --error=/d/hpc/home/dd25660/ibb/archive_extract/dlord_rgbnir_verification/slurm_logs/eval_pid_%j.err

set -euo pipefail

cd /d/hpc/home/dd25660/ibb/archive_extract/dlord_rgbnir_verification
source /d/hpc/home/dd25660/myenv/bin/activate

bash 01_extract_embeddings_pid.sh
bash 02_eval_protocol_A_with_npz.sh > pid_diffusion_results.txt

cat pid_diffusion_results.txt
