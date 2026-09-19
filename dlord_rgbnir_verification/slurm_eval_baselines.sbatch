#!/bin/bash
#SBATCH --job-name=pid_eval_baselines
#SBATCH --account=fri-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/d/hpc/home/dd25660/ibb/archive_extract/dlord_rgbnir_verification/slurm_logs/eval_baselines_%j.out
#SBATCH --error=/d/hpc/home/dd25660/ibb/archive_extract/dlord_rgbnir_verification/slurm_logs/eval_baselines_%j.err

set -euo pipefail

cd /d/hpc/home/dd25660/ibb/archive_extract/dlord_rgbnir_verification
source /d/hpc/home/dd25660/myenv/bin/activate

OUT=analysis/npz

python evaluate_protocol_A.py --source_embs dlord_embeddings/real_rgb_arcface_ir101_ms1mv3_baseline_mid_fid2emb.pt --target_embs dlord_embeddings/real_nir_arcface_ir101_ms1mv3_baseline_mid_fid2emb.pt --experiment_name "arcface RGB2NIR baseline" --save_npz $OUT/arcface_rgb_baseline.npz

python evaluate_protocol_A.py --source_embs dlord_embeddings/synthetic_nir_arcface_ir101_ms1mv3_pix2pix_scratch_run_mid_fid2emb.pt --target_embs dlord_embeddings/real_nir_arcface_ir101_ms1mv3_baseline_mid_fid2emb.pt --experiment_name "arcface pix2pix synthetic NIR to real NIR" --save_npz $OUT/arcface_pix2pix.npz

python evaluate_protocol_A.py --source_embs dlord_embeddings/real_rgb_adaface_ir101_ms1mv3_baseline_mid_fid2emb.pt --target_embs dlord_embeddings/real_nir_adaface_ir101_ms1mv3_baseline_mid_fid2emb.pt --experiment_name "adaface RGB2NIR baseline" --save_npz $OUT/adaface_rgb_baseline.npz

python evaluate_protocol_A.py --source_embs dlord_embeddings/synthetic_nir_adaface_ir101_ms1mv3_pix2pix_scratch_run_mid_fid2emb.pt --target_embs dlord_embeddings/real_nir_adaface_ir101_ms1mv3_baseline_mid_fid2emb.pt --experiment_name "adaface pix2pix synthetic NIR to real NIR" --save_npz $OUT/adaface_pix2pix.npz
