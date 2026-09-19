#!/bin/bash
# Re-runs the mentor's baseline + pix2pix evaluations AND our PID diffusion
# run, each also saving raw scores to analysis/npz/ for plotting (ROC
# curves, bar charts with CI) alongside the original stdout report.
# Uses already-cached embeddings, so each call is fast (~1 min on a GPU).
set -e

OUT=analysis/npz

python evaluate_protocol_A.py --source_embs dlord_embeddings/real_rgb_arcface_ir101_ms1mv3_baseline_mid_fid2emb.pt --target_embs dlord_embeddings/real_nir_arcface_ir101_ms1mv3_baseline_mid_fid2emb.pt --experiment_name "arcface RGB2NIR baseline" --save_npz $OUT/arcface_rgb_baseline.npz

python evaluate_protocol_A.py --source_embs dlord_embeddings/synthetic_nir_arcface_ir101_ms1mv3_pix2pix_scratch_run_mid_fid2emb.pt --target_embs dlord_embeddings/real_nir_arcface_ir101_ms1mv3_baseline_mid_fid2emb.pt --experiment_name "arcface pix2pix synthetic NIR to real NIR" --save_npz $OUT/arcface_pix2pix.npz

python evaluate_protocol_A.py --source_embs dlord_embeddings/real_rgb_adaface_ir101_ms1mv3_baseline_mid_fid2emb.pt --target_embs dlord_embeddings/real_nir_adaface_ir101_ms1mv3_baseline_mid_fid2emb.pt --experiment_name "adaface RGB2NIR baseline" --save_npz $OUT/adaface_rgb_baseline.npz

python evaluate_protocol_A.py --source_embs dlord_embeddings/synthetic_nir_adaface_ir101_ms1mv3_pix2pix_scratch_run_mid_fid2emb.pt --target_embs dlord_embeddings/real_nir_adaface_ir101_ms1mv3_baseline_mid_fid2emb.pt --experiment_name "adaface pix2pix synthetic NIR to real NIR" --save_npz $OUT/adaface_pix2pix.npz

python evaluate_protocol_A.py --source_embs dlord_embeddings/synthetic_nir_arcface_ir101_ms1mv3_pid_diffusion_run_mid_fid2emb.pt --target_embs dlord_embeddings/real_nir_arcface_ir101_ms1mv3_baseline_mid_fid2emb.pt --experiment_name "arcface PID diffusion synthetic NIR to real NIR" --save_npz $OUT/arcface_pid.npz

python evaluate_protocol_A.py --source_embs dlord_embeddings/synthetic_nir_adaface_ir101_ms1mv3_pid_diffusion_run_mid_fid2emb.pt --target_embs dlord_embeddings/real_nir_adaface_ir101_ms1mv3_baseline_mid_fid2emb.pt --experiment_name "adaface PID diffusion synthetic NIR to real NIR" --save_npz $OUT/adaface_pid.npz
