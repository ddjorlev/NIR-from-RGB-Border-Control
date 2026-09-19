set -e

#! --target_embs should always be real_nir
# Evaluate PID synthetic nir - nir Arcface
python evaluate_protocol_A.py --source_embs dlord_embeddings/synthetic_nir_arcface_ir101_ms1mv3_pid_diffusion_run_mid_fid2emb.pt --target_embs dlord_embeddings/real_nir_arcface_ir101_ms1mv3_baseline_mid_fid2emb.pt --experiment_name "arcface PID diffusion synthetic NIR to real NIR"
# Evaluate PID synthetic nir - nir Adaface
python evaluate_protocol_A.py --source_embs dlord_embeddings/synthetic_nir_adaface_ir101_ms1mv3_pid_diffusion_run_mid_fid2emb.pt --target_embs dlord_embeddings/real_nir_adaface_ir101_ms1mv3_baseline_mid_fid2emb.pt --experiment_name "adaface PID diffusion synthetic NIR to real NIR"
