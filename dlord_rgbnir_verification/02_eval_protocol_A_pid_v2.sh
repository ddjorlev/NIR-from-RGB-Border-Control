set -e

OUT=analysis/npz

python evaluate_protocol_A.py --source_embs dlord_embeddings/synthetic_nir_arcface_ir101_ms1mv3_pid_diffusion_v2_run_mid_fid2emb.pt --target_embs dlord_embeddings/real_nir_arcface_ir101_ms1mv3_baseline_mid_fid2emb.pt --experiment_name "arcface PID diffusion v2 synthetic NIR to real NIR" --save_npz $OUT/arcface_pid_v2.npz
python evaluate_protocol_A.py --source_embs dlord_embeddings/synthetic_nir_adaface_ir101_ms1mv3_pid_diffusion_v2_run_mid_fid2emb.pt --target_embs dlord_embeddings/real_nir_adaface_ir101_ms1mv3_baseline_mid_fid2emb.pt --experiment_name "adaface PID diffusion v2 synthetic NIR to real NIR" --save_npz $OUT/adaface_pid_v2.npz
