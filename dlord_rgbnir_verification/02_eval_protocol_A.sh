set -e

#! --target_embs should always be real_nir
# Evaluate rgb - nir Arcface
python evaluate_protocol_A.py --source_embs dlord_embeddings/real_rgb_arcface_ir101_ms1mv3_baseline_mid_fid2emb.pt --target_embs dlord_embeddings/real_nir_arcface_ir101_ms1mv3_baseline_mid_fid2emb.pt --experiment_name "arcface RGB2NIR baseline"
# Evaluate synthetic nir - nir Arcface
python evaluate_protocol_A.py --source_embs dlord_embeddings/synthetic_nir_arcface_ir101_ms1mv3_pix2pix_scratch_run_mid_fid2emb.pt --target_embs dlord_embeddings/real_nir_arcface_ir101_ms1mv3_baseline_mid_fid2emb.pt --experiment_name "arcface pix2pix GAN synthetic NIR to real NIR"
# Evaluate rgb - nir Adaface
python evaluate_protocol_A.py --source_embs dlord_embeddings/real_rgb_adaface_ir101_ms1mv3_baseline_mid_fid2emb.pt --target_embs dlord_embeddings/real_nir_adaface_ir101_ms1mv3_baseline_mid_fid2emb.pt --experiment_name "adaface RGB2NIR baseline"
# Evaluate synthetic nir - nir Adaface
python evaluate_protocol_A.py --source_embs dlord_embeddings/synthetic_nir_adaface_ir101_ms1mv3_pix2pix_scratch_run_mid_fid2emb.pt --target_embs dlord_embeddings/real_nir_adaface_ir101_ms1mv3_baseline_mid_fid2emb.pt --experiment_name "adaface pix2pix GAN synthetic NIR to real NIR"
