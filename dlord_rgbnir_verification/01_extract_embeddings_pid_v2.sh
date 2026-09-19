#!/bin/bash
set -e

python extract_embeddings.py \
  --model arcface \
  --dataset_type synthetic_nir \
  --tag pid_diffusion_v2_run \
  --root_dir DLORD_synthetic_nir_PID_v2 \
  --out_dir dlord_embeddings

python extract_embeddings.py \
  --model adaface \
  --dataset_type synthetic_nir \
  --tag pid_diffusion_v2_run \
  --root_dir DLORD_synthetic_nir_PID_v2 \
  --out_dir dlord_embeddings
