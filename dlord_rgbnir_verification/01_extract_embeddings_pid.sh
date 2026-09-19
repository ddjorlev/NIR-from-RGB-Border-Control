#!/bin/bash

set -e

python extract_embeddings.py \
  --model arcface \
  --dataset_type synthetic_nir \
  --tag pid_diffusion_run \
  --root_dir DLORD_synthetic_nir_PID \
  --out_dir dlord_embeddings

python extract_embeddings.py \
  --model adaface \
  --dataset_type synthetic_nir \
  --tag pid_diffusion_run \
  --root_dir DLORD_synthetic_nir_PID \
  --out_dir dlord_embeddings
