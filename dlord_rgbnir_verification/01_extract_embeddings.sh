#!/bin/bash

set -e

python extract_embeddings.py \
  --model arcface \
  --dataset_type real_rgb \
  --tag baseline \
  --root_dir DLORD_rgb2nir \
  --out_dir dlord_embeddings

python extract_embeddings.py \
  --model arcface \
  --dataset_type real_nir \
  --tag baseline \
  --root_dir DLORD_rgb2nir \
  --out_dir dlord_embeddings

python extract_embeddings.py \
  --model arcface \
  --dataset_type synthetic_nir \
  --tag pix2pix_scratch_run \
  --root_dir DLORD_synthetic_nir \
  --out_dir dlord_embeddings

python extract_embeddings.py \
  --model adaface \
  --dataset_type real_rgb \
  --tag baseline \
  --root_dir DLORD_rgb2nir \
  --out_dir dlord_embeddings

python extract_embeddings.py \
  --model adaface \
  --dataset_type real_nir \
  --tag baseline \
  --root_dir DLORD_rgb2nir \
  --out_dir dlord_embeddings

python extract_embeddings.py \
  --model adaface \
  --dataset_type synthetic_nir \
  --tag pix2pix_scratch_run \
  --root_dir DLORD_synthetic_nir \
  --out_dir dlord_embeddings