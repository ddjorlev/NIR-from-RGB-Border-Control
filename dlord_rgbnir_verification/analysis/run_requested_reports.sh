#!/bin/bash
# Generates the two report variants requested for the mini-paper:
#   1. v2 vs pix2pix vs baseline (v1 excluded)
#   2. v1 vs v2 only (isolates the effect of the attention/model-size fix)
set -e
cd "$(dirname "$0")"
source /d/hpc/home/dd25660/myenv/bin/activate

python generate_report.py \
  --stems arcface_rgb_baseline,arcface_pix2pix,arcface_pid_v2,adaface_rgb_baseline,adaface_pix2pix,adaface_pid_v2 \
  --tag v2_vs_baselines \
  --title-suffix "(v2 vs pix2pix vs baseline)"

python generate_report.py \
  --stems arcface_pid,arcface_pid_v2,adaface_pid,adaface_pid_v2 \
  --tag v1_vs_v2 \
  --title-suffix "(v1 vs v2)"

echo "Both report variants written to analysis/figures/ and analysis/results_table_*.csv"
