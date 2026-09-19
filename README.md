# RGB2NIR: Physics-Informed Diffusion for Face NIR Synthesis

A conditional diffusion model that generates near-infrared (NIR) face images
from RGB input, for cross-spectral face verification in a border-control
setting (OnMoveID). The approach is inspired by Physics-Informed Diffusion
(PID, originally proposed for RGB-to-thermal/LWIR generation), adapted here
to NIR: since NIR is reflected light rather than emitted heat, the
Planck's-law / black-body radiation prior used in the original PID paper
does not apply, and is replaced with a learned reflectance-prior head
trained jointly with the denoiser.

Full training details, results, and discussion are in the project report;
this README documents the codebase itself.

## Architecture

A conditional U-Net DDPM operating directly in pixel space (no VAE / latent
space, unlike the original PID paper):

- **RGB conditioning**: the RGB image is channel-concatenated with the noisy
  NIR image at the U-Net's input.
- **Timestep conditioning**: sinusoidal embeddings injected into every
  residual block.
- **Self-attention**: multi-head attention at selected resolutions, tracked
  by actual feature-map resolution (not level index) so it lands where
  configured.
- **Reflectance-prior head**: a 4-layer convolutional network (GroupNorm +
  ReLU) predicts NIR reflectance directly from the RGB image and supplies an
  auxiliary L1 consistency target for the denoiser's clean-image estimate
  $\hat{x}_0$, in place of PID's separately pretrained physical-decomposition
  network. It is used only during training; DDIM sampling at inference time
  conditions on RGB alone.
- **Optional extensions** (`config/config.py`, disabled by default): FiLM
  conditioning at every internal resolution, a channel-consistency loss, and
  a perceptual (LPIPS) loss — implemented and evaluated as an ablation, but
  did not improve on the base model (see report).

## Project Structure

```
NIR-from-RGB-Border-Control/
├── config/
│   └── config.py                    # Dataclass configs; get_config_for_tufts_faces_v2()
│                                     # is the reported/final model
├── src/
│   ├── model.py                     # PIDModel, U-Net, reflectance-prior head, FiLM
│   ├── data.py                      # PairedFolderDataset/DataModule (Tufts layout)
│   ├── dataloader.py                # Dataloader construction, batch preparation
│   ├── train.py                     # Training loop, checkpointing, resume logic
│   ├── main.py                      # Entry point (reads PID_CONFIG env var)
│   └── export_dlord_synthetic_nir.py  # Batched RGB->synthetic-NIR export for DLORD
├── analysis/
│   ├── make_sample_grid.py          # Qualitative RGB / real-NIR / generated-NIR grid
│   ├── compute_validation_metrics.py  # PSNR / SSIM / RMSE / LPIPS on Tufts val split
│   └── plot_training_curve.py
├── dlord_rgbnir_verification/       # Mentor-provided DLORD Protocol A evaluation
│   ├── DLORD_rgb2nir/                 # Raw RGB/NIR video dataset (per identity)
│   ├── DLORD_synthetic_nir_PID_v2/    # Our model's synthetic NIR output (reported)
│   ├── tufts_faces_rgb_nir/            # Aligned RGB-NIR face pairs used for training
│   ├── extract_embeddings.py / evaluate_protocol_A.py  # ArcFace/AdaFace verification
│   └── pid_diffusion_v2_results.txt    # Reported TPR@FPR results
├── slurm/                           # SLURM job scripts (training, export, eval, analysis)
├── checkpoints_tufts_v2/            # Final model checkpoints
├── samples_tufts_v2/                # Periodic training-time sample grids
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Dataset: `dlord_rgbnir_verification/tufts_faces_rgb_nir/` (Tufts Face
Database RGB-NIR pairs, matched by filename: 1,970 train / 143 validation
pairs). DLORD evaluation data lives in
`dlord_rgbnir_verification/DLORD_rgb2nir/` (see that folder's own README for
the evaluation protocol, provided by the course).

## Training

Configs are selected via the `PID_CONFIG` environment variable
(`src/main.py` / `src/train.py`):

```bash
export PID_CONFIG=tufts_faces_v2   # the reported/final model
python -m src.main
```

or via SLURM: `sbatch slurm/train_tufts_v2.sbatch`.

Key hyperparameters for the reported model (`get_config_for_tufts_faces_v2`
in `config/config.py`): 128x128 images, batch size 32, AdamW with cosine
learning-rate schedule, mixed precision, 1,000 diffusion steps (linear
noise schedule 1e-4 to 0.02), 64 base U-Net channels, channel multipliers
(1,2,2,4), attention at resolutions 32 and 16, reflectance-prior loss weight
0.5, 300 epochs. Checkpoint selection: minimum validation loss. Raw
(non-EMA) weights are used for inference/evaluation — EMA decay would need
far more steps than this run's budget to converge away from random init.

Two other configs are also implemented and documented in
`config/config.py`: `tufts_faces` (an earlier iteration with a coarser
attention-resolution heuristic and a larger, more overparameterized U-Net)
and `tufts_faces_v3` (an ablation adding FiLM conditioning, a
channel-consistency loss, a perceptual loss, and longer training — did not
improve on v2; kept for documentation of what was tried).

## Inference / Export

`src/export_dlord_synthetic_nir.py` converts RGB frames to synthetic NIR in
batches, with `--shard-index`/`--num-shards` for parallelizing across a
SLURM job array (`slurm/export_dlord_v2_array.sbatch`). Output is written
to `dlord_rgbnir_verification/DLORD_synthetic_nir_PID_v2/`, preserving the
input directory structure as required by the evaluation protocol.

## Evaluation

Two complementary evaluations are used:

1. **Paired image-quality metrics** on the Tufts validation split (PSNR,
   SSIM, RMSE, LPIPS) via `analysis/compute_validation_metrics.py`, since
   Tufts pairs are spatially aligned.
2. **DLORD Protocol A** cross-spectral face verification (`ArcFace`/
   `AdaFace` embeddings, TPR@FPR with stratified bootstrap 95% CIs) via
   `dlord_rgbnir_verification/`, since DLORD's RGB and NIR videos are
   *not* spatially/temporally aligned and require identity-level rather
   than pixel-level comparison.

```bash
python analysis/make_sample_grid.py --checkpoint checkpoints_tufts_v2/checkpoint_best.pt \
    --out dlord_rgbnir_verification/analysis/figures/qualitative.png
python analysis/compute_validation_metrics.py --checkpoint checkpoints_tufts_v2/checkpoint_best.pt --tag v2

cd dlord_rgbnir_verification
bash 01_extract_embeddings_pid_v2.sh
bash 02_eval_protocol_A_pid_v2.sh
```

## Results (summary)

See the project report for full tables, plots, and discussion. On DLORD
Protocol A (ArcFace, TPR@FPR), the model improves on a Tufts-trained
Pix2Pix reference at every operating point, though both translation
methods remain below the untranslated RGB-vs-NIR baseline. On the Tufts
validation split: PSNR 15.3 dB, SSIM 0.657, RMSE 0.181, LPIPS 0.322.
Qualitatively, facial structure and pose are recovered well; a mild
color-tint artifact and some blur remain (see report for discussion).

## Acknowledgments

- Tufts Face Database (RGB-NIR pairs) for training data.
- DLORD dataset and Protocol A evaluation framework, provided by the course.
- PID (arXiv:2407.09299) for the physics-informed diffusion formulation
  this work adapts from LWIR to NIR.
