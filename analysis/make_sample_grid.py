"""
Generates a qualitative RGB | Real-NIR | Generated-NIR comparison grid from a
given checkpoint, using RAW (non-EMA) weights by default.

Why not EMA: with ema_decay=0.9999 and this run's short step budget (~18k
steps for 300 epochs), EMA weights stay dominated by the random init for
most/all of training and produce noise instead of real samples (verified
empirically). Raw weights reflect what the model actually learned. Pass
--use-ema only if you've separately confirmed EMA has converged.

Usage:
  python analysis/make_sample_grid.py --checkpoint checkpoints_tufts/checkpoint_latest.pt \
      --out ../archive_extract/dlord_rgbnir_verification/analysis/figures/qualitative_latest.png
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))

from src.model import PIDModel
from src.data import PairedFolderDataModule, denormalize_image, save_image_grid
from src.dataloader import create_dataloaders, prepare_diffusion_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--num-samples', type=int, default=8)
    parser.add_argument('--num-steps', type=int, default=50)
    parser.add_argument('--use-ema', action='store_true')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42, help='Fixes which val examples are shown, for fair comparison across checkpoints')
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint['config']
    print(f"Checkpoint epoch: {checkpoint['epoch']}")

    model = PIDModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if args.use_ema and 'ema_shadow' in checkpoint:
        state_dict = model.state_dict()
        for name, param in checkpoint['ema_shadow'].items():
            if name in state_dict:
                state_dict[name] = param
        model.load_state_dict(state_dict)
        print("Using EMA weights.")
    else:
        print("Using raw (non-EMA) weights.")

    model.eval()

    dm = PairedFolderDataModule(config)
    train_loader, val_loader, test_loader = create_dataloaders(config, dm)

    torch.manual_seed(args.seed)
    batch = next(iter(val_loader))
    batch = prepare_diffusion_batch(batch, device)

    rgb = batch['rgb'][:args.num_samples]
    nir = batch['nir'][:args.num_samples]

    with torch.no_grad():
        gen = model.sample(rgb, num_steps=args.num_steps, eta=0.0)

    rgb_vis = denormalize_image(rgb)
    nir_vis = denormalize_image(nir)
    gen_vis = denormalize_image(gen)
    comparison = torch.cat([rgb_vis, nir_vis, gen_vis], dim=0)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_image_grid(comparison, args.out, nrow=args.num_samples)
    print(f"Saved {args.out} (rows: RGB input / real NIR / generated NIR)")


if __name__ == '__main__':
    main()
