"""
Computes paired image-quality metrics (PSNR, SSIM, RMSE, LPIPS) between
generated NIR and ground-truth NIR on the Tufts validation set (paired
data, unlike DLORD). Mirrors the metrics colleagues reported for
StegoGAN/CycleGAN (SSIM, PSNR, FID) and Pix2Next (PSNR, SSIM, RMSE, SAM,
LPIPS, DISTS), for a like-for-like comparison table in the report.

Uses RAW (non-EMA) weights by default -- see make_sample_grid.py / the
export script for why.

Usage:
  python analysis/compute_validation_metrics.py --checkpoint checkpoints_tufts/checkpoint_best.pt --tag v1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import lpips
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

sys.path.append(str(Path(__file__).parent.parent))

from src.model import PIDModel
from src.data import PairedFolderDataModule, denormalize_image
from src.dataloader import create_dataloaders, prepare_diffusion_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--tag', required=True, help='Name for this run, e.g. v1 or v2 (used in output filename)')
    parser.add_argument('--num-steps', type=int, default=50)
    parser.add_argument('--use-ema', action='store_true')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-samples', type=int, default=None, help='Cap number of val pairs evaluated (default: all)')
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
    model.eval()

    dm = PairedFolderDataModule(config)
    train_loader, val_loader, test_loader = create_dataloaders(config, dm)

    lpips_model = lpips.LPIPS(net='alex').to(device)

    psnr_vals, ssim_vals, rmse_vals, lpips_vals = [], [], [], []
    n_seen = 0

    for batch in val_loader.dataloader if hasattr(val_loader, 'dataloader') else val_loader:
        batch = prepare_diffusion_batch(batch, device)
        rgb = batch['rgb']
        nir_gt = batch['nir']

        with torch.no_grad():
            nir_gen = model.sample(rgb, num_steps=args.num_steps, eta=0.0)
            lp = lpips_model(nir_gen, nir_gt).squeeze()
            if lp.dim() == 0:
                lp = lp.unsqueeze(0)

        nir_gt_vis = denormalize_image(nir_gt).cpu().numpy()
        nir_gen_vis = denormalize_image(nir_gen).clamp(0, 1).cpu().numpy()

        for i in range(nir_gt_vis.shape[0]):
            gt = np.transpose(nir_gt_vis[i], (1, 2, 0))
            gen = np.transpose(nir_gen_vis[i], (1, 2, 0))

            psnr_vals.append(psnr_fn(gt, gen, data_range=1.0))
            ssim_vals.append(ssim_fn(gt, gen, data_range=1.0, channel_axis=2))
            rmse_vals.append(np.sqrt(np.mean((gt - gen) ** 2)))
            lpips_vals.append(lp[i].item())

            n_seen += 1
            if args.max_samples and n_seen >= args.max_samples:
                break
        if args.max_samples and n_seen >= args.max_samples:
            break

    results = {
        'tag': args.tag,
        'checkpoint_epoch': checkpoint['epoch'],
        'n_samples': n_seen,
        'PSNR_mean': float(np.mean(psnr_vals)),
        'PSNR_std': float(np.std(psnr_vals)),
        'SSIM_mean': float(np.mean(ssim_vals)),
        'SSIM_std': float(np.std(ssim_vals)),
        'RMSE_mean': float(np.mean(rmse_vals)),
        'RMSE_std': float(np.std(rmse_vals)),
        'LPIPS_mean': float(np.mean(lpips_vals)),
        'LPIPS_std': float(np.std(lpips_vals)),
    }

    print(json.dumps(results, indent=2))

    out_dir = Path(__file__).parent.parent.parent / 'archive_extract' / 'dlord_rgbnir_verification' / 'analysis'
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'validation_metrics_{args.tag}.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == '__main__':
    main()
