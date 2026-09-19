"""
Convert all DLORD_rgb2nir RGB video frames to synthetic NIR using a trained
PID model, mirroring the exact input directory structure so the output can be
dropped straight into the mentor's dlord_rgbnir_verification evaluation
pipeline (DLORD_synthetic_nir_<tag>/<subject>/<rgbn_*>/<frame>.png).

Only `rgbn_*` video folders are converted; `irn_*` folders are the original
ground-truth NIR videos and are left untouched (Protocol A compares your
synthetic NIR against them, it does not need them regenerated).
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.model import PIDModel


def denormalize(tensor, mean, std):
    mean = torch.tensor(mean, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(1, 3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)


def load_model(checkpoint_path: str, device: torch.device, use_ema: bool = False):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    model = PIDModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # NOTE: EMA (decay=0.9999) needs ~10k steps to converge, but a short run
    # (e.g. 300 epochs x ~61 iters/epoch = ~18.3k steps) barely clears that,
    # so EMA weights can still be dominated by the random init and produce
    # noise instead of real samples. Verified empirically on this run: raw
    # weights at epoch 165 produced recognizable faces, EMA weights did not.
    # Raw weights are used by default; pass --use-ema only once you've
    # confirmed (e.g. via a quick sample check) that EMA has caught up.
    if use_ema and 'ema_shadow' in checkpoint:
        ema_shadow = checkpoint['ema_shadow']
        state_dict = model.state_dict()
        for name, param in ema_shadow.items():
            if name in state_dict:
                state_dict[name] = param
        model.load_state_dict(state_dict)
        print("Loaded EMA weights for inference.")
    else:
        print("Loaded raw (non-EMA) trained weights for inference.")

    model.eval()
    return model, config


def collect_rgb_frames(input_root: Path, shard_index: int = 0, num_shards: int = 1):
    """
    Yield (subject_dir_name, video_dir_name, frame_path) for all rgbn_* frames.

    When num_shards > 1, only every num_shards-th subject (offset by
    shard_index) is yielded, so N parallel processes each with a different
    shard_index collectively cover every subject exactly once. Sharding by
    subject (not by frame) keeps each subject's frames together, which
    doesn't matter for correctness here but makes progress easier to reason
    about per shard.
    """
    subjects = sorted([d for d in input_root.iterdir() if d.is_dir()])
    subjects = subjects[shard_index::num_shards]
    for subject_dir in subjects:
        for video_dir in sorted(subject_dir.iterdir()):
            if not video_dir.is_dir() or not video_dir.name.startswith('rgbn'):
                continue
            for frame_path in sorted(video_dir.iterdir()):
                if frame_path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                    yield subject_dir.name, video_dir.name, frame_path


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input-root', required=True, help='DLORD_rgb2nir directory')
    parser.add_argument('--output-root', required=True, help='DLORD_synthetic_nir_PID output directory')
    parser.add_argument('--output-size', type=int, default=112, help='Required output resolution (protocol expects 112x112)')
    parser.add_argument('--num-steps', type=int, default=25, help='DDIM sampling steps')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--limit', type=int, default=None, help='Optional cap on number of frames (debugging)')
    parser.add_argument('--use-ema', action='store_true', help='Use EMA weights instead of raw trained weights (only if EMA has converged)')
    parser.add_argument('--shard-index', type=int, default=0, help='This process\'s shard (0-indexed), for splitting work across parallel jobs')
    parser.add_argument('--num-shards', type=int, default=1, help='Total number of parallel shards (e.g. SLURM array size)')
    args = parser.parse_args()

    device = torch.device(args.device)
    model, config = load_model(args.checkpoint, device, use_ema=args.use_ema)
    img_size = tuple(config.data.img_size)
    mean = tuple(config.data.normalize_mean)
    std = tuple(config.data.normalize_std)

    to_tensor = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    frames = list(collect_rgb_frames(input_root, shard_index=args.shard_index, num_shards=args.num_shards))
    print(f"Shard {args.shard_index}/{args.num_shards}")
    if args.limit:
        frames = frames[:args.limit]
    print(f"Found {len(frames)} rgbn_* frames to convert.")

    batch = []
    for item in tqdm(frames, desc="Converting DLORD RGB -> synthetic NIR"):
        batch.append(item)
        if len(batch) == args.batch_size:
            _process_batch(batch, model, to_tensor, mean, std, args, input_root, output_root, device)
            batch = []
    if batch:
        _process_batch(batch, model, to_tensor, mean, std, args, input_root, output_root, device)

    print("Done.")


def _process_batch(batch, model, to_tensor, mean, std, args, input_root, output_root, device):
    imgs = []
    for _, _, frame_path in batch:
        img = Image.open(frame_path).convert('RGB')
        imgs.append(to_tensor(img))
    rgb_batch = torch.stack(imgs).to(device)

    generated = model.sample(rgb_batch, num_steps=args.num_steps, eta=0.0)
    generated = denormalize(generated, mean, std)  # [B,3,H,W] in [0,1]

    if generated.shape[-1] != args.output_size or generated.shape[-2] != args.output_size:
        generated = F.interpolate(
            generated, size=(args.output_size, args.output_size),
            mode='bilinear', align_corners=False
        )

    for (subject, video, frame_path), out_tensor in zip(batch, generated):
        out_dir = output_root / subject / video
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / frame_path.name

        arr = (out_tensor.clamp(0, 1) * 255).round().byte().cpu().permute(1, 2, 0).numpy()
        Image.fromarray(arr).save(out_path)


if __name__ == '__main__':
    main()
