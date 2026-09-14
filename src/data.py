"""
Data handling for NIR-from-RGB face translation (PID model).

Loads paired RGB(VIS)-NIR face crops. Two layouts are supported:

  * "tufts" (default): the aligned Tufts RGB-NIR faces shipped in the eval package.
        data_root/
            train/VIS/*.jpg   train/NIR/*.jpg   (paired by identical filename)
            val/VIS/*.jpg     val/NIR/*.jpg
  * "ranus" (legacy): data_root/RANUS/RGB/<subject>/* and RANUS/NIR/<subject>/*

Images are resized to img_size (112x112 for the FR-verification eval), scaled to
[0, 1], then normalized to [-1, 1] (the diffusion convention). NIR is loaded as a
single luminance band and, when nir_as_gray is set, replicated across 3 channels so
it matches the 3-channel FR backbone and the eval's PNG save format.
"""

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
from PIL import Image
from typing import Tuple, Optional, Dict, List
from pathlib import Path
import logging
import random
import numpy as np

logger = logging.getLogger(__name__)

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class PairedRGBNIRDataset(Dataset):
    """Paired RGB-NIR face dataset for RGB->NIR diffusion training."""

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        layout: str = "tufts",
        img_size: Tuple[int, int] = (112, 112),
        normalize_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        normalize_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        nir_as_gray: bool = True,
        use_augmentation: bool = True,
        train_split: float = 0.8,
        val_split: float = 0.1,
        seed: int = 42,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.layout = layout
        self.img_size = tuple(img_size)
        self.mean = normalize_mean
        self.std = normalize_std
        self.nir_as_gray = nir_as_gray
        self.use_augmentation = use_augmentation and split == "train"

        if layout == "tufts":
            self.image_pairs = self._load_tufts()
        elif layout == "ranus":
            self.image_pairs = self._load_ranus(train_split, val_split, seed)
        else:
            raise ValueError(f"Unknown dataset layout: {layout}")

        if len(self.image_pairs) == 0:
            raise ValueError(
                f"No paired RGB-NIR images found for split='{split}' under {self.data_root} "
                f"(layout='{layout}'). Check the dataset path/structure."
            )

        logger.info(
            f"[{layout}] Loaded {len(self.image_pairs)} RGB-NIR pairs for '{split}' split"
        )

    # ------------------------------------------------------------------ #
    #  Pair discovery
    # ------------------------------------------------------------------ #
    def _split_dirs_tufts(self) -> Tuple[Path, Path]:
        """Map the requested split onto Tufts' train/ and val/ folders."""
        sub = "train" if self.split == "train" else "val"  # valid/test -> val
        return self.data_root / sub / "VIS", self.data_root / sub / "NIR"

    def _load_tufts(self) -> List[Dict]:
        vis_dir, nir_dir = self._split_dirs_tufts()
        if not vis_dir.exists() or not nir_dir.exists():
            raise ValueError(
                f"Tufts dataset not found. Expected {vis_dir} and {nir_dir}."
            )

        nir_map = {p.name: p for p in nir_dir.iterdir() if p.suffix.lower() in _IMG_EXTS}
        pairs = []
        for vis_path in sorted(vis_dir.iterdir()):
            if vis_path.suffix.lower() not in _IMG_EXTS:
                continue
            # exact filename match, then same-stem fallback
            nir_path = nir_map.get(vis_path.name)
            if nir_path is None:
                cand = list(nir_dir.glob(f"{vis_path.stem}.*"))
                nir_path = cand[0] if cand else None
            if nir_path is None:
                logger.warning(f"No NIR match for {vis_path.name}, skipping")
                continue
            pairs.append({"rgb": vis_path, "nir": nir_path, "id": vis_path.stem})
        return pairs

    def _load_ranus(self, train_split: float, val_split: float, seed: int) -> List[Dict]:
        rgb_root = self.data_root / "RANUS" / "RGB"
        nir_root = self.data_root / "RANUS" / "NIR"
        if not rgb_root.exists() or not nir_root.exists():
            raise ValueError(
                f"RANUS dataset not found. Expected {rgb_root} and {nir_root}."
            )

        pairs = []
        for subject in sorted(d for d in rgb_root.iterdir() if d.is_dir()):
            nir_subject = nir_root / subject.name
            if not nir_subject.exists():
                continue
            nir_map = {p.name: p for p in nir_subject.iterdir() if p.suffix.lower() in _IMG_EXTS}
            for rgb_path in sorted(p for p in subject.iterdir() if p.suffix.lower() in _IMG_EXTS):
                nir_path = nir_map.get(rgb_path.name)
                if nir_path is None:
                    cand = list(nir_subject.glob(f"{rgb_path.stem}.*"))
                    nir_path = cand[0] if cand else None
                if nir_path is None:
                    continue
                pairs.append({"rgb": rgb_path, "nir": nir_path, "id": f"{subject.name}/{rgb_path.stem}"})

        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(pairs))
        train_end = int(len(idx) * train_split)
        val_end = int(len(idx) * (train_split + val_split))
        if self.split == "train":
            sel = idx[:train_end]
        elif self.split in ("valid", "val"):
            sel = idx[train_end:val_end]
        else:  # test
            sel = idx[val_end:]
        return [pairs[i] for i in sel]

    # ------------------------------------------------------------------ #
    #  Loading / transforms
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.image_pairs)

    def _to_tensor_norm(self, img: Image.Image) -> torch.Tensor:
        t = TF.to_tensor(img)  # [0, 1], shape [C, H, W]
        t = TF.normalize(t, mean=self.mean[: t.shape[0]], std=self.std[: t.shape[0]])
        return t

    def _paired_augment(self, rgb: Image.Image, nir: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """Geometric transforms are applied identically to both modalities;
        photometric jitter is applied to RGB only."""
        # Random horizontal flip (shared)
        if random.random() < 0.5:
            rgb, nir = TF.hflip(rgb), TF.hflip(nir)

        # Small shared affine (rotation + translation + scale)
        angle = random.uniform(-10, 10)
        max_dx = 0.08 * self.img_size[1]
        max_dy = 0.08 * self.img_size[0]
        translate = (random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy))
        scale = random.uniform(0.9, 1.1)
        rgb = TF.affine(rgb, angle=angle, translate=translate, scale=scale, shear=[0.0, 0.0])
        nir = TF.affine(nir, angle=angle, translate=translate, scale=scale, shear=[0.0, 0.0])

        # Photometric jitter on RGB only (NIR has no meaningful color/hue)
        if random.random() < 0.5:
            rgb = TF.adjust_brightness(rgb, random.uniform(0.8, 1.2))
            rgb = TF.adjust_contrast(rgb, random.uniform(0.8, 1.2))
            rgb = TF.adjust_saturation(rgb, random.uniform(0.8, 1.2))

        return rgb, nir

    def __getitem__(self, idx: int) -> Dict:
        pair = self.image_pairs[idx]

        rgb_img = Image.open(pair["rgb"]).convert("RGB").resize(
            (self.img_size[1], self.img_size[0]), Image.BILINEAR
        )
        # NIR: single luminance band
        nir_img = Image.open(pair["nir"]).convert("L").resize(
            (self.img_size[1], self.img_size[0]), Image.BILINEAR
        )

        if self.use_augmentation:
            rgb_img, nir_img = self._paired_augment(rgb_img, nir_img)

        rgb_tensor = self._to_tensor_norm(rgb_img)  # [3, H, W] in [-1, 1]
        nir_tensor = self._to_tensor_norm(nir_img)  # [1, H, W] in [-1, 1]

        if self.nir_as_gray and nir_tensor.shape[0] == 1:
            nir_tensor = nir_tensor.repeat(3, 1, 1)  # replicate luminance to 3 channels

        return {
            "rgb": rgb_tensor,
            "nir": nir_tensor,
            "metadata": {"subject_id": pair["id"], "filename": Path(pair["rgb"]).name, "split": self.split},
        }


# Backward-compatible alias (train.py / older code imports RANUSDataset)
RANUSDataset = PairedRGBNIRDataset


class RANUSDataModule:
    """Builds train/val/test datasets from config."""

    def __init__(self, config):
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _make(self, split: str, augment: bool) -> PairedRGBNIRDataset:
        dc = self.config.data
        return PairedRGBNIRDataset(
            data_root=dc.data_root,
            split=split,
            layout=getattr(dc, "dataset_layout", "tufts"),
            img_size=tuple(dc.img_size),
            normalize_mean=tuple(dc.normalize_mean),
            normalize_std=tuple(dc.normalize_std),
            nir_as_gray=getattr(dc, "nir_as_gray", True),
            use_augmentation=augment and dc.use_augmentation,
            train_split=dc.train_split,
            val_split=dc.val_split,
            seed=dc.seed,
        )

    def setup(self):
        self.train_dataset = self._make("train", augment=True)
        self.val_dataset = self._make("valid", augment=False)
        # Tufts has no separate test split -> reuse val; RANUS carves a real test split.
        self.test_dataset = self._make("test", augment=False)
        logger.info(
            f"Dataset setup complete - Train: {len(self.train_dataset)}, "
            f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}"
        )

    def train_dataloader(self):
        return self.train_dataset

    def val_dataloader(self):
        return self.val_dataset

    def test_dataloader(self):
        return self.test_dataset


# ---------------------------------------------------------------------- #
#  Utilities
# ---------------------------------------------------------------------- #
def denormalize_image(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> torch.Tensor:
    """Undo [-1, 1] normalization back to [0, 1]."""
    c = tensor.shape[-3]
    mean_t = torch.tensor(mean[:c], device=tensor.device).view(c, 1, 1)
    std_t = torch.tensor(std[:c], device=tensor.device).view(c, 1, 1)
    return tensor * std_t + mean_t


def save_image_grid(images: torch.Tensor, path: str, nrow: int = 4):
    """Save a batch of images as a grid (denormalizing if needed)."""
    if images.min() < 0:
        images = denormalize_image(images)
    images = torch.clamp(images, 0, 1)
    vutils.save_image(images, path, nrow=nrow, normalize=False)


__all__ = [
    "PairedRGBNIRDataset",
    "RANUSDataset",
    "RANUSDataModule",
    "denormalize_image",
    "save_image_grid",
]
