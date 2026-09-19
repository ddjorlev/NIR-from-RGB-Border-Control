"""
Data handling for NIR-from-RGB Border Control Diffusion Model
Implements dataset loading and preprocessing for RANUS paired RGB-NIR images
"""

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from typing import Tuple, Optional, Dict, List
from pathlib import Path
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class RANUSDataset(Dataset):
    """
    RANUS dataset for NIR-from-RGB diffusion model training.
    
    This dataset loads paired RGB and NIR images from the RANUS border control dataset.
    Each subject has corresponding RGB and NIR images in separate folders.
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        img_size: Tuple[int, int] = (256, 256),
        normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        use_augmentation: bool = True,
        train_split: float = 0.8,
        val_split: float = 0.1,
        seed: int = 42
    ):
        """
        Initialize RANUS dataset.
        
        Args:
            data_root: Root directory of the dataset (should contain RANUS/RGB and RANUS/NIR)
            split: Dataset split ("train", "valid", "test")
            img_size: Target image size (H, W)
            normalize_mean: RGB normalization means
            normalize_std: RGB normalization standard deviations
            use_augmentation: Whether to apply data augmentation
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            seed: Random seed for reproducibility
        """
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.use_augmentation = use_augmentation and split == "train"
        
        # Paths to RGB and NIR folders
        self.rgb_root = self.data_root / "RANUS" / "RGB"
        self.nir_root = self.data_root / "RANUS" / "NIR"
        
        if not self.rgb_root.exists() or not self.nir_root.exists():
            raise ValueError(f"Dataset not found at {self.data_root}. "
                           f"Expected RANUS/RGB and RANUS/NIR folders.")
        
        # Load paired image paths
        self.image_pairs = self._load_paired_images(train_split, val_split, seed)
        
        # Setup transforms
        self.transform = self._setup_transforms(normalize_mean, normalize_std)
        self.augment_transform = self._setup_augmentation() if self.use_augmentation else None
        
        logger.info(f"Loaded {len(self.image_pairs)} image pairs from {split} split")
    
# ...existing code...
    def _load_paired_images(
        self, 
        train_split: float, 
        val_split: float, 
        seed: int
    ) -> List[Dict[str, Path]]:
        """
        Load paired RGB-NIR images and split into train/val/test.
        Improved pairing: check multiple extensions and fallback to index-based pairing if necessary.
        """
        paired_images = []
        
        # Get all subject folders (01, 02, ..., 50)
        rgb_subjects = sorted([d for d in self.rgb_root.iterdir() if d.is_dir()])
        
        for subject_folder in rgb_subjects:
            subject_id = subject_folder.name
            nir_subject_folder = self.nir_root / subject_id
            
            if not nir_subject_folder.exists():
                logger.warning(f"Missing NIR folder for subject {subject_id}, skipping...")
                continue
            
            # Collect RGB and NIR files (common extensions)
            rgb_images = sorted([p for p in subject_folder.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')])
            nir_images = sorted([p for p in nir_subject_folder.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')])
            
            if len(rgb_images) == 0 or len(nir_images) == 0:
                logger.warning(f"No images in RGB or NIR for subject {subject_id}, skipping...")
                continue
            
            # Build a name->path map for NIR to try exact match first
            nir_map = {p.name: p for p in nir_images}
            
            for idx, rgb_path in enumerate(rgb_images):
                # 1) exact filename match
                nir_path = nir_map.get(rgb_path.name, None)
                
                # 2) try same stem with any extension
                if nir_path is None:
                    candidates = list(nir_subject_folder.glob(f"{rgb_path.stem}.*"))
                    if candidates:
                        nir_path = candidates[0]
                
                # 3) fallback: pair by index if counts are equal-ish (best-effort)
                if nir_path is None:
                    if len(nir_images) == len(rgb_images):
                        nir_path = nir_images[idx]
                    elif len(nir_images) > idx:
                        nir_path = nir_images[idx]
                
                if nir_path is None or not nir_path.exists():
                    logger.warning(f"Missing NIR image for {rgb_path}, skipping...")
                    continue
                
                paired_images.append({
                    'rgb': rgb_path,
                    'nir': nir_path,
                    'subject_id': subject_id,
                    'filename': rgb_path.name
                })
        
        # Shuffle & split
        import numpy as _np
        _np.random.seed(seed)
        indices = _np.random.permutation(len(paired_images))
        
        train_end = int(len(indices) * train_split)
        val_end = int(len(indices) * (train_split + val_split))
        
        if self.split == "train":
            selected_indices = indices[:train_end]
        elif self.split == "valid":
            selected_indices = indices[train_end:val_end]
        elif self.split == "test":
            selected_indices = indices[val_end:]
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
        return [paired_images[i] for i in selected_indices]


    def _setup_transforms(
        self, 
        mean: Tuple[float, ...], 
        std: Tuple[float, ...]
    ) -> transforms.Compose:
        """Setup image preprocessing transforms."""
        return transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    def _setup_augmentation(self) -> transforms.Compose:
        """Setup data augmentation transforms."""
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))
        ])
    
    def _load_image(self, image_path: Path) -> Image.Image:
        """Load and convert image to RGB."""
        try:
            img = Image.open(image_path).convert('RGB')
            return img
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a blank image as fallback
            return Image.new('RGB', self.img_size, color=(128, 128, 128))
    
    def __len__(self) -> int:
        return len(self.image_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a paired RGB-NIR sample.
        
        Returns:
            Dictionary containing:
                - rgb: RGB image tensor [3, H, W]
                - nir: NIR image tensor [3, H, W]
                - metadata: Additional information
        """
        pair = self.image_pairs[idx]
        
        # Load images
        rgb_img = self._load_image(pair['rgb'])
        nir_img = self._load_image(pair['nir'])
        
        # Apply augmentation (same transform to both images for consistency)
        if self.augment_transform is not None:
            # Set random seed for consistent augmentation
            seed = np.random.randint(2147483647)
            
            torch.manual_seed(seed)
            np.random.seed(seed)
            rgb_img = self.augment_transform(rgb_img)
            
            torch.manual_seed(seed)
            np.random.seed(seed)
            nir_img = self.augment_transform(nir_img)
        
        # Apply normalization
        rgb_tensor = self.transform(rgb_img)
        nir_tensor = self.transform(nir_img)
        
        return {
            'rgb': rgb_tensor,
            'nir': nir_tensor,
            'metadata': {
                'subject_id': pair['subject_id'],
                'filename': pair['filename'],
                'split': self.split
            }
        }

class PairedFolderDataset(Dataset):
    """
    Generic paired RGB(VIS)-NIR dataset for folder layouts where each split
    has flat `VIS/` and `NIR/` subfolders with matching filenames (e.g. the
    Tufts Face RGB-NIR dataset provided for finetuning/training).

    Expected layout:
        data_root/<split>/VIS/*.jpg
        data_root/<split>/NIR/*.jpg
    where `<split>` is one of "train", "val". "test" falls back to "val"
    since the Tufts package does not ship a separate test split.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        img_size: Tuple[int, int] = (256, 256),
        normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        use_augmentation: bool = True,
        train_split: float = 0.8,
        val_split: float = 0.1,
        seed: int = 42
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.use_augmentation = use_augmentation and split == "train"

        folder_split = "val" if split in ("val", "valid", "test") else split
        self.vis_root = self.data_root / folder_split / "VIS"
        self.nir_root = self.data_root / folder_split / "NIR"

        if not self.vis_root.exists() or not self.nir_root.exists():
            raise ValueError(
                f"Dataset not found at {self.data_root}/{folder_split}. "
                f"Expected VIS/ and NIR/ folders with matching filenames."
            )

        self.image_pairs = self._load_paired_images()

        self.transform = self._setup_transforms(normalize_mean, normalize_std)
        self.augment_transform = self._setup_augmentation() if self.use_augmentation else None

        logger.info(f"Loaded {len(self.image_pairs)} image pairs from {split} split ({self.vis_root})")

    def _load_paired_images(self) -> List[Dict[str, Path]]:
        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        vis_images = sorted([p for p in self.vis_root.iterdir() if p.suffix.lower() in exts])
        nir_map = {p.name: p for p in self.nir_root.iterdir() if p.suffix.lower() in exts}

        paired_images = []
        for vis_path in vis_images:
            nir_path = nir_map.get(vis_path.name)
            if nir_path is None:
                candidates = list(self.nir_root.glob(f"{vis_path.stem}.*"))
                nir_path = candidates[0] if candidates else None

            if nir_path is None or not nir_path.exists():
                logger.warning(f"Missing NIR image for {vis_path}, skipping...")
                continue

            paired_images.append({
                'rgb': vis_path,
                'nir': nir_path,
                'subject_id': vis_path.stem,
                'filename': vis_path.name
            })

        return paired_images

    def _setup_transforms(self, mean, std) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def _setup_augmentation(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))
        ])

    def _load_image(self, image_path: Path) -> Image.Image:
        try:
            return Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return Image.new('RGB', self.img_size, color=(128, 128, 128))

    def __len__(self) -> int:
        return len(self.image_pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.image_pairs[idx]

        rgb_img = self._load_image(pair['rgb'])
        nir_img = self._load_image(pair['nir'])

        if self.augment_transform is not None:
            seed = np.random.randint(2147483647)

            torch.manual_seed(seed)
            np.random.seed(seed)
            rgb_img = self.augment_transform(rgb_img)

            torch.manual_seed(seed)
            np.random.seed(seed)
            nir_img = self.augment_transform(nir_img)

        rgb_tensor = self.transform(rgb_img)
        nir_tensor = self.transform(nir_img)

        return {
            'rgb': rgb_tensor,
            'nir': nir_tensor,
            'metadata': {
                'subject_id': pair['subject_id'],
                'filename': pair['filename'],
                'split': self.split
            }
        }


class PairedFolderDataModule:
    """Data module for the generic paired VIS/NIR folder dataset (e.g. Tufts)."""

    def __init__(self, config):
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self):
        data_config = self.config.data

        self.train_dataset = PairedFolderDataset(
            data_root=data_config.data_root,
            split="train",
            img_size=tuple(data_config.img_size),
            normalize_mean=tuple(data_config.normalize_mean),
            normalize_std=tuple(data_config.normalize_std),
            use_augmentation=data_config.use_augmentation,
            train_split=data_config.train_split,
            val_split=data_config.val_split,
            seed=data_config.seed
        )

        self.val_dataset = PairedFolderDataset(
            data_root=data_config.data_root,
            split="val",
            img_size=tuple(data_config.img_size),
            normalize_mean=tuple(data_config.normalize_mean),
            normalize_std=tuple(data_config.normalize_std),
            use_augmentation=False,
            train_split=data_config.train_split,
            val_split=data_config.val_split,
            seed=data_config.seed
        )

        # Tufts ships no separate test split; reuse val.
        self.test_dataset = self.val_dataset

        logger.info(f"Dataset setup complete - Train: {len(self.train_dataset)}, "
                   f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")

    def train_dataloader(self):
        return self.train_dataset

    def val_dataloader(self):
        return self.val_dataset

    def test_dataloader(self):
        return self.test_dataset


class RANUSDataModule:
    """
    Data module for RANUS dataset handling train/val/test splits.
    """
    
    def __init__(self, config):
        """
        Initialize data module.
        
        Args:
            config: Configuration object with data settings
        """
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self):
        """Setup train, validation, and test datasets."""
        data_config = self.config.data
        
        self.train_dataset = RANUSDataset(
            data_root=data_config.data_root,
            split="train",
            img_size=tuple(data_config.img_size),
            normalize_mean=tuple(data_config.normalize_mean),
            normalize_std=tuple(data_config.normalize_std),
            use_augmentation=data_config.use_augmentation,
            train_split=data_config.train_split,
            val_split=data_config.val_split,
            seed=data_config.seed
        )
        
        self.val_dataset = RANUSDataset(
            data_root=data_config.data_root,
            split="valid",
            img_size=tuple(data_config.img_size),
            normalize_mean=tuple(data_config.normalize_mean),
            normalize_std=tuple(data_config.normalize_std),
            use_augmentation=False,
            train_split=data_config.train_split,
            val_split=data_config.val_split,
            seed=data_config.seed
        )
        
        self.test_dataset = RANUSDataset(
            data_root=data_config.data_root,
            split="test",
            img_size=tuple(data_config.img_size),
            normalize_mean=tuple(data_config.normalize_mean),
            normalize_std=tuple(data_config.normalize_std),
            use_augmentation=False,
            train_split=data_config.train_split,
            val_split=data_config.val_split,
            seed=data_config.seed
        )
        
        logger.info(f"Dataset setup complete - Train: {len(self.train_dataset)}, "
                   f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
    
    def train_dataloader(self):
        """Return training dataset (dataloader creation handled by dataloader.py)."""
        return self.train_dataset
    
    def val_dataloader(self):
        """Return validation dataset."""
        return self.val_dataset
    
    def test_dataloader(self):
        """Return test dataset."""
        return self.test_dataset

# Utility functions
def denormalize_image(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> torch.Tensor:
    """Denormalize image tensor back to [0, 1] range."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    if tensor.is_cuda:
        mean = mean.cuda()
        std = std.cuda()
    
    return tensor * std + mean

def save_image_grid(images: torch.Tensor, path: str, nrow: int = 4):
    """Save batch of images as a grid."""
    # Denormalize if needed
    if images.min() < 0:
        images = denormalize_image(images)
    
    # Clamp to valid range
    images = torch.clamp(images, 0, 1)
    
    # Save grid
    vutils.save_image(images, path, nrow=nrow, normalize=False)

__all__ = [
    'RANUSDataset',
    'RANUSDataModule',
    'PairedFolderDataset',
    'PairedFolderDataModule',
    'denormalize_image',
    'save_image_grid'
]