"""
DataLoader utilities for NIR-from-RGB Border Control Diffusion Model
Implements advanced data loading, augmentation, and batch processing for RANUS dataset
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from typing import Iterator, Optional, List, Dict, Any, Tuple
import numpy as np
import random
from PIL import Image, ImageFilter
import logging

logger = logging.getLogger(__name__)

class DiffusionDataLoader:
    """
    Advanced DataLoader for diffusion model training with specialized augmentations
    and batch processing optimized for NIR-RGB image pairs.
    """
    
    def __init__(self, dataset, config, is_training: bool = True):
        """
        Initialize diffusion dataloader.
        
        Args:
            dataset: RANUS dataset instance
            config: Configuration object
            is_training: Whether this is for training (affects augmentations)
        """
        self.dataset = dataset
        self.config = config
        self.is_training = is_training
        
        # Setup dataloader
        self.dataloader = self._create_dataloader()
        
        # Advanced augmentation pipeline
        if is_training:
            self.advanced_aug = AdvancedAugmentation(config)
        else:
            self.advanced_aug = None
    
    def _create_dataloader(self) -> DataLoader:
        """Create the underlying PyTorch DataLoader."""
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.config.training.batch_size,
            shuffle=self.is_training,
            num_workers=self.config.system.num_workers,
            pin_memory=self.config.system.pin_memory,
            drop_last=self.is_training,
            persistent_workers=self.config.system.num_workers > 0,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for batching with advanced augmentations.
        
        Args:
            batch: List of dataset items
            
        Returns:
            Batched dictionary with augmented data
        """
        # Stack basic tensors
        rgb_images = torch.stack([item['rgb'] for item in batch])
        nir_images = torch.stack([item['nir'] for item in batch])
        
        # Apply advanced augmentations if training
        if self.is_training and self.advanced_aug is not None:
            rgb_images, nir_images = self.advanced_aug(rgb_images, nir_images)
        
        # Create metadata batch
        metadata = {
            'subject_ids': [item['metadata']['subject_id'] for item in batch],
            'filenames': [item['metadata']['filename'] for item in batch],
            'splits': [item['metadata']['split'] for item in batch]
        }
        
        return {
            'rgb': rgb_images,
            'nir': nir_images,
            'metadata': metadata
        }
    
    def __iter__(self):
        """Iterator interface."""
        return iter(self.dataloader)
    
    def __len__(self):
        """Length interface."""
        return len(self.dataloader)

class AdvancedAugmentation:
    """
    Advanced augmentation pipeline specifically designed for diffusion model training.
    Includes spatial augmentations, color transformations, and noise injection.
    """
    
    def __init__(self, config):
        """Initialize augmentation pipeline."""
        self.config = config
        self.rng = np.random.default_rng()
        
        # Augmentation probabilities
        self.aug_probs = {
            'mixup': 0.2,
            'cutmix': 0.2,
            'noise_injection': 0.3,
            'blur': 0.15,
            'brightness': 0.4,
            'contrast': 0.4,
        }
    
    def __call__(self, rgb_batch: torch.Tensor, nir_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply augmentations to batch of RGB-NIR pairs.
        
        Args:
            rgb_batch: Batch of RGB images [B, 3, H, W]
            nir_batch: Batch of NIR images [B, 3, H, W]
            
        Returns:
            Augmented RGB and NIR batches
        """
        batch_size = rgb_batch.size(0)
        
        # Apply per-sample augmentations
        for i in range(batch_size):
            # Individual augmentations
            if random.random() < self.aug_probs['noise_injection']:
                rgb_batch[i] = self._noise_injection(rgb_batch[i])
                nir_batch[i] = self._noise_injection(nir_batch[i], noise_level=0.01)
            
            if random.random() < self.aug_probs['blur']:
                if random.random() < 0.5:
                    rgb_batch[i] = self._gaussian_blur(rgb_batch[i])
            
            # Color augmentations (only for RGB)
            if random.random() < self.aug_probs['brightness']:
                rgb_batch[i] = self._brightness_adjustment(rgb_batch[i])
            
            if random.random() < self.aug_probs['contrast']:
                rgb_batch[i] = self._contrast_adjustment(rgb_batch[i])
        
        # Batch-level augmentations
        if batch_size > 1:
            if random.random() < self.aug_probs['mixup']:
                rgb_batch, nir_batch = self._mixup(rgb_batch, nir_batch)
            elif random.random() < self.aug_probs['cutmix']:
                rgb_batch, nir_batch = self._cutmix(rgb_batch, nir_batch)
        
        return rgb_batch, nir_batch
    
    def _noise_injection(self, img: torch.Tensor, noise_level: float = 0.02) -> torch.Tensor:
        """Add Gaussian noise to image."""
        noise = torch.randn_like(img) * noise_level
        return torch.clamp(img + noise, -3, 3)  # Allow some overflow for normalized images
    
    def _gaussian_blur(self, img: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        """Apply Gaussian blur using conv2d."""
        # Create Gaussian kernel
        sigma = random.uniform(0.5, 1.5)
        kernel = self._get_gaussian_kernel(kernel_size, sigma)
        kernel = kernel.to(img.device).type(img.dtype)
        
        # Apply to each channel
        img_blurred = F.conv2d(
            img.unsqueeze(0), 
            kernel.repeat(3, 1, 1, 1),
            padding=kernel_size//2,
            groups=3
        ).squeeze(0)
        
        return img_blurred
    
    def _get_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """Generate Gaussian kernel."""
        x = torch.arange(kernel_size).float() - kernel_size // 2
        gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        kernel = gauss.unsqueeze(0) * gauss.unsqueeze(1)
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)
    
    def _brightness_adjustment(self, img: torch.Tensor, factor_range: Tuple[float, float] = (0.8, 1.2)) -> torch.Tensor:
        """Adjust brightness randomly."""
        factor = random.uniform(*factor_range)
        return torch.clamp(img * factor, -3, 3)
    
    def _contrast_adjustment(self, img: torch.Tensor, factor_range: Tuple[float, float] = (0.8, 1.2)) -> torch.Tensor:
        """Adjust contrast randomly."""
        factor = random.uniform(*factor_range)
        mean = img.mean(dim=(1, 2), keepdim=True)
        return torch.clamp((img - mean) * factor + mean, -3, 3)
    
    def _mixup(self, rgb_batch: torch.Tensor, nir_batch: torch.Tensor, alpha: float = 0.4) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply MixUp augmentation."""
        batch_size = rgb_batch.size(0)
        
        # Sample mixing coefficient
        lam = self.rng.beta(alpha, alpha)
        
        # Shuffle indices
        indices = torch.randperm(batch_size)
        
        # Mix images
        rgb_mixed = lam * rgb_batch + (1 - lam) * rgb_batch[indices]
        nir_mixed = lam * nir_batch + (1 - lam) * nir_batch[indices]
        
        return rgb_mixed, nir_mixed
    
    def _cutmix(self, rgb_batch: torch.Tensor, nir_batch: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply CutMix augmentation."""
        batch_size = rgb_batch.size(0)
        _, _, H, W = rgb_batch.shape
        
        # Sample mixing coefficient
        lam = self.rng.beta(alpha, alpha)
        
        # Calculate cut region
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Random center
        cx = self.rng.integers(0, W)
        cy = self.rng.integers(0, H)
        
        # Calculate bounds
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Shuffle indices
        indices = torch.randperm(batch_size)
        
        # Apply cut and mix
        rgb_batch[:, :, bby1:bby2, bbx1:bbx2] = rgb_batch[indices, :, bby1:bby2, bbx1:bbx2]
        nir_batch[:, :, bby1:bby2, bbx1:bbx2] = nir_batch[indices, :, bby1:bby2, bbx1:bbx2]
        
        return rgb_batch, nir_batch

class ValidationDataLoader:
    """
    Specialized dataloader for validation with deterministic ordering and no augmentations.
    """
    
    def __init__(self, dataset, config):
        """Initialize validation dataloader."""
        self.dataset = dataset
        self.config = config
        
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.system.num_workers,
            pin_memory=config.system.pin_memory,
            drop_last=False,
            persistent_workers=config.system.num_workers > 0
        )
    
    def __iter__(self):
        """Iterator interface."""
        return iter(self.dataloader)
    
    def __len__(self):
        """Length interface."""
        return len(self.dataloader)

def create_dataloaders(config, data_module):
    """
    Factory function to create train and validation dataloaders.
    
    Args:
        config: Configuration object
        data_module: RANUSDataModule instance
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Setup datasets
    data_module.setup()
    
    # Create training dataloader
    train_loader = DiffusionDataLoader(
        dataset=data_module.train_dataloader(),
        config=config,
        is_training=True
    )
    
    # Create validation dataloader
    val_loader = ValidationDataLoader(
        dataset=data_module.val_dataloader(),
        config=config
    )
    
    # Create test dataloader
    test_loader = ValidationDataLoader(
        dataset=data_module.test_dataloader(),
        config=config
    )
    
    logger.info(f"Created dataloaders - Train: {len(train_loader)}, "
               f"Val: {len(val_loader)}, Test: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader

# Utility functions for data processing
def prepare_diffusion_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Prepare batch for diffusion model training.
    
    Args:
        batch: Batch from dataloader
        device: Target device
        
    Returns:
        Prepared batch with proper device placement
    """
    return {
        'rgb': batch['rgb'].to(device, non_blocking=True),
        'nir': batch['nir'].to(device, non_blocking=True),
        'metadata': batch['metadata']  # Keep metadata on CPU
    }

def compute_batch_statistics(batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Compute statistics for a batch (useful for monitoring).
    
    Args:
        batch: Batch from dataloader
        
    Returns:
        Dictionary of batch statistics
    """
    rgb = batch['rgb']
    nir = batch['nir']
    
    stats = {
        'batch_size': rgb.size(0),
        'rgb_mean': rgb.mean().item(),
        'rgb_std': rgb.std().item(),
        'rgb_min': rgb.min().item(),
        'rgb_max': rgb.max().item(),
        'nir_mean': nir.mean().item(),
        'nir_std': nir.std().item(),
        'nir_min': nir.min().item(),
        'nir_max': nir.max().item(),
    }
    
    return stats

__all__ = [
    'DiffusionDataLoader',
    'AdvancedAugmentation',
    'ValidationDataLoader',
    'create_dataloaders',
    'prepare_diffusion_batch',
    'compute_batch_statistics'
]