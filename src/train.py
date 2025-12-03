"""
Training script for Physics-Informed Diffusion Model (PID)
RGB-to-NIR generation for border control
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import logging
import os
import sys
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional, Tuple
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model import PIDModel
from src.data import RANUSDataModule, denormalize_image, save_image_grid
from src.dataloader import create_dataloaders, prepare_diffusion_batch
from config.config import Config

# Optional: Weights & Biases integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, logging to tensorboard only")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EMAModel:
    """
    Exponential Moving Average of model parameters.
    Helps stabilize training and improve sample quality.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply EMA parameters to model (for inference)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class DiffusionTrainer:
    """
    Complete training pipeline for PID model.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.system.device)
        
        # Set random seeds for reproducibility
        self._set_seed(config.system.seed)
        
        # Create directories
        self._create_directories()
        
        # Initialize model
        logger.info("Initializing PID model...")
        self.model = PIDModel(config).to(self.device)
        
        # Initialize EMA
        if config.training.use_ema:
            logger.info("Initializing EMA model...")
            self.ema_model = EMAModel(self.model, decay=config.training.ema_decay)
        else:
            self.ema_model = None
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize gradient scaler for mixed precision
        self.scaler = GradScaler() if config.training.use_amp else None
        
        # Initialize data
        logger.info("Loading data...")
        self.data_module = RANUSDataModule(config)
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            config, self.data_module
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Initialize logging
        self.writer = SummaryWriter(log_dir=os.path.join(config.system.output_dir, 'logs'))
        
        if config.system.use_wandb and WANDB_AVAILABLE:
            self._init_wandb()
        
        logger.info(f"Trainer initialized. Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)
        
        # Make cudnn deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _create_directories(self):
        """Create necessary directories."""
        dirs = [
            self.config.system.checkpoint_dir,
            self.config.system.output_dir,
            self.config.system.sample_dir,
            os.path.join(self.config.system.output_dir, 'logs')
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        config = self.config.training
        
        if config.optimizer.lower() == "adamw":
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                betas=(config.adam_beta1, config.adam_beta2),
                weight_decay=config.weight_decay
            )
        elif config.optimizer.lower() == "adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=config.learning_rate,
                betas=(config.adam_beta1, config.adam_beta2),
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")
        
        logger.info(f"Created optimizer: {config.optimizer}")
        return optimizer
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        config = self.config.training
        
        if config.lr_scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.num_epochs,
                eta_min=config.lr_min
            )
        elif config.lr_scheduler == "step":
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.num_epochs // 3,
                gamma=0.1
            )
        elif config.lr_scheduler == "constant":
            scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {config.lr_scheduler}")
        
        if scheduler:
            logger.info(f"Created scheduler: {config.lr_scheduler}")
        
        return scheduler
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        wandb.init(
            project=self.config.system.wandb_project,
            entity=self.config.system.wandb_entity,
            config={
                'model': {
                    'type': 'PID',
                    'channels': self.config.model.model_channels,
                    'timesteps': self.config.model.num_timesteps
                },
                'training': {
                    'batch_size': self.config.training.batch_size,
                    'learning_rate': self.config.training.learning_rate,
                    'num_epochs': self.config.training.num_epochs
                },
                'data': {
                    'img_size': self.config.data.img_size
                }
            }
        )
        wandb.watch(self.model, log='all', log_freq=100)
        logger.info("Initialized Weights & Biases logging")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {
            'loss': 0.0,
            'mse': 0.0
        }
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch}/{self.config.training.num_epochs}"
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Prepare batch
            batch = prepare_diffusion_batch(batch, self.device)
            rgb_images = batch['rgb']
            nir_images = batch['nir']
            
            # Forward pass with mixed precision
            if self.config.training.use_amp:
                with autocast():
                    loss_dict = self.model.compute_loss(
                        nir_images, rgb_images, return_dict=True
                    )
                    loss = loss_dict['loss']
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.training.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip
                    )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_dict = self.model.compute_loss(
                    nir_images, rgb_images, return_dict=True
                )
                loss = loss_dict['loss']
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.training.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip
                    )
                
                # Optimizer step
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            # Update EMA
            if self.ema_model is not None:
                self.ema_model.update()
            
            # Update metrics
            epoch_metrics['loss'] += loss.item()
            epoch_metrics['mse'] += loss_dict['mse']
            
            # Logging
            if self.global_step % self.config.system.log_every_n_steps == 0:
                self._log_step(loss_dict, batch_idx)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            self.global_step += 1
        
        # Average metrics
        num_batches = len(self.train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        val_metrics = {
            'loss': 0.0,
            'mse': 0.0
        }
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        for batch in pbar:
            batch = prepare_diffusion_batch(batch, self.device)
            rgb_images = batch['rgb']
            nir_images = batch['nir']
            
            # Compute loss
            loss_dict = self.model.compute_loss(
                nir_images, rgb_images, return_dict=True
            )
            
            val_metrics['loss'] += loss_dict['loss'].item()
            val_metrics['mse'] += loss_dict['mse']
        
        # Average metrics
        num_batches = len(self.val_loader)
        for key in val_metrics:
            val_metrics[key] /= num_batches
        
        return val_metrics
    
    @torch.no_grad()
    def generate_samples(self, num_samples: int = 8):
        """Generate sample images for visualization."""
        self.model.eval()
        
        # Apply EMA if available
        if self.ema_model is not None:
            self.ema_model.apply_shadow()
        
        # Get a batch from validation set
        val_batch = next(iter(self.val_loader))
        val_batch = prepare_diffusion_batch(val_batch, self.device)
        
        rgb_images = val_batch['rgb'][:num_samples]
        nir_images = val_batch['nir'][:num_samples]
        
        # Generate NIR images from RGB
        generated_nir = self.model.sample(
            rgb_images,
            num_steps=50,  # Faster sampling
            eta=0.0  # Deterministic
        )
        
        # Denormalize for visualization
        rgb_vis = denormalize_image(rgb_images)
        nir_vis = denormalize_image(nir_images)
        gen_vis = denormalize_image(generated_nir)
        
        # Create comparison grid
        comparison = torch.cat([rgb_vis, nir_vis, gen_vis], dim=0)
        
        # Save image grid
        save_path = os.path.join(
            self.config.system.sample_dir,
            f'samples_epoch_{self.current_epoch:04d}.png'
        )
        save_image_grid(comparison, save_path, nrow=num_samples)
        
        # Log to tensorboard
        self.writer.add_images(
            'samples/rgb', rgb_vis, self.current_epoch
        )
        self.writer.add_images(
            'samples/nir_real', nir_vis, self.current_epoch
        )
        self.writer.add_images(
            'samples/nir_generated', gen_vis, self.current_epoch
        )
        
        # Log to wandb
        if self.config.system.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'samples': [wandb.Image(save_path, caption=f'Epoch {self.current_epoch}')]
            }, step=self.global_step)
        
        # Restore original parameters
        if self.ema_model is not None:
            self.ema_model.restore()
        
        logger.info(f"Generated samples saved to {save_path}")
    
    def _log_step(self, metrics: Dict, batch_idx: int):
        """Log training step metrics."""
        # Tensorboard
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'train/{key}', value, self.global_step)
        
        self.writer.add_scalar(
            'train/learning_rate',
            self.optimizer.param_groups[0]['lr'],
            self.global_step
        )
        
        # Wandb
        if self.config.system.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'train/loss': metrics['loss'],
                'train/mse': metrics['mse'],
                'train/learning_rate': self.optimizer.param_groups[0]['lr']
            }, step=self.global_step)
    
    def _log_epoch(self, train_metrics: Dict, val_metrics: Dict):
        """Log epoch metrics."""
        logger.info(f"\nEpoch {self.current_epoch} Summary:")
        logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")
        logger.info(f"  Val Loss: {val_metrics['loss']:.4f}")
        
        # Tensorboard
        for key, value in train_metrics.items():
            self.writer.add_scalar(f'epoch/train_{key}', value, self.current_epoch)
        
        for key, value in val_metrics.items():
            self.writer.add_scalar(f'epoch/val_{key}', value, self.current_epoch)
        
        # Wandb
        if self.config.system.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'epoch/train_loss': train_metrics['loss'],
                'epoch/val_loss': val_metrics['loss'],
                'epoch': self.current_epoch
            }, step=self.global_step)
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.ema_model is not None:
            checkpoint['ema_shadow'] = self.ema_model.shadow
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save latest checkpoint
        latest_path = os.path.join(
            self.config.system.checkpoint_dir,
            'checkpoint_latest.pt'
        )
        torch.save(checkpoint, latest_path)
        
        # Save periodic checkpoint
        if self.current_epoch % self.config.system.save_every_n_epochs == 0:
            epoch_path = os.path.join(
                self.config.system.checkpoint_dir,
                f'checkpoint_epoch_{self.current_epoch:04d}.pt'
            )
            torch.save(checkpoint, epoch_path)
            logger.info(f"Saved checkpoint: {epoch_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(
                self.config.system.checkpoint_dir,
                'checkpoint_best.pt'
            )
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint: {best_path}")
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent N."""
        checkpoint_dir = Path(self.config.system.checkpoint_dir)
        checkpoints = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        
        # Keep only the most recent N checkpoints
        if len(checkpoints) > self.config.system.keep_n_checkpoints:
            for ckpt in checkpoints[:-self.config.system.keep_n_checkpoints]:
                ckpt.unlink()
                logger.info(f"Removed old checkpoint: {ckpt}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.ema_model is not None and 'ema_shadow' in checkpoint:
            self.ema_model.shadow = checkpoint['ema_shadow']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Resumed from epoch {self.current_epoch}")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Training for {self.config.training.num_epochs} epochs")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        
        try:
            for epoch in range(self.current_epoch, self.config.training.num_epochs):
                self.current_epoch = epoch
                
                # Train epoch
                train_metrics = self.train_epoch()
                
                # Validation
                if epoch % self.config.training.val_every_n_epochs == 0:
                    val_metrics = self.validate()
                    
                    # Log metrics
                    self._log_epoch(train_metrics, val_metrics)
                    
                    # Save checkpoint
                    is_best = val_metrics['loss'] < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_metrics['loss']
                    
                    self.save_checkpoint(is_best=is_best)
                else:
                    # Just log training metrics
                    logger.info(f"Epoch {epoch}: Train Loss = {train_metrics['loss']:.4f}")
                    self.save_checkpoint(is_best=False)
                
                # Generate samples
                if epoch % self.config.training.save_samples_every_n_epochs == 0:
                    self.generate_samples(
                        num_samples=self.config.training.num_sample_images
                    )
                
                # Update learning rate
                if self.scheduler is not None:
                    self.scheduler.step()
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self.save_checkpoint()
        
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        
        finally:
            self.writer.close()
            if self.config.system.use_wandb and WANDB_AVAILABLE:
                wandb.finish()
        
        logger.info("Training completed!")


def main():
    """Main entry point."""
    # Load configuration
    from config.config import get_default_config
    config = get_default_config()
    
    # Override config for quick testing if needed
    # config = get_config_for_quick_test()
    
    # Create trainer
    trainer = DiffusionTrainer(config)
    
    # Check for existing checkpoint to resume from
    latest_checkpoint = os.path.join(
        config.system.checkpoint_dir,
        'checkpoint_latest.pt'
    )
    
    if os.path.exists(latest_checkpoint):
        response = input(f"Found checkpoint at {latest_checkpoint}. Resume? (y/n): ")
        if response.lower() == 'y':
            trainer.load_checkpoint(latest_checkpoint)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()