"""
Configuration management for NIR-from-RGB Border Control Diffusion Model
Uses dataclasses for type-safe, hierarchical configuration
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import torch
import os

@dataclass
class DataConfig:
    """Data-related configuration"""
    data_root: str = "./data"
    dataset_type: str = "ranus"  # "ranus" or "paired_folder" (e.g. Tufts VIS/NIR layout)
    img_size: List[int] = field(default_factory=lambda: [256, 256])
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    use_augmentation: bool = True
    train_split: float = 0.8
    val_split: float = 0.1
    seed: int = 42

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # UNet architecture
    in_channels: int = 3  # RGB input
    out_channels: int = 3  # NIR output
    model_channels: int = 128
    num_res_blocks: int = 2
    attention_resolutions: List[int] = field(default_factory=lambda: [16, 8])
    channel_mult: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    dropout: float = 0.1
    use_checkpoint: bool = False

    # Physics-informed settings
    physics_prior_type: str = "learned"   # "learned" or "planck"
    physics_loss_weight: float = 1.0      # weight for physics loss term

    # Diffusion parameters
    num_timesteps: int = 1000
    beta_schedule: str = "linear"  # "linear" or "cosine"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    
    # Conditioning
    use_conditional: bool = True
    condition_on_rgb: bool = True

    # Multi-scale RGB conditioning (FiLM at every U-Net resolution, on top
    # of the input-level channel-concat)
    use_film_conditioning: bool = False

    # Optional extra loss terms (see PIDModel.compute_loss)
    use_channel_consistency_loss: bool = False
    channel_consistency_weight: float = 0.1
    use_perceptual_loss: bool = False
    perceptual_loss_weight: float = 0.1

@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    
    # Optimizer
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    
    # Learning rate schedule
    lr_scheduler: str = "cosine"  # "cosine", "step", or "constant"
    lr_warmup_steps: int = 1000
    lr_min: float = 1e-6
    
    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9999
    
    # Mixed precision
    use_amp: bool = True
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 1
    
    # Validation
    val_every_n_epochs: int = 5
    save_samples_every_n_epochs: int = 5
    num_sample_images: int = 8

@dataclass
class SystemConfig:
    """System and logging configuration"""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 42
    
    # Logging
    use_wandb: bool = False
    wandb_project: str = "nir-from-rgb-diffusion"
    wandb_entity: Optional[str] = None
    log_every_n_steps: int = 10
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every_n_epochs: int = 10
    keep_n_checkpoints: int = 3
    
    # Output
    output_dir: str = "./output"
    sample_dir: str = "./samples"

@dataclass
class Config:
    """Main configuration class combining all sub-configs"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        # Ensure directories exist
        os.makedirs(self.system.checkpoint_dir, exist_ok=True)
        os.makedirs(self.system.output_dir, exist_ok=True)
        os.makedirs(self.system.sample_dir, exist_ok=True)
        
        # Validate data splits
        assert 0 < self.data.train_split < 1, "train_split must be between 0 and 1"
        assert 0 < self.data.val_split < 1, "val_split must be between 0 and 1"
        assert self.data.train_split + self.data.val_split < 1, "train_split + val_split must be < 1"
        
        # Adjust settings based on device
        if self.system.device == "cpu":
            self.training.use_amp = False
            self.system.pin_memory = False
            if self.training.batch_size > 4:
                print(f"Warning: Reducing batch size from {self.training.batch_size} to 4 for CPU training")
                self.training.batch_size = 4
    
    def to_dict(self):
        """Convert config to dictionary for logging."""
        return {
            'data': {
                'data_root': self.data.data_root,
                'img_size': self.data.img_size,
                'train_split': self.data.train_split,
                'val_split': self.data.val_split
            },
            'model': {
                'model_channels': self.model.model_channels,
                'num_timesteps': self.model.num_timesteps,
                'beta_schedule': self.model.beta_schedule,
                'condition_on_rgb': self.model.condition_on_rgb
            },
            'training': {
                'batch_size': self.training.batch_size,
                'num_epochs': self.training.num_epochs,
                'learning_rate': self.training.learning_rate,
                'optimizer': self.training.optimizer,
                'lr_scheduler': self.training.lr_scheduler,
                'use_ema': self.training.use_ema,
                'use_amp': self.training.use_amp
            },
            'system': {
                'device': self.system.device,
                'num_workers': self.system.num_workers
            }
        }
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON file into nested dataclasses."""
        import json
        from dataclasses import is_dataclass

        def _update_dataclass(obj, data: dict):
            """Recursively update dataclass instance fields from dict."""
            for key, val in data.items():
                if not hasattr(obj, key):
                    # ignore unknown keys
                    continue
                current = getattr(obj, key)
                if is_dataclass(current) and isinstance(val, dict):
                    _update_dataclass(current, val)
                else:
                    setattr(obj, key, val)

        with open(path, 'r') as f:
            config_dict = json.load(f)

        config = cls()
        if isinstance(config_dict, dict):
            _update_dataclass(config, config_dict)
        return config

def get_default_config() -> Config:
    """Get default configuration for full training."""
    return Config()

def get_config_for_quick_test() -> Config:
    """Get configuration optimized for quick testing."""
    config = Config()
    
    # Reduce model size
    config.model.model_channels = 64
    config.model.num_timesteps = 100
    config.model.channel_mult = [1, 2, 4]
    
    # Reduce training time
    config.training.batch_size = 4
    config.training.num_epochs = 10
    config.training.val_every_n_epochs = 2
    config.training.save_samples_every_n_epochs = 2
    
    # Reduce workers
    config.system.num_workers = 2
    
    return config

def get_config_for_cpu() -> Config:
    """Get configuration optimized for CPU training."""
    config = Config()
    
    # CPU-specific settings
    config.system.device = "cpu"
    config.training.use_amp = False
    config.system.pin_memory = False
    config.system.num_workers = 2
    
    # Reduce model/batch size
    config.model.model_channels = 64
    config.model.num_timesteps = 200
    config.training.batch_size = 2
    
    return config

def get_config_for_tufts_faces() -> Config:
    """
    Config for training PID on the Tufts Face RGB-NIR paired dataset
    (aligned face crops, provided for finetuning/training the RGB2NIR generator
    used in the DLORD Protocol A face-verification evaluation).
    """
    config = Config()

    config.data.data_root = "../archive_extract/dlord_rgbnir_verification/tufts_faces_rgb_nir"
    config.data.dataset_type = "paired_folder"
    config.data.img_size = [128, 128]

    # NIR here is reflected-light (700-1000nm), not thermal emission, so the
    # "learned" RGB->NIR prior head is used instead of the Planck/blackbody one.
    config.model.physics_prior_type = "learned"
    config.model.physics_loss_weight = 0.5
    config.model.model_channels = 128
    config.model.channel_mult = [1, 2, 2, 4]
    config.model.attention_resolutions = [16, 8]

    config.training.batch_size = 32
    config.training.num_epochs = 300
    config.training.val_every_n_epochs = 5
    config.training.save_samples_every_n_epochs = 5
    config.training.save_every_n_epochs = 10
    # EMA decay=0.9999 (the dataclass default) needs ~10k steps to converge;
    # this run only has ~300 epochs x ~61 iters/epoch = ~18.3k steps total,
    # so a slower decay would leave EMA weights still dominated by the random
    # init for most of training (verified: raw weights at epoch 165 produced
    # real faces, default-decay EMA weights were still noise). Use a decay
    # whose ~1/(1-decay) time constant fits this run's step budget.
    config.training.ema_decay = 0.995

    config.system.checkpoint_dir = "./checkpoints_tufts"
    config.system.output_dir = "./output_tufts"
    config.system.sample_dir = "./samples_tufts"
    config.system.use_wandb = False

    return config

def get_config_for_tufts_faces_v2() -> Config:
    """
    v2 of the Tufts training config, addressing two issues found while
    auditing the v1 run:
      1. attention_resolutions was matched against level-index heuristics
         (2**level) instead of actual feature-map resolution, so
         self-attention (which helps capture long-range facial structure
         instead of just local blur) was under-applied. Now fixed in
         PhysicsInformedUNet to track real resolution; this config also
         requests two attention points (32 and 16) that are both reachable
         with this depth, instead of one.
      2. 82M params for only 1970 training pairs is a large model:data
         ratio and a plausible cause of the loss plateau seen in v1;
         model_channels is roughly halved here.
    physics_loss_weight is left unchanged from v1 to isolate the effect of
    these two changes for comparison.
    """
    config = get_config_for_tufts_faces()

    config.model.model_channels = 64
    config.model.attention_resolutions = [32, 16]

    config.system.checkpoint_dir = "./checkpoints_tufts_v2"
    config.system.output_dir = "./output_tufts_v2"
    config.system.sample_dir = "./samples_tufts_v2"

    return config

def get_config_for_tufts_faces_v3() -> Config:
    """
    v3: builds on v2 (which already includes the deeper, 4-layer reflectance
    prior head -- see PIDModel.__init__) and adds three further changes,
    each independently motivated by an audit of v2's results:
      1. Multi-scale RGB FiLM conditioning: v2 only injects RGB via a
         channel-concat at the input, which has to survive many downsample/
         upsample steps to influence deep or late-decoder layers. FiLM gives
         every internal resolution direct access to RGB features.
      2. Channel-consistency loss: penalizes per-pixel variance across the
         3 output channels, directly targeting the color-tint artifact
         v2 still showed despite its deeper physics head.
      3. Perceptual (LPIPS) loss on the x0 reconstruction, to combat the
         blurriness inherent to pure pixel-MSE diffusion training.
    Also trains longer (450 vs 300 epochs) since v2's best checkpoints kept
    landing late in training; the cosine schedule is naturally rescaled to
    the new epoch count since this is a fresh run, not a resumed one.
    """
    config = get_config_for_tufts_faces_v2()

    config.model.use_film_conditioning = True
    config.model.use_channel_consistency_loss = True
    config.model.channel_consistency_weight = 0.1
    config.model.use_perceptual_loss = True
    config.model.perceptual_loss_weight = 0.1

    config.training.num_epochs = 450

    config.system.checkpoint_dir = "./checkpoints_tufts_v3"
    config.system.output_dir = "./output_tufts_v3"
    config.system.sample_dir = "./samples_tufts_v3"

    return config

def get_config_for_high_res() -> Config:
    """Get configuration for high-resolution training (512x512)."""
    config = Config()
    
    # High resolution
    config.data.img_size = [512, 512]
    
    # Adjust batch size
    config.training.batch_size = 4
    
    # Increase model capacity
    config.model.model_channels = 192
    config.model.channel_mult = [1, 2, 3, 4, 5]
    
    return config

# Export all config functions
__all__ = [
    'Config',
    'DataConfig',
    'ModelConfig',
    'TrainingConfig',
    'SystemConfig',
    'get_default_config',
    'get_config_for_quick_test',
    'get_config_for_cpu',
    'get_config_for_high_res',
    'get_config_for_tufts_faces',
    'get_config_for_tufts_faces_v2',
    'get_config_for_tufts_faces_v3'
]