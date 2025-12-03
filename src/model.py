"""
Physics-Informed Diffusion Model for NIR-from-RGB Generation
Implementation based on PID paper (arXiv:2407.09299) adapted for border control
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Time Embedding Layers
# ============================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal time embeddings for diffusion timesteps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TimeEmbedding(nn.Module):
    """Time embedding with MLP projection."""
    
    def __init__(self, time_dim: int, hidden_dim: int):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.time_mlp(t)


# ============================================================================
# Physics-Informed Modules
# ============================================================================

class PhysicsGuidedAttention(nn.Module):
    """
    Physics-guided attention module that incorporates thermal radiation physics.
    Based on Planck's law and Stefan-Boltzmann principles.
    """
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        # Multi-head attention
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)
        
        # Physics-informed temperature estimation
        self.temp_estimator = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, 1),
            nn.Sigmoid()  # Normalize temperature to [0, 1]
        )
        
        # Wavelength-dependent response (for NIR simulation)
        self.wavelength_weight = nn.Parameter(torch.ones(1, channels, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Estimate temperature map
        temp_map = self.temp_estimator(x)  # [B, 1, H, W]
        
        # Apply physics-based modulation (simplified Planck's law)
        # I(λ, T) ∝ 1 / (exp(hc/λkT) - 1)
        physics_weight = torch.exp(-1.0 / (temp_map + 1e-6))
        x_modulated = x * physics_weight
        
        # Multi-head self-attention
        qkv = self.qkv(x_modulated).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(1)  # [B, num_heads, head_dim, H*W]
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        out = out.reshape(B, C, H, W)
        
        # Apply wavelength-dependent modulation
        out = out * self.wavelength_weight
        
        return self.proj(out) + x


class SpectralTransformBlock(nn.Module):
    """
    Spectral transformation block for RGB to NIR conversion.
    Uses frequency domain processing to model spectral response.
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        
        # Learnable spectral filters
        self.spectral_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1)
        )
        
        # Channel attention for spectral weighting
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply spectral transformation
        spectral_feat = self.spectral_conv(x)
        
        # Apply channel attention for spectral response
        channel_weight = self.channel_attn(spectral_feat)
        spectral_feat = spectral_feat * channel_weight
        
        return spectral_feat + x


# ============================================================================
# UNet Building Blocks
# ============================================================================

class ResidualBlock(nn.Module):
    """Residual block with time embedding and group normalization."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        time_emb_dim: int,
        dropout: float = 0.1,
        use_physics: bool = True
    ):
        super().__init__()
        self.use_physics = use_physics
        
        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()
        
        # Physics-informed module
        if use_physics:
            self.physics_module = SpectralTransformBlock(out_channels)
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        
        # Add time embedding
        time_emb = self.time_emb(time_emb)[:, :, None, None]
        h = h + time_emb
        
        h = self.conv2(h)
        
        # Apply physics-informed transformation
        if self.use_physics:
            h = self.physics_module(h)
        
        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    """Self-attention block for capturing global dependencies."""
    
    def __init__(self, channels: int, num_heads: int = 8, use_physics: bool = True):
        super().__init__()
        self.use_physics = use_physics
        
        if use_physics:
            self.attention = PhysicsGuidedAttention(channels, num_heads)
        else:
            self.norm = nn.GroupNorm(32, channels)
            self.attention = nn.MultiheadAttention(
                channels, num_heads, batch_first=True
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_physics:
            return self.attention(x)
        else:
            B, C, H, W = x.shape
            h = self.norm(x)
            h = h.reshape(B, C, H * W).transpose(1, 2)
            h, _ = self.attention(h, h, h)
            h = h.transpose(1, 2).reshape(B, C, H, W)
            return x + h


class Downsample(nn.Module):
    """Downsampling layer using strided convolution."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling layer using transposed convolution."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ============================================================================
# Main UNet Model
# ============================================================================

class PhysicsInformedUNet(nn.Module):
    """
    Physics-Informed UNet for RGB-to-NIR diffusion model.
    Incorporates thermal radiation physics and spectral transformation.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16, 8),
        dropout: float = 0.1,
        time_embed_dim: int = 512,
        use_physics: bool = True,
        condition_on_rgb: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.use_physics = use_physics
        self.condition_on_rgb = condition_on_rgb
        
        # Time embedding
        self.time_embed = TimeEmbedding(model_channels, time_embed_dim)
        
        # Initial convolution
        if condition_on_rgb:
            # Concatenate noisy NIR with RGB condition
            self.input_conv = nn.Conv2d(
                in_channels * 2, model_channels, 3, padding=1
            )
        else:
            self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Encoder (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        ch = model_channels
        input_block_chans = [ch]
        
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            
            for block_idx in range(num_res_blocks):
                layers = ResidualBlock(
                    ch, out_ch, time_embed_dim, dropout, use_physics
                )
                self.encoder_blocks.append(layers)
                ch = out_ch
                input_block_chans.append(ch)
                
                # Add attention at specified resolutions
                if level in attention_resolutions or (2**level) in attention_resolutions:
                    self.encoder_attns.append(
                        AttentionBlock(ch, num_heads=8, use_physics=use_physics)
                    )
                else:
                    self.encoder_attns.append(nn.Identity())
                
                # For each residual block append an explicit downsample placeholder:
                # only the last block in a level performs actual downsampling (except last level)
                if (block_idx == num_res_blocks - 1) and (level != len(channel_mult) - 1):
                    self.downsample_layers.append(Downsample(ch))
                    input_block_chans.append(ch)
                else:
                    self.downsample_layers.append(nn.Identity())
        
        # Middle blocks
        self.middle_block1 = ResidualBlock(
            ch, ch, time_embed_dim, dropout, use_physics
        )
        self.middle_attn = AttentionBlock(ch, num_heads=8, use_physics=use_physics)
        self.middle_block2 = ResidualBlock(
            ch, ch, time_embed_dim, dropout, use_physics
        )
        
        # Decoder (upsampling path)
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        
        for level, mult in enumerate(reversed(channel_mult)):
            out_ch = model_channels * mult
            
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = ResidualBlock(
                    ch + ich, out_ch, time_embed_dim, dropout, use_physics
                )
                self.decoder_blocks.append(layers)
                ch = out_ch
                
                # Add attention at specified resolutions
                res_level = len(channel_mult) - 1 - level
                if res_level in attention_resolutions or (2**res_level) in attention_resolutions:
                    self.decoder_attns.append(
                        AttentionBlock(ch, num_heads=8, use_physics=use_physics)
                    )
                else:
                    self.decoder_attns.append(nn.Identity())
            
            # Upsample (except last level)
            if level != len(channel_mult) - 1:
                self.upsample_layers.append(Upsample(ch))
            else:
                self.upsample_layers.append(nn.Identity())
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        timesteps: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the UNet.
        
        Args:
            x: Noisy NIR image [B, 3, H, W]
            timesteps: Diffusion timesteps [B]
            condition: RGB conditioning image [B, 3, H, W]
            
        Returns:
            Predicted noise [B, 3, H, W]
        """
        # Time embedding
        t_emb = self.time_embed(timesteps)
        
        # Concatenate with RGB condition if using conditional generation
        if self.condition_on_rgb and condition is not None:
            x = torch.cat([x, condition], dim=1)
        
        # Initial convolution
        h = self.input_conv(x)
        
        # Encoder
        encoder_features = [h]
        
        for i, (block, attn, downsample) in enumerate(
            zip(self.encoder_blocks, self.encoder_attns, self.downsample_layers)
        ):
            h = block(h, t_emb)
            h = attn(h)
            encoder_features.append(h)
            h = downsample(h)
            if not isinstance(downsample, nn.Identity):
                encoder_features.append(h)
        
        # Middle
        h = self.middle_block1(h, t_emb)
        h = self.middle_attn(h)
        h = self.middle_block2(h, t_emb)
        
        # Decoder
        for i, (block, attn, upsample) in enumerate(
            zip(self.decoder_blocks, self.decoder_attns, self.upsample_layers)
        ):
            skip = encoder_features.pop()
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_emb)
            h = attn(h)
            h = upsample(h)
        
        # Output
        return self.output_layers(h)


# ============================================================================
# Diffusion Process
# ============================================================================

class DiffusionProcess:
    """
    Diffusion process implementing DDPM/DDIM sampling.
    Based on the PID paper's physics-informed approach.
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02
    ):
        self.num_timesteps = num_timesteps
        
        # Create beta schedule
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Precompute useful quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # Posterior variance for DDPM
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine beta schedule from Improved DDPM paper."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def q_sample(
        self, 
        x_start: torch.Tensor, 
        t: torch.Tensor, 
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward diffusion process: q(x_t | x_0).
        Add noise to clean images according to timestep.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        device = x_start.device if isinstance(x_start, torch.Tensor) else t.device

        # ensure constants live on the same device
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)

        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Reverse diffusion process: p(x_{t-1} | x_t).
        Single denoising step (DDPM sampling).
        """
        device = x_t.device

        # Predict noise
        predicted_noise = model(x_t, t, condition)
        
        # move coeff tensors to device before indexing
        alphas = self.alphas.to(device)
        alphas_cumprod = self.alphas_cumprod.to(device)
        alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        posterior_variance = self.posterior_variance.to(device)

        alpha_t = alphas[t][:, None, None, None]
        alpha_cumprod_t = alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        sqrt_recip_alpha_t = sqrt_recip_alphas[t][:, None, None, None]
        
        # Predict x_0
        pred_x0 = sqrt_recip_alpha_t * (
            x_t - ((1 - alpha_t) / (sqrt_one_minus_alpha_cumprod_t + 1e-12)) * predicted_noise
        )
        pred_x0 = torch.clamp(pred_x0, -1, 1)
        
        # Get posterior mean
        posterior_mean = (
            torch.sqrt(alphas_cumprod_prev[t][:, None, None, None]) * pred_x0 +
            torch.sqrt(1 - alphas_cumprod_prev[t][:, None, None, None]) * predicted_noise
        )
        
        # Add noise (except for t=0)
        if (t[0].item() if isinstance(t, torch.Tensor) else t[0]) > 0:
            noise = torch.randn_like(x_t)
            posterior_variance_t = posterior_variance[t][:, None, None, None]
            return posterior_mean + torch.sqrt(posterior_variance_t) * noise
        else:
            return posterior_mean
        
    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        condition: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        eta: float = 0.0,
        device: str = "cuda"
    ) -> torch.Tensor:
        """
        DDIM sampling for faster generation.
        """
        # Create timestep sequence
        skip = max(1, self.num_timesteps // num_steps)
        timesteps = torch.arange(0, self.num_timesteps, skip, device=device)
        timesteps = torch.flip(timesteps, [0])
        
        # Start from random noise
        x = torch.randn(shape, device=device)

        # ensure arrays moved to device once
        alphas_cumprod = self.alphas_cumprod.to(device)
        alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = model(x, t_batch, condition)
            
            # Get alpha values on device
            alpha_t = alphas_cumprod[t]
            
            if i < len(timesteps) - 1:
                alpha_t_prev = alphas_cumprod[timesteps[i + 1]]
            else:
                alpha_t_prev = torch.tensor(1.0, device=device)
            
            # Predict x_0
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            # Direction pointing to x_t
            # guard against division by zero and ensure tensors on device
            eps = 1e-12
            sqrt_alpha_t = torch.sqrt(alpha_t + eps)
            sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev + eps)
            denom = torch.sqrt(1 - alpha_t + eps)
            dir_xt = torch.sqrt(torch.clamp(1 - alpha_t_prev - eta ** 2 * (1 - alpha_t_prev) / max(eps, (1 - alpha_t).item()) * (1 - (alpha_t / (alpha_t_prev + eps)).item()), 0.0)) * predicted_noise
            # Random noise
            noise = torch.randn_like(x) if eta > 0 else 0
            
            # DDIM update (keep operations on device)
            x = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t + eps)) * torch.sqrt(max(0.0, 1 - (alpha_t / (alpha_t_prev + eps)))) * (noise if isinstance(noise, torch.Tensor) else 0)
        
        return x


# ============================================================================
# Full Model with Training Utilities
# ============================================================================

class PIDModel(nn.Module):
    """
    Complete Physics-Informed Diffusion model for RGB-to-NIR generation.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Initialize UNet
        self.unet = PhysicsInformedUNet(
            in_channels=config.model.in_channels,
            out_channels=config.model.out_channels,
            model_channels=config.model.model_channels,
            channel_mult=tuple(config.model.channel_mult),
            num_res_blocks=config.model.num_res_blocks,
            attention_resolutions=tuple(config.model.attention_resolutions),
            dropout=config.model.dropout,
            use_physics=True,
            condition_on_rgb=config.model.condition_on_rgb
        )

        # Physics prior: either learned mapping (RGB -> NIR) or simple planck-like prior
        prior_type = getattr(self.config.model, 'physics_prior_type', 'learned')
        if prior_type == 'learned':
            # small learned physics head: RGB -> NIR prior
            self.physics_head = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, self.unet.out_channels, kernel_size=3, padding=1)
            )
        elif prior_type == 'planck':
            # simple physics-ish prior: estimate per-pixel "temperature" from RGB, map -> NIR channels
            self.temp_estimator = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, 3, padding=1),
                nn.Sigmoid()  # normalized temperature map in [0,1]
            )
            # small conv to convert temperature map -> NIR channels and a learnable scale
            self.temp_to_nir = nn.Conv2d(1, self.unet.out_channels, 1)
            self.planck_scale = nn.Parameter(torch.tensor(1.0))
        else:
            raise ValueError(f"Unknown physics_prior_type: {prior_type}")

        # Initialize diffusion process
        self.diffusion = DiffusionProcess(
            num_timesteps=config.model.num_timesteps,
            beta_schedule=config.model.beta_schedule,
            beta_start=config.model.beta_start,
            beta_end=config.model.beta_end
        )

        logger.info(f"Initialized PID model with {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through the model."""
        return self.unet(x, t, condition)
    
    def compute_loss(
        self,
        nir_images: torch.Tensor,
        rgb_images: torch.Tensor,
        return_dict: bool = False
    ) -> torch.Tensor:
        """
        Compute diffusion training loss.
        Combines standard DDPM epsilon prediction MSE with a configurable physics loss.
        """
        batch_size = nir_images.shape[0]
        device = nir_images.device

        # Sample random timesteps
        t = torch.randint(
            0, self.config.model.num_timesteps, (batch_size,), device=device
        ).long()

        # Sample noise
        noise = torch.randn_like(nir_images)

        # Add noise to NIR images (forward diffusion)
        noisy_nir = self.diffusion.q_sample(nir_images, t, noise)

        # Predict noise
        predicted_noise = self(noisy_nir, t, rgb_images)

        # Standard DDPM MSE loss on noise
        mse_loss = F.mse_loss(predicted_noise, noise)

        # --- physics-informed term: reconstruct x0 and compare to physics prior ---
        # compute predicted x0 from predicted_noise (same formula used in DDPM)
        alpha_t = self.diffusion.alphas[t][:, None, None, None].to(device)
        sqrt_one_minus_alphas_cumprod_t = self.diffusion.sqrt_one_minus_alphas_cumprod[t][:, None, None, None].to(device)
        sqrt_recip_alpha_t = self.diffusion.sqrt_recip_alphas[t][:, None, None, None].to(device)

        pred_x0 = sqrt_recip_alpha_t * (
            noisy_nir - ((1 - alpha_t) / (sqrt_one_minus_alphas_cumprod_t + 1e-12)) * predicted_noise
        )
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        prior_type = getattr(self.config.model, 'physics_prior_type', 'learned')
        if prior_type == 'learned':
            physics_pred = self.physics_head(rgb_images)
        else:
            # planck-like prior: estimate per-pixel temperature and map to channels
            temp_map = self.temp_estimator(rgb_images)  # [B,1,H,W] in [0,1]
            physics_pred = self.temp_to_nir(temp_map) * (self.planck_scale.view(1,1,1,1) if self.planck_scale.dim()==0 else self.planck_scale)
            # match range of pred_x0: apply tanh scaling if needed
            physics_pred = torch.tanh(physics_pred)

        # physics loss (L1)
        phys_weight = getattr(self.config.model, 'physics_loss_weight', 0.0)
        physics_loss = F.l1_loss(pred_x0, physics_pred)

        # total loss
        loss = mse_loss + phys_weight * physics_loss

        if return_dict:
            return {
                'loss': loss,
                'mse': mse_loss.item(),
                'physics_loss': physics_loss.item() if isinstance(physics_loss, torch.Tensor) else float(physics_loss),
                'predicted_noise_mean': predicted_noise.mean().item(),
                'predicted_noise_std': predicted_noise.std().item()
            }

        return loss

__all__ = [
    'PIDModel',
    'PhysicsInformedUNet',
    'DiffusionProcess',
    'PhysicsGuidedAttention',
    'SpectralTransformBlock'
]