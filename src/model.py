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

class ReflectanceGuidedAttention(nn.Module):
    """
    Reflectance-guided attention for RGB->NIR *face* translation.

    NIR (700-1000nm) is reflected, not emitted, light: pixel intensity is governed by
    per-material reflectance/albedo, NOT by thermal (blackbody) emission. We therefore
    replace the thermal Planck-law temperature modulation of the original PID with a
    learned per-pixel *reflectance* map in [0, 1] that multiplicatively gates the
    features (reflected intensity = incident light x reflectance). A learnable
    per-channel spectral weight retains the "band-emphasis" idea for the NIR window.
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

        # Per-pixel NIR reflectance estimation (albedo-like), normalized to [0, 1]
        self.reflectance_estimator = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, 1),
            nn.Sigmoid()  # reflectance fraction in [0, 1]
        )

        # Learnable spectral (NIR-band) channel response
        self.spectral_weight = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Estimate per-pixel reflectance and gate features by it
        reflectance_map = self.reflectance_estimator(x)  # [B, 1, H, W] in [0, 1]
        x_modulated = x * reflectance_map

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

        # Apply spectral (NIR-band) channel modulation
        out = out * self.spectral_weight

        return self.proj(out) + x


# Backward-compatible alias (older configs / imports reference the thermal name)
PhysicsGuidedAttention = ReflectanceGuidedAttention


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

class TimestepSequential(nn.Sequential):
    """Sequential container that passes the time embedding only to modules that
    accept it (ResidualBlock); attention / sampling layers receive just x."""

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, ResidualBlock):
                x = layer(x, t_emb)
            else:
                x = layer(x)
        return x


class PhysicsInformedUNet(nn.Module):
    """
    Physics-informed conditional UNet for RGB->NIR diffusion.

    Standard DDPM encoder/decoder skip topology (num_res_blocks+1 decoder blocks per
    level) with reflectance-guided attention at the configured resolutions and a
    spectral-transform physics module inside each residual block. Conditioning is done
    by channel-concatenating the RGB image with the noisy NIR input.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (14, 7),
        dropout: float = 0.1,
        time_embed_dim: int = 512,
        use_physics: bool = True,
        condition_on_rgb: bool = True,
        image_size: int = 112,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.use_physics = use_physics
        self.condition_on_rgb = condition_on_rgb
        attn_res = set(attention_resolutions)
        num_levels = len(channel_mult)

        # Time embedding
        self.time_embed = TimeEmbedding(model_channels, time_embed_dim)

        # Initial convolution (concatenate noisy NIR with RGB condition when conditioning)
        in_total = in_channels * 2 if condition_on_rgb else in_channels
        self.input_conv = nn.Conv2d(in_total, model_channels, 3, padding=1)

        def make_attn(ch):
            return AttentionBlock(ch, num_heads=8, use_physics=use_physics)

        # ---- Encoder (downsampling path) ----
        self.input_blocks = nn.ModuleList()
        ch = model_channels
        skip_channels = [ch]          # channels of each stored skip (input_conv output first)
        resolution = image_size
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                layers = [ResidualBlock(ch, out_ch, time_embed_dim, dropout, use_physics)]
                ch = out_ch
                if resolution in attn_res:
                    layers.append(make_attn(ch))
                self.input_blocks.append(TimestepSequential(*layers))
                skip_channels.append(ch)
            if level != num_levels - 1:
                self.input_blocks.append(TimestepSequential(Downsample(ch)))
                skip_channels.append(ch)
                resolution //= 2

        # ---- Middle ----
        self.middle_block1 = ResidualBlock(ch, ch, time_embed_dim, dropout, use_physics)
        self.middle_attn = make_attn(ch)
        self.middle_block2 = ResidualBlock(ch, ch, time_embed_dim, dropout, use_physics)

        # ---- Decoder (upsampling path) ----
        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            out_ch = model_channels * mult
            for i in range(num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                layers = [ResidualBlock(ch + skip_ch, out_ch, time_embed_dim, dropout, use_physics)]
                ch = out_ch
                if resolution in attn_res:
                    layers.append(make_attn(ch))
                if level != 0 and i == num_res_blocks:
                    layers.append(Upsample(ch))
                    resolution *= 2
                self.output_blocks.append(TimestepSequential(*layers))

        # ---- Output ----
        self.output_layers = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Noisy NIR image [B, C_out, H, W]
            timesteps: Diffusion timesteps [B]
            condition: RGB conditioning image [B, 3, H, W]
        Returns:
            Predicted noise [B, C_out, H, W]
        """
        t_emb = self.time_embed(timesteps)

        if self.condition_on_rgb and condition is not None:
            x = torch.cat([x, condition], dim=1)

        h = self.input_conv(x)
        skips = [h]
        for module in self.input_blocks:
            h = module(h, t_emb)
            skips.append(h)

        h = self.middle_block1(h, t_emb)
        h = self.middle_attn(h)
        h = self.middle_block2(h, t_emb)

        for module in self.output_blocks:
            h = torch.cat([h, skips.pop()], dim=1)
            h = module(h, t_emb)

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
# NIR Reflectance Prior
# ============================================================================

class ReflectancePrior(nn.Module):
    """
    Physics-informed NIR prior for faces, based on *reflectance* rather than thermal
    emission. It produces a coarse NIR estimate directly from the RGB image.

    Inductive bias: skin/hair NIR reflectance in the 700-1000nm band tracks the visible
    channels unevenly -- it is dominated by the red channel and rises where melanin
    absorption falls, so a red-weighted luminance is a good first-order NIR guess. We
    initialize a 1x1 "reflectance mixing" conv toward that red-heavy grayscale
    (R,G,B ~ 0.6/0.3/0.1) and let a small residual CNN refine local structure. Output is
    tanh-mapped to [-1, 1] to match the diffusion target range.
    """

    def __init__(self, out_channels: int = 3):
        super().__init__()
        self.out_channels = out_channels

        # First-order reflectance mixing: RGB -> single NIR reflectance band.
        self.reflectance_mix = nn.Conv2d(3, 1, kernel_size=1, bias=True)
        with torch.no_grad():
            self.reflectance_mix.weight.copy_(
                torch.tensor([0.6, 0.3, 0.1]).view(1, 3, 1, 1)
            )
            self.reflectance_mix.bias.zero_()

        # Local residual refinement (subsurface scattering / vein structure differs in NIR)
        self.refine = nn.Sequential(
            nn.Conv2d(3 + 1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        # rgb expected in [-1, 1]; map to [0, 1] for a reflectance-fraction interpretation
        rgb01 = (rgb + 1.0) * 0.5
        coarse = self.reflectance_mix(rgb01)                    # [B, 1, H, W]
        refined = self.refine(torch.cat([rgb01, coarse], dim=1))  # [B, out, H, W]
        # coarse (broadcast) provides the physical prior; refine adds a learned residual
        out = coarse + refined
        return torch.tanh(out)  # [-1, 1]


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
            condition_on_rgb=config.model.condition_on_rgb,
            image_size=int(config.data.img_size[0]),
        )

        # Physics prior producing a coarse NIR estimate from RGB (used by the
        # physics-consistency loss). "reflectance" is NIR-appropriate; "learned" is a
        # generic head; "planck" is the legacy thermal prior kept for ablation.
        self.prior_type = getattr(self.config.model, 'physics_prior_type', 'reflectance')
        if self.prior_type == 'reflectance':
            self.reflectance_prior = ReflectancePrior(out_channels=self.unet.out_channels)
        elif self.prior_type == 'learned':
            self.physics_head = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, self.unet.out_channels, kernel_size=3, padding=1)
            )
        elif self.prior_type == 'planck':
            # legacy thermal-emission prior (physically wrong for NIR; ablation only)
            self.temp_estimator = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, 3, padding=1),
                nn.Sigmoid()
            )
            self.temp_to_nir = nn.Conv2d(1, self.unet.out_channels, 1)
            self.planck_scale = nn.Parameter(torch.tensor(1.0))
        else:
            raise ValueError(f"Unknown physics_prior_type: {self.prior_type}")

        # Optional frozen face-recognition backbone for the identity-preserving loss.
        self._init_identity_backbone()

        # Initialize diffusion process
        self.diffusion = DiffusionProcess(
            num_timesteps=config.model.num_timesteps,
            beta_schedule=config.model.beta_schedule,
            beta_start=config.model.beta_start,
            beta_end=config.model.beta_end
        )

        logger.info(f"Initialized PID model with {sum(p.numel() for p in self.parameters()):,} parameters")

    def _init_identity_backbone(self):
        """Load a frozen FR backbone if an identity loss is configured.

        The backbone and preprocessor are held inside a plain dict so nn.Module does NOT
        register them as submodules -- their frozen 260MB weights stay out of this model's
        state_dict, optimizer and EMA. Because they are unregistered, .to(device) will not
        move them; we move them lazily to the input's device in _identity_embeddings.
        """
        self._identity = None  # dict with 'backbone'/'preprocessor' when enabled

        weight_path = getattr(self.config.model, 'identity_weights_path', None)
        weight = getattr(self.config.model, 'identity_loss_weight', 0.0)
        if not weight_path or weight <= 0:
            return

        import os
        if not os.path.exists(weight_path):
            logger.warning(
                f"identity_weights_path '{weight_path}' not found; disabling identity loss."
            )
            return

        try:
            from src.fr_model import get_arcface_ir101_model, get_adaface_ir101_model
        except ImportError:
            from fr_model import get_arcface_ir101_model, get_adaface_ir101_model

        which = getattr(self.config.model, 'identity_model', 'arcface')
        getter = get_adaface_ir101_model if which == 'adaface' else get_arcface_ir101_model
        backbone, preprocessor, tag = getter(weights_path=weight_path)

        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        # Skip the preprocessor's range check: AMP + clamped x0 can drift a hair past [0,1].
        preprocessor.do_range_check = False

        self._identity = {'backbone': backbone, 'preprocessor': preprocessor}
        logger.info(f"Identity loss enabled using frozen FR backbone: {tag}")

    @property
    def identity_backbone(self):
        """Convenience flag/accessor: None when identity loss is disabled."""
        return None if self._identity is None else self._identity['backbone']

    def _physics_prior(self, rgb_images: torch.Tensor) -> torch.Tensor:
        """Coarse NIR estimate from RGB, in [-1, 1]."""
        if self.prior_type == 'reflectance':
            return self.reflectance_prior(rgb_images)
        if self.prior_type == 'learned':
            return self.physics_head(rgb_images)
        # planck (legacy)
        temp_map = self.temp_estimator(rgb_images)
        physics_pred = self.temp_to_nir(temp_map) * self.planck_scale
        return torch.tanh(physics_pred)

    def _identity_embeddings(self, imgs_pm1: torch.Tensor) -> torch.Tensor:
        """Embed images given in [-1, 1] with the frozen FR backbone (normalized embeddings).

        Runs in fp32 with autocast disabled: the backbone asserts unit-norm embeddings to
        within 1e-5, which fp16/AMP can violate.
        """
        backbone = self._identity['backbone']
        preprocessor = self._identity['preprocessor']
        # Move the (unregistered) frozen modules onto the data's device on first use.
        if next(backbone.parameters()).device != imgs_pm1.device:
            backbone.to(imgs_pm1.device)
            preprocessor.to(imgs_pm1.device)
        with torch.amp.autocast(device_type=imgs_pm1.device.type, enabled=False):
            imgs01 = torch.clamp((imgs_pm1.float() + 1.0) * 0.5, 0.0, 1.0)  # [-1,1] -> [0,1]
            if imgs01.shape[1] == 1:
                imgs01 = imgs01.repeat(1, 3, 1, 1)
            x = preprocessor(imgs01)
            return backbone(x).embedding  # L2-normalized [B, D]

    @torch.no_grad()
    def sample(self, rgb_images: torch.Tensor, num_steps: int = 50, eta: float = 0.0) -> torch.Tensor:
        """Generate NIR images from RGB conditioning via DDIM sampling. Output in [-1, 1]."""
        shape = (
            rgb_images.shape[0],
            self.unet.out_channels,
            rgb_images.shape[2],
            rgb_images.shape[3],
        )
        return self.diffusion.ddim_sample(
            self.unet, shape, condition=rgb_images, num_steps=num_steps,
            eta=eta, device=rgb_images.device,
        )
    
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
        # Move constants to device BEFORE indexing (t lives on `device`).
        alpha_t = self.diffusion.alphas.to(device)[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.diffusion.sqrt_one_minus_alphas_cumprod.to(device)[t][:, None, None, None]
        sqrt_recip_alpha_t = self.diffusion.sqrt_recip_alphas.to(device)[t][:, None, None, None]

        pred_x0 = sqrt_recip_alpha_t * (
            noisy_nir - ((1 - alpha_t) / (sqrt_one_minus_alphas_cumprod_t + 1e-12)) * predicted_noise
        )
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        # Reflectance-consistency (physics) loss: predicted NIR should agree with the
        # NIR-reflectance prior derived from RGB.
        physics_pred = self._physics_prior(rgb_images)
        phys_weight = getattr(self.config.model, 'physics_loss_weight', 0.0)
        physics_loss = F.l1_loss(pred_x0, physics_pred)

        # Identity-preserving loss: embeddings of the generated NIR should match those of
        # the ground-truth NIR (and optionally the RGB source). Weighted per-sample by the
        # signal level alpha_bar_t so noisy-timestep x0 estimates contribute little.
        id_weight = getattr(self.config.model, 'identity_loss_weight', 0.0)
        identity_loss = torch.zeros((), device=device)
        if self.identity_backbone is not None and id_weight > 0:
            alpha_bar_t = self.diffusion.alphas_cumprod.to(device)[t]  # [B] in [0,1]
            gen_emb = self._identity_embeddings(pred_x0)
            targets = []
            if getattr(self.config.model, 'identity_use_gt_nir', True):
                targets.append(nir_images)
            if getattr(self.config.model, 'identity_use_rgb', False):
                targets.append(rgb_images)
            for tgt in targets:
                with torch.no_grad():
                    tgt_emb = self._identity_embeddings(tgt)
                cos = (gen_emb * tgt_emb).sum(dim=1)          # [B], cosine similarity
                identity_loss = identity_loss + (alpha_bar_t * (1.0 - cos)).mean()
            if targets:
                identity_loss = identity_loss / len(targets)

        # total loss
        loss = mse_loss + phys_weight * physics_loss + id_weight * identity_loss

        if return_dict:
            return {
                'loss': loss,
                'mse': mse_loss.item(),
                'physics_loss': physics_loss.item() if isinstance(physics_loss, torch.Tensor) else float(physics_loss),
                'identity_loss': identity_loss.item() if isinstance(identity_loss, torch.Tensor) else float(identity_loss),
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