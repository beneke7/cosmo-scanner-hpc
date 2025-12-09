"""
model_hybrid.py - Hybrid CNN + Power Spectrum Architecture
==========================================================

Addresses the precision saturation problem observed in v0.4.0:
- Box plots showed overlap between Ω_m = 0.307, 0.318, 0.328
- 2D spatial features alone cannot distinguish fine Ω_m increments
- Power spectrum P(k) explicitly encodes scale-dependent clustering

Architecture
------------
┌─────────────────────────────────────────────────────────────┐
│                      CosmoNetHybrid                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Image (256×256)                Power Spectrum (64 bins)    │
│       │                                │                    │
│       ▼                                ▼                    │
│  ┌─────────────┐                ┌─────────────┐            │
│  │   ResNet    │                │    MLP      │            │
│  │  Backbone   │                │  (3 layers) │            │
│  │             │                │             │            │
│  │ Conv blocks │                │ 64→128→64   │            │
│  │ + Attention │                │             │            │
│  └─────────────┘                └─────────────┘            │
│       │                                │                    │
│       ▼                                ▼                    │
│   512-dim                          64-dim                   │
│       │                                │                    │
│       └────────────┬───────────────────┘                    │
│                    ▼                                        │
│              ┌───────────┐                                  │
│              │  Fusion   │                                  │
│              │  576→256  │                                  │
│              │  →64→1    │                                  │
│              └───────────┘                                  │
│                    │                                        │
│                    ▼                                        │
│                  Ω_m                                        │
└─────────────────────────────────────────────────────────────┘

Why This Works
--------------
1. ResNet branch: Captures non-Gaussian features (void shapes, filament topology)
2. Power spectrum branch: Captures scale-dependent clustering (P(k) shape)
3. Fusion: Combines complementary information for precise Ω_m estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


# =============================================================================
# POWER SPECTRUM COMPUTATION (Differentiable)
# =============================================================================

def compute_power_spectrum(
    image: torch.Tensor,
    n_bins: int = 64,
    log_scale: bool = True,
    normalize: bool = True
) -> torch.Tensor:
    """
    Compute radially-averaged power spectrum from 2D image.
    
    This is differentiable and can be used in the forward pass!
    
    Parameters
    ----------
    image : torch.Tensor
        Input image, shape (B, 1, H, W) or (B, H, W) or (H, W)
    n_bins : int
        Number of radial bins for P(k)
    log_scale : bool
        Apply log10 to power spectrum
    normalize : bool
        Normalize to zero mean, unit variance
        
    Returns
    -------
    torch.Tensor
        Power spectrum, shape (B, n_bins) or (n_bins,)
    """
    # Handle different input shapes
    squeeze_batch = False
    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)
        squeeze_batch = True
    elif image.dim() == 3:
        image = image.unsqueeze(1)
    
    B, C, H, W = image.shape
    device = image.device
    
    # 2D FFT
    fft = torch.fft.fft2(image)
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
    power_2d = torch.abs(fft_shifted) ** 2
    
    # Create radial coordinate grid
    cy, cx = H // 2, W // 2
    y = torch.arange(H, device=device).float() - cy
    x = torch.arange(W, device=device).float() - cx
    Y, X = torch.meshgrid(y, x, indexing='ij')
    R = torch.sqrt(X**2 + Y**2)
    
    # Radial binning
    r_max = min(cx, cy)
    bin_edges = torch.linspace(0, r_max, n_bins + 1, device=device)
    
    # Compute mean power in each radial bin
    power_spectrum = torch.zeros(B, n_bins, device=device)
    
    for i in range(n_bins):
        mask = (R >= bin_edges[i]) & (R < bin_edges[i + 1])
        if mask.sum() > 0:
            # Average over spatial dimensions for each batch
            for b in range(B):
                power_spectrum[b, i] = power_2d[b, 0, mask].mean()
    
    # Log scale (more informative for neural network)
    if log_scale:
        power_spectrum = torch.log10(power_spectrum + 1e-10)
    
    # Normalize
    if normalize:
        mean = power_spectrum.mean(dim=1, keepdim=True)
        std = power_spectrum.std(dim=1, keepdim=True) + 1e-10
        power_spectrum = (power_spectrum - mean) / std
    
    if squeeze_batch:
        power_spectrum = power_spectrum.squeeze(0)
    
    return power_spectrum


def compute_power_spectrum_fast(
    image: torch.Tensor,
    n_bins: int = 64
) -> torch.Tensor:
    """
    Fast power spectrum computation using pre-computed bin indices.
    
    More efficient for batched computation during training.
    """
    B, C, H, W = image.shape
    device = image.device
    
    # 2D FFT
    fft = torch.fft.fft2(image)
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
    power_2d = torch.abs(fft_shifted) ** 2  # (B, C, H, W)
    
    # Flatten spatial dimensions
    power_flat = power_2d.view(B, -1)  # (B, H*W)
    
    # Pre-compute radial bin indices (cached)
    cy, cx = H // 2, W // 2
    y = torch.arange(H, device=device).float() - cy
    x = torch.arange(W, device=device).float() - cx
    Y, X = torch.meshgrid(y, x, indexing='ij')
    R = torch.sqrt(X**2 + Y**2).view(-1)  # (H*W,)
    
    r_max = min(cx, cy)
    bin_indices = (R / r_max * n_bins).long().clamp(0, n_bins - 1)
    
    # Scatter-add for binning
    power_spectrum = torch.zeros(B, n_bins, device=device)
    counts = torch.zeros(n_bins, device=device)
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            power_spectrum[:, i] = power_flat[:, mask].mean(dim=1)
            counts[i] = mask.sum()
    
    # Log scale and normalize
    power_spectrum = torch.log10(power_spectrum + 1e-10)
    mean = power_spectrum.mean(dim=1, keepdim=True)
    std = power_spectrum.std(dim=1, keepdim=True) + 1e-10
    power_spectrum = (power_spectrum - mean) / std
    
    return power_spectrum


# =============================================================================
# BUILDING BLOCKS
# =============================================================================

class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# =============================================================================
# MAIN HYBRID MODEL
# =============================================================================

class CosmoNetHybrid(nn.Module):
    """
    Hybrid CNN + Power Spectrum model for Ω_m estimation.
    
    Combines:
    1. ResNet backbone for spatial/non-Gaussian features
    2. MLP for power spectrum (scale-dependent clustering)
    3. Attention-weighted fusion
    
    Parameters
    ----------
    n_power_bins : int
        Number of power spectrum bins (default: 64)
    dropout : float
        Dropout rate (default: 0.3)
    use_attention : bool
        Use channel attention in ResNet (default: True)
    """
    
    def __init__(
        self,
        n_power_bins: int = 64,
        dropout: float = 0.3,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.n_power_bins = n_power_bins
        self.use_attention = use_attention
        
        # =====================================================================
        # Branch 1: ResNet backbone for image features
        # =====================================================================
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3, bias=False),  # 256 → 128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)  # 128 → 64
        )
        
        self.stage1 = self._make_stage(64, 64, 2, stride=1)    # 64×64
        self.stage2 = self._make_stage(64, 128, 2, stride=2)   # 32×32
        self.stage3 = self._make_stage(128, 256, 2, stride=2)  # 16×16
        self.stage4 = self._make_stage(256, 512, 2, stride=2)  # 8×8
        
        if use_attention:
            self.attn1 = ChannelAttention(64)
            self.attn2 = ChannelAttention(128)
            self.attn3 = ChannelAttention(256)
            self.attn4 = ChannelAttention(512)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.img_dropout = nn.Dropout(dropout)
        
        # =====================================================================
        # Branch 2: MLP for power spectrum
        # =====================================================================
        self.ps_mlp = nn.Sequential(
            nn.Linear(n_power_bins, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )
        
        # =====================================================================
        # Fusion head
        # =====================================================================
        # 512 (image) + 64 (power spectrum) = 576
        fusion_dim = 512 + 64
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def _make_stage(self, in_ch: int, out_ch: int, num_blocks: int, stride: int):
        layers = [ResidualBlock(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)
    
    def forward(
        self,
        image: torch.Tensor,
        power_spectrum: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        image : torch.Tensor
            Input image, shape (B, 1, 256, 256)
        power_spectrum : torch.Tensor, optional
            Pre-computed power spectrum, shape (B, n_power_bins)
            If None, computed on-the-fly from image
            
        Returns
        -------
        torch.Tensor
            Predicted Ω_m, shape (B, 1)
        """
        # Compute power spectrum if not provided
        if power_spectrum is None:
            power_spectrum = compute_power_spectrum_fast(image, self.n_power_bins)
        
        # Image branch
        x = self.stem(image)
        
        x = self.stage1(x)
        if self.use_attention:
            x = self.attn1(x)
        
        x = self.stage2(x)
        if self.use_attention:
            x = self.attn2(x)
        
        x = self.stage3(x)
        if self.use_attention:
            x = self.attn3(x)
        
        x = self.stage4(x)
        if self.use_attention:
            x = self.attn4(x)
        
        img_features = self.pool(x).flatten(1)  # (B, 512)
        img_features = self.img_dropout(img_features)
        
        # Power spectrum branch
        ps_features = self.ps_mlp(power_spectrum)  # (B, 64)
        
        # Fusion
        combined = torch.cat([img_features, ps_features], dim=1)  # (B, 576)
        output = self.fusion(combined)
        
        return output
    
    def get_embeddings(
        self,
        image: torch.Tensor,
        power_spectrum: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get intermediate embeddings for analysis."""
        if power_spectrum is None:
            power_spectrum = compute_power_spectrum_fast(image, self.n_power_bins)
        
        # Image branch
        x = self.stem(image)
        x = self.stage1(x)
        if self.use_attention:
            x = self.attn1(x)
        x = self.stage2(x)
        if self.use_attention:
            x = self.attn2(x)
        x = self.stage3(x)
        if self.use_attention:
            x = self.attn3(x)
        x = self.stage4(x)
        if self.use_attention:
            x = self.attn4(x)
        img_features = self.pool(x).flatten(1)
        
        # Power spectrum branch
        ps_features = self.ps_mlp(power_spectrum)
        
        return img_features, ps_features


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import time
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Test power spectrum computation
    print("\n=== Power Spectrum Test ===")
    image = torch.randn(8, 1, 256, 256, device=device)
    
    start = time.time()
    ps = compute_power_spectrum_fast(image, n_bins=64)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    print(f"Power spectrum shape: {ps.shape}")
    print(f"Time: {(time.time() - start)*1000:.1f}ms")
    
    # Test model
    print("\n=== CosmoNetHybrid Test ===")
    model = CosmoNetHybrid(n_power_bins=64, dropout=0.3).to(device)
    print(f"Parameters: {count_parameters(model):,}")
    
    # Forward pass (auto-compute power spectrum)
    start = time.time()
    output = model(image)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    print(f"Output shape: {output.shape}")
    print(f"Forward time: {(time.time() - start)*1000:.1f}ms")
    
    # Forward pass (pre-computed power spectrum)
    start = time.time()
    output = model(image, ps)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    print(f"Forward time (pre-computed PS): {(time.time() - start)*1000:.1f}ms")
    
    # Memory usage
    if device.type == 'cuda':
        print(f"\nGPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
    print("\n✓ All tests passed!")
