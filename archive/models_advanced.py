"""
models_advanced.py - Advanced Neural Network Architectures
==========================================================

Contains more sophisticated models for cosmological parameter estimation:
1. CosmoResNet - ResNet-style architecture with skip connections
2. CosmoAttentionNet - Adds spatial attention mechanisms
3. CosmoNetV2 - Hybrid with improved feature fusion

These models are designed for:
- Better gradient flow (skip connections)
- Multi-scale feature extraction
- Attention to cosmologically relevant regions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# =============================================================================
# BUILDING BLOCKS
# =============================================================================


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.
    
    x ──► Conv ──► BN ──► ReLU ──► Conv ──► BN ──► (+) ──► ReLU ──► out
    │                                               ↑
    └───────────────────────────────────────────────┘
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection (identity or projection)
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
        out = F.relu(out + identity)
        
        return out


class SpatialAttention(nn.Module):
    """
    Spatial attention module - learns where to focus.
    
    Useful for cosmology: attention to high-density regions (clusters)
    and low-density regions (voids) which are most informative.
    """
    
    def __init__(self, in_channels: int):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.conv(x)
        return x * attention


class ChannelAttention(nn.Module):
    """
    Channel attention (SE-Net style) - learns which features are important.
    """
    
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


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Combines channel and spatial attention.
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# =============================================================================
# MAIN MODELS
# =============================================================================


class CosmoResNet(nn.Module):
    """
    ResNet-style architecture for Ω_m estimation.
    
    Architecture:
    - Initial conv layer
    - 4 ResNet stages with increasing channels
    - Global average pooling
    - FC head for regression
    
    Total depth: ~18 layers (similar to ResNet-18)
    """
    
    def __init__(self, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        
        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False),  # 256 -> 128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)  # 128 -> 64
        )
        
        # ResNet stages
        self.stage1 = self._make_stage(64, 64, 2, stride=1)    # 64x64
        self.stage2 = self._make_stage(64, 128, 2, stride=2)   # 32x32
        self.stage3 = self._make_stage(128, 256, 2, stride=2)  # 16x16
        self.stage4 = self._make_stage(256, 512, 2, stride=2)  # 8x8
        
        # Global pooling and head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
    
    def _make_stage(self, in_ch: int, out_ch: int, num_blocks: int, stride: int):
        layers = [ResidualBlock(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class CosmoAttentionNet(nn.Module):
    """
    ResNet with attention mechanisms.
    
    Adds CBAM attention after each stage to focus on
    cosmologically relevant features.
    """
    
    def __init__(self, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        
        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        
        # Stages with attention
        self.stage1 = self._make_stage(64, 64, 2, stride=1)
        self.attn1 = CBAM(64)
        
        self.stage2 = self._make_stage(64, 128, 2, stride=2)
        self.attn2 = CBAM(128)
        
        self.stage3 = self._make_stage(128, 256, 2, stride=2)
        self.attn3 = CBAM(256)
        
        self.stage4 = self._make_stage(256, 512, 2, stride=2)
        self.attn4 = CBAM(512)
        
        # Head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
    
    def _make_stage(self, in_ch: int, out_ch: int, num_blocks: int, stride: int):
        layers = [ResidualBlock(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        
        x = self.stage1(x)
        x = self.attn1(x)
        
        x = self.stage2(x)
        x = self.attn2(x)
        
        x = self.stage3(x)
        x = self.attn3(x)
        
        x = self.stage4(x)
        x = self.attn4(x)
        
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class CosmoNetV2Hybrid(nn.Module):
    """
    Advanced hybrid model: ResNet + Power Spectrum + Multi-scale features.
    
    Architecture:
    - ResNet backbone for image features
    - Power spectrum MLP branch
    - Multi-scale feature aggregation
    - Attention-based fusion
    """
    
    def __init__(self, in_channels: int = 1, n_power_bins: int = 32, dropout: float = 0.3):
        super().__init__()
        
        self.n_power_bins = n_power_bins
        
        # =====================================================================
        # Image branch: ResNet backbone
        # =====================================================================
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        
        self.stage1 = self._make_stage(64, 64, 2, stride=1)
        self.stage2 = self._make_stage(64, 128, 2, stride=2)
        self.stage3 = self._make_stage(128, 256, 2, stride=2)
        self.stage4 = self._make_stage(256, 512, 2, stride=2)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Multi-scale feature aggregation
        self.ms_pool1 = nn.AdaptiveAvgPool2d(1)
        self.ms_pool2 = nn.AdaptiveAvgPool2d(1)
        self.ms_pool3 = nn.AdaptiveAvgPool2d(1)
        
        # Project multi-scale features to same dimension
        self.ms_proj1 = nn.Linear(64, 64)
        self.ms_proj2 = nn.Linear(128, 64)
        self.ms_proj3 = nn.Linear(256, 64)
        
        # =====================================================================
        # Power spectrum branch
        # =====================================================================
        self.ps_mlp = nn.Sequential(
            nn.Linear(n_power_bins, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )
        
        # =====================================================================
        # Fusion with attention
        # =====================================================================
        # Features: 512 (stage4) + 64*3 (multi-scale) + 64 (power spectrum) = 768
        fusion_dim = 512 + 64 * 3 + 64
        
        self.fusion_attention = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim // 4, fusion_dim),
            nn.Sigmoid()
        )
        
        self.fusion_head = nn.Sequential(
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
    
    def forward(self, image: torch.Tensor, power_spectrum: torch.Tensor) -> torch.Tensor:
        # Image branch with multi-scale extraction
        x = self.stem(image)
        
        x1 = self.stage1(x)
        ms1 = self.ms_proj1(self.ms_pool1(x1).flatten(1))
        
        x2 = self.stage2(x1)
        ms2 = self.ms_proj2(self.ms_pool2(x2).flatten(1))
        
        x3 = self.stage3(x2)
        ms3 = self.ms_proj3(self.ms_pool3(x3).flatten(1))
        
        x4 = self.stage4(x3)
        img_features = self.pool(x4).flatten(1)  # 512-dim
        
        # Power spectrum branch
        ps_features = self.ps_mlp(power_spectrum)  # 64-dim
        
        # Concatenate all features
        combined = torch.cat([img_features, ms1, ms2, ms3, ps_features], dim=1)  # 768-dim
        
        # Attention-weighted fusion
        attention = self.fusion_attention(combined)
        combined = combined * attention
        
        # Final prediction
        output = self.fusion_head(combined)
        
        return output


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test models
    print("Testing advanced models...")
    
    # Test inputs
    batch_size = 4
    image = torch.randn(batch_size, 1, 256, 256)
    power_spectrum = torch.randn(batch_size, 32)
    
    # CosmoResNet
    model1 = CosmoResNet()
    out1 = model1(image)
    print(f"CosmoResNet: {count_parameters(model1):,} params, output shape: {out1.shape}")
    
    # CosmoAttentionNet
    model2 = CosmoAttentionNet()
    out2 = model2(image)
    print(f"CosmoAttentionNet: {count_parameters(model2):,} params, output shape: {out2.shape}")
    
    # CosmoNetV2Hybrid
    model3 = CosmoNetV2Hybrid()
    out3 = model3(image, power_spectrum)
    print(f"CosmoNetV2Hybrid: {count_parameters(model3):,} params, output shape: {out3.shape}")
    
    print("\n✓ All models working!")
