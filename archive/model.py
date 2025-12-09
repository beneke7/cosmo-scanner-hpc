"""
model.py - The Neural Network Architecture
==========================================

This module defines CosmoNet, a Convolutional Neural Network for estimating
cosmological parameters from dark matter density maps.

THEORETICAL PERSPECTIVE
-----------------------
A neural network can be understood as a parameterized family of functions:
    f_θ: ℝ^{H×W} → ℝ

The goal is to find parameters θ* such that f_θ*(δ(x)) ≈ Ω_m for fields 
generated with matter density Ω_m.

ARCHITECTURE CHOICES & PHYSICS ANALOGIES
----------------------------------------

1. CONVOLUTION AS A GREEN'S FUNCTION
   The convolution operation (δ * W)(x) = Σ_y W(y) δ(x-y) is a linear,
   translation-invariant operator. This is exactly the structure of a 
   Green's function solution: G * source = field.
   
   The kernel W is our learnable "Green's function" that extracts
   local patterns from the field.

2. POOLING AS RENORMALIZATION
   MaxPooling performs coarse-graining: we reduce resolution by keeping
   only the maximum activation in each block. This is analogous to
   block-spin renormalization in statistical mechanics:
   - Discard UV (high-frequency) information
   - Retain IR (low-frequency) structure
   - Flow toward fixed points that capture universal behavior

3. NONLINEARITY AS SYMMETRY BREAKING
   ReLU activation f(x) = max(0, x) breaks the x ↔ -x symmetry.
   Without nonlinearities, the network would be a linear operator
   (just a single large matrix), unable to learn complex patterns.

4. THE LINEAR HEAD AS PROJECTION
   The final linear layer projects from the high-dimensional feature
   manifold (after convolutions) down to the 1D parameter manifold (Ω_m).
   This is like projecting a wavefunction onto an observable's eigenbasis.
"""

import torch
import torch.nn as nn
from typing import Tuple


def compute_conv_output_size(input_size: int, kernel_size: int, stride: int = 1, padding: int = 0) -> int:
    """
    Compute the output spatial dimension after a convolution.
    
    Formula: output = floor((input + 2*padding - kernel) / stride) + 1
    """
    return (input_size + 2 * padding - kernel_size) // stride + 1


def compute_pool_output_size(input_size: int, kernel_size: int, stride: int = None) -> int:
    """
    Compute the output spatial dimension after pooling.
    
    By default, stride = kernel_size for non-overlapping pooling.
    """
    if stride is None:
        stride = kernel_size
    return input_size // stride


class CosmoNet(nn.Module):
    """
    Convolutional Neural Network for cosmological parameter estimation.
    
    Architecture (for input_size=64):
    ---------------------------------
    Input: (B, 1, 64, 64) - Batch of density fields
    
    Conv Block 1:
        Conv2d(1→32, k=3, p=1)  : (B, 32, 64, 64)  [preserves size]
        ReLU                     : nonlinear activation
        MaxPool2d(2)            : (B, 32, 32, 32)  [2× downsampling]
    
    Conv Block 2:
        Conv2d(32→64, k=3, p=1) : (B, 64, 32, 32)
        ReLU
        MaxPool2d(2)            : (B, 64, 16, 16)  [4× total downsampling]
    
    Conv Block 3:
        Conv2d(64→128, k=3, p=1): (B, 128, 16, 16)
        ReLU
        MaxPool2d(2)            : (B, 128, 8, 8)   [8× total downsampling]
    
    Global Average Pooling:
        AdaptiveAvgPool2d(1)    : (B, 128, 1, 1)   [spatial → scalar]
    
    Linear Head:
        Flatten                 : (B, 128)
        Linear(128→1)           : (B, 1)          [feature → parameter]
    
    RECEPTIVE FIELD ANALYSIS
    ------------------------
    The receptive field is the region of input that affects a single output.
    After 3 conv layers with k=3 and 3 pooling layers with k=2:
    - Each conv with k=3, p=1 adds 1 pixel to each side
    - Each pool with k=2 doubles the effective receptive field
    
    Effective RF ≈ 3 + 2*(3-1) + 4*(3-1) + 8*(3-1) = 3 + 4 + 8 + 16 = 31 pixels
    (This is approximate; exact calculation requires tracking through pooling)
    
    Parameters
    ----------
    input_size : int
        Spatial dimension of input (assumes square input).
    """
    
    def __init__(self, input_size: int = 64):
        super().__init__()
        
        self.input_size = input_size
        
        # =====================================================
        # CONVOLUTIONAL FEATURE EXTRACTOR
        # Each block: Conv → ReLU → MaxPool
        # Think of this as a hierarchy of "Green's functions"
        # learning correlations at different scales
        # =====================================================
        
        # Block 1: Input channels=1 (grayscale field), Output=32 feature maps
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2: 32 → 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3: 64 → 128 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global average pooling: (B, C, H, W) → (B, C, 1, 1)
        # This makes the network invariant to small translations
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # =====================================================
        # LINEAR PROJECTION HEAD
        # Projects from 128-dim feature space to 1-dim parameter space
        # This is the final "measurement" that extracts Ω_m
        # =====================================================
        self.fc = nn.Linear(128, 1)
        
        # Activation function (ReLU = Rectified Linear Unit)
        self.relu = nn.ReLU()
        
        # Print architecture summary with dimensional analysis
        self._print_architecture_summary()
    
    def _print_architecture_summary(self) -> None:
        """Print detailed dimensional analysis of the network."""
        print("\n" + "=" * 70)
        print("COSMONET ARCHITECTURE SUMMARY")
        print("=" * 70)
        print(f"Input: (B, 1, {self.input_size}, {self.input_size})")
        print("-" * 70)
        
        # Track spatial dimensions through the network
        size = self.input_size
        
        # Block 1
        size_after_conv1 = compute_conv_output_size(size, 3, padding=1)
        size_after_pool1 = compute_pool_output_size(size_after_conv1, 2)
        print(f"Layer 1 (Conv+Pool): {size}×{size} → {size_after_pool1}×{size_after_pool1}")
        print(f"         Channels: 1 → 32")
        print(f"         Feature maps: 32 × {size_after_pool1}² = {32 * size_after_pool1**2} neurons")
        size = size_after_pool1
        
        # Block 2
        size_after_conv2 = compute_conv_output_size(size, 3, padding=1)
        size_after_pool2 = compute_pool_output_size(size_after_conv2, 2)
        print(f"Layer 2 (Conv+Pool): {size}×{size} → {size_after_pool2}×{size_after_pool2}")
        print(f"         Channels: 32 → 64")
        print(f"         Feature maps: 64 × {size_after_pool2}² = {64 * size_after_pool2**2} neurons")
        size = size_after_pool2
        
        # Block 3
        size_after_conv3 = compute_conv_output_size(size, 3, padding=1)
        size_after_pool3 = compute_pool_output_size(size_after_conv3, 2)
        print(f"Layer 3 (Conv+Pool): {size}×{size} → {size_after_pool3}×{size_after_pool3}")
        print(f"         Channels: 64 → 128")
        print(f"         Feature maps: 128 × {size_after_pool3}² = {128 * size_after_pool3**2} neurons")
        
        # Global pool
        print(f"Global Avg Pool: {size_after_pool3}×{size_after_pool3} → 1×1")
        print(f"         Output: 128-dimensional feature vector")
        
        # FC
        print(f"Linear Head: 128 → 1 (scalar prediction Ω_m)")
        
        # Parameter count
        total_params = sum(p.numel() for p in self.parameters())
        print("-" * 70)
        print(f"Total trainable parameters: {total_params:,}")
        print(f"This defines a {total_params}-dimensional optimization problem")
        print("=" * 70 + "\n")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 1, H, W) where:
            - B = batch size (number of universe realizations)
            - 1 = single channel (density field)
            - H, W = spatial dimensions
            
        Returns
        -------
        torch.Tensor
            Predicted Ω_m values with shape (B, 1).
            
        Notes
        -----
        The forward pass can be viewed as a composition of operators:
            f(x) = Linear ∘ GlobalPool ∘ Block3 ∘ Block2 ∘ Block1 (x)
        
        Each block transforms the representation, extracting increasingly
        abstract features (like going to higher orders in perturbation theory).
        """
        # Input validation
        assert x.dim() == 4, f"Expected 4D input (B, C, H, W), got {x.dim()}D"
        assert x.size(1) == 1, f"Expected 1 input channel, got {x.size(1)}"
        
        # Block 1: Initial feature extraction
        x = self.conv1(x)           # Learnable Green's function
        x = self.relu(x)            # Nonlinear activation (symmetry breaking)
        x = self.pool1(x)           # Coarse graining (RG step)
        
        # Block 2: Intermediate features
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)           # Another RG step
        
        # Block 3: High-level features
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)           # Final RG step
        
        # Global averaging → translation invariant summary
        x = self.global_pool(x)     # (B, 128, 1, 1)
        
        # Flatten and project to scalar
        x = x.view(x.size(0), -1)   # (B, 128)
        x = self.fc(x)              # (B, 1) - the "measurement"
        
        return x
    
    def get_feature_maps(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Extract intermediate feature maps for visualization.
        
        Useful for understanding what the network "sees" at each layer.
        
        Returns
        -------
        Tuple of tensors: (after_block1, after_block2, after_block3, final)
        """
        # Block 1
        x1 = self.pool1(self.relu(self.conv1(x)))
        
        # Block 2
        x2 = self.pool2(self.relu(self.conv2(x1)))
        
        # Block 3
        x3 = self.pool3(self.relu(self.conv3(x2)))
        
        # Global pool and FC
        x4 = self.global_pool(x3)
        out = self.fc(x4.view(x4.size(0), -1))
        
        return x1, x2, x3, out


if __name__ == "__main__":
    # Quick test
    model = CosmoNet(input_size=64)
    
    # Test forward pass
    x = torch.randn(4, 1, 64, 64)  # Batch of 4 random "fields"
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Sample predictions: {y.squeeze().tolist()}")
