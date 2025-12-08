"""
config.py - Training Configuration
==================================

All hyperparameters, paths, and model architecture in one place.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import numpy as np
from PIL import Image
import csv
from typing import Tuple, List, Dict

# =============================================================================
# PATHS
# =============================================================================

DATA_DIR = "data/images"
METADATA_FILE = "data/metadata.csv"
LOG_DIR = "train/logs"
MODEL_DIR = "train/models"

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

# Optimized for RTX 5090 (32GB VRAM)
BATCH_SIZE = 128  # Reduced for more stable training
LEARNING_RATE = 3e-4  # Lower LR for stability
WEIGHT_DECAY = 1e-3  # Increased regularization
NUM_EPOCHS = 15
PATIENCE = 5  # Early stopping patience

# Dropout rates
CONV_DROPOUT = 0.1  # Dropout after conv blocks
FC_DROPOUT = 0.4  # Dropout in FC layers
GRADIENT_ACCUMULATION_STEPS = 1  # Can increase if OOM occurs
USE_AMP = True  # Automatic Mixed Precision for faster training

# Data split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# Scheduler
LR_START_FACTOR = 1.0
LR_END_FACTOR = 0.1
LR_TOTAL_ITERS = NUM_EPOCHS

# Random seed
SEED = 42

# Workers - optimized for 60 CPU cores
NUM_WORKERS = 16  # Increased from 4 to utilize multi-core CPU
PREFETCH_FACTOR = 4  # Prefetch batches for faster data loading
PERSISTENT_WORKERS = True  # Keep workers alive between epochs

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

# Input: 256x256 grayscale image
IMAGE_SIZE = 256
IN_CHANNELS = 1

# CNN Architecture - deeper for 256x256 images
# Conv layers: [out_channels, kernel_size, stride, padding]
CONV_LAYERS = [
    (32, 5, 1, 2),   # 256 -> 256, kernel=5x5
    (64, 5, 2, 2),   # 256 -> 128, kernel=5x5, stride=2
    (128, 3, 2, 1),  # 128 -> 64,  kernel=3x3, stride=2
    (256, 3, 2, 1),  # 64 -> 32,   kernel=3x3, stride=2
    (512, 3, 2, 1),  # 32 -> 16,   kernel=3x3, stride=2
]

# Fully connected layers after global average pooling
FC_LAYERS = [256, 64]

# Output: single omega_m value
OUTPUT_SIZE = 1

# =============================================================================
# WANDB
# =============================================================================

WANDB_PROJECT = "learner"

# =============================================================================
# DEVICE
# =============================================================================

def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# =============================================================================
# DATASET
# =============================================================================


class CosmoDataset(Dataset):
    """Dataset loading pre-generated density fields from disk."""
    
    def __init__(self, samples: List[Dict], augment: bool = False):
        self.samples = samples
        self.augment = augment
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        filepath = os.path.join(DATA_DIR, sample['filename'])
        
        img = Image.open(filepath)
        field = np.array(img, dtype=np.float32) / 255.0
        
        # Data augmentation (physics-preserving: flips and 90-degree rotations)
        if self.augment:
            # Random horizontal flip
            if np.random.random() > 0.5:
                field = np.flip(field, axis=1).copy()
            # Random vertical flip
            if np.random.random() > 0.5:
                field = np.flip(field, axis=0).copy()
            # Random 90-degree rotation (0, 1, 2, or 3 times)
            k = np.random.randint(0, 4)
            field = np.rot90(field, k).copy()
        
        field_tensor = torch.from_numpy(field).unsqueeze(0)  # (1, H, W)
        omega_tensor = torch.tensor([sample['omega_m']], dtype=torch.float32)
        
        return field_tensor, omega_tensor


def load_metadata() -> List[Dict]:
    """Load all samples from metadata CSV."""
    samples = []
    with open(METADATA_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append({
                'filename': row['filename'],
                'omega_m': float(row['omega_m'])
            })
    return samples


def create_dataloaders(seed: int = SEED) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders with 70/20/10 split.
    """
    samples = load_metadata()
    
    # Shuffle and split
    np.random.seed(seed)
    indices = np.random.permutation(len(samples))
    
    n_train = int(TRAIN_RATIO * len(samples))
    n_val = int(VAL_RATIO * len(samples))
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]
    
    train_loader = DataLoader(
        CosmoDataset(train_samples, augment=True),  # Augmentation ON for training
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=PERSISTENT_WORKERS
    )
    
    # Larger batch for validation = more stable loss estimate
    val_loader = DataLoader(
        CosmoDataset(val_samples, augment=False),  # No augmentation for validation
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=PERSISTENT_WORKERS
    )
    
    test_loader = DataLoader(
        CosmoDataset(test_samples, augment=False),  # No augmentation for test
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=PERSISTENT_WORKERS
    )
    
    return train_loader, val_loader, test_loader

# =============================================================================
# MODEL
# =============================================================================


class CosmoNet(nn.Module):
    """
    CNN for cosmological parameter estimation.
    
    Architecture:
    - 5 Conv blocks with BatchNorm, ReLU, and Dropout
    - Global Average Pooling
    - 2 FC layers with Dropout
    - Output: omega_m prediction
    """
    
    def __init__(self):
        super().__init__()
        
        # Build conv layers with dropout
        self.conv_blocks = nn.ModuleList()
        in_ch = IN_CHANNELS
        
        for out_ch, kernel, stride, pad in CONV_LAYERS:
            block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel, stride, pad),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout2d(CONV_DROPOUT)
            )
            self.conv_blocks.append(block)
            in_ch = out_ch
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # FC layers with dropout
        fc_layers = []
        fc_in = CONV_LAYERS[-1][0]  # Last conv output channels
        
        for fc_out in FC_LAYERS:
            fc_layers.extend([
                nn.Linear(fc_in, fc_out),
                nn.ReLU(inplace=True),
                nn.Dropout(FC_DROPOUT)
            ])
            fc_in = fc_out
        
        fc_layers.append(nn.Linear(fc_in, OUTPUT_SIZE))
        self.fc = nn.Sequential(*fc_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.conv_blocks:
            x = block(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
