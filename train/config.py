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

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
PATIENCE = 10  # Early stopping patience

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

# Workers
NUM_WORKERS = 4

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

# Input: 128x128 grayscale image
IMAGE_SIZE = 128
IN_CHANNELS = 1

# CNN Architecture
# Conv layers: [out_channels, kernel_size, stride, padding]
CONV_LAYERS = [
    (32, 5, 1, 2),   # 128 -> 128, kernel=5x5
    (64, 5, 2, 2),   # 128 -> 64,  kernel=5x5, stride=2
    (128, 3, 2, 1),  # 64 -> 32,   kernel=3x3, stride=2
    (256, 3, 2, 1),  # 32 -> 16,   kernel=3x3, stride=2
]

# Fully connected layers after global average pooling
FC_LAYERS = [128, 32]

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
    
    def __init__(self, samples: List[Dict]):
        self.samples = samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        filepath = os.path.join(DATA_DIR, sample['filename'])
        
        img = Image.open(filepath)
        field = np.array(img, dtype=np.float32) / 255.0
        
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
        CosmoDataset(train_samples),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        CosmoDataset(val_samples),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        CosmoDataset(test_samples),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

# =============================================================================
# MODEL
# =============================================================================


class CosmoNet(nn.Module):
    """
    CNN for cosmological parameter estimation.
    
    Architecture:
    - 4 Conv blocks with BatchNorm and ReLU
    - Global Average Pooling
    - 2 FC layers
    - Output: omega_m prediction
    
    Kernel sizes: 5x5 (first two), 3x3 (last two)
    """
    
    def __init__(self):
        super().__init__()
        
        # Build conv layers
        layers = []
        in_ch = IN_CHANNELS
        
        for out_ch, kernel, stride, pad in CONV_LAYERS:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel, stride, pad),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ])
            in_ch = out_ch
        
        self.conv = nn.Sequential(*layers)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # FC layers
        fc_layers = []
        fc_in = CONV_LAYERS[-1][0]  # Last conv output channels
        
        for fc_out in FC_LAYERS:
            fc_layers.extend([
                nn.Linear(fc_in, fc_out),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2)
            ])
            fc_in = fc_out
        
        fc_layers.append(nn.Linear(fc_in, OUTPUT_SIZE))
        self.fc = nn.Sequential(*fc_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
