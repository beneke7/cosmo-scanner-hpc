"""
dataset.py - The PyTorch Data Pipeline
=======================================

Loads pre-generated images from disk using metadata.csv for labels.

Usage:
    # Generate dataset first
    python -m src.generate_dataset
    
    # Then use in training
    from src.dataset import CosmoDataset, create_dataloaders
"""

import os
import csv
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from typing import Tuple, List, Dict

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = "data"
METADATA_FILE = "data/metadata.csv"

# =============================================================================
# DATASET CLASS
# =============================================================================


class CosmoDataset(Dataset):
    """
    PyTorch Dataset that loads pre-generated dark matter density fields.
    
    Reads image paths and Omega_m values from metadata.csv.
    
    Parameters
    ----------
    data_dir : str
        Directory containing the JPG images.
    metadata_file : str
        Path to the CSV file with columns: filename, omega_m
    split : str
        One of 'train', 'val', or 'all'. Splits data 80/20.
    seed : int
        Random seed for reproducible train/val split.
    """
    
    def __init__(
        self,
        data_dir: str = DATA_DIR,
        metadata_file: str = METADATA_FILE,
        split: str = 'all',
        seed: int = 42
    ):
        self.data_dir = data_dir
        self.metadata_file = metadata_file
        
        # Load metadata from CSV
        self.samples = self._load_metadata()
        
        if len(self.samples) == 0:
            raise ValueError(
                f"No samples found. Check that {metadata_file} exists. "
                f"Run 'python -m src.generate_dataset' first."
            )
        
        # Train/val split
        if split in ['train', 'val']:
            np.random.seed(seed)
            indices = np.random.permutation(len(self.samples))
            split_idx = int(0.8 * len(self.samples))
            
            if split == 'train':
                indices = indices[:split_idx]
            else:  # val
                indices = indices[split_idx:]
            
            self.samples = [self.samples[i] for i in indices]
    
    def _load_metadata(self) -> List[Dict]:
        """Load metadata from CSV file."""
        samples = []
        
        if not os.path.exists(self.metadata_file):
            return samples
        
        with open(self.metadata_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                samples.append({
                    'filename': row['filename'],
                    'omega_m': float(row['omega_m'])
                })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load a single (field, omega_m) sample from disk.
        """
        sample = self.samples[index]
        filepath = os.path.join(self.data_dir, sample['filename'])
        
        # Load image
        img = Image.open(filepath)
        field = np.array(img, dtype=np.float32)
        
        # Normalize from [0, 255] to [0, 1]
        field = field / 255.0
        
        # Convert to tensor: (H, W) -> (1, H, W)
        field_tensor = torch.from_numpy(field).unsqueeze(0)
        omega_m_tensor = torch.tensor([sample['omega_m']], dtype=torch.float32)
        
        return field_tensor, omega_m_tensor


def create_dataloaders(
    data_dir: str = DATA_DIR,
    metadata_file: str = METADATA_FILE,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders.
    
    Returns
    -------
    train_loader : DataLoader
        Training data (80%).
    val_loader : DataLoader
        Validation data (20%).
    """
    train_dataset = CosmoDataset(
        data_dir=data_dir,
        metadata_file=metadata_file,
        split='train',
        seed=seed
    )
    
    val_dataset = CosmoDataset(
        data_dir=data_dir,
        metadata_file=metadata_file,
        split='val',
        seed=seed
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    print("=== Testing CosmoDataset ===\n")
    
    try:
        dataset = CosmoDataset(split='all')
        print(f"Total samples: {len(dataset)}")
        
        field, omega_m = dataset[0]
        print(f"Field shape: {field.shape}")
        print(f"Omega_m: {omega_m.item():.5f}")
        
        train = CosmoDataset(split='train')
        val = CosmoDataset(split='val')
        print(f"\nTrain: {len(train)} | Val: {len(val)}")
        
        print("\n✓ Dataset ready!")
        
    except ValueError as e:
        print(f"Error: {e}")
