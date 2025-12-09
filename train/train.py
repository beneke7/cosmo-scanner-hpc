"""
main_v5.py - v0.5.0 Training Pipeline
======================================

Optimized for RTX 5090 with:
1. CosineAnnealingWarmRestarts scheduler (handles loss spikes)
2. channels_last memory format (faster convolutions)
3. TF32 precision for matmul (RTX 30/40/50 series)
4. 2LPT physics for realistic synthetic data
5. Hybrid model with power spectrum branch

Usage:
    python train/main_v5.py --run_name hybrid_2lpt_v1 --epochs 50
"""

import os
import sys
import argparse
import logging
import time
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import wandb
from pathlib import Path
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.model_hybrid import CosmoNetHybrid, compute_power_spectrum_fast, count_parameters
from src.physics_lpt import generate_2lpt_numpy
from src.physics_lensing import generate_des_like_numpy

# =============================================================================
# RTX 5090 OPTIMIZATIONS
# =============================================================================

def setup_rtx_optimizations():
    """Configure PyTorch for optimal RTX 5090 performance."""
    
    # TF32 for faster matmul on Ampere/Ada/Blackwell
    torch.set_float32_matmul_precision('high')
    
    # Enable cudnn benchmarking for consistent input sizes
    torch.backends.cudnn.benchmark = True
    
    # Enable TF32 for convolutions
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print("RTX optimizations enabled:")
    print(f"  - TF32 matmul precision: high")
    print(f"  - cuDNN benchmark: True")
    print(f"  - TF32 for conv/matmul: True")


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Data
    'data_dir': 'data/images',
    'metadata_file': 'data/metadata.csv',
    'image_size': 256,
    'n_power_bins': 64,
    
    # Training
    'batch_size': 64,
    'learning_rate': 3e-4,
    'weight_decay': 1e-3,
    'num_epochs': 50,
    'patience': 15,
    
    # Scheduler (CosineAnnealingWarmRestarts)
    'T_0': 10,        # Initial restart period
    'T_mult': 2,      # Period multiplier after each restart
    'eta_min': 1e-6,  # Minimum learning rate
    
    # Model
    'dropout': 0.3,
    'use_attention': True,
    
    # Mixed precision
    'use_amp': True,
    
    # Data augmentation
    'augment': True,
    'noise_std': 0.05,
    
    # Paths
    'log_dir': 'train/logs',
    'model_dir': 'train/models',
}


# =============================================================================
# DATASET
# =============================================================================

class CosmoDatasetV5(Dataset):
    """
    Dataset for v0.5.0 training.
    
    Features:
    - Pre-computed power spectrum
    - On-the-fly augmentation
    - channels_last memory format
    """
    
    def __init__(
        self,
        data_dir: str,
        metadata_file: str,
        n_power_bins: int = 64,
        augment: bool = False,
        noise_std: float = 0.05
    ):
        self.data_dir = Path(data_dir)
        self.n_power_bins = n_power_bins
        self.augment = augment
        self.noise_std = noise_std
        
        # Load metadata
        self.samples = []
        with open(metadata_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append({
                    'filename': row['filename'],
                    'omega_m': float(row['omega_m'])
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        img_path = self.data_dir / sample['filename']
        img = Image.open(img_path).convert('L')
        field = np.array(img, dtype=np.float32) / 255.0
        
        # Augmentation
        if self.augment:
            # Random flips
            if np.random.random() > 0.5:
                field = np.flip(field, axis=1).copy()
            if np.random.random() > 0.5:
                field = np.flip(field, axis=0).copy()
            
            # Random rotation (90° increments)
            k = np.random.randint(0, 4)
            field = np.rot90(field, k).copy()
            
            # Add noise
            if self.noise_std > 0:
                noise = np.random.normal(0, self.noise_std, field.shape).astype(np.float32)
                field = np.clip(field + noise, 0, 1)
        
        # Convert to tensor
        field_tensor = torch.from_numpy(field).unsqueeze(0)  # (1, H, W)
        target_tensor = torch.tensor([sample['omega_m']], dtype=torch.float32)
        
        return field_tensor, target_tensor


class OnlineDataset(Dataset):
    """
    Dataset that generates fields on-the-fly.
    
    Supports multiple data types:
    - '2lpt': N-body like fields (for Quijote)
    - 'des': Weak lensing κ maps (for DES)
    - 'grf': Simple Gaussian Random Fields
    """
    
    def __init__(
        self,
        num_samples: int,
        omega_m_range: tuple = (0.1, 0.5),
        size: int = 256,
        augment: bool = True,
        noise_std: float = 0.05,
        data_type: str = 'des',  # 'des', '2lpt', or 'grf'
        smoothing_arcmin: float = 10.0,  # For DES-like
    ):
        self.num_samples = num_samples
        self.omega_m_range = omega_m_range
        self.size = size
        self.augment = augment
        self.noise_std = noise_std
        self.data_type = data_type
        self.smoothing_arcmin = smoothing_arcmin
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Random Ω_m
        omega_m = np.random.uniform(*self.omega_m_range)
        
        # Generate field based on data type
        if self.data_type == '2lpt':
            field = generate_2lpt_numpy(omega_m, size=self.size, seed=None)
        elif self.data_type == 'des':
            field = generate_des_like_numpy(
                omega_m, 
                size=self.size, 
                seed=None,
                smoothing_arcmin=self.smoothing_arcmin,
                add_noise=True
            )
        else:  # 'grf' - use pre-generated data logic
            field = generate_des_like_numpy(
                omega_m, size=self.size, seed=None,
                smoothing_arcmin=0.1, add_noise=False
            )
        
        # Normalize to [0, 1]
        field = (field - field.min()) / (field.max() - field.min() + 1e-10)
        
        # Augmentation
        if self.augment:
            if np.random.random() > 0.5:
                field = np.flip(field, axis=1).copy()
            if np.random.random() > 0.5:
                field = np.flip(field, axis=0).copy()
            k = np.random.randint(0, 4)
            field = np.rot90(field, k).copy()
            
            if self.noise_std > 0 and self.data_type != 'des':
                # DES already has noise built in
                noise = np.random.normal(0, self.noise_std, field.shape).astype(np.float32)
                field = np.clip(field + noise, 0, 1)
        
        field_tensor = torch.from_numpy(field).unsqueeze(0).float()
        target_tensor = torch.tensor([omega_m], dtype=torch.float32)
        
        return field_tensor, target_tensor


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool = True
) -> float:
    """Train for one epoch with mixed precision."""
    model.train()
    total_loss = 0.0
    
    for images, targets in loader:
        # Move to device with channels_last format
        images = images.to(device, memory_format=torch.channels_last)
        targets = targets.to(device)
        
        optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
        
        with autocast('cuda', enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)
        
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True
) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, memory_format=torch.channels_last)
            targets = targets.to(device)
            
            with autocast('cuda', enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            total_loss += loss.item()
    
    return total_loss / len(loader)


def setup_logging(run_name: str) -> str:
    """Setup logging."""
    os.makedirs(CONFIG['log_dir'], exist_ok=True)
    log_file = os.path.join(CONFIG['log_dir'], f"{run_name}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file


# =============================================================================
# MAIN TRAINING
# =============================================================================

def train(
    run_name: str,
    data_type: str = 'des',
    num_samples: int = 100000,
    smoothing_arcmin: float = 10.0
):
    """
    Main training function.
    
    Parameters
    ----------
    run_name : str
        Name for this training run
    data_type : str
        Type of synthetic data: 'des' (weak lensing), '2lpt' (N-body), 'disk' (pre-generated)
    num_samples : int
        Number of samples for online generation
    smoothing_arcmin : float
        Smoothing scale for DES-like data (arcmin)
    """
    setup_logging(run_name)
    setup_rtx_optimizations()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CONFIG['model_dir'], exist_ok=True)
    
    logging.info(f"=== Training: {run_name} ===")
    logging.info(f"Device: {device}")
    logging.info(f"Data type: {data_type}")
    if data_type == 'des':
        logging.info(f"Smoothing: {smoothing_arcmin} arcmin")
    
    # Create model with channels_last format
    model = CosmoNetHybrid(
        n_power_bins=CONFIG['n_power_bins'],
        dropout=CONFIG['dropout'],
        use_attention=CONFIG['use_attention']
    ).to(device, memory_format=torch.channels_last)
    
    logging.info(f"Model parameters: {count_parameters(model):,}")
    
    # Create datasets
    if data_type in ['des', '2lpt', 'grf']:
        logging.info(f"Using online {data_type.upper()} generation")
        train_dataset = OnlineDataset(
            num_samples=int(num_samples * 0.8),
            omega_m_range=(0.1, 0.5),
            augment=CONFIG['augment'],
            noise_std=CONFIG['noise_std'],
            data_type=data_type,
            smoothing_arcmin=smoothing_arcmin
        )
        val_dataset = OnlineDataset(
            num_samples=int(num_samples * 0.2),
            omega_m_range=(0.1, 0.5),
            augment=False,
            noise_std=0,
            data_type=data_type,
            smoothing_arcmin=smoothing_arcmin
        )
    else:  # 'disk' - use pre-generated data
        logging.info("Using pre-generated dataset")
        full_dataset = CosmoDatasetV5(
            CONFIG['data_dir'],
            CONFIG['metadata_file'],
            n_power_bins=CONFIG['n_power_bins'],
            augment=CONFIG['augment'],
            noise_std=CONFIG['noise_std']
        )
        
        n_total = len(full_dataset)
        n_train = int(0.8 * n_total)
        n_val = n_total - n_train
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'] * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    logging.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # CosineAnnealingWarmRestarts - handles loss spikes better
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=CONFIG['T_0'],
        T_mult=CONFIG['T_mult'],
        eta_min=CONFIG['eta_min']
    )
    
    scaler = GradScaler('cuda', enabled=CONFIG['use_amp'])
    
    # Initialize wandb
    wandb.init(project="cosmo-scanner", name=run_name, config={
        **CONFIG,
        'data_type': data_type,
        'num_samples': num_samples,
        'smoothing_arcmin': smoothing_arcmin
    })
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(CONFIG['model_dir'], f"{run_name}_best.pt")
    
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        epoch_start = time.time()
        
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, CONFIG['use_amp']
        )
        val_loss = validate(model, val_loader, criterion, device, CONFIG['use_amp'])
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]['lr']
        
        logging.info(
            f"Epoch {epoch:03d} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f} | "
            f"LR: {lr:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr,
            "epoch_time": epoch_time
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': CONFIG
            }, best_model_path)
            logging.info(f"  -> Saved best model (RMSE: {np.sqrt(val_loss):.4f})")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                logging.info(f"Early stopping at epoch {epoch}")
                break
    
    logging.info(f"Training complete. Best val loss: {best_val_loss:.6f}")
    logging.info(f"Best RMSE: {np.sqrt(best_val_loss):.4f}")
    logging.info(f"Model saved: {best_model_path}")
    
    wandb.finish()
    
    return best_val_loss


# =============================================================================
# MAIN
# =============================================================================

def main():
    import datetime
    
    parser = argparse.ArgumentParser(description="Cosmo Scanner v0.5.0 Training")
    parser.add_argument("--run_name", type=str, default=None, help="Run name (auto-generated if not provided)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--data_type", type=str, default="des", 
                        choices=["des", "2lpt", "grf", "disk"],
                        help="Data type: des (weak lensing), 2lpt (N-body), grf (Gaussian), disk (pre-generated)")
    parser.add_argument("--num_samples", type=int, default=100000, help="Number of samples")
    parser.add_argument("--smoothing", type=float, default=10.0, help="Smoothing scale for DES (arcmin)")
    
    args = parser.parse_args()
    
    # Auto-generate run name if not provided
    if args.run_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"v5_{args.data_type}_{timestamp}"
    
    # Update config
    CONFIG['num_epochs'] = args.epochs
    CONFIG['batch_size'] = args.batch_size
    CONFIG['learning_rate'] = args.lr
    
    train(
        args.run_name, 
        data_type=args.data_type, 
        num_samples=args.num_samples,
        smoothing_arcmin=args.smoothing
    )


if __name__ == "__main__":
    main()
