"""
main_finetune.py - Fine-tuning Pipeline
========================================

Two-stage training:
1. Pre-train on synthetic data (large dataset, fast generation)
2. Fine-tune on real simulations (Quijote - limited but accurate)

Fine-tuning techniques:
- Freeze early layers (keep general features)
- Lower learning rate (preserve pre-trained knowledge)
- Gradual unfreezing (optional)
- Discriminative learning rates (different LR per layer group)

Usage:
    # Stage 1: Pre-train on synthetic
    python train/main_finetune.py --stage pretrain --run_name resnet_v1
    
    # Stage 2: Fine-tune on Quijote
    python train/main_finetune.py --stage finetune --run_name resnet_v1_ft --pretrained resnet_v1
"""

import os
import sys
import argparse
import logging
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import Dataset, DataLoader
import wandb
from pathlib import Path
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from train.models_advanced import CosmoResNet, CosmoAttentionNet, CosmoNetV2Hybrid, count_parameters

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
LOG_DIR = "train/logs"
MODEL_DIR = "train/models"
SYNTHETIC_DATA_DIR = "data/images"
SYNTHETIC_METADATA = "data/metadata.csv"
QUIJOTE_DATA_DIR = "data/real/quijote"
QUIJOTE_METADATA = "data/real/quijote/metadata.csv"

# Training hyperparameters
PRETRAIN_CONFIG = {
    'batch_size': 64,        # Smaller batch = more regularization
    'learning_rate': 3e-4,   # Lower LR for stability
    'weight_decay': 1e-2,    # 100x stronger weight decay
    'num_epochs': 50,
    'patience': 10,
    'label_smoothing': 0.1,  # Prevent overconfidence
}

FINETUNE_CONFIG = {
    'batch_size': 32,  # Smaller batch for limited data
    'learning_rate': 1e-4,  # 10x lower than pre-training
    'weight_decay': 1e-3,  # Stronger regularization
    'num_epochs': 50,
    'patience': 10,
    'freeze_stages': 2,  # Freeze first N stages
    'unfreeze_epoch': 10,  # Unfreeze after this epoch
}

# Device
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =============================================================================
# POWER SPECTRUM (copied for standalone use)
# =============================================================================

def compute_power_spectrum(field: np.ndarray, n_bins: int = 32) -> np.ndarray:
    """Compute radially-averaged 2D power spectrum."""
    fft = np.fft.fft2(field)
    fft_shifted = np.fft.fftshift(fft)
    power_2d = np.abs(fft_shifted) ** 2
    
    h, w = field.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    
    r_max = min(cx, cy)
    bin_edges = np.linspace(0, r_max, n_bins + 1)
    power_spectrum = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
        if np.sum(mask) > 0:
            power_spectrum[i] = np.mean(power_2d[mask])
    
    power_spectrum = np.log10(power_spectrum + 1e-10)
    power_spectrum = (power_spectrum - np.mean(power_spectrum)) / (np.std(power_spectrum) + 1e-10)
    
    return power_spectrum.astype(np.float32)


# =============================================================================
# DATASETS
# =============================================================================

class SyntheticDataset(Dataset):
    """Dataset for synthetic training data."""
    
    def __init__(self, data_dir: str, metadata_file: str, augment: bool = False):
        self.data_dir = Path(data_dir)
        self.augment = augment
        
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
            if np.random.random() > 0.5:
                field = np.flip(field, axis=1).copy()
            if np.random.random() > 0.5:
                field = np.flip(field, axis=0).copy()
            k = np.random.randint(0, 4)
            field = np.rot90(field, k).copy()
        
        # Compute power spectrum
        power_spectrum = compute_power_spectrum(field)
        
        # Convert to tensors
        field_tensor = torch.from_numpy(field).unsqueeze(0)
        ps_tensor = torch.from_numpy(power_spectrum)
        target_tensor = torch.tensor([sample['omega_m']], dtype=torch.float32)
        
        return field_tensor, ps_tensor, target_tensor


class QuijoteDataset(Dataset):
    """Dataset for Quijote simulation data."""
    
    def __init__(self, data_dir: str, metadata_file: str, augment: bool = False):
        self.data_dir = Path(data_dir)
        self.augment = augment
        
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
        
        # Augmentation (more aggressive for small dataset)
        if self.augment:
            if np.random.random() > 0.5:
                field = np.flip(field, axis=1).copy()
            if np.random.random() > 0.5:
                field = np.flip(field, axis=0).copy()
            k = np.random.randint(0, 4)
            field = np.rot90(field, k).copy()
            
            # Add slight noise for regularization
            noise = np.random.normal(0, 0.02, field.shape).astype(np.float32)
            field = np.clip(field + noise, 0, 1)
        
        # Compute power spectrum
        power_spectrum = compute_power_spectrum(field)
        
        # Convert to tensors
        field_tensor = torch.from_numpy(field).unsqueeze(0)
        ps_tensor = torch.from_numpy(power_spectrum)
        target_tensor = torch.tensor([sample['omega_m']], dtype=torch.float32)
        
        return field_tensor, ps_tensor, target_tensor


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def setup_logging(run_name: str) -> str:
    """Setup logging."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, f"{run_name}.log")
    
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


def freeze_stages(model: nn.Module, num_stages: int):
    """Freeze the first N stages of the model."""
    # Freeze stem
    if num_stages >= 1:
        for param in model.stem.parameters():
            param.requires_grad = False
        logging.info("Frozen: stem")
    
    # Freeze stages
    for i in range(1, num_stages + 1):
        stage_name = f"stage{i}"
        if hasattr(model, stage_name):
            for param in getattr(model, stage_name).parameters():
                param.requires_grad = False
            logging.info(f"Frozen: {stage_name}")


def unfreeze_all(model: nn.Module):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True
    logging.info("Unfrozen all parameters")


def train_epoch(model, loader, criterion, optimizer, device, use_ps: bool = True):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch in loader:
        if use_ps:
            images, power_spectra, targets = batch
            images = images.to(device)
            power_spectra = power_spectra.to(device)
            targets = targets.to(device)
            outputs = model(images, power_spectra)
        else:
            images, _, targets = batch
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
        
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(model, loader, criterion, device, use_ps: bool = True):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in loader:
            if use_ps:
                images, power_spectra, targets = batch
                images = images.to(device)
                power_spectra = power_spectra.to(device)
                targets = targets.to(device)
                outputs = model(images, power_spectra)
            else:
                images, _, targets = batch
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
            
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    return total_loss / len(loader)


# =============================================================================
# MAIN TRAINING LOOPS
# =============================================================================

def pretrain(run_name: str, model_type: str = "resnet"):
    """Stage 1: Pre-train on synthetic data."""
    setup_logging(run_name)
    device = get_device()
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    logging.info(f"=== PRE-TRAINING: {run_name} ===")
    logging.info(f"Device: {device}")
    logging.info(f"Model type: {model_type}")
    
    # Create model
    if model_type == "resnet":
        model = CosmoResNet().to(device)
        use_ps = False
    elif model_type == "attention":
        model = CosmoAttentionNet().to(device)
        use_ps = False
    elif model_type == "hybrid":
        model = CosmoNetV2Hybrid().to(device)
        use_ps = True
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    logging.info(f"Model parameters: {count_parameters(model):,}")
    
    # Load data
    logging.info("Loading synthetic data...")
    full_dataset = SyntheticDataset(SYNTHETIC_DATA_DIR, SYNTHETIC_METADATA, augment=True)
    
    # Split
    n_total = len(full_dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=PRETRAIN_CONFIG['batch_size'], 
                              shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=PRETRAIN_CONFIG['batch_size'] * 2,
                            shuffle=False, num_workers=8, pin_memory=True)
    
    logging.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=PRETRAIN_CONFIG['learning_rate'],
                      weight_decay=PRETRAIN_CONFIG['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=PRETRAIN_CONFIG['num_epochs'])
    
    # Initialize wandb
    wandb.init(project="cosmo-scanner", name=run_name, config={
        "stage": "pretrain",
        "model_type": model_type,
        **PRETRAIN_CONFIG
    })
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(MODEL_DIR, f"{run_name}_best.pt")
    
    for epoch in range(1, PRETRAIN_CONFIG['num_epochs'] + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, use_ps)
        val_loss = validate(model, val_loader, criterion, device, use_ps)
        scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        logging.info(f"Epoch {epoch:03d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr:.6f}")
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr})
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'model_type': model_type,
            }, best_model_path)
            logging.info(f"  -> Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= PRETRAIN_CONFIG['patience']:
                logging.info(f"Early stopping at epoch {epoch}")
                break
    
    logging.info(f"Pre-training complete. Best val loss: {best_val_loss:.6f}")
    logging.info(f"Model saved: {best_model_path}")
    wandb.finish()


def finetune(run_name: str, pretrained_name: str):
    """Stage 2: Fine-tune on Quijote data."""
    setup_logging(run_name)
    device = get_device()
    
    logging.info(f"=== FINE-TUNING: {run_name} ===")
    logging.info(f"Pre-trained model: {pretrained_name}")
    logging.info(f"Device: {device}")
    
    # Load pre-trained model
    pretrained_path = os.path.join(MODEL_DIR, f"{pretrained_name}_best.pt")
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pre-trained model not found: {pretrained_path}")
    
    checkpoint = torch.load(pretrained_path, weights_only=True)
    model_type = checkpoint.get('model_type', 'resnet')
    
    # Create model
    if model_type == "resnet":
        model = CosmoResNet().to(device)
        use_ps = False
    elif model_type == "attention":
        model = CosmoAttentionNet().to(device)
        use_ps = False
    elif model_type == "hybrid":
        model = CosmoNetV2Hybrid().to(device)
        use_ps = True
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Loaded pre-trained weights from epoch {checkpoint['epoch']}")
    logging.info(f"Pre-trained val loss: {checkpoint['val_loss']:.6f}")
    
    # Freeze early stages
    freeze_stages(model, FINETUNE_CONFIG['freeze_stages'])
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Trainable parameters: {trainable_params:,} / {count_parameters(model):,}")
    
    # Load Quijote data
    logging.info("Loading Quijote data...")
    full_dataset = QuijoteDataset(QUIJOTE_DATA_DIR, QUIJOTE_METADATA, augment=True)
    
    # Split (80/20 for small dataset)
    n_total = len(full_dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=FINETUNE_CONFIG['batch_size'],
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=FINETUNE_CONFIG['batch_size'] * 2,
                            shuffle=False, num_workers=4, pin_memory=True)
    
    logging.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # Training setup with discriminative learning rates
    # Lower LR for frozen layers (when unfrozen), higher for head
    param_groups = [
        {'params': model.fc.parameters(), 'lr': FINETUNE_CONFIG['learning_rate']},
    ]
    
    # Add other parameters with lower LR
    other_params = [p for n, p in model.named_parameters() 
                    if 'fc' not in n and p.requires_grad]
    if other_params:
        param_groups.append({
            'params': other_params, 
            'lr': FINETUNE_CONFIG['learning_rate'] * 0.1
        })
    
    criterion = nn.MSELoss()
    optimizer = AdamW(param_groups, weight_decay=FINETUNE_CONFIG['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=FINETUNE_CONFIG['num_epochs'])
    
    # Initialize wandb
    wandb.init(project="cosmo-scanner", name=run_name, config={
        "stage": "finetune",
        "pretrained": pretrained_name,
        "model_type": model_type,
        **FINETUNE_CONFIG
    })
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(MODEL_DIR, f"{run_name}_best.pt")
    
    for epoch in range(1, FINETUNE_CONFIG['num_epochs'] + 1):
        # Gradual unfreezing
        if epoch == FINETUNE_CONFIG['unfreeze_epoch']:
            logging.info(f"Epoch {epoch}: Unfreezing all layers")
            unfreeze_all(model)
            # Reset optimizer with all parameters
            optimizer = AdamW(model.parameters(), 
                            lr=FINETUNE_CONFIG['learning_rate'] * 0.1,
                            weight_decay=FINETUNE_CONFIG['weight_decay'])
            scheduler = CosineAnnealingLR(optimizer, 
                                         T_max=FINETUNE_CONFIG['num_epochs'] - epoch)
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, use_ps)
        val_loss = validate(model, val_loader, criterion, device, use_ps)
        scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        logging.info(f"Epoch {epoch:03d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr:.6f}")
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr})
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'model_type': model_type,
                'pretrained': pretrained_name,
            }, best_model_path)
            logging.info(f"  -> Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= FINETUNE_CONFIG['patience']:
                logging.info(f"Early stopping at epoch {epoch}")
                break
    
    logging.info(f"Fine-tuning complete. Best val loss: {best_val_loss:.6f}")
    logging.info(f"Model saved: {best_model_path}")
    wandb.finish()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tuning pipeline")
    parser.add_argument("--stage", type=str, required=True, 
                       choices=["pretrain", "finetune"],
                       help="Training stage")
    parser.add_argument("--run_name", type=str, required=True,
                       help="Name for this run")
    parser.add_argument("--pretrained", type=str, default=None,
                       help="Pre-trained model name (for finetune stage)")
    parser.add_argument("--model", type=str, default="resnet",
                       choices=["resnet", "attention", "hybrid"],
                       help="Model architecture (for pretrain stage)")
    
    args = parser.parse_args()
    
    if args.stage == "pretrain":
        pretrain(args.run_name, args.model)
    elif args.stage == "finetune":
        if args.pretrained is None:
            raise ValueError("--pretrained required for finetune stage")
        finetune(args.run_name, args.pretrained)


if __name__ == "__main__":
    main()
