#!/usr/bin/env python3
"""
train_hpc.py - HPC-Optimized Training Pipeline
===============================================

Full-featured training script optimized for HPC clusters with:
- Distributed Data Parallel (DDP) for multi-GPU/multi-node
- Mixed precision training (AMP)
- torch.compile for kernel fusion
- Gradient checkpointing for memory efficiency
- W&B integration for experiment tracking
- Automatic checkpointing and resume
- SLURM-aware configuration

Usage:
    # Single GPU
    python hpc/train_hpc.py --epochs 100
    
    # Multi-GPU (via torchrun)
    torchrun --nproc_per_node=4 hpc/train_hpc.py --epochs 100
    
    # Multi-node (via SLURM)
    srun python hpc/train_hpc.py --epochs 100

Author: Cosmo Scanner HPC Team
Version: 1.0.0
"""

import os
import sys
import argparse
import logging
import time
import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_hybrid import CosmoNetHybrid, count_parameters
from hpc.config_hpc import HPCConfig, get_single_gpu_config, get_debug_config

# Optional: wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


# =============================================================================
# DISTRIBUTED UTILITIES
# =============================================================================

def setup_distributed() -> Tuple[int, int, int]:
    """
    Initialize distributed training.
    
    Returns:
        (rank, local_rank, world_size)
    """
    # Check if we're in a distributed environment
    if 'RANK' in os.environ:
        # Launched via torchrun or srun
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    elif 'SLURM_PROCID' in os.environ:
        # Launched via SLURM without torchrun
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ.get('SLURM_LOCALID', rank))
        world_size = int(os.environ['SLURM_NTASKS'])
        
        # Set environment for torch.distributed
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        
        # Get master address from SLURM
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = os.environ.get('SLURM_LAUNCH_NODE_IPADDR', 'localhost')
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '29500'
    else:
        # Single GPU
        rank = 0
        local_rank = 0
        world_size = 1
    
    # Initialize process group if distributed
    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
    
    return rank, local_rank, world_size


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """Check if this is the main process."""
    return rank == 0


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(rank: int, log_dir: Path, run_name: str) -> logging.Logger:
    """Setup logging for distributed training."""
    logger = logging.getLogger('cosmo_hpc')
    logger.setLevel(logging.INFO if is_main_process(rank) else logging.WARNING)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler (main process only)
    if is_main_process(rank):
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        ))
        logger.addHandler(console)
        
        # File handler
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / f'{run_name}.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(file_handler)
    
    return logger


# =============================================================================
# DATASET
# =============================================================================

class PreGeneratedDataset(torch.utils.data.Dataset):
    """
    Dataset that loads pre-generated JPG images from disk.
    Much faster than online generation - enables high GPU utilization.
    """
    
    def __init__(
        self,
        data_dir: Path,
        num_samples: Optional[int] = None,
        augment: bool = True,
        noise_std: float = 0.02,
        is_validation: bool = False
    ):
        import pandas as pd
        from PIL import Image
        
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / 'images'
        self.augment = augment and not is_validation
        self.noise_std = noise_std if not is_validation else 0.0
        self.is_validation = is_validation
        
        # Load metadata
        metadata_path = self.data_dir / 'metadata.csv'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        self.metadata = pd.read_csv(metadata_path)
        
        # Limit samples if specified
        if num_samples is not None and num_samples < len(self.metadata):
            if is_validation:
                # Use last N samples for validation (deterministic)
                self.metadata = self.metadata.tail(num_samples).reset_index(drop=True)
            else:
                # Use first N samples for training
                self.metadata = self.metadata.head(num_samples).reset_index(drop=True)
        
        print(f"Loaded {len(self.metadata)} samples from {data_dir}")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        from PIL import Image
        
        row = self.metadata.iloc[idx]
        img_path = self.images_dir / row['filename']
        
        # Load image
        img = Image.open(img_path)
        field = np.array(img, dtype=np.float32) / 255.0
        
        # Augmentation
        if self.augment:
            if np.random.random() > 0.5:
                field = np.flip(field, axis=1).copy()
            if np.random.random() > 0.5:
                field = np.flip(field, axis=0).copy()
            k = np.random.randint(0, 4)
            field = np.rot90(field, k).copy()
            
            if self.noise_std > 0:
                noise = np.random.normal(0, self.noise_std, field.shape).astype(np.float32)
                field = np.clip(field + noise, 0, 1)
        
        field_tensor = torch.from_numpy(field).unsqueeze(0).float()
        target_tensor = torch.tensor([row['omega_m']], dtype=torch.float32)
        
        return field_tensor, target_tensor


class OnlineDatasetHPC(torch.utils.data.Dataset):
    """
    HPC-optimized online dataset with deterministic validation.
    
    Generates cosmological fields on-the-fly for training,
    uses fixed seeds for validation reproducibility.
    
    NOTE: This is slower than PreGeneratedDataset. Use only if you need
    fresh random samples each epoch.
    """
    
    def __init__(
        self,
        num_samples: int,
        omega_m_range: Tuple[float, float] = (0.1, 0.5),
        size: int = 256,
        data_type: str = 'des',
        smoothing_arcmin: float = 10.0,
        augment: bool = True,
        noise_std: float = 0.05,
        is_validation: bool = False,
        rank: int = 0,
        world_size: int = 1
    ):
        self.num_samples = num_samples
        self.omega_m_range = omega_m_range
        self.size = size
        self.data_type = data_type
        self.smoothing_arcmin = smoothing_arcmin
        self.augment = augment and not is_validation
        self.noise_std = noise_std if not is_validation else 0.0
        self.is_validation = is_validation
        
        # Pre-generate omega_m values for validation (deterministic)
        if is_validation:
            np.random.seed(42)
            self.omega_m_values = np.random.uniform(*omega_m_range, num_samples)
        else:
            self.omega_m_values = None
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Import here to avoid issues with multiprocessing
        from src.physics_lensing import generate_des_like_numpy
        from src.physics_lpt import generate_2lpt_numpy
        
        # Get omega_m
        if self.is_validation:
            omega_m = self.omega_m_values[idx]
            seed = 10000 + idx
        else:
            omega_m = np.random.uniform(*self.omega_m_range)
            seed = None
        
        # Generate field
        if self.data_type == '2lpt':
            field = generate_2lpt_numpy(omega_m, size=self.size, seed=seed)
            field = (field - field.min()) / (field.max() - field.min() + 1e-10)
        else:  # 'des'
            field = generate_des_like_numpy(
                omega_m, 
                size=self.size, 
                seed=seed,
                smoothing_arcmin=self.smoothing_arcmin,
                add_noise=True
            )
            field = (field - field.min()) / (field.max() - field.min() + 1e-10)
        
        # Augmentation
        if self.augment:
            if np.random.random() > 0.5:
                field = np.flip(field, axis=1).copy()
            if np.random.random() > 0.5:
                field = np.flip(field, axis=0).copy()
            k = np.random.randint(0, 4)
            field = np.rot90(field, k).copy()
            
            if self.noise_std > 0:
                noise = np.random.normal(0, self.noise_std, field.shape).astype(np.float32)
                field = np.clip(field + noise, 0, 1)
        
        field_tensor = torch.from_numpy(field).unsqueeze(0).float()
        target_tensor = torch.tensor([omega_m], dtype=torch.float32)
        
        return field_tensor, target_tensor


def create_dataloaders(
    config: HPCConfig,
    rank: int = 0,
    world_size: int = 1,
    data_dir: Optional[Path] = None
) -> Tuple[DataLoader, DataLoader]:
    """Create distributed dataloaders.
    
    Args:
        config: HPC configuration
        rank: Process rank for distributed training
        world_size: Total number of processes
        data_dir: Path to pre-generated data (if None, uses online generation)
    """
    
    # Check for pre-generated data
    if data_dir is None:
        data_dir = PROJECT_ROOT / 'data' / 'synthetic'
    
    use_pregenerated = (data_dir / 'metadata.csv').exists() and (data_dir / 'images').exists()
    
    if use_pregenerated:
        print(f"Using pre-generated dataset from: {data_dir}")
        train_dataset = PreGeneratedDataset(
            data_dir=data_dir,
            num_samples=config.data.train_samples,
            augment=config.data.augment,
            noise_std=config.data.noise_std,
            is_validation=False
        )
        
        val_dataset = PreGeneratedDataset(
            data_dir=data_dir,
            num_samples=config.data.val_samples,
            augment=False,
            noise_std=0.0,
            is_validation=True
        )
    else:
        print("WARNING: No pre-generated data found. Using slow online generation.")
        print(f"  Run './run_hpc.sh --generate-data' first for better performance.")
        train_dataset = OnlineDatasetHPC(
            num_samples=config.data.train_samples,
            omega_m_range=(config.data.omega_m_min, config.data.omega_m_max),
            size=config.data.image_size,
            data_type=config.data.data_type,
            smoothing_arcmin=config.data.smoothing_arcmin,
            augment=config.data.augment,
            noise_std=config.data.noise_std,
            is_validation=False,
            rank=rank,
            world_size=world_size
        )
        
        val_dataset = OnlineDatasetHPC(
            num_samples=config.data.val_samples,
            omega_m_range=(config.data.omega_m_min, config.data.omega_m_max),
            size=config.data.image_size,
            data_type=config.data.data_type,
            smoothing_arcmin=config.data.smoothing_arcmin,
            augment=False,
            noise_std=0.0,
            is_validation=True,
            rank=rank,
            world_size=world_size
        )
    
    # Distributed samplers
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=True
    ) if world_size > 1 else None
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    ) if world_size > 1 else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers,
        prefetch_factor=config.data.prefetch_factor,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size * 2,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.data.num_workers // 2,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers,
        prefetch_factor=config.data.prefetch_factor
    )
    
    return train_loader, val_loader


# =============================================================================
# MODEL SETUP
# =============================================================================

def setup_model(
    config: HPCConfig,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1
) -> nn.Module:
    """Create and setup model with all optimizations."""
    
    # Create model
    model = CosmoNetHybrid(
        n_power_bins=config.model.n_power_bins,
        dropout=config.model.dropout,
        use_attention=config.model.use_attention
    )
    
    # Move to device with channels_last memory format
    model = model.to(device, memory_format=torch.channels_last)
    
    # Gradient checkpointing
    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable() if hasattr(model, 'gradient_checkpointing_enable') else None
    
    # Compile model (PyTorch 2.0+)
    if config.model.compile_model and hasattr(torch, 'compile'):
        model = torch.compile(model, mode=config.model.compile_mode)
    
    # Distributed Data Parallel
    if world_size > 1:
        # Sync batch norm
        if config.distributed.sync_batchnorm:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        model = DDP(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=config.distributed.find_unused_parameters
        )
    
    return model


def setup_optimizations(device: torch.device):
    """Setup GPU optimizations."""
    if device.type == 'cuda':
        # TF32 for faster matmul on Ampere+
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # cuDNN benchmarking
        torch.backends.cudnn.benchmark = True


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    config: HPCConfig,
    epoch: int,
    logger: logging.Logger
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = len(loader)
    
    for batch_idx, (images, targets) in enumerate(loader):
        # Move to device with channels_last
        images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Forward pass with AMP
        with autocast('cuda', enabled=config.training.use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss = loss / config.training.accumulation_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % config.training.accumulation_steps == 0:
            # Gradient clipping
            if config.training.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    config.training.grad_clip_norm
                )
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        total_loss += loss.item() * config.training.accumulation_steps
        
        # Logging
        if batch_idx % config.wandb.log_interval == 0:
            logger.debug(f"Epoch {epoch} | Batch {batch_idx}/{num_batches} | Loss: {loss.item():.6f}")
    
    return total_loss / num_batches


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: HPCConfig
) -> Tuple[float, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    for images, targets in loader:
        images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with autocast('cuda', enabled=config.training.use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)
        
        total_loss += loss.item()
        all_preds.append(outputs.cpu())
        all_targets.append(targets.cpu())
    
    # Compute RMSE
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    rmse = torch.sqrt(torch.mean((preds - targets) ** 2)).item()
    
    return total_loss / len(loader), rmse


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: GradScaler,
    epoch: int,
    val_loss: float,
    config: HPCConfig,
    checkpoint_dir: Path,
    run_name: str,
    is_best: bool = False
):
    """Save training checkpoint."""
    # Get model state (handle DDP)
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'val_loss': val_loss,
        'config': config.to_dict()
    }
    
    # Save latest
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_dir / f'{run_name}_latest.pt')
    
    # Save best
    if is_best:
        torch.save(checkpoint, checkpoint_dir / f'{run_name}_best.pt')


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[GradScaler] = None
) -> int:
    """Load checkpoint and return starting epoch."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle DDP
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint['epoch']


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train(config: HPCConfig, resume_from: Optional[Path] = None):
    """Main training function."""
    
    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    
    # Device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    
    # Setup optimizations
    setup_optimizations(device)
    
    # Run name
    if config.wandb.run_name is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        config.wandb.run_name = f'hpc_{config.data.data_type}_{timestamp}'
    run_name = config.wandb.run_name
    
    # Logging
    log_dir = config.paths.project_root / config.paths.log_dir
    logger = setup_logging(rank, log_dir, run_name)
    
    # Log configuration
    if is_main_process(rank):
        logger.info("=" * 70)
        logger.info("COSMO-SCANNER HPC TRAINING")
        logger.info("=" * 70)
        logger.info(f"Run name: {run_name}")
        logger.info(f"Device: {device}")
        logger.info(f"World size: {world_size}")
        logger.info(f"Rank: {rank}")
        logger.info(f"Data type: {config.data.data_type}")
        logger.info(f"Train samples: {config.data.train_samples:,}")
        logger.info(f"Batch size (per GPU): {config.training.batch_size}")
        logger.info(f"Effective batch size: {config.training.batch_size * world_size * config.training.accumulation_steps}")
        logger.info("=" * 70)
    
    # Initialize W&B
    if is_main_process(rank) and config.wandb.enabled and WANDB_AVAILABLE:
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=run_name,
            tags=config.wandb.tags,
            config=config.to_dict(),
            mode='offline' if config.wandb.offline else 'online'
        )
    
    # Seed
    torch.manual_seed(config.seed + rank)
    np.random.seed(config.seed + rank)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config, rank, world_size)
    
    if is_main_process(rank):
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")
    
    # Create model
    model = setup_model(config, device, rank, world_size)
    
    if is_main_process(rank):
        param_count = count_parameters(model.module if hasattr(model, 'module') else model)
        logger.info(f"Model parameters: {param_count:,}")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Scheduler with warmup
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=config.training.warmup_start_lr / config.training.learning_rate,
        end_factor=1.0,
        total_iters=config.training.warmup_epochs
    )
    
    main_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.training.T_0,
        T_mult=config.training.T_mult,
        eta_min=config.training.eta_min
    )
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[config.training.warmup_epochs]
    )
    
    # Loss and scaler
    criterion = nn.MSELoss()
    scaler = GradScaler('cuda', enabled=config.training.use_amp)
    
    # Resume from checkpoint
    start_epoch = 1
    if resume_from is not None and resume_from.exists():
        start_epoch = load_checkpoint(resume_from, model, optimizer, scheduler, scaler) + 1
        if is_main_process(rank):
            logger.info(f"Resumed from epoch {start_epoch - 1}")
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    checkpoint_dir = config.paths.project_root / config.paths.checkpoint_dir
    
    for epoch in range(start_epoch, config.training.epochs + 1):
        epoch_start = time.time()
        
        # Set epoch for distributed sampler
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, config, epoch, logger
        )
        
        # Validate
        val_loss, val_rmse = validate(model, val_loader, criterion, device, config)
        
        # Scheduler step
        scheduler.step()
        
        # Gather metrics across processes
        if world_size > 1:
            train_loss_tensor = torch.tensor([train_loss], device=device)
            val_loss_tensor = torch.tensor([val_loss], device=device)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.AVG)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
            train_loss = train_loss_tensor.item()
            val_loss = val_loss_tensor.item()
        
        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]['lr']
        
        # Logging
        if is_main_process(rank):
            logger.info(
                f"Epoch {epoch:03d} | "
                f"Train: {train_loss:.6f} | "
                f"Val: {val_loss:.6f} | "
                f"RMSE: {val_rmse:.4f} | "
                f"LR: {lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )
            
            # W&B logging
            if config.wandb.enabled and WANDB_AVAILABLE:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_rmse': val_rmse,
                    'learning_rate': lr,
                    'epoch_time': epoch_time
                })
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
                logger.info(f"  -> New best model! RMSE: {val_rmse:.4f}")
            else:
                patience_counter += 1
            
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch, val_loss, config, checkpoint_dir, run_name, is_best
            )
            
            # Early stopping
            if patience_counter >= config.training.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
    
    # Final logging
    if is_main_process(rank):
        logger.info("=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Best validation loss: {best_val_loss:.6f}")
        logger.info(f"Best RMSE: {np.sqrt(best_val_loss):.4f}")
        logger.info(f"Checkpoints saved to: {checkpoint_dir}")
        
        if config.wandb.enabled and WANDB_AVAILABLE:
            wandb.finish()
    
    # Cleanup
    cleanup_distributed()
    
    return best_val_loss


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='HPC-optimized training for Cosmo Scanner',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    
    # Data
    parser.add_argument('--data-type', type=str, default='des',
                        choices=['des', '2lpt', 'synthetic'],
                        help='Data type')
    parser.add_argument('--train-samples', type=int, default=100000, help='Training samples')
    parser.add_argument('--val-samples', type=int, default=10000, help='Validation samples')
    parser.add_argument('--num-workers', type=int, default=8, help='DataLoader workers per GPU')
    
    # Model
    parser.add_argument('--no-compile', action='store_true', help='Disable torch.compile')
    parser.add_argument('--gradient-checkpointing', action='store_true', help='Enable gradient checkpointing')
    
    # W&B
    parser.add_argument('--wandb-project', type=str, default='cosmo-scanner-hpc', help='W&B project')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='W&B run name')
    parser.add_argument('--no-wandb', action='store_true', help='Disable W&B')
    parser.add_argument('--wandb-offline', action='store_true', help='W&B offline mode')
    
    # Misc
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--debug', action='store_true', help='Debug mode (small dataset)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create config
    if args.debug:
        config = get_debug_config()
    else:
        config = HPCConfig.from_slurm_env()
    
    # Override with CLI args
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.data.data_type = args.data_type
    config.data.train_samples = args.train_samples
    config.data.val_samples = args.val_samples
    config.data.num_workers = args.num_workers
    config.model.compile_model = not args.no_compile
    config.training.gradient_checkpointing = args.gradient_checkpointing
    config.wandb.project = args.wandb_project
    config.wandb.run_name = args.wandb_run_name
    config.wandb.enabled = not args.no_wandb
    config.wandb.offline = args.wandb_offline
    config.seed = args.seed
    
    # Resume path
    resume_from = Path(args.resume) if args.resume else None
    
    # Train
    train(config, resume_from)


if __name__ == '__main__':
    main()
