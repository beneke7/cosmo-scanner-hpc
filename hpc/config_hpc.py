"""
config_hpc.py - HPC-Optimized Configuration
============================================

Configuration for HPC training with full optimizations:
- Multi-GPU (DDP) support
- Mixed precision (AMP)
- Gradient checkpointing
- torch.compile
- Optimized DataLoader
- W&B integration

Environment Variables:
    WANDB_API_KEY: Your W&B API key
    WANDB_PROJECT: Project name (default: cosmo-scanner-hpc)
    SLURM_JOB_ID: Auto-set by SLURM
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class DataConfig:
    """Data generation and loading configuration."""
    # Data type: 'des', '2lpt', 'synthetic', 'grf'
    data_type: str = 'des'
    
    # Image properties
    image_size: int = 256
    n_power_bins: int = 64
    
    # Dataset sizes (for 100K dataset: 90K train, 10K val)
    train_samples: int = 90_000
    val_samples: int = 10_000
    
    # Omega_m range
    omega_m_min: float = 0.1
    omega_m_max: float = 0.5
    
    # DES-specific
    smoothing_arcmin: float = 10.0
    
    # Augmentation
    augment: bool = True
    noise_std: float = 0.05
    
    # DataLoader (conservative - leave CPU headroom for other users)
    num_workers: int = 32  # Adjust based on available cores
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Model type
    model_type: str = 'hybrid'  # 'hybrid', 'resnet', 'simple'
    
    # Architecture
    n_power_bins: int = 64
    dropout: float = 0.5
    use_attention: bool = True
    
    # torch.compile settings (PyTorch 2.0+)
    # NOTE: 'reduce-overhead' uses CUDA graphs which are incompatible with dynamic FFT ops
    compile_model: bool = False  # Disabled - causes CUDA graph errors with power spectrum
    compile_mode: str = 'default'  # 'default', 'reduce-overhead', 'max-autotune'


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Basic
    epochs: int = 100
    batch_size: int = 128  # Per GPU (256 total for 2x RTX PRO 6000 96GB)
    
    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    
    # Scheduler (CosineAnnealingWarmRestarts)
    scheduler_type: str = 'cosine_warm_restarts'
    T_0: int = 10
    T_mult: int = 2
    eta_min: float = 1e-6
    
    # Warmup
    warmup_epochs: int = 5
    warmup_start_lr: float = 1e-6
    
    # Early stopping
    patience: int = 20
    min_delta: float = 1e-6
    
    # Gradient clipping
    grad_clip_norm: float = 1.0
    
    # Mixed precision
    use_amp: bool = True
    
    # Gradient checkpointing (saves memory, slower)
    gradient_checkpointing: bool = False
    
    # Gradient accumulation (effective batch = batch_size * accumulation_steps * num_gpus)
    accumulation_steps: int = 1


@dataclass
class DistributedConfig:
    """Distributed training configuration."""
    # Backend
    backend: str = 'nccl'  # 'nccl' for GPU, 'gloo' for CPU
    
    # Multi-node
    master_addr: str = 'localhost'
    master_port: str = '29500'
    
    # Sync batch norm
    sync_batchnorm: bool = True
    
    # Find unused parameters (for complex models)
    find_unused_parameters: bool = False


@dataclass
class WandbConfig:
    """Weights & Biases configuration."""
    enabled: bool = True
    project: str = 'cosmo-scanner-hpc'
    entity: Optional[str] = None  # Your W&B username/team
    
    # Run naming
    run_name: Optional[str] = None  # Auto-generated if None
    tags: List[str] = field(default_factory=lambda: ['hpc', 'v1.0'])
    
    # Logging
    log_interval: int = 10  # Log every N batches
    log_gradients: bool = False  # Expensive but useful for debugging
    log_model: bool = True  # Save model to W&B
    
    # Offline mode (for clusters without internet)
    offline: bool = False


@dataclass
class PathConfig:
    """Path configuration."""
    # Project root (auto-detected)
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    
    # Output directories
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'
    
    # Data directories
    data_dir: str = 'data'
    synthetic_data_dir: str = 'data/synthetic'
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        (self.project_root / self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        (self.project_root / self.log_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class HPCConfig:
    """Master configuration combining all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False  # True = slower but reproducible
    
    def to_dict(self) -> dict:
        """Convert to flat dictionary for W&B logging."""
        result = {}
        for field_name in ['data', 'model', 'training', 'distributed', 'wandb']:
            sub_config = getattr(self, field_name)
            for key, value in sub_config.__dict__.items():
                result[f'{field_name}.{key}'] = value
        result['seed'] = self.seed
        result['deterministic'] = self.deterministic
        return result
    
    @classmethod
    def from_slurm_env(cls) -> 'HPCConfig':
        """Create config from SLURM environment variables."""
        config = cls()
        
        # Auto-detect number of workers based on CPUs
        cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 8))
        config.data.num_workers = max(4, cpus // 2)
        
        # Auto-set W&B run name from SLURM job
        job_id = os.environ.get('SLURM_JOB_ID', 'local')
        job_name = os.environ.get('SLURM_JOB_NAME', 'cosmo')
        config.wandb.run_name = f'{job_name}_{job_id}'
        
        # Add SLURM info to tags
        if 'SLURM_JOB_ID' in os.environ:
            config.wandb.tags.append(f'slurm_{job_id}')
        
        return config


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

def get_debug_config() -> HPCConfig:
    """Small config for debugging."""
    config = HPCConfig()
    config.data.train_samples = 1000
    config.data.val_samples = 200
    config.training.epochs = 5
    config.training.batch_size = 16
    config.model.compile_model = False
    config.wandb.enabled = False
    return config


def get_single_gpu_config() -> HPCConfig:
    """Optimized for single RTX PRO 6000 (96GB)."""
    config = HPCConfig()
    config.training.batch_size = 256  # Large batch for 96GB VRAM
    config.data.num_workers = 48  # Use ~half of 128 cores
    return config


def get_dual_gpu_config() -> HPCConfig:
    """Optimized for 2x RTX PRO 6000 (96GB each) - YOUR SETUP."""
    config = HPCConfig()
    config.training.batch_size = 128  # Per GPU (256 effective)
    config.data.num_workers = 32  # Per GPU (64 total, leaves headroom)
    config.data.train_samples = 800_000
    config.data.val_samples = 100_000
    config.distributed.sync_batchnorm = True
    return config


def get_multi_gpu_config(num_gpus: int = 2) -> HPCConfig:
    """Optimized for multi-GPU on single node."""
    config = HPCConfig()
    config.training.batch_size = 128  # Per GPU
    config.data.num_workers = 32  # Per GPU
    config.distributed.sync_batchnorm = True
    return config


def get_multi_node_config(num_nodes: int = 2, gpus_per_node: int = 4) -> HPCConfig:
    """Optimized for multi-node training."""
    config = HPCConfig()
    config.training.batch_size = 64  # Per GPU (smaller for communication overhead)
    config.data.num_workers = 16
    config.distributed.sync_batchnorm = True
    config.training.accumulation_steps = 2  # Reduce communication frequency
    return config


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    # Print default config
    config = HPCConfig()
    print("Default HPC Configuration:")
    print("=" * 60)
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
