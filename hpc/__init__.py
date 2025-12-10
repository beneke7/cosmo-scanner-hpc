"""
HPC-optimized training module for Cosmo Scanner.

This module provides:
- Distributed Data Parallel (DDP) training
- Mixed precision (AMP) support
- torch.compile optimization
- W&B integration
- SLURM-aware configuration
"""

from .config_hpc import (
    HPCConfig,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    DistributedConfig,
    WandbConfig,
    PathConfig,
    get_debug_config,
    get_single_gpu_config,
    get_dual_gpu_config,
    get_multi_gpu_config,
    get_multi_node_config
)

__all__ = [
    'HPCConfig',
    'DataConfig',
    'ModelConfig', 
    'TrainingConfig',
    'DistributedConfig',
    'WandbConfig',
    'PathConfig',
    'get_debug_config',
    'get_single_gpu_config',
    'get_multi_gpu_config',
    'get_multi_node_config'
]
