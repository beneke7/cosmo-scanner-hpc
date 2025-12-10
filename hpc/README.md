# HPC Training Module

**Fully optimized training pipeline for HPC clusters with NVIDIA GPUs (H100/A100).**

## Features

- **Distributed Data Parallel (DDP)** - Multi-GPU and multi-node training
- **Mixed Precision (AMP)** - FP16/BF16 for 2x speedup
- **torch.compile** - Kernel fusion for additional speedup
- **Gradient Checkpointing** - Trade compute for memory
- **W&B Integration** - Full experiment tracking
- **SLURM Support** - Ready-to-use job scripts
- **Automatic Checkpointing** - Resume from any point

## Quick Start

### Single GPU
```bash
# Direct execution
python hpc/train_hpc.py --epochs 100 --batch-size 128

# Via SLURM
sbatch hpc/slurm/single_gpu.sh
```

### Multi-GPU (Single Node)
```bash
# Direct execution with torchrun
torchrun --nproc_per_node=4 hpc/train_hpc.py --epochs 100

# Via SLURM
sbatch hpc/slurm/multi_gpu.sh
```

### Multi-Node
```bash
# Via SLURM (recommended)
sbatch hpc/slurm/multi_node.sh
```

## Directory Structure

```
hpc/
├── __init__.py          # Module exports
├── config_hpc.py        # Configuration dataclasses
├── train_hpc.py         # Main training script
├── README.md            # This file
└── slurm/
    ├── single_gpu.sh    # Single GPU job script
    ├── multi_gpu.sh     # Multi-GPU (single node) job script
    ├── multi_node.sh    # Multi-node job script
    └── generate_data.sh # Data generation job script
```

## Configuration

### Command Line Arguments

```bash
python hpc/train_hpc.py --help

# Key arguments:
--epochs          # Number of training epochs (default: 100)
--batch-size      # Batch size per GPU (default: 64)
--lr              # Learning rate (default: 3e-4)
--data-type       # Data type: des, 2lpt, synthetic (default: des)
--train-samples   # Number of training samples (default: 100000)
--val-samples     # Number of validation samples (default: 10000)
--num-workers     # DataLoader workers per GPU (default: 8)
--no-compile      # Disable torch.compile
--gradient-checkpointing  # Enable gradient checkpointing
--wandb-project   # W&B project name
--wandb-run-name  # W&B run name
--no-wandb        # Disable W&B logging
--resume          # Resume from checkpoint path
--debug           # Debug mode (small dataset)
```

### Configuration Presets

```python
from hpc import get_single_gpu_config, get_multi_gpu_config, get_debug_config

# Debug (fast iteration)
config = get_debug_config()

# Single GPU (H100/A100)
config = get_single_gpu_config()

# Multi-GPU (4x GPUs)
config = get_multi_gpu_config(num_gpus=4)

# Multi-node (2 nodes × 4 GPUs)
config = get_multi_node_config(num_nodes=2, gpus_per_node=4)
```

## SLURM Job Submission

### Customize Job Parameters

Override parameters via `--export`:

```bash
# Custom epochs and data type
sbatch hpc/slurm/single_gpu.sh --export=EPOCHS=200,DATA_TYPE=2lpt

# Custom batch size and learning rate
sbatch hpc/slurm/multi_gpu.sh --export=BATCH_SIZE=128,LR=1e-4

# More GPUs
sbatch hpc/slurm/multi_gpu.sh --export=GPUS=8
```

### Modify SLURM Directives

Edit the scripts to change:
- `--partition` - Your cluster's GPU partition name
- `--gres` - GPU type (e.g., `gpu:h100:1`, `gpu:a100:4`)
- `--time` - Wall time limit
- `--mem` - Memory allocation
- Module loads - Your cluster's module names

### Monitor Jobs

```bash
# Check queue
squeue -u $USER

# Check job output
tail -f logs/slurm/cosmo-1gpu_<JOB_ID>.out

# Cancel job
scancel <JOB_ID>
```

## W&B Integration

### Setup

```bash
# Login to W&B (one-time)
wandb login

# Or set API key
export WANDB_API_KEY="your-api-key"
```

### Offline Mode

For clusters without internet access:

```bash
python hpc/train_hpc.py --wandb-offline

# Sync later
wandb sync logs/wandb/offline-run-*
```

### Disable W&B

```bash
python hpc/train_hpc.py --no-wandb
```

## Performance Optimization

### Batch Size Scaling

| Setup | Batch Size/GPU | Effective Batch | Notes |
|-------|---------------|-----------------|-------|
| 1 GPU | 128 | 128 | Maximum for H100 80GB |
| 4 GPU | 64 | 256 | Good balance |
| 8 GPU | 32 | 256 | Communication overhead |
| 2×4 GPU | 32 | 256 | Multi-node overhead |

### Learning Rate Scaling

For linear scaling with batch size:
```
lr_new = lr_base × (effective_batch / base_batch)
```

Example: If `lr=3e-4` works for batch 64, use `lr=6e-4` for batch 128.

### Memory Optimization

If running out of memory:

1. **Reduce batch size** - Most effective
2. **Enable gradient checkpointing** - `--gradient-checkpointing`
3. **Reduce workers** - `--num-workers 4`
4. **Disable compile** - `--no-compile`

### Speed Optimization

1. **Increase batch size** - Better GPU utilization
2. **More workers** - `--num-workers 12`
3. **Enable compile** - Default on (PyTorch 2.0+)
4. **Use BF16** - Faster than FP16 on H100

## Checkpointing

### Automatic Saves

- `checkpoints/<run_name>_latest.pt` - Every epoch
- `checkpoints/<run_name>_best.pt` - Best validation loss

### Resume Training

```bash
python hpc/train_hpc.py --resume checkpoints/my_run_latest.pt
```

### Checkpoint Contents

```python
{
    'epoch': int,
    'model_state_dict': dict,
    'optimizer_state_dict': dict,
    'scheduler_state_dict': dict,
    'scaler_state_dict': dict,
    'val_loss': float,
    'config': dict
}
```

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
--batch-size 32

# Enable gradient checkpointing
--gradient-checkpointing

# Reduce workers
--num-workers 4
```

### NCCL Timeout (Multi-node)

```bash
# Increase timeout
export NCCL_TIMEOUT=1800

# Check network interface
export NCCL_SOCKET_IFNAME=eth0  # or ib0 for InfiniBand
```

### Slow DataLoader

```bash
# Increase workers
--num-workers 12

# Pre-generate data
sbatch hpc/slurm/generate_data.sh
python hpc/train_hpc.py --data-type synthetic
```

### W&B Connection Issues

```bash
# Use offline mode
--wandb-offline

# Sync later
wandb sync logs/wandb/offline-run-*
```

## Example Workflows

### Quick Test
```bash
python hpc/train_hpc.py --debug --no-wandb
```

### Production Single GPU
```bash
sbatch hpc/slurm/single_gpu.sh --export=EPOCHS=200,TRAIN_SAMPLES=200000
```

### Production Multi-GPU
```bash
sbatch hpc/slurm/multi_gpu.sh --export=GPUS=4,EPOCHS=100,BATCH_SIZE=64
```

### Hyperparameter Sweep
```bash
for lr in 1e-4 3e-4 1e-3; do
    sbatch hpc/slurm/single_gpu.sh --export=LR=$lr
done
```

## Comparison: Original vs HPC

| Feature | Original (`train/train.py`) | HPC (`hpc/train_hpc.py`) |
|---------|----------------------------|--------------------------|
| Multi-GPU | ❌ | ✅ DDP |
| Multi-node | ❌ | ✅ |
| torch.compile | ❌ | ✅ |
| Gradient checkpointing | ❌ | ✅ |
| Warmup scheduler | ❌ | ✅ |
| SLURM integration | Basic | Full |
| Resume training | Basic | Full |
| Config system | Dict | Dataclasses |
