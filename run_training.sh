#!/bin/bash
# =============================================================================
# run_training.sh - Launch training with a named run
# =============================================================================
#
# Optimized for:
#   - GPU: NVIDIA RTX 5090 (32GB VRAM)
#   - CPU: 60 cores
#   - RAM: 40GB
#
# Optimizations applied:
#   - Batch size: 256 (up from 32)
#   - DataLoader workers: 16 (up from 4)
#   - Mixed precision training (AMP) enabled
#   - Persistent workers for faster data loading
#   - Prefetch factor: 4
#
# Usage:
#     ./run_training.sh <run_name>
#
# Example:
#     ./run_training.sh baseline_v1
#
# =============================================================================

set -e

# Check argument
if [ -z "$1" ]; then
    echo "Usage: ./run_training.sh <run_name>"
    echo "Example: ./run_training.sh baseline_v1"
    exit 1
fi

RUN_NAME=$1

# Navigate to project root
cd "$(dirname "$0")"

# Set environment variables
export KMP_DUPLICATE_LIB_OK=TRUE

# Optional: Set CUDA optimizations
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run training
echo "Starting optimized training run: $RUN_NAME"
echo "=========================================="
echo "GPU: RTX 5090 (32GB) | Batch: 256 | AMP: ON"
echo "CPU Workers: 16 | Prefetch: 4"
echo "=========================================="

python train/main.py --run_name "$RUN_NAME"

echo "=========================================="
echo "Training complete: $RUN_NAME"
