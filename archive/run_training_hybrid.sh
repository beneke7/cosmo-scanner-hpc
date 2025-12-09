#!/bin/bash
# =============================================================================
# run_training_hybrid.sh - Launch hybrid model training
# =============================================================================
#
# Hybrid Model: CNN + Power Spectrum
# - Branch 1: CNN extracts spatial features from images
# - Branch 2: MLP processes power spectrum (scale-dependent info)
# - Fusion: Combined prediction for improved accuracy
#
# Expected improvement: RMSE 0.05 → ~0.01-0.02
#
# Usage:
#     ./run_training_hybrid.sh <run_name>
#
# Example:
#     ./run_training_hybrid.sh hybrid_v1
#
# =============================================================================

set -e

# Check argument
if [ -z "$1" ]; then
    echo "Usage: ./run_training_hybrid.sh <run_name>"
    echo "Example: ./run_training_hybrid.sh hybrid_v1"
    exit 1
fi

RUN_NAME=$1

# Navigate to project root
cd "$(dirname "$0")"

# Set environment variables
export KMP_DUPLICATE_LIB_OK=TRUE
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run training
echo "Starting HYBRID model training: $RUN_NAME"
echo "=============================================="
echo "Model: CNN + Power Spectrum (32 bins)"
echo "GPU: RTX 5090 (32GB) | Batch: 128 | AMP: ON"
echo "=============================================="

python3 train/main_hybrid.py --run_name "$RUN_NAME"

echo "=============================================="
echo "Hybrid training complete: $RUN_NAME"
