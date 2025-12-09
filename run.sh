#!/bin/bash
# =============================================================================
# Cosmo Scanner - Training Script (v0.5.0)
# =============================================================================
#
# Usage:
#   ./run.sh                           # DES-like weak lensing (default)
#   ./run.sh --data_type des           # DES-like weak lensing
#   ./run.sh --data_type 2lpt          # N-body like (for Quijote)
#   ./run.sh --data_type disk          # Pre-generated data
#   ./run.sh --smoothing 5.0           # Change smoothing scale
#   ./run.sh --epochs 100              # More epochs
#
# =============================================================================

set -e

cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run training
echo "=============================================="
echo "Cosmo Scanner v0.5.0 Training"
echo "=============================================="

python3 train/train.py "$@"

echo "=============================================="
echo "Training complete!"
