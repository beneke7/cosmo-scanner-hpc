#!/bin/bash
# =============================================================================
# Cosmo Scanner - Training Script (v5.3)
# =============================================================================
#
# STEP 1: Generate synthetic data (run once)
#   python scripts/generate_synthetic_data.py --num_samples 100000
#   python scripts/generate_synthetic_data.py --preview_only  # Just preview
#
# STEP 2: Train on pre-generated data
#   ./run.sh --epochs 50               # Train v5.3 on synthetic data
#   ./run.sh --run_name my_exp         # Custom run name
#
# v5.3 Features:
#   - Pre-generated synthetic data (faster training)
#   - Tunable parameters in scripts/generate_synthetic_data.py
#   - Red power spectrum, smoothing, masks, grain
#
# =============================================================================

set -e

cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Set environment variables for optimal GPU performance
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run training
echo "=============================================="
echo "Cosmo Scanner v5.3 Training"
echo "=============================================="

python3 train/train.py "$@"

echo "=============================================="
echo "Training complete!"
