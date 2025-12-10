#!/bin/bash
# =============================================================================
# Cosmo Scanner - HPC Training Script (Direct Run)
# =============================================================================
#
# Run training directly without SLURM. For SLURM jobs, use:
#   sbatch hpc/slurm/single_gpu.sh
#
# USAGE:
#   ./run_hpc.sh                          # Single GPU, default settings
#   ./run_hpc.sh --epochs 200             # Custom epochs
#   ./run_hpc.sh --debug                  # Quick test run
#   ./run_hpc.sh --generate-data          # Generate 1M dataset first
#
# =============================================================================

#===============================================================================
# >>> USER CONFIG - EDIT THESE VALUES <<<
#===============================================================================

# GPU selection (0 = first GPU, 1 = second GPU)
GPU_ID=${GPU_ID:-0}

# Training defaults (can be overridden via command line)
DEFAULT_EPOCHS=100
DEFAULT_BATCH_SIZE=128
DEFAULT_TRAIN_SAMPLES=90000       # 90K train from 100K dataset
DEFAULT_VAL_SAMPLES=10000         # 10K val from 100K dataset
DEFAULT_NUM_WORKERS=32

#===============================================================================

set -e
cd "$(dirname "$0")"

# Parse special arguments
GENERATE_DATA=false
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --generate-data)
            GENERATE_DATA=true
            shift
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Print header
echo "=============================================="
echo "Cosmo Scanner HPC Training"
echo "=============================================="
echo "GPU: $GPU_ID"
echo "=============================================="

# Data generation mode
if [ "$GENERATE_DATA" = true ]; then
    echo "Generating 100K synthetic dataset (JPG format)..."
    python3 hpc/generate_data_parallel.py \
        --num_samples 100000 \
        --num_workers 60 \
        --output_dir data/synthetic
    echo "Data generation complete!"
    exit 0
fi

# Run training with defaults (can be overridden)
python3 hpc/train_hpc.py \
    --epochs $DEFAULT_EPOCHS \
    --batch-size $DEFAULT_BATCH_SIZE \
    --train-samples $DEFAULT_TRAIN_SAMPLES \
    --val-samples $DEFAULT_VAL_SAMPLES \
    --num-workers $DEFAULT_NUM_WORKERS \
    $EXTRA_ARGS

echo "=============================================="
echo "Training complete!"
echo "=============================================="
