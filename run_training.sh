#!/bin/bash
# =============================================================================
# run_training.sh - Launch training with a named run
# =============================================================================
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

# Run training
echo "Starting training run: $RUN_NAME"
echo "================================"

python train/main.py --run_name "$RUN_NAME"

echo "================================"
echo "Training complete: $RUN_NAME"
