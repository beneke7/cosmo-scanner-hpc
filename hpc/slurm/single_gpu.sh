#!/bin/bash
#===============================================================================
# SLURM Job Script: Single GPU Training
#===============================================================================
# Optimized for RTX PRO 6000 (96GB) / H100 / A100 training.
#
# Usage:
#   sbatch hpc/slurm/single_gpu.sh
#   # Or override via command line:
#   sbatch hpc/slurm/single_gpu.sh --export=EPOCHS=200,DATA_TYPE=2lpt
#===============================================================================

#SBATCH --job-name=cosmo-1gpu
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1
#SBATCH --mem=128G

# Email notifications (uncomment and set your email)
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=your.email@domain.edu

#===============================================================================
# >>> USER CONFIG - EDIT THESE VALUES <<<
#===============================================================================

# Training parameters
EPOCHS=${EPOCHS:-100}                    # Number of epochs
BATCH_SIZE=${BATCH_SIZE:-128}            # Batch size (128-256 for 96GB GPU)
LR=${LR:-3e-4}                           # Learning rate

# Dataset
DATA_TYPE=${DATA_TYPE:-des}              # 'des', '2lpt', or 'synthetic'
TRAIN_SAMPLES=${TRAIN_SAMPLES:-80000}    # Training samples (from 100K dataset)
VAL_SAMPLES=${VAL_SAMPLES:-10000}        # Validation samples

# CPU workers (leave headroom for other users)
NUM_WORKERS=${NUM_WORKERS:-32}           # DataLoader workers

# W&B settings
WANDB_PROJECT=${WANDB_PROJECT:-cosmo-scanner-hpc}
WANDB_OFFLINE=${WANDB_OFFLINE:-false}    # Set to 'true' for offline mode

# GPU selection (0 or 1)
GPU_ID=${GPU_ID:-0}

#===============================================================================

#-------------------------------------------------------------------------------
# ENVIRONMENT SETUP
#-------------------------------------------------------------------------------
set -e

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

# Create log directories
mkdir -p logs/slurm checkpoints

# Print job info
echo "==============================================================================="
echo "COSMO-SCANNER HPC: Single GPU Training"
echo "==============================================================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "CPUs:          $SLURM_CPUS_PER_TASK"
echo "GPU:           $CUDA_VISIBLE_DEVICES"
echo "Time:          $(date)"
echo "Project Root:  $PROJECT_ROOT"
echo "==============================================================================="

#-------------------------------------------------------------------------------
# LOAD MODULES (adjust for your HPC system)
#-------------------------------------------------------------------------------
# Uncomment and modify based on your cluster's module system:
# module purge
# module load python/3.10
# module load cuda/12.1
# module load cudnn/8.9

#-------------------------------------------------------------------------------
# PYTHON ENVIRONMENT
#-------------------------------------------------------------------------------
VENV_DIR="$PROJECT_ROOT/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "[ENV] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source "$VENV_DIR/bin/activate"
fi

echo "[ENV] Python: $(which python)"
echo "[ENV] PyTorch: $(python -c 'import torch; print(torch.__version__)')"

#-------------------------------------------------------------------------------
# GPU SETUP
#-------------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo ""
echo "[GPU] Diagnostics (GPU $GPU_ID):"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv -i $GPU_ID
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

#-------------------------------------------------------------------------------
# RUN TRAINING
#-------------------------------------------------------------------------------
echo ""
echo "==============================================================================="
echo "[TRAIN] Starting single-GPU training..."
echo "==============================================================================="
echo "  Epochs:        $EPOCHS"
echo "  Batch size:    $BATCH_SIZE"
echo "  Learning rate: $LR"
echo "  Data type:     $DATA_TYPE"
echo "  Train samples: $TRAIN_SAMPLES"
echo "  Val samples:   $VAL_SAMPLES"
echo "  Workers:       $NUM_WORKERS"
echo "  GPU:           $GPU_ID"
echo "  W&B project:   $WANDB_PROJECT"
echo "==============================================================================="

# Build wandb args
WANDB_ARGS="--wandb-project $WANDB_PROJECT"
if [ "$WANDB_OFFLINE" = "true" ]; then
    WANDB_ARGS="$WANDB_ARGS --wandb-offline"
fi

python hpc/train_hpc.py \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --data-type $DATA_TYPE \
    --train-samples $TRAIN_SAMPLES \
    --val-samples $VAL_SAMPLES \
    --num-workers $NUM_WORKERS \
    $WANDB_ARGS \
    --wandb-run-name "1gpu_${DATA_TYPE}_${SLURM_JOB_ID:-local}"

#-------------------------------------------------------------------------------
# CLEANUP
#-------------------------------------------------------------------------------
echo ""
echo "==============================================================================="
echo "[DONE] Training complete!"
echo "==============================================================================="
echo "End time: $(date)"
echo "Checkpoints: $PROJECT_ROOT/checkpoints/"

deactivate
exit 0
