#!/bin/bash
#===============================================================================
# SLURM Job Script for cosmo-scanner-hpc
#===============================================================================
# This script is designed for HPC clusters with NVIDIA H100 GPUs.
# It implements a hybrid environment strategy:
#   1. Check if a virtual environment exists
#   2. If not, create one and install dependencies
#   3. Activate the venv and run training
#
# This approach isolates the HPC environment from any local Conda environments
# while using the same requirements.txt for consistency.
#===============================================================================

#-------------------------------------------------------------------------------
# SLURM DIRECTIVES
#-------------------------------------------------------------------------------
#SBATCH --job-name=cosmo-scanner          # Job name (visible in queue)
#SBATCH --output=logs/%x_%j.out           # Standard output (%x=job name, %j=job id)
#SBATCH --error=logs/%x_%j.err            # Standard error
#SBATCH --time=24:00:00                   # Wall time limit (HH:MM:SS)
#SBATCH --partition=gpu                   # Partition (queue) name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks (1 for single-GPU)
#SBATCH --cpus-per-task=16                # CPU cores for DataLoader workers
#SBATCH --gres=gpu:h100:1                 # GPU resources (1x H100)
#SBATCH --mem=64G                         # Memory per node
#SBATCH --mail-type=BEGIN,END,FAIL        # Email notifications
#SBATCH --mail-user=your.email@domain.edu # Your email (change this!)

#-------------------------------------------------------------------------------
# ENVIRONMENT CONFIGURATION
#-------------------------------------------------------------------------------

# Exit on any error
set -e

# Get the directory where this script lives
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Virtual environment path
VENV_DIR="$PROJECT_ROOT/.venv"

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/logs"

# Print job information
echo "==============================================================================="
echo "COSMO-SCANNER-HPC JOB STARTING"
echo "==============================================================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Job Name:      $SLURM_JOB_NAME"
echo "Node:          $SLURM_NODELIST"
echo "CPUs:          $SLURM_CPUS_PER_TASK"
echo "GPUs:          $CUDA_VISIBLE_DEVICES"
echo "Time:          $(date)"
echo "Working Dir:   $PROJECT_ROOT"
echo "==============================================================================="

#-------------------------------------------------------------------------------
# HYBRID ENVIRONMENT SETUP
#-------------------------------------------------------------------------------
# Philosophy: We want an isolated Python environment that doesn't depend on
# the user's local Conda setup, but uses the same requirements.txt.
# This ensures reproducibility across different HPC systems.
#-------------------------------------------------------------------------------

echo ""
echo "[ENV] Setting up Python environment..."

# Load required modules (adjust for your HPC system)
# Common module names - uncomment and modify as needed:
# module purge
# module load python/3.10
# module load cuda/12.1
# module load cudnn/8.9

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "[ENV] Virtual environment not found. Creating..."
    
    # Create virtual environment
    python3 -m venv "$VENV_DIR"
    
    # Activate it
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install dependencies
    echo "[ENV] Installing dependencies from requirements.txt..."
    pip install -r "$PROJECT_ROOT/requirements.txt"
    
    # Install the project itself in editable mode (if setup.py exists)
    if [ -f "$PROJECT_ROOT/setup.py" ] || [ -f "$PROJECT_ROOT/pyproject.toml" ]; then
        pip install -e "$PROJECT_ROOT"
    fi
    
    echo "[ENV] Virtual environment created and configured."
else
    echo "[ENV] Virtual environment found. Activating..."
    source "$VENV_DIR/bin/activate"
    
    # Optional: Check if requirements have changed and update
    # pip install -r "$PROJECT_ROOT/requirements.txt" --upgrade
fi

# Verify Python and key packages
echo "[ENV] Python: $(which python)"
echo "[ENV] Python version: $(python --version)"
echo "[ENV] PyTorch version: $(python -c 'import torch; print(torch.__version__)')"

#-------------------------------------------------------------------------------
# GPU DIAGNOSTICS
#-------------------------------------------------------------------------------

echo ""
echo "[GPU] Running GPU diagnostics..."
nvidia-smi

echo ""
python -c "
import torch
print(f'[GPU] CUDA available: {torch.cuda.is_available()}')
print(f'[GPU] CUDA version: {torch.version.cuda}')
print(f'[GPU] Device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'[GPU] Device {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)')
"

#-------------------------------------------------------------------------------
# TRAINING CONFIGURATION
#-------------------------------------------------------------------------------
# Adjust these parameters for your experiment

EPOCHS=100
BATCH_SIZE=64
LEARNING_RATE=1e-3
TRAIN_SAMPLES=50000
VAL_SAMPLES=5000
NUM_WORKERS=8  # Use fewer than SLURM_CPUS_PER_TASK to leave headroom

# Weights & Biases configuration (optional)
# export WANDB_API_KEY="your-api-key-here"
WANDB_PROJECT="cosmo-scanner-hpc"
WANDB_RUN_NAME="h100-${SLURM_JOB_ID}"

#-------------------------------------------------------------------------------
# RUN TRAINING
#-------------------------------------------------------------------------------

echo ""
echo "==============================================================================="
echo "[TRAIN] Starting training..."
echo "==============================================================================="
echo "  Epochs:        $EPOCHS"
echo "  Batch size:    $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Train samples: $TRAIN_SAMPLES"
echo "  Val samples:   $VAL_SAMPLES"
echo "  Workers:       $NUM_WORKERS"
echo "==============================================================================="

cd "$PROJECT_ROOT"

# Run training
# The training script will auto-detect CUDA and use the GPU
python -m src.train \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --train-samples $TRAIN_SAMPLES \
    --val-samples $VAL_SAMPLES \
    --num-workers $NUM_WORKERS \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "$WANDB_RUN_NAME"

# Alternative: Run training as a module with default settings
# python -c "from src.train import train; train(epochs=$EPOCHS, batch_size=$BATCH_SIZE)"

#-------------------------------------------------------------------------------
# CLEANUP AND SUMMARY
#-------------------------------------------------------------------------------

echo ""
echo "==============================================================================="
echo "[DONE] Training complete!"
echo "==============================================================================="
echo "End time: $(date)"
echo "Checkpoints saved to: $PROJECT_ROOT/checkpoints/"
echo "Logs saved to: $PROJECT_ROOT/logs/"

# Deactivate virtual environment
deactivate

exit 0
