#!/bin/bash
#===============================================================================
# SLURM Job Script: Multi-GPU Training (Single Node)
#===============================================================================
# Distributed Data Parallel training on multiple GPUs within a single node.
# Uses torchrun for process management.
#
# Usage:
#   sbatch hpc/slurm/multi_gpu.sh
#   sbatch hpc/slurm/multi_gpu.sh --export=GPUS=8,EPOCHS=200
#===============================================================================

#SBATCH --job-name=cosmo-multi
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=256G

# Email notifications (uncomment and set your email)
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=your.email@domain.edu

#-------------------------------------------------------------------------------
# CONFIGURATION (can be overridden via --export)
#-------------------------------------------------------------------------------
GPUS=${GPUS:-4}
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-64}        # Per GPU
LR=${LR:-3e-4}
DATA_TYPE=${DATA_TYPE:-des}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-100000}
VAL_SAMPLES=${VAL_SAMPLES:-10000}
NUM_WORKERS=${NUM_WORKERS:-8}       # Per GPU

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
echo "COSMO-SCANNER HPC: Multi-GPU Training (DDP)"
echo "==============================================================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "CPUs:          $SLURM_CPUS_PER_TASK"
echo "GPUs:          $GPUS"
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
# module load nccl/2.18

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
# GPU DIAGNOSTICS
#-------------------------------------------------------------------------------
echo ""
echo "[GPU] Diagnostics:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'Device count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

#-------------------------------------------------------------------------------
# DISTRIBUTED SETUP
#-------------------------------------------------------------------------------
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / GPUS))

# NCCL optimizations
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

echo ""
echo "[DDP] Distributed configuration:"
echo "  Master addr:  $MASTER_ADDR"
echo "  Master port:  $MASTER_PORT"
echo "  OMP threads:  $OMP_NUM_THREADS"

#-------------------------------------------------------------------------------
# RUN TRAINING
#-------------------------------------------------------------------------------
EFFECTIVE_BATCH=$((BATCH_SIZE * GPUS))

echo ""
echo "==============================================================================="
echo "[TRAIN] Starting multi-GPU training..."
echo "==============================================================================="
echo "  GPUs:              $GPUS"
echo "  Epochs:            $EPOCHS"
echo "  Batch size/GPU:    $BATCH_SIZE"
echo "  Effective batch:   $EFFECTIVE_BATCH"
echo "  Learning rate:     $LR"
echo "  Data type:         $DATA_TYPE"
echo "  Train samples:     $TRAIN_SAMPLES"
echo "  Val samples:       $VAL_SAMPLES"
echo "  Workers/GPU:       $NUM_WORKERS"
echo "==============================================================================="

torchrun \
    --standalone \
    --nproc_per_node=$GPUS \
    hpc/train_hpc.py \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --data-type $DATA_TYPE \
    --train-samples $TRAIN_SAMPLES \
    --val-samples $VAL_SAMPLES \
    --num-workers $NUM_WORKERS \
    --wandb-project cosmo-scanner-hpc \
    --wandb-run-name "${GPUS}gpu_${DATA_TYPE}_${SLURM_JOB_ID}"

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
