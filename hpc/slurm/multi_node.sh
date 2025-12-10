#!/bin/bash
#===============================================================================
# SLURM Job Script: Multi-Node Training
#===============================================================================
# Distributed Data Parallel training across multiple nodes.
# Uses srun for cross-node process management.
#
# Usage:
#   sbatch hpc/slurm/multi_node.sh
#   sbatch hpc/slurm/multi_node.sh --export=NODES=4,GPUS_PER_NODE=4
#===============================================================================

#SBATCH --job-name=cosmo-nodes
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --exclusive

# Email notifications (uncomment and set your email)
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=your.email@domain.edu

#-------------------------------------------------------------------------------
# CONFIGURATION (can be overridden via --export)
#-------------------------------------------------------------------------------
NODES=${SLURM_NNODES:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-32}        # Per GPU (smaller for multi-node)
LR=${LR:-3e-4}
DATA_TYPE=${DATA_TYPE:-des}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-200000}
VAL_SAMPLES=${VAL_SAMPLES:-20000}
NUM_WORKERS=${NUM_WORKERS:-6}       # Per GPU

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

# Calculate total GPUs
TOTAL_GPUS=$((NODES * GPUS_PER_NODE))

# Print job info
echo "==============================================================================="
echo "COSMO-SCANNER HPC: Multi-Node Training"
echo "==============================================================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Nodes:         $SLURM_NODELIST ($NODES nodes)"
echo "GPUs/node:     $GPUS_PER_NODE"
echo "Total GPUs:    $TOTAL_GPUS"
echo "CPUs/task:     $SLURM_CPUS_PER_TASK"
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
# module load openmpi/4.1

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
# DISTRIBUTED SETUP
#-------------------------------------------------------------------------------
# Get master node address
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

export MASTER_ADDR
export MASTER_PORT
export WORLD_SIZE=$TOTAL_GPUS
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# NCCL optimizations for multi-node
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_SOCKET_IFNAME=ib0  # InfiniBand interface (adjust for your cluster)

# For clusters with InfiniBand
# export NCCL_IB_HCA=mlx5_0

echo ""
echo "[DDP] Multi-node configuration:"
echo "  Master addr:  $MASTER_ADDR"
echo "  Master port:  $MASTER_PORT"
echo "  World size:   $WORLD_SIZE"
echo "  OMP threads:  $OMP_NUM_THREADS"

#-------------------------------------------------------------------------------
# GPU DIAGNOSTICS (on master node)
#-------------------------------------------------------------------------------
echo ""
echo "[GPU] Diagnostics (master node):"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'Device count: {torch.cuda.device_count()}')
"

#-------------------------------------------------------------------------------
# RUN TRAINING
#-------------------------------------------------------------------------------
EFFECTIVE_BATCH=$((BATCH_SIZE * TOTAL_GPUS))

echo ""
echo "==============================================================================="
echo "[TRAIN] Starting multi-node training..."
echo "==============================================================================="
echo "  Nodes:             $NODES"
echo "  GPUs/node:         $GPUS_PER_NODE"
echo "  Total GPUs:        $TOTAL_GPUS"
echo "  Epochs:            $EPOCHS"
echo "  Batch size/GPU:    $BATCH_SIZE"
echo "  Effective batch:   $EFFECTIVE_BATCH"
echo "  Learning rate:     $LR"
echo "  Data type:         $DATA_TYPE"
echo "  Train samples:     $TRAIN_SAMPLES"
echo "  Val samples:       $VAL_SAMPLES"
echo "  Workers/GPU:       $NUM_WORKERS"
echo "==============================================================================="

# Use srun to launch across nodes
srun --ntasks=$TOTAL_GPUS \
     --ntasks-per-node=$GPUS_PER_NODE \
     --cpus-per-task=$SLURM_CPUS_PER_TASK \
     python hpc/train_hpc.py \
     --epochs $EPOCHS \
     --batch-size $BATCH_SIZE \
     --lr $LR \
     --data-type $DATA_TYPE \
     --train-samples $TRAIN_SAMPLES \
     --val-samples $VAL_SAMPLES \
     --num-workers $NUM_WORKERS \
     --wandb-project cosmo-scanner-hpc \
     --wandb-run-name "${NODES}n${GPUS_PER_NODE}g_${DATA_TYPE}_${SLURM_JOB_ID}"

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
