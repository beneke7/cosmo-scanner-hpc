#!/bin/bash
#===============================================================================
# SLURM Job Script: Data Generation
#===============================================================================
# Pre-generate synthetic training data using parallel CPU processing.
#
# Usage:
#   sbatch hpc/slurm/generate_data.sh
#   # Or override via command line:
#   sbatch hpc/slurm/generate_data.sh --export=NUM_SAMPLES=500000
#===============================================================================

#SBATCH --job-name=cosmo-datagen
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G

#===============================================================================
# >>> USER CONFIG - EDIT THESE VALUES <<<
#===============================================================================

# Dataset size
NUM_SAMPLES=${NUM_SAMPLES:-100000}       # Total samples to generate

# CPU usage (leave headroom for other users)
NUM_WORKERS=${NUM_WORKERS:-60}           # Parallel workers (< cpus-per-task)

# Output settings
OUTPUT_DIR=${OUTPUT_DIR:-data/synthetic}
IMAGE_SIZE=${IMAGE_SIZE:-256}

#===============================================================================

#-------------------------------------------------------------------------------
# ENVIRONMENT SETUP
#-------------------------------------------------------------------------------
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

mkdir -p logs/slurm

echo "==============================================================================="
echo "COSMO-SCANNER HPC: Data Generation"
echo "==============================================================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "CPUs:          $SLURM_CPUS_PER_TASK"
echo "Workers:       $NUM_WORKERS"
echo "Samples:       $NUM_SAMPLES"
echo "Output:        $OUTPUT_DIR"
echo "Time:          $(date)"
echo "==============================================================================="

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

#-------------------------------------------------------------------------------
# GENERATE DATA
#-------------------------------------------------------------------------------
echo ""
echo "[DATA] Generating $NUM_SAMPLES synthetic samples with $NUM_WORKERS workers..."

python hpc/generate_data_parallel.py \
    --num_samples $NUM_SAMPLES \
    --num_workers $NUM_WORKERS \
    --output_dir $OUTPUT_DIR \
    --size $IMAGE_SIZE

#-------------------------------------------------------------------------------
# SUMMARY
#-------------------------------------------------------------------------------
echo ""
echo "==============================================================================="
echo "[DONE] Data generation complete!"
echo "==============================================================================="
echo "Output directory: $PROJECT_ROOT/$OUTPUT_DIR"
echo "End time: $(date)"

# Show generated files
ls -lh "$OUTPUT_DIR"
wc -l "$OUTPUT_DIR/metadata.csv"

deactivate
exit 0
