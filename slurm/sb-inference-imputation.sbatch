#!/bin/bash
#SBATCH --job-name=SEDD-Imputation
#SBATCH --account=gts-agarg35
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=rtx_6000:1
#SBATCH --mem-per-gpu=80G
#SBATCH --time=4:00:00
#SBATCH --qos=embers
#SBATCH --output=slurm_out/sedd_imputation-%A.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amete7@gatech.edu

# SLURM batch script for running imputation inference with trained SEDD model
#
# Usage:
#   sbatch slurm/sb-inference-imputation.sbatch --experiment_dir experiments/rnaseq_diffusion_20240115_123456
#
# Or with custom parameters:
#   sbatch slurm/sb-inference-imputation.sbatch --experiment_dir experiments/my_run --mask_ratio 0.3 --num_steps 100

# Change to project directory
cd /storage/project/r-agark35-0/amete7/vla/st3 || exit 1

# Create slurm output directory if it doesn't exist
mkdir -p slurm_out

# Activate virtual environment (adjust path as needed)
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Warning: No virtual environment found. Using system Python."
fi

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Print job information
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo ""

# Default inference parameters (can be overridden by environment variables or passing arguments to sbatch)
EXPERIMENT_DIR="${EXPERIMENT_DIR:-}"
MASK_RATIO="${MASK_RATIO:-0.2}"
NUM_STEPS="${NUM_STEPS:-50}"
TEMPERATURE="${TEMPERATURE:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_BATCHES="${NUM_BATCHES:-}"
NUM_CELLS_VISUALIZE="${NUM_CELLS_VISUALIZE:-3}"

# Print inference configuration
echo "Imputation Inference Configuration:"
echo "Experiment dir: $EXPERIMENT_DIR"
echo "Mask ratio: $MASK_RATIO"
echo "Sampling steps: $NUM_STEPS"
echo "Temperature: $TEMPERATURE"
echo "Batch size: $BATCH_SIZE"
echo "Num batches: ${NUM_BATCHES:-all}"
echo "Num cells to visualize: $NUM_CELLS_VISUALIZE"
echo ""

# Check if experiment directory is provided
if [ -z "$EXPERIMENT_DIR" ]; then
    echo "Error: EXPERIMENT_DIR not provided"
    echo "Usage: sbatch slurm/sb-inference-imputation.sbatch --experiment_dir experiments/your_run"
    exit 1
fi

# Build the command arguments
CMD_ARGS=(
    --experiment_dir "$EXPERIMENT_DIR"
    --mask_ratio "$MASK_RATIO"
    --num_steps "$NUM_STEPS"
    --temperature "$TEMPERATURE"
    --batch_size "$BATCH_SIZE"
    --num_cells_visualize "$NUM_CELLS_VISUALIZE"
)

# Add optional arguments if set
if [ -n "$NUM_BATCHES" ]; then
    CMD_ARGS+=(--num_batches "$NUM_BATCHES")
fi

# Run inference with srun
srun python scripts/inference_imputation.py "${CMD_ARGS[@]}" "$@"

# Print completion information
echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
