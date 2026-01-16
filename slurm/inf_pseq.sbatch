#!/bin/bash
#SBATCH --job-name=SEDD-PerturbSeq-Inference
#SBATCH --account=gts-agarg35
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=rtx_6000:1
#SBATCH --mem-per-gpu=80G
#SBATCH --time=4:00:00
#SBATCH --qos=embers
#SBATCH --output=slurm_out/sedd_perturbseq_inference-%A.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amete7@gatech.edu

# SLURM batch script for running perturbation prediction inference with trained SEDD model
#
# Usage:
#   sbatch slurm/inf_pseq.sbatch --experiment_dir experiments/perturbseq_20240115_123456
#
# Or with custom parameters:
#   sbatch slurm/inf_pseq.sbatch --experiment_dir experiments/my_run --config configs/perturbseq_dry_run.yaml --num_steps 50

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
CONFIG="${CONFIG:-configs/perturbseq_dry_run.yaml}"
TEST_DATA_PATH="${TEST_DATA_PATH:-}"
NUM_STEPS="${NUM_STEPS:-50}"
TEMPERATURE="${TEMPERATURE:-1.0}"
NUM_BATCHES="${NUM_BATCHES:-}"
NUM_CELLS_VISUALIZE="${NUM_CELLS_VISUALIZE:-3}"
BATCH_SIZE="${BATCH_SIZE:-32}"

# Print inference configuration
echo "Perturbation Prediction Inference Configuration:"
echo "Experiment dir: $EXPERIMENT_DIR"
echo "Config file: $CONFIG"
echo "Test data path: $TEST_DATA_PATH"
echo "Sampling steps: $NUM_STEPS"
echo "Temperature: $TEMPERATURE"
echo "Batch size: $BATCH_SIZE"
echo "Num batches: $NUM_BATCHES"
echo "Num cells to visualize: $NUM_CELLS_VISUALIZE"
echo ""

# Check if experiment directory is provided
if [ -z "$EXPERIMENT_DIR" ]; then
    echo "Error: EXPERIMENT_DIR not provided"
    echo "Usage: sbatch slurm/inf_pseq.sbatch --experiment_dir experiments/your_run"
    exit 1
fi

# Build the command arguments
CMD_ARGS=(
    --config "$CONFIG"
    --experiment_dir "$EXPERIMENT_DIR"
    --num_steps "$NUM_STEPS"
    --temperature "$TEMPERATURE"
    --batch_size "$BATCH_SIZE"
    --num_cells_visualize "$NUM_CELLS_VISUALIZE"
)

# Add optional arguments if provided
if [ -n "$TEST_DATA_PATH" ]; then
    CMD_ARGS+=(--data_path "$TEST_DATA_PATH")
fi

if [ -n "$NUM_BATCHES" ]; then
    CMD_ARGS+=(--num_batches "$NUM_BATCHES")
fi

# Run inference with srun
srun python scripts/inference_perturbseq.py "${CMD_ARGS[@]}" "$@"

# Print completion information
echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
