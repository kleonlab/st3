#!/bin/bash
#SBATCH --job-name=SEDD-Generation
#SBATCH --account=gts-agarg35
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=rtx_6000:1
#SBATCH --mem-per-gpu=80G
#SBATCH --time=4:00:00
#SBATCH --qos=embers
#SBATCH --output=slurm_out/sedd_generation-%A.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amete7@gatech.edu

# SLURM batch script for running generation inference with trained SEDD model
#
# Usage:
#   sbatch slurm/sb-inference-generation.sbatch --experiment_dir experiments/rnaseq_diffusion_20240115_123456
#
# Or with custom parameters:
#   sbatch slurm/sb-inference-generation.sbatch --experiment_dir experiments/my_run --num_generate 500 --num_steps 100

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
NUM_GENERATE="${NUM_GENERATE:-100}"
NUM_STEPS="${NUM_STEPS:-100}"
TEMPERATURE="${TEMPERATURE:-1.0}"
NUM_CELLS_VISUALIZE="${NUM_CELLS_VISUALIZE:-3}"
NUM_REAL_VISUALIZE="${NUM_REAL_VISUALIZE:-3}"

# Print inference configuration
echo "Generation Inference Configuration:"
echo "Experiment dir: $EXPERIMENT_DIR"
echo "Num cells to generate: $NUM_GENERATE"
echo "Sampling steps: $NUM_STEPS"
echo "Temperature: $TEMPERATURE"
echo "Num generated cells to visualize: $NUM_CELLS_VISUALIZE"
echo "Num real cells to visualize: $NUM_REAL_VISUALIZE"
echo ""

# Check if experiment directory is provided
if [ -z "$EXPERIMENT_DIR" ]; then
    echo "Error: EXPERIMENT_DIR not provided"
    echo "Usage: sbatch slurm/sb-inference-generation.sbatch --experiment_dir experiments/your_run"
    exit 1
fi

# Build the command arguments
CMD_ARGS=(
    --experiment_dir "$EXPERIMENT_DIR"
    --num_generate "$NUM_GENERATE"
    --num_steps "$NUM_STEPS"
    --temperature "$TEMPERATURE"
    --num_cells_visualize "$NUM_CELLS_VISUALIZE"
    --num_real_visualize "$NUM_REAL_VISUALIZE"
)

# Run inference with srun
srun python scripts/inference_generation.py "${CMD_ARGS[@]}" "$@"

# Print completion information
echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
