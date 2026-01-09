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

# SLURM batch script for running imputation inference with trained SEDD model using YAML config
#
# Usage:
#   sbatch slurm/sb-inference-imputation.sbatch --config configs/rnaseq_small.yaml --experiment_dir experiments/rnaseq_small
#
# Or with environment variables:
#   CONFIG=configs/rnaseq_small.yaml EXPERIMENT_DIR=experiments/rnaseq_small sbatch slurm/sb-inference-imputation.sbatch

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

# Default parameters (can be overridden by environment variables)
CONFIG="${CONFIG:-configs/rnaseq_small.yaml}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-}"
DATA_PATH="${DATA_PATH:-}"

# Print inference configuration
echo "Imputation Inference Configuration:"
echo "Config file: $CONFIG"
echo "Experiment dir: $EXPERIMENT_DIR"
if [ -n "$DATA_PATH" ]; then
    echo "Data path: $DATA_PATH"
fi
echo ""

# Check if experiment directory is provided
if [ -z "$EXPERIMENT_DIR" ]; then
    echo "Error: EXPERIMENT_DIR not provided"
    echo "Usage: sbatch slurm/sb-inference-imputation.sbatch --experiment_dir experiments/your_run"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

# Build the command
CMD="python scripts/inference_imputation.py --config $CONFIG --experiment_dir $EXPERIMENT_DIR"

# Add data path if provided
if [ -n "$DATA_PATH" ]; then
    CMD="$CMD --data_path $DATA_PATH"
fi

# Add any additional command line arguments passed to sbatch
CMD="$CMD $@"

# Run inference with srun
echo "Running command: $CMD"
echo ""
srun $CMD

# Print completion information
echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
