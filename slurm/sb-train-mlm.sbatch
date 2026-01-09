#!/bin/bash
#SBATCH --job-name=SEDD-Train
#SBATCH --account=gts-agarg35
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=rtx_6000:1
#SBATCH --mem-per-gpu=80G
#SBATCH --time=48:00:00
#SBATCH --qos=embers
#SBATCH --output=slurm_out/sedd_train-%A.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amete7@gatech.edu

# SLURM batch script for training SEDD RNA-seq model using YAML config
#
# Usage:
#   sbatch slurm/sb-train-mlm.sbatch --config configs/rnaseq_small.yaml --data_path /path/to/data.h5ad
#
# Or with environment variables:
#   CONFIG=configs/rnaseq_large.yaml sbatch slurm/sb-train-mlm.sbatch

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
DATA_PATH="${DATA_PATH:-}"

# Print training configuration
echo "Training Configuration:"
echo "Config file: $CONFIG"
if [ -n "$DATA_PATH" ]; then
    echo "Data path: $DATA_PATH"
fi
echo ""

# Check if config file exists
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

# Build the command
CMD="python scripts/train_rnaseq.py --config $CONFIG"

# Add data path if provided
if [ -n "$DATA_PATH" ]; then
    CMD="$CMD --data_path $DATA_PATH"
fi

# Add any additional command line arguments passed to sbatch
CMD="$CMD $@"

# Run training with srun
echo "Running command: $CMD"
echo ""
srun $CMD

# Print completion information
echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
