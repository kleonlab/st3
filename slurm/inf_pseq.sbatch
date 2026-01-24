#!/bin/bash
#SBATCH --job-name=SEDD-PSeq-Inf
#SBATCH --account=brics.b5cc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=64G
#SBATCH --time=8:00:00
#SBATCH --qos=normal
#SBATCH --output=slurm_out/sedd_pseq_inference-%A.out
#SBATCH --mail-type=END,FAIL

# SLURM batch script for SEDD Perturbation-seq inference using trained checkpoint
#
# Usage:
#   sbatch slurm/sb-inference-pseq.sbatch --config configs/perturbseq_dry_run.yaml --experiment_dir experiments/psed_demo/perturbseq_dry_run --data_path /path/to/test_data.h5ad
#
# Or with environment variables:
#   CONFIG=configs/perturbseq_dry_run.yaml EXPERIMENT_DIR=experiments/psed_demo/perturbseq_dry_run TEST_DATA_PATH=/path/to/test_data.h5ad sbatch slurm/sb-inference-pseq.sbatch

# Change to project directory
cd /home/b5cc/sanjukta.b5cc/st3 || exit 1

# Create slurm output directory if it doesn't exist
mkdir -p slurm_out

# Load CUDA module
module load cuda/12.6 2>/dev/null || module load cuda 2>/dev/null || echo "Warning: Could not load CUDA module"

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

# Memory optimization for large sequences
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Check CUDA availability
echo "CUDA check:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')" 2>/dev/null || echo "Could not check CUDA"

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
CONFIG="${CONFIG:-configs/perturbseq_small.yaml}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-/home/b5cc/sanjukta.b5cc/st3/experiments/5k_psed}"
TEST_DATA_PATH="${TEST_DATA_PATH:-}"

# Print inference configuration
echo "Inference Configuration:"
echo "Config file    : $CONFIG"
echo "Experiment dir : $EXPERIMENT_DIR"
if [ -n "$TEST_DATA_PATH" ]; then
    echo "Test data path : $TEST_DATA_PATH"
else
    echo "Test data path : (from config file)"
fi
echo ""

# Check if config file exists
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

# Check if experiment directory exists
if [ ! -d "$EXPERIMENT_DIR" ]; then
    echo "Error: Experiment directory not found: $EXPERIMENT_DIR"
    exit 1
fi

# Build the command
CMD="python scripts/inference_perturbseq.py --config $CONFIG --experiment_dir $EXPERIMENT_DIR"

# Add test data path if provided
if [ -n "$TEST_DATA_PATH" ]; then
    CMD="$CMD --data_path $TEST_DATA_PATH"
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
echo "Perturbation prediction complete!"
echo "Check ${EXPERIMENT_DIR}/perturbseq_inference_results for outputs"
echo "Job completed at: $(date)"
echo "=========================================="