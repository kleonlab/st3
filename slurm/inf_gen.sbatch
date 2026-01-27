#!/bin/bash
#SBATCH --job-name=SEDD-Gen-Inf
#SBATCH --account=brics.b5cc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=128G
#SBATCH --time=24:00:00
#SBATCH --qos=normal
#SBATCH --output=slurm_out/sedd_generation_inference-%A.out
#SBATCH --mail-type=END,FAIL

# SLURM batch script for SEDD Conditional inference using trained checkpoint
#
# Usage:
#   sbatch slurm/sb-inference-conditional.sbatch EXPERIMENT_DIR=experiments/your_exp [OPTIONS]
#
#
# perturbations.txt should contain one gene/perturbation name per line:
#   KRAS
#   TP53
#   EGFR
#   non-targeting

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

echo "=========================================="
echo "Generation Inference - SEDD"
echo "=========================================="
echo ""

if [ -z "$EXPERIMENT_DIR" ]; then
    echo "ERROR: EXPERIMENT_DIR is required"
    echo "Usage: EXPERIMENT_DIR=path/to/experiment sbatch slurm/inf_gen.sbatch"
    echo "   or: sbatch slurm/inf_gen.sbatch EXPERIMENT_DIR=path/to/experiment"
    exit 1
fi

# Default parameters (can be overridden by environment variables)
# Only CONFIG has a default - all other paths come from YAML config unless explicitly overridden
CONFIG="${CONFIG:-/home/b5cc/sanjukta.b5cc/st3/configs/rnaseq_small.yaml}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-/home/b5cc/sanjukta.b5cc/st3/experiments/mlm/128,4,4b}"  # Required
CHECKPOINT="${CHECKPOINT}"  # Optional specific checkpoint path
REFERENCE_DATA="${REFERENCE_DATA:-/home/b5cc/sanjukta.b5cc/st3/datasets/dentate/dentate.h5ad}"
OUTPUT_PATH="${OUTPUT_PATH}"  # Optional, defaults to experiment_dir/generated_cells.h5ad

NUM_GENERATE="${NUM_GENERATE:-100}"
NUM_STEPS="${NUM_STEPS:-50}"
TEMPERATURE="${TEMPERATURE:-1.0}"
SEED="${SEED:-42}"
KEEP_SPARSE="${KEEP_SPARSE:-false}"

echo "Experiment dir : ${EXPERIMENT_DIR}"
echo "Checkpoint     : ${CHECKPOINT:-auto (best/final/latest)}"
echo "Reference data : ${REFERENCE_DATA}"
echo "Output path    : ${OUTPUT_PATH:-${EXPERIMENT_DIR}/inference/generated_cells.h5ad}"
echo "Num generate   : ${NUM_GENERATE}"
echo "Num steps      : ${NUM_STEPS}"
echo "Temperature    : ${TEMPERATURE}"
echo "Seed           : ${SEED}"
echo "Keep sparse    : ${KEEP_SPARSE}"
echo "=========================================="
echo ""


if [ ! -d "$EXPERIMENT_DIR" ]; then
    echo "Error: Experiment directory not found: $EXPERIMENT_DIR"
    exit 1
fi

# Check if reference data exists
if [ ! -f "$REFERENCE_DATA" ]; then
    echo "Error: Reference data file not found: $REFERENCE_DATA"
    exit 1
fi

# Build the command
CMD="python scripts/inference_generation.py"

# Add required parameters
CMD="$CMD --experiment_dir $EXPERIMENT_DIR"
CMD="$CMD --config $CONFIG"
CMD="$CMD --reference_data $REFERENCE_DATA"
CMD="$CMD --num_generate $NUM_GENERATE"
CMD="$CMD --num_steps $NUM_STEPS"
CMD="$CMD --temperature $TEMPERATURE"
CMD="$CMD --seed $SEED"

# Add optional checkpoint if provided
if [ -n "$CHECKPOINT" ]; then
    CMD="$CMD --checkpoint $CHECKPOINT"
fi

# Add optional output path if provided
if [ -n "$OUTPUT_PATH" ]; then
    CMD="$CMD --output_path $OUTPUT_PATH"
fi

# Add keep_sparse flag if true
if [ "$KEEP_SPARSE" = "true" ]; then
    CMD="$CMD --keep_sparse"
fi

# Add any additional command line arguments passed to sbatch
CMD="$CMD $@"

# Run generation with srun
echo "Running command:"
echo "$CMD"
echo ""
srun $CMD

# Print completion information
echo ""
echo "=========================================="
echo "Generation complete!"
if [ -n "$OUTPUT_PATH" ]; then
    echo "Output saved to: ${OUTPUT_PATH}"
else
    echo "Output saved to: ${EXPERIMENT_DIR}/inference/generated_cells.h5ad"
fi
echo "Job completed at: $(date)"
echo "=========================================="