#!/bin/bash
#SBATCH --job-name=celltype-inf
#SBATCH --account=brics.b5cc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=128G
#SBATCH --time=04:00:00
#SBATCH --qos=normal
#SBATCH --output=slurm_out/celltype_inf-%A.out
#SBATCH --mail-type=END,FAIL

# SLURM batch script for Cell-Type Conditioned Inference
#
# This script generates perturbed cells for a SPECIFIC CELL TYPE.
# The key parameter is CELL_TYPE which controls which cell type to generate.
#
# Usage:
#   # Generate for T cells
#   CELL_TYPE="T cell" EXPERIMENT_DIR=experiments/celltype_model \
#       sbatch celltype_cond/slurm/sb-inference-celltype.sbatch
#
#   # Generate for all cell types (one job per cell type - use array jobs or loop)
#   for ct in "T cell" "B cell" "Monocyte"; do
#       CELL_TYPE="$ct" EXPERIMENT_DIR=experiments/celltype_model \
#           sbatch celltype_cond/slurm/sb-inference-celltype.sbatch
#   done

# Change to project directory
cd /home/b5cc/sanjukta.b5cc/st3 || exit 1

# Create slurm output directory
mkdir -p slurm_out

# Load CUDA
module load cuda/12.6 2>/dev/null || module load cuda 2>/dev/null || echo "Warning: Could not load CUDA module"

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Check CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || true

# Print job information
echo "=========================================="
echo "Cell-Type Conditioned Inference"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Parameters
CONFIG="${CONFIG:-celltype_cond/configs/celltype_inference_example.yaml}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-}"
CELL_TYPE="${CELL_TYPE:-}"
DATA_PATH="${DATA_PATH:-}"
PERTURBATIONS_FILE="${PERTURBATIONS_FILE:-}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

# Print configuration
echo "Inference Configuration:"
echo "Config: $CONFIG"
echo "Experiment dir: $EXPERIMENT_DIR"
if [ -n "$CELL_TYPE" ]; then
    echo "Cell type: $CELL_TYPE"
else
    echo "Cell type: ALL"
fi
if [ -n "$DATA_PATH" ]; then
    echo "Data path: $DATA_PATH"
fi
echo "Samples per condition: $NUM_SAMPLES"
echo ""

# Validate required arguments
if [ -z "$EXPERIMENT_DIR" ]; then
    echo "Error: EXPERIMENT_DIR is required"
    exit 1
fi

# Build command
CMD="python celltype_cond/scripts/inference_celltype_perturbseq.py"
CMD="$CMD --config $CONFIG"
CMD="$CMD --experiment_dir $EXPERIMENT_DIR"
CMD="$CMD --num_samples_per_condition $NUM_SAMPLES"

if [ -n "$CELL_TYPE" ]; then
    CMD="$CMD --cell_type \"$CELL_TYPE\""
fi
if [ -n "$DATA_PATH" ]; then
    CMD="$CMD --data_path $DATA_PATH"
fi
if [ -n "$PERTURBATIONS_FILE" ]; then
    CMD="$CMD --perturbations_file $PERTURBATIONS_FILE"
fi
if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output_dir $OUTPUT_DIR"
fi

# Run inference
echo "Running: $CMD"
echo ""
srun bash -c "$CMD"

# Print completion
echo ""
echo "=========================================="
echo "Inference completed at: $(date)"
echo "=========================================="
