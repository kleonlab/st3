#!/bin/bash
#SBATCH --job-name=celltype-train
#SBATCH --account=brics.b5cc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=256G
#SBATCH --time=24:00:00
#SBATCH --qos=normal
#SBATCH --output=slurm_out/celltype_train-%A.out
#SBATCH --mail-type=END,FAIL

# SLURM batch script for training Cell-Type Conditioned SEDD model
#
# Usage:
#   sbatch celltype_cond/slurm/sb-train-celltype.sbatch
#
# With environment variables:
#   CONFIG=celltype_cond/configs/celltype_perturbseq_medium.yaml \
#   TRAIN_DATA_PATH=/path/to/data.h5ad \
#   sbatch celltype_cond/slurm/sb-train-celltype.sbatch

# Change to project directory
cd /home/b5cc/sanjukta.b5cc/st3 || exit 1

# Create slurm output directory
mkdir -p slurm_out

# Load CUDA module
module load cuda/12.6 2>/dev/null || module load cuda 2>/dev/null || echo "Warning: Could not load CUDA module"

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Warning: No virtual environment found. Using system Python."
fi

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Check CUDA
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
CONFIG="${CONFIG:-celltype_cond/configs/celltype_perturbseq_small.yaml}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-}"
CELL_TYPE_COL="${CELL_TYPE_COL:-cell_type}"
COND_LABELS_PT_PATH="${COND_LABELS_PT_PATH:-}"
CELLTYPE_LABELS_PT_PATH="${CELLTYPE_LABELS_PT_PATH:-}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
RESUME="${RESUME:-auto}"

# Print configuration
echo "Training Configuration:"
echo "Config file: $CONFIG"
if [ -n "$TRAIN_DATA_PATH" ]; then
    echo "Data path: $TRAIN_DATA_PATH"
fi
echo "Cell-type column: $CELL_TYPE_COL"
if [ -n "$COND_LABELS_PT_PATH" ]; then
    echo "Perturbation embeddings: $COND_LABELS_PT_PATH"
fi
if [ -n "$CELLTYPE_LABELS_PT_PATH" ]; then
    echo "Cell-type embeddings: $CELLTYPE_LABELS_PT_PATH"
fi
if [ -n "$CHECKPOINT_DIR" ]; then
    echo "Checkpoint dir: $CHECKPOINT_DIR"
fi
echo "Resume: $RESUME"
echo ""

# Check config file
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

# Build command
CMD="python celltype_cond/scripts/train_celltype_perturbseq.py --config $CONFIG"

if [ -n "$TRAIN_DATA_PATH" ]; then
    CMD="$CMD --train_data_path $TRAIN_DATA_PATH"
fi
if [ -n "$CELL_TYPE_COL" ]; then
    CMD="$CMD --cell_type_col $CELL_TYPE_COL"
fi
if [ -n "$COND_LABELS_PT_PATH" ]; then
    CMD="$CMD --cond_labels_pt_path $COND_LABELS_PT_PATH"
fi
if [ -n "$CELLTYPE_LABELS_PT_PATH" ]; then
    CMD="$CMD --celltype_labels_pt_path $CELLTYPE_LABELS_PT_PATH"
fi
if [ -n "$CHECKPOINT_DIR" ]; then
    CMD="$CMD --checkpoint_dir $CHECKPOINT_DIR"
fi
if [ -n "$RESUME" ]; then
    CMD="$CMD --resume $RESUME"
fi

# Run training
echo "Running command: $CMD"
echo ""
srun $CMD

# Print completion information
echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
