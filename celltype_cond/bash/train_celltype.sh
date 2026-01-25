#!/bin/bash
#
# Training script for Cell-Type Conditioned SEDD Perturbation Prediction
#
# Usage:
#   ./celltype_cond/bash/train_celltype.sh --config celltype_cond/configs/celltype_perturbseq_small.yaml
#
# Environment variables (override config):
#   CONFIG                   - Path to YAML config file
#   TRAIN_DATA_PATH          - Path to training h5ad file
#   CELL_TYPE_COL            - Column name for cell types (default: cell_type)
#   COND_LABELS_PT_PATH      - Path to perturbation embeddings .pt file
#   CELLTYPE_LABELS_PT_PATH  - Path to cell-type embeddings .pt file
#   CHECKPOINT_DIR           - Directory for saving checkpoints
#   RESUME                   - Checkpoint to resume from ('auto' for latest)
#

set -e

# Default config file (can be overridden by command line or environment)
CONFIG="${CONFIG:-celltype_cond/configs/celltype_perturbseq_small.yaml}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-}"
CELL_TYPE_COL="${CELL_TYPE_COL:-}"
COND_LABELS_PT_PATH="${COND_LABELS_PT_PATH:-}"
CELLTYPE_LABELS_PT_PATH="${CELLTYPE_LABELS_PT_PATH:-}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
RESUME="${RESUME:-}"

# Print configuration
echo "=========================================="
echo "Cell-Type Conditioned SEDD Training"
echo "=========================================="
echo "Config file: $CONFIG"
if [ -n "$TRAIN_DATA_PATH" ]; then
    echo "Train data path: $TRAIN_DATA_PATH"
else
    echo "Train data path: (from config file)"
fi
if [ -n "$CELL_TYPE_COL" ]; then
    echo "Cell-type column: $CELL_TYPE_COL"
else
    echo "Cell-type column: (from config file)"
fi
if [ -n "$COND_LABELS_PT_PATH" ]; then
    echo "Perturbation embeddings: $COND_LABELS_PT_PATH"
fi
if [ -n "$CELLTYPE_LABELS_PT_PATH" ]; then
    echo "Cell-type embeddings: $CELLTYPE_LABELS_PT_PATH"
fi
if [ -n "$CHECKPOINT_DIR" ]; then
    echo "Checkpoint dir: $CHECKPOINT_DIR"
fi
if [ -n "$RESUME" ]; then
    echo "Resume from: $RESUME"
fi
echo "=========================================="
echo ""

# Activate virtual environment if available
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Build command
CMD="python celltype_cond/scripts/train_celltype_perturbseq.py --config $CONFIG"

# Add optional arguments if provided
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

# Add any additional command line arguments
CMD="$CMD $@"

# Run training
echo "Running: $CMD"
echo ""
eval $CMD

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
