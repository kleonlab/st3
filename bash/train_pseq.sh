#!/bin/bash
#
# Training script for SEDD Perturbation Prediction model using YAML config
# Usage: ./bash/train_pseq.sh --config configs/perturbseq_dry_run.yaml --data_path /path/to/data.h5ad
#        or: CONFIG=configs/perturbseq_dry_run.yaml DATA_PATH=/path/to/data.h5ad ./bash/train_pseq.sh
#
# Example dry run:
#   ./bash/train_pseq.sh --config configs/perturbseq_dry_run.yaml
#   CONFIG=configs/perturbseq_dry_run.yaml ./bash/train_pseq.sh
#

# Default config file (can be overridden by command line or environment)
CONFIG="${CONFIG:-configs/perturbseq_dry_run.yaml}"
DATA_PATH="${DATA_PATH:-}"
PERT_COL="${PERT_COL:-}"
CONTROL_NAME="${CONTROL_NAME:-}"

# Print configuration
echo "=========================================="
echo "SEDD Perturbation Prediction Training"
echo "=========================================="
echo "Config file: $CONFIG"
if [ -n "$DATA_PATH" ]; then
    echo "Data path: $DATA_PATH"
fi
if [ -n "$PERT_COL" ]; then
    echo "Perturbation column: $PERT_COL"
fi
if [ -n "$CONTROL_NAME" ]; then
    echo "Control name: $CONTROL_NAME"
fi
echo "=========================================="
echo ""

# Build command
CMD="python scripts/train_perturbseq.py --config $CONFIG"

# Add data path if provided
if [ -n "$DATA_PATH" ]; then
    CMD="$CMD --data_path $DATA_PATH"
fi

# Add perturbation column if provided
if [ -n "$PERT_COL" ]; then
    CMD="$CMD --pert_col $PERT_COL"
fi

# Add control name if provided
if [ -n "$CONTROL_NAME" ]; then
    CMD="$CMD --control_name $CONTROL_NAME"
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
