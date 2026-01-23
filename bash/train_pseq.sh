#!/bin/bash
#
# Training script for SEDD Perturbation-seq prediction using YAML config
# Usage: ./bash/train_pseq.sh --config configs/perturbseq_dry_run.yaml --data_path /path/to/train_data.h5ad
#        or: CONFIG=configs/perturbseq_dry_run.yaml TRAIN_DATA_PATH=/path/to/train_data.h5ad ./bash/train_pseq.sh
#

# Default config file (can be overridden by command line or environment)
CONFIG="${CONFIG:-configs/perturbseq_medium.yaml}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-}"
COND_LABELS_PT_PATH="${COND_LABELS_PT_PATH:-}"

# Print configuration
echo "=========================================="
echo "SEDD Perturbation-seq Training"
echo "=========================================="
echo "Config file: $CONFIG"
if [ -n "$TRAIN_DATA_PATH" ]; then
    echo "Train data path: $TRAIN_DATA_PATH"
else
    echo "Train data path: (from config file)"
fi
if [ -n "$COND_LABELS_PT_PATH" ]; then
    echo "Cond labels .pt: $COND_LABELS_PT_PATH"
else
    echo "Cond labels .pt: (from config file)"
fi
echo "=========================================="
echo ""

source .venv/bin/activate
# Build command
CMD="python scripts/train_perturbseq.py --config $CONFIG"

# Add train data path if provided (overrides config)
if [ -n "$TRAIN_DATA_PATH" ]; then
    CMD="$CMD --train_data_path $TRAIN_DATA_PATH"
fi
if [ -n "$COND_LABELS_PT_PATH" ]; then
    CMD="$CMD --cond_labels_pt_path $COND_LABELS_PT_PATH"
fi

# Add any additional command line arguments
CMD="$CMD $@"

# Run training
eval $CMD

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
