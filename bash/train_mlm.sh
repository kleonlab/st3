#!/bin/bash
#
# Training script for SEDD RNA-seq diffusion model using YAML config
# Usage: ./bash/train_mlm.sh --config configs/rnaseq_small.yaml --data_path /path/to/data.h5ad
#        or: CONFIG=configs/rnaseq_large.yaml DATA_PATH=/path/to/data.h5ad ./bash/train_mlm.sh
#

# Default config file (can be overridden by command line or environment)
CONFIG="${CONFIG:-configs/rnaseq_small.yaml}"
DATA_PATH="${DATA_PATH:-}"

# Print configuration
echo "=========================================="
echo "SEDD RNA-seq Training"
echo "=========================================="
echo "Config file: $CONFIG"
if [ -n "$DATA_PATH" ]; then
    echo "Data path: $DATA_PATH"
fi
echo "=========================================="
echo ""

# Build command
CMD="python scripts/train_rnaseq.py --config $CONFIG"

# Add data path if provided
if [ -n "$DATA_PATH" ]; then
    CMD="$CMD --data_path $DATA_PATH"
fi

# Add any additional command line arguments
CMD="$CMD $@"

# Run training
eval $CMD

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
