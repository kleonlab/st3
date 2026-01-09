#!/bin/bash
#
# Inference script for imputation using a trained SEDD checkpoint with YAML config
# Usage: ./bash/inf_imputation.sh --config configs/rnaseq_small.yaml --experiment_dir experiments/rnaseq_small --data_path /path/to/data.h5ad
#        or: CONFIG=configs/rnaseq_small.yaml EXPERIMENT_DIR=experiments/rnaseq_small ./bash/inf_imputation.sh
#

echo "=========================================="
echo "Imputation Inference - SEDD RNA-seq"
echo "=========================================="
echo ""

# Defaults (override via environment variables or command line)
CONFIG="${CONFIG:-configs/rnaseq_small.yaml}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-experiments/quick_test}"
DATA_PATH="${DATA_PATH:-}"

echo "Config file    : ${CONFIG}"
echo "Experiment dir : ${EXPERIMENT_DIR}"
if [ -n "$DATA_PATH" ]; then
    echo "Data path      : ${DATA_PATH}"
fi
echo "=========================================="
echo ""

# Build command
CMD="python scripts/inference_imputation.py --config $CONFIG --experiment_dir $EXPERIMENT_DIR"

# Add data path if provided
if [ -n "$DATA_PATH" ]; then
    CMD="$CMD --data_path $DATA_PATH"
fi

# Add any additional command line arguments
CMD="$CMD $@"

# Run inference
eval $CMD

echo ""
echo "=========================================="
echo "Imputation complete!"
echo "Check ${EXPERIMENT_DIR}/imputation_results for outputs"
echo "=========================================="

