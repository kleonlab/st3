#!/bin/bash
#
# Inference script for perturbation prediction using a trained SEDD checkpoint with YAML config
# Usage: ./bash/inf_pseq.sh --config configs/perturbseq_dry_run.yaml --experiment_dir experiments/psed_demo/perturbseq_dry_run --data_path /path/to/test_data.h5ad
#        or: CONFIG=configs/perturbseq_dry_run.yaml EXPERIMENT_DIR=experiments/psed_demo/perturbseq_dry_run TEST_DATA_PATH=/path/to/test_data.h5ad ./bash/inf_pseq.sh
#

echo "=========================================="
echo "Perturbation Prediction Inference - SEDD"
echo "=========================================="
echo ""

# Defaults (override via environment variables or command line)
CONFIG="${CONFIG:-configs/perturbseq_dry_run.yaml}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-experiments/psed_demo/perturbseq_dry_run}"
TEST_DATA_PATH="${TEST_DATA_PATH:-}"

echo "Config file    : ${CONFIG}"
echo "Experiment dir : ${EXPERIMENT_DIR}"
if [ -n "$TEST_DATA_PATH" ]; then
    echo "Test data path : ${TEST_DATA_PATH}"
else
    echo "Test data path : (from config file)"
fi
echo "=========================================="
echo ""

# Build command
CMD="python scripts/inference_perturbseq.py --config $CONFIG --experiment_dir $EXPERIMENT_DIR"

# Add test data path if provided (overrides config)
if [ -n "$TEST_DATA_PATH" ]; then
    CMD="$CMD --data_path $TEST_DATA_PATH"
fi

# Add any additional command line arguments
CMD="$CMD $@"

# Run inference
eval $CMD

echo ""
echo "=========================================="
echo "Perturbation prediction complete!"
echo "Check ${EXPERIMENT_DIR}/perturbseq_inference_results for outputs"
echo "=========================================="
