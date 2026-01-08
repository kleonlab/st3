#!/bin/bash
#
# Inference script for imputation using scGPT model.
# This script runs scGPT imputation on the same dataset used for SEDD inference.
#
# Usage:
#   bash bash/inf_scgpt_imputation.sh
#
# Override defaults with environment variables:
#   MODEL_DIR=models/scGPT DATA_PATH=/path/to/data.h5ad bash bash/inf_scgpt_imputation.sh
#
# Note: The models/ directory is typically a symlink to GPU storage.
# Ensure your scGPT model files are placed in models/scGPT/ on the GPU node.
#

echo "=========================================="
echo "scGPT Imputation Inference"
echo "=========================================="
echo ""

# Default arguments (override via environment variables)
MODEL_DIR="${MODEL_DIR:-models/scGPT}"
DATA_PATH="${DATA_PATH:-/home/b5cc/sanjukta.b5cc/aracneseq/datasets/k562_5k.h5ad}"
MASK_RATIO="${MASK_RATIO:-0.2}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_BATCHES="${NUM_BATCHES:-}"  # empty = all batches
NUM_CELLS_VIZ="${NUM_CELLS_VIZ:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-}"  # defaults to MODEL_DIR/imputation_results
VAL_FRACTION="${VAL_FRACTION:-0.1}"
SEED="${SEED:-42}"

echo "Configuration:"
echo "  Model directory: ${MODEL_DIR}"
echo "  Data path      : ${DATA_PATH}"
echo "  Mask ratio     : ${MASK_RATIO}"
echo "  Batch size     : ${BATCH_SIZE}"
echo "  Num batches    : ${NUM_BATCHES:-all}"
echo "  Val fraction   : ${VAL_FRACTION}"
echo "  Cells to plot  : ${NUM_CELLS_VIZ}"
echo "  Output dir     : ${OUTPUT_DIR:-${MODEL_DIR}/imputation_results}"
echo "  Seed           : ${SEED}"
echo "=========================================="
echo ""

# Build command
CMD="uv run scripts/inference_scgpt_imputation.py"
CMD="${CMD} --model_dir ${MODEL_DIR}"
CMD="${CMD} --data_path ${DATA_PATH}"
CMD="${CMD} --mask_ratio ${MASK_RATIO}"
CMD="${CMD} --batch_size ${BATCH_SIZE}"
CMD="${CMD} --val_fraction ${VAL_FRACTION}"
CMD="${CMD} --seed ${SEED}"

if [ -n "${NUM_BATCHES}" ]; then
    CMD="${CMD} --num_batches ${NUM_BATCHES}"
fi

CMD="${CMD} --num_cells_visualize ${NUM_CELLS_VIZ}"

if [ -n "${OUTPUT_DIR}" ]; then
    CMD="${CMD} --output_dir ${OUTPUT_DIR}"
fi

# Run inference
echo "Running: ${CMD}"
echo ""
eval ${CMD}

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "scGPT Imputation complete!"
    if [ -n "${OUTPUT_DIR}" ]; then
        echo "Check ${OUTPUT_DIR} for outputs"
    else
        echo "Check ${MODEL_DIR}/imputation_results for outputs"
    fi
else
    echo "scGPT Imputation FAILED with exit code ${EXIT_CODE}"
fi
echo "=========================================="

exit ${EXIT_CODE}
