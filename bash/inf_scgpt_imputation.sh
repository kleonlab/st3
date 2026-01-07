#!/bin/bash
#
# Inference script for imputation using scGPT model.
# This script runs scGPT imputation on the same dataset used for SEDD inference.
#
# Note: The models/ directory is typically a symlink to GPU storage.
# Ensure your scGPT model files are placed in models/scGPT/ on the GPU node.
#

echo "=========================================="
echo "scGPT Imputation Inference"
echo "=========================================="
echo "Running imputation with scGPT model"
echo "=========================================="
echo ""

# Defaults (override via environment variables before running)
MODEL_DIR="${MODEL_DIR:-models/scGPT}"
DATA_PATH="${DATA_PATH:-/home/b5cc/sanjukta.b5cc/aracneseq/datasets/k562_5k.h5ad}"
MASK_RATIO="${MASK_RATIO:-0.2}"
NUM_STEPS="${NUM_STEPS:-1}"  # scGPT typically uses 1 step
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_BATCHES="${NUM_BATCHES:-}"  # empty = all batches
NUM_CELLS_VIZ="${NUM_CELLS_VIZ:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-}"  # defaults to MODEL_DIR/imputation_results

echo "Model directory: ${MODEL_DIR}"
echo "Data path      : ${DATA_PATH}"
echo "Mask ratio     : ${MASK_RATIO}"
echo "Num steps      : ${NUM_STEPS}"
echo "Batch size     : ${BATCH_SIZE}"
echo "Num batches    : ${NUM_BATCHES:-all}"
echo "Cells to plot  : ${NUM_CELLS_VIZ}"
echo "Output dir     : ${OUTPUT_DIR:-${MODEL_DIR}/imputation_results}"
echo "=========================================="
echo ""

# Build command
CMD="uv run scripts/inference_scgpt_imputation.py"
CMD="${CMD} --model_dir \"${MODEL_DIR}\""
CMD="${CMD} --data_path \"${DATA_PATH}\""
CMD="${CMD} --mask_ratio ${MASK_RATIO}"
CMD="${CMD} --num_steps ${NUM_STEPS}"
CMD="${CMD} --batch_size ${BATCH_SIZE}"
if [ -n "${NUM_BATCHES}" ]; then
    CMD="${CMD} --num_batches ${NUM_BATCHES}"
fi
CMD="${CMD} --num_cells_visualize ${NUM_CELLS_VIZ}"
if [ -n "${OUTPUT_DIR}" ]; then
    CMD="${CMD} --output_dir \"${OUTPUT_DIR}\""
fi

# Run inference
echo "Running: ${CMD}"
echo ""
eval ${CMD}

echo ""
echo "=========================================="
echo "scGPT Imputation complete!"
if [ -n "${OUTPUT_DIR}" ]; then
    echo "Check ${OUTPUT_DIR} for outputs"
else
    echo "Check ${MODEL_DIR}/imputation_results for outputs"
fi
echo "=========================================="
