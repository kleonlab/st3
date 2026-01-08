#!/bin/bash
#
# Quick inference script for imputation using a trained SEDD checkpoint.
# Mirrors train_quick.sh style: sets sensible defaults and runs inference.
#

echo "=========================================="
echo "Quick Imputation - SEDD RNA-seq"
echo "=========================================="
echo "This will run imputation on a small batch"
echo "using an existing trained checkpoint."
echo "=========================================="
echo ""

# Defaults (override via environment variables before running)
EXPERIMENT_DIR="${EXPERIMENT_DIR:-experiments/quick_test}"
CHECKPOINT="${CHECKPOINT:-""}"  # optional specific checkpoint path
DATA_PATH="${DATA_PATH:-/home/b5cc/sanjukta.b5cc/aracneseq/datasets/k562_5k.h5ad}"
MASK_RATIO="${MASK_RATIO:-0.2}"
NUM_STEPS="${NUM_STEPS:-50}"
TEMPERATURE="${TEMPERATURE:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_BATCHES="${NUM_BATCHES:-1}"          # evaluate first N batches (set empty for all)
NUM_CELLS_VIZ="${NUM_CELLS_VIZ:-3}"

echo "Experiment dir : ${EXPERIMENT_DIR}"
echo "Checkpoint     : ${CHECKPOINT:-auto (best/final/latest)}"
echo "Data path      : ${DATA_PATH}"
echo "Mask ratio     : ${MASK_RATIO}"
echo "Num steps      : ${NUM_STEPS}"
echo "Batch size     : ${BATCH_SIZE}"
echo "Num batches    : ${NUM_BATCHES}"
echo "Cells to plot  : ${NUM_CELLS_VIZ}"
echo "=========================================="
echo ""

# Run inference
uv run scripts/inference_imputation.py \
    --experiment_dir "${EXPERIMENT_DIR}" \
    ${CHECKPOINT:+--checkpoint "${CHECKPOINT}"} \
    --data_path "${DATA_PATH}" \
    --mask_ratio "${MASK_RATIO}" \
    --num_steps "${NUM_STEPS}" \
    --temperature "${TEMPERATURE}" \
    --batch_size "${BATCH_SIZE}" \
    ${NUM_BATCHES:+--num_batches "${NUM_BATCHES}"} \
    --num_cells_visualize "${NUM_CELLS_VIZ}"

echo ""
echo "=========================================="
echo "Imputation complete!"
echo "Check ${EXPERIMENT_DIR}/imputation_results for outputs"
echo "=========================================="

