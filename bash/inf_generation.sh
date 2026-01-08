#!/bin/bash
#
# Quick generation script for creating synthetic cells with a trained SEDD checkpoint.
# Mirrors train_quick.sh style: sets sensible defaults and runs generation.
#

echo "=========================================="
echo "Quick Generation - SEDD RNA-seq"
echo "=========================================="
echo "This will generate synthetic cells from a"
echo "trained checkpoint with reasonable defaults."
echo "=========================================="
echo ""

# Defaults (override via environment variables before running)
EXPERIMENT_DIR="${EXPERIMENT_DIR:-experiments/quick_test}"
CHECKPOINT="${CHECKPOINT:-""}"  # optional specific checkpoint path
DATA_PATH="${DATA_PATH:-/home/b5cc/sanjukta.b5cc/aracneseq/datasets/k562_5k.h5ad}"

NUM_GENERATE="${NUM_GENERATE:-5}"
NUM_STEPS="${NUM_STEPS:-100}"
TEMPERATURE="${TEMPERATURE:-1.0}"

NUM_CELLS_VIZ="${NUM_CELLS_VIZ:-3}"
NUM_REAL_VIZ="${NUM_REAL_VIZ:-3}"

echo "Experiment dir : ${EXPERIMENT_DIR}"
echo "Checkpoint     : ${CHECKPOINT:-auto (best/final/latest)}"
echo "Data path      : ${DATA_PATH}"
echo "Num generate   : ${NUM_GENERATE}"
echo "Num steps      : ${NUM_STEPS}"
echo "Temperature    : ${TEMPERATURE}"
echo "Cells to plot  : ${NUM_CELLS_VIZ} (generated), ${NUM_REAL_VIZ} (real)"
echo "=========================================="
echo ""

# Run generation
uv run scripts/inference_generation.py \
    --experiment_dir "${EXPERIMENT_DIR}" \
    ${CHECKPOINT:+--checkpoint "${CHECKPOINT}"} \
    --data_path "${DATA_PATH}" \
    --num_generate "${NUM_GENERATE}" \
    --num_steps "${NUM_STEPS}" \
    --temperature "${TEMPERATURE}" \
    --num_cells_visualize "${NUM_CELLS_VIZ}" \
    --num_real_visualize "${NUM_REAL_VIZ}"

echo ""
echo "=========================================="
echo "Generation complete!"
echo "Check ${EXPERIMENT_DIR}/generation_results for outputs"
echo "=========================================="

