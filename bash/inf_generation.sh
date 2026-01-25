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
EXPERIMENT_DIR="${EXPERIMENT_DIR:-experiments/mlm_demo/rnaseq_small}"
CHECKPOINT="${CHECKPOINT:-""}"  # optional specific checkpoint path
REFERENCE_DATA="${REFERENCE_DATA:-/home/b5cc/sanjukta.b5cc/aracneseq/datasets/k562_5k.h5ad}"
OUTPUT_PATH="${OUTPUT_PATH:-""}"  # optional, defaults to experiment_dir/generated_cells.h5ad

NUM_GENERATE="${NUM_GENERATE:-10}"
NUM_STEPS="${NUM_STEPS:-10}"
TEMPERATURE="${TEMPERATURE:-1.0}"
SEED="${SEED:-42}"
KEEP_SPARSE="${KEEP_SPARSE:-false}"

echo "Experiment dir : ${EXPERIMENT_DIR}"
echo "Checkpoint     : ${CHECKPOINT:-auto (best/final/latest)}"
echo "Reference data : ${REFERENCE_DATA}"
echo "Output path    : ${OUTPUT_PATH:-${EXPERIMENT_DIR}/generated_cells.h5ad}"
echo "Num generate   : ${NUM_GENERATE}"
echo "Num steps      : ${NUM_STEPS}"
echo "Temperature    : ${TEMPERATURE}"
echo "Seed           : ${SEED}"
echo "Keep sparse    : ${KEEP_SPARSE}"
echo "=========================================="
echo ""

# Run generation
#uv run scripts/inference_generation.py \
#    --experiment_dir "${EXPERIMENT_DIR}" \
#    ${CHECKPOINT:+--checkpoint "${CHECKPOINT}"} \
#    --data_path "${DATA_PATH}" \
#    --num_generate "${NUM_GENERATE}" \
#    --num_steps "${NUM_STEPS}" \
#    --temperature "${TEMPERATURE}" \
#    --num_cells_visualize "${NUM_CELLS_VIZ}" \
#    --num_real_visualize "${NUM_REAL_VIZ}"

CMD="uv run scripts/inference_generation.py \
    --experiment_dir \"${EXPERIMENT_DIR}\" \
    --reference_data \"${REFERENCE_DATA}\" \
    --num_generate ${NUM_GENERATE} \
    --num_steps ${NUM_STEPS} \
    --temperature ${TEMPERATURE} \
    --seed ${SEED}"


if [ -n "${CHECKPOINT}" ]; then
    CMD="${CMD} --checkpoint \"${CHECKPOINT}\""
fi

if [ -n "${OUTPUT_PATH}" ]; then
    CMD="${CMD} --output_path \"${OUTPUT_PATH}\""
fi
 
if [ "${KEEP_SPARSE}" = "true" ]; then
    CMD="${CMD} --keep_sparse"
fi

eval ${CMD}

echo ""
echo "=========================================="
echo "Generation complete!"
if [ -n "${OUTPUT_PATH}" ]; then
    echo "Output saved to: ${OUTPUT_PATH}"
else
    echo "Output saved to: ${EXPERIMENT_DIR}/generated_cells.h5ad"
fi
echo "=========================================="