#!/bin/bash
#
# Inference script for perturbation prediction using a trained SEDD checkpoint with YAML config
# Usage: ./bash/inf_pseq.sh --config configs/perturbseq_dry_run.yaml --experiment_dir experiments/psed_demo/perturbseq_dry_run --perturbations_file perturbations.txt
#        or: CONFIG=configs/perturbseq_dry_run.yaml EXPERIMENT_DIR=experiments/psed_demo/perturbseq_dry_run PERTURBATIONS_FILE=perturbations.txt ./bash/inf_pseq.sh
#
# perturbations.txt should contain one gene/perturbation name per line, e.g.:
#   KRAS
#   TP53
#   EGFR
#   non-targeting
#

echo "=========================================="
echo "Perturbation Prediction Inference - SEDD"
echo "=========================================="
echo ""

# Defaults (override via environment variables or command line)
CONFIG="${CONFIG:-configs/perturbseq_dry_run.yaml}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-experiments/psed_demo/perturbseq_dry_run}"
PERTURBATIONS_FILE="${PERTURBATIONS_FILE:-/home/b5cc/sanjukta.b5cc/st3/datasets/dataset/perturbation_names_5k.txt}"
MAPPING_DATA_PATH="${MAPPING_DATA_PATH:-/home/b5cc/sanjukta.b5cc/st3/datasets/dataset/k562_5k_train_split.h5ad}"
TEST_DATA_PATH="${TEST_DATA_PATH:-/home/b5cc/sanjukta.b5cc/st3/datasets/dataset/k562_5k_test_split.h5ad}"
NUM_SAMPLES_PER_PERT="${NUM_SAMPLES_PER_PERT:-10}"
NUM_STEPS="${NUM_STEPS:-50}"
TEMPERATURE="${TEMPERATURE:-1.0}"
EVALUATE="${EVALUATE:-false}"

echo "Config file           : ${CONFIG}"
echo "Experiment dir        : ${EXPERIMENT_DIR}"
if [ -n "$PERTURBATIONS_FILE" ]; then
    echo "Perturbations file    : ${PERTURBATIONS_FILE}"
    # Show first few lines of the file
    if [ -f "$PERTURBATIONS_FILE" ]; then
        echo "  First perturbations:"
        head -n 5 "$PERTURBATIONS_FILE" | sed 's/^/    /'
        TOTAL_LINES=$(wc -l < "$PERTURBATIONS_FILE")
        echo "  Total: ${TOTAL_LINES} perturbations"
    fi
else
    echo "Perturbations file    : (from config file or test data)"
fi
if [ -n "$MAPPING_DATA_PATH" ]; then
    echo "Mapping data path     : ${MAPPING_DATA_PATH}"
else
    echo "Mapping data path     : (from config file)"
fi
if [ -n "$TEST_DATA_PATH" ]; then
    echo "Test data path        : ${TEST_DATA_PATH}"
else
    echo "Test data path        : (from config file if needed)"
fi
echo "Samples per pert      : ${NUM_SAMPLES_PER_PERT}"
echo "Num steps             : ${NUM_STEPS}"
echo "Temperature           : ${TEMPERATURE}"
echo "Evaluate              : ${EVALUATE}"
echo "=========================================="
echo ""

# Build command
CMD="python scripts/inference_conditional.py --config $CONFIG --experiment_dir $EXPERIMENT_DIR"

# Add perturbations file if provided (overrides config)
if [ -n "$PERTURBATIONS_FILE" ]; then
    CMD="$CMD --perturbations_file $PERTURBATIONS_FILE"
fi

# Add mapping data path if provided (overrides config)
if [ -n "$MAPPING_DATA_PATH" ]; then
    CMD="$CMD --mapping_data_path $MAPPING_DATA_PATH"
fi

# Add test data path if provided (overrides config)
if [ -n "$TEST_DATA_PATH" ]; then
    CMD="$CMD --test_data_path $TEST_DATA_PATH"
fi

# Add generation parameters
CMD="$CMD --num_samples_per_pert $NUM_SAMPLES_PER_PERT"
CMD="$CMD --num_steps $NUM_STEPS"
CMD="$CMD --temperature $TEMPERATURE"

# Add evaluate flag if true
if [ "$EVALUATE" = "true" ]; then
    CMD="$CMD --evaluate"
fi

# Add any additional command line arguments
CMD="$CMD $@"

# Run inference
echo "Running command:"
echo "$CMD"
echo ""
eval $CMD

echo ""
echo "=========================================="
echo "Perturbation prediction complete!"
echo "Check ${EXPERIMENT_DIR}/inference_results for outputs"
echo "==========================================