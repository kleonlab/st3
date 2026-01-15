#!/bin/bash
#
# Inference script for SEDD Perturbation Prediction model using YAML config
# Usage: ./bash/inf_pseq.sh --config configs/perturbseq_dry_run.yaml --experiment_dir experiments/perturbseq_dry_run
#        or: CONFIG=configs/perturbseq_dry_run.yaml EXPERIMENT_DIR=experiments/perturbseq_dry_run ./bash/inf_pseq.sh
#
# Example dry run:
#   ./bash/inf_pseq.sh --config configs/perturbseq_dry_run.yaml --experiment_dir experiments/perturbseq_dry_run
#   CONFIG=configs/perturbseq_dry_run.yaml EXPERIMENT_DIR=experiments/perturbseq_dry_run ./bash/inf_pseq.sh
#

# Default config file (can be overridden by command line or environment)
CONFIG="${CONFIG:-configs/perturbseq_dry_run.yaml}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-experiments/perturbseq_dry_run}"
DATA_PATH="${DATA_PATH:-}"
CHECKPOINT="${CHECKPOINT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-inference_results}"

# Print configuration
echo "=========================================="
echo "SEDD Perturbation Prediction Inference"
echo "=========================================="
echo "Config file: $CONFIG"
echo "Experiment dir: $EXPERIMENT_DIR"
if [ -n "$DATA_PATH" ]; then
    echo "Data path: $DATA_PATH"
fi
if [ -n "$CHECKPOINT" ]; then
    echo "Checkpoint: $CHECKPOINT"
fi
echo "Output dir: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Build command
CMD="python scripts/inference_perturbseq.py --config $CONFIG"

# Add experiment directory
CMD="$CMD --experiment_dir $EXPERIMENT_DIR"

# Add output directory
CMD="$CMD --output_dir $OUTPUT_DIR"

# Add data path if provided
if [ -n "$DATA_PATH" ]; then
    CMD="$CMD --data_path $DATA_PATH"
fi

# Add checkpoint if provided (overrides experiment_dir)
if [ -n "$CHECKPOINT" ]; then
    CMD="$CMD --checkpoint $CHECKPOINT"
fi

# Add save predictions flag
CMD="$CMD --save_predictions"

# Add any additional command line arguments
CMD="$CMD $@"

# Run inference
echo "Running: $CMD"
echo ""
eval $CMD

echo ""
echo "=========================================="
echo "Inference complete!"
echo "Check results in: $OUTPUT_DIR"
echo "=========================================="
