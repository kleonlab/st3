#!/bin/bash
#
# Inference script for Cell-Type Conditioned SEDD Perturbation Prediction
#
# This script generates perturbed cells for a specific cell type.
# The key feature is the CELL_TYPE environment variable that controls
# which cell type to generate predictions for.
#
# Usage:
#   # Generate for a specific cell type
#   CELL_TYPE="T cell" ./celltype_cond/bash/inference_celltype.sh \
#       --experiment_dir experiments/celltype_model
#
#   # Generate for all cell types
#   ./celltype_cond/bash/inference_celltype.sh --experiment_dir experiments/celltype_model
#
# Environment variables:
#   CONFIG                   - Path to YAML config file
#   CELL_TYPE                - Cell type to generate for (key parameter!)
#   EXPERIMENT_DIR           - Path to trained model directory
#   DATA_PATH                - Path to test h5ad file
#   PERTURBATIONS_FILE       - File with perturbation names to generate
#   NUM_SAMPLES              - Number of samples per condition
#   OUTPUT_DIR               - Output directory for results
#

set -e

# Default values
CONFIG="${CONFIG:-celltype_cond/configs/celltype_inference_example.yaml}"
CELL_TYPE="${CELL_TYPE:-}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-}"
DATA_PATH="${DATA_PATH:-}"
PERTURBATIONS_FILE="${PERTURBATIONS_FILE:-}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

# Print configuration
echo "=========================================="
echo "Cell-Type Conditioned Inference"
echo "=========================================="
echo "Config file: $CONFIG"
if [ -n "$CELL_TYPE" ]; then
    echo "Cell type: $CELL_TYPE"
else
    echo "Cell type: ALL (generating for all cell types)"
fi
if [ -n "$EXPERIMENT_DIR" ]; then
    echo "Experiment dir: $EXPERIMENT_DIR"
fi
if [ -n "$DATA_PATH" ]; then
    echo "Data path: $DATA_PATH"
fi
if [ -n "$PERTURBATIONS_FILE" ]; then
    echo "Perturbations file: $PERTURBATIONS_FILE"
fi
echo "Samples per condition: $NUM_SAMPLES"
if [ -n "$OUTPUT_DIR" ]; then
    echo "Output dir: $OUTPUT_DIR"
fi
echo "=========================================="
echo ""

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Build command
CMD="python celltype_cond/scripts/inference_celltype_perturbseq.py"

if [ -n "$CONFIG" ]; then
    CMD="$CMD --config $CONFIG"
fi
if [ -n "$EXPERIMENT_DIR" ]; then
    CMD="$CMD --experiment_dir $EXPERIMENT_DIR"
fi
if [ -n "$CELL_TYPE" ]; then
    CMD="$CMD --cell_type \"$CELL_TYPE\""
fi
if [ -n "$DATA_PATH" ]; then
    CMD="$CMD --data_path $DATA_PATH"
fi
if [ -n "$PERTURBATIONS_FILE" ]; then
    CMD="$CMD --perturbations_file $PERTURBATIONS_FILE"
fi
if [ -n "$NUM_SAMPLES" ]; then
    CMD="$CMD --num_samples_per_condition $NUM_SAMPLES"
fi
if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output_dir $OUTPUT_DIR"
fi

# Add any additional command line arguments
CMD="$CMD $@"

# Run inference
echo "Running: $CMD"
echo ""
eval $CMD

echo ""
echo "=========================================="
echo "Inference complete!"
echo "=========================================="
