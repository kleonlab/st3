#!/bin/bash
#
# Batch inference script for generating predictions for ALL cell types
#
# This script iterates through all cell types defined in the trained model
# and generates predictions for each one separately.
#
# Usage:
#   ./celltype_cond/bash/inference_all_celltypes.sh \
#       --experiment_dir experiments/celltype_model \
#       --cell_types "T cell" "B cell" "Monocyte" "NK cell"
#
# Or specify cell types via environment variable:
#   CELL_TYPES="T cell,B cell,Monocyte" ./celltype_cond/bash/inference_all_celltypes.sh ...
#
# Environment variables:
#   CONFIG                   - Path to YAML config file
#   EXPERIMENT_DIR           - Path to trained model directory
#   CELL_TYPES               - Comma-separated list of cell types
#   DATA_PATH                - Path to test h5ad file
#   NUM_SAMPLES              - Number of samples per condition
#

set -e

# Default values
CONFIG="${CONFIG:-celltype_cond/configs/celltype_inference_example.yaml}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-}"
CELL_TYPES="${CELL_TYPES:-}"
DATA_PATH="${DATA_PATH:-}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment_dir)
            EXPERIMENT_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --cell_types)
            shift
            CELL_TYPES=""
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                if [ -n "$CELL_TYPES" ]; then
                    CELL_TYPES="$CELL_TYPES,$1"
                else
                    CELL_TYPES="$1"
                fi
                shift
            done
            ;;
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$EXPERIMENT_DIR" ]; then
    echo "Error: --experiment_dir is required"
    exit 1
fi

# Print configuration
echo "=========================================="
echo "Batch Cell-Type Inference"
echo "=========================================="
echo "Config: $CONFIG"
echo "Experiment dir: $EXPERIMENT_DIR"
echo "Cell types: $CELL_TYPES"
echo "Samples per condition: $NUM_SAMPLES"
echo "=========================================="
echo ""

# If no cell types specified, try to get them from the trained model's args.json
if [ -z "$CELL_TYPES" ]; then
    ARGS_FILE="$EXPERIMENT_DIR/args.json"
    if [ -f "$ARGS_FILE" ]; then
        echo "Reading cell types from $ARGS_FILE..."
        # Use Python to extract cell types from JSON
        CELL_TYPES=$(python -c "
import json
with open('$ARGS_FILE', 'r') as f:
    args = json.load(f)
cell_types = args.get('cell_types', [])
print(','.join(cell_types))
")
        echo "Found cell types: $CELL_TYPES"
    else
        echo "Error: No cell types specified and args.json not found at $ARGS_FILE"
        exit 1
    fi
fi

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Convert comma-separated to array
IFS=',' read -ra CELL_TYPE_ARRAY <<< "$CELL_TYPES"

echo ""
echo "Generating predictions for ${#CELL_TYPE_ARRAY[@]} cell types..."
echo ""

# Iterate through cell types
for cell_type in "${CELL_TYPE_ARRAY[@]}"; do
    echo "=========================================="
    echo "Processing: $cell_type"
    echo "=========================================="

    CMD="python celltype_cond/scripts/inference_celltype_perturbseq.py"
    CMD="$CMD --config $CONFIG"
    CMD="$CMD --experiment_dir $EXPERIMENT_DIR"
    CMD="$CMD --cell_type \"$cell_type\""
    CMD="$CMD --num_samples_per_condition $NUM_SAMPLES"

    if [ -n "$DATA_PATH" ]; then
        CMD="$CMD --data_path $DATA_PATH"
    fi

    echo "Running: $CMD"
    eval $CMD

    echo ""
    echo "Completed: $cell_type"
    echo ""
done

echo "=========================================="
echo "All cell types processed!"
echo "=========================================="
echo ""
echo "Results saved to: $EXPERIMENT_DIR/inference_results/"
ls -la "$EXPERIMENT_DIR/inference_results/" 2>/dev/null || echo "(directory may have different structure)"
