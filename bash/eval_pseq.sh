#!/bin/bash
#
# Evaluation script for perturbation prediction using eval_pseq.py
# Usage: ./bash/eval_pseq.sh
#        or: OUTPUT_DIR=... TEST_DATA_PATH=... PREDICTED_DATA_PATH=... ./bash/eval_pseq.sh
#

echo "=========================================="
echo "Perturbation Prediction Evaluation - SEDD"
echo "=========================================="
echo ""

# Optional overrides via environment variables (eval_pseq.py reads hardcoded paths)
OUTPUT_DIR="${OUTPUT_DIR:-}"
TEST_DATA_PATH="${TEST_DATA_PATH:-}"
PREDICTED_DATA_PATH="${PREDICTED_DATA_PATH:-}"

if [ -n "$OUTPUT_DIR" ] || [ -n "$TEST_DATA_PATH" ] || [ -n "$PREDICTED_DATA_PATH" ]; then
    echo "Note: eval_pseq.py currently uses hardcoded paths."
    echo "Env overrides provided:"
    echo "  OUTPUT_DIR         : ${OUTPUT_DIR:-<unset>}"
    echo "  TEST_DATA_PATH     : ${TEST_DATA_PATH:-<unset>}"
    echo "  PREDICTED_DATA_PATH: ${PREDICTED_DATA_PATH:-<unset>}"
    echo ""
fi

# Activate virtual environment if available
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Run evaluation
CMD="python scripts/eval_pseq.py"
CMD="$CMD $@"
eval $CMD

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="

