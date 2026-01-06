#!/bin/bash
#
# Quick sanity test for SEDD RNA-seq training
# Runs a small model for a few epochs to verify everything works
#

echo "=========================================="
echo "Quick Sanity Test - SEDD RNA-seq"
echo "=========================================="
echo "This will train a small model for 5 epochs"
echo "to verify the training pipeline works."
echo "=========================================="
echo ""

# Run with small parameters for quick test
DATA_PATH="${DATA_PATH:-/home/b5cc/sanjukta.b5cc/aracneseq/datasets/k562_5k.h5ad}" \
CHECKPOINT_DIR="experiments/quick_test" \
NUM_EPOCHS=5 \
BATCH_SIZE=16 \
LEARNING_RATE=1e-4 \
HIDDEN_DIM=64 \
NUM_LAYERS=2 \
NUM_HEADS=2 \
./bash/train.sh \
    --log_interval 10 \
    --val_interval 1

echo ""
echo "=========================================="
echo "Quick test complete!"
echo "Check experiments/quick_test/ for outputs"
echo "=========================================="
