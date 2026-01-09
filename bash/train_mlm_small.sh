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

# Use small config and override for quick test
DATA_PATH="${DATA_PATH:-/home/b5cc/sanjukta.b5cc/aracneseq/datasets/k562_5k.h5ad}"
CONFIG="${CONFIG:-configs/rnaseq_small.yaml}"

python scripts/train_rnaseq.py \
    --config "$CONFIG" \
    --data_path "$DATA_PATH" \
    --checkpoint_dir "experiments/quick_test" \
    --num_epochs 5 \
    --batch_size 16 \
    --hidden_dim 64 \
    --num_layers 2 \
    --num_heads 2 \
    --log_interval 10 \
    --val_interval 1

echo ""
echo "=========================================="
echo "Quick test complete!"
echo "Check experiments/quick_test/ for outputs"
echo "=========================================="
