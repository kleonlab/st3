#!/bin/bash
#
# Use small config and override for quick test
DATA_PATH="${DATA_PATH:-/home/b5cc/sanjukta.b5cc/aracneseq/datasets/k562_5k.h5ad}"
CONFIG="${CONFIG:-configs/rnaseq_small.yaml}"

python scripts/train_rnaseq.py \
    --config "$CONFIG" \
    --data_path "$DATA_PATH" \
    --checkpoint_dir "experiments/mlm/128,4,4" \
    --num_epochs 5 \
    --batch_size 16 \
    --hidden_dim 128 \
    --num_layers 4 \
    --num_heads 4 \
    --log_interval 2 \
    --val_interval 1

echo "Check experiments/mlm/ for outputs"
