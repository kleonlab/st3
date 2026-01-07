#!/bin/bash
#
# Training script for SEDD RNA-seq diffusion model
# Usage: ./bash/train.sh [--data_path PATH] [--num_epochs N] [other args...]
#

# Default parameters (can be overridden by command line arguments)
DATA_PATH="${DATA_PATH:-/home/b5cc/sanjukta.b5cc/aracneseq/datasets/k562.h5ad}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-experiments/rnaseq_diffusion}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:- 8}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
NUM_LAYERS="${NUM_LAYERS:-4}"
NUM_HEADS="${NUM_HEADS:-4}"

# Print configuration
echo "=========================================="
echo "SEDD RNA-seq Training Configuration"
echo "=========================================="
echo "Data path: $DATA_PATH"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Epochs: $NUM_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Model: hidden_dim=$HIDDEN_DIM, layers=$NUM_LAYERS, heads=$NUM_HEADS"
echo "=========================================="
echo ""

# Run training
python scripts/train_rnaseq.py \
    --data_path "$DATA_PATH" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --hidden_dim "$HIDDEN_DIM" \
    --num_layers "$NUM_LAYERS" \
    --num_heads "$NUM_HEADS" \
    --mask_ratio 0.15 \
    --gradient_clip 1.0 \
    --val_fraction 0.1 \
    --log_interval 50 \
    --val_interval 1 \
    --save_interval 10 \
    --num_workers 4 \
    --seed 42 \
    "$@"  # Pass any additional command line arguments

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
