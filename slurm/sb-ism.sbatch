#!/bin/bash
#SBATCH --job-name=SEDD-RNAseq
#SBATCH --account=gts-agarg35
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=rtx_6000:1
#SBATCH --mem-per-gpu=80G
#SBATCH --time=24:00:00
#SBATCH --qos=embers
#SBATCH --output=slurm_out/sedd_rnaseq-%A.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amete7@gatech.edu

# SLURM batch script for training SEDD discrete diffusion model on RNA-seq data
#
# Usage:
#   sbatch slurm/sb-ism.sbatch
#
# Or with custom parameters:
#   sbatch slurm/sb-ism.sbatch --data_path /path/to/data.h5ad --num_epochs 200

# Change to project directory
cd /storage/project/r-agark35-0/amete7/vla/st3 || exit 1

# Create slurm output directory if it doesn't exist
mkdir -p slurm_out

# Activate virtual environment (adjust path as needed)
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Warning: No virtual environment found. Using system Python."
fi

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Print job information
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo ""

# Default training parameters (can be overridden by passing arguments to sbatch)
DATA_PATH="${DATA_PATH:-/home/b5cc/sanjukta.b5cc/aracneseq/datasets/k562_5k.h5ad}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-experiments/rnaseq_diffusion_$(date +%Y%m%d_%H%M%S)}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
NUM_LAYERS="${NUM_LAYERS:-6}"
NUM_HEADS="${NUM_HEADS:-8}"

# Print training configuration
echo "Training Configuration:"
echo "Data path: $DATA_PATH"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Epochs: $NUM_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Model: hidden_dim=$HIDDEN_DIM, layers=$NUM_LAYERS, heads=$NUM_HEADS"
echo ""

# Run training with srun
srun python scripts/train_rnaseq.py \
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
    --num_workers 8 \
    --seed 42 \
    "$@"

# Print completion information
echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
