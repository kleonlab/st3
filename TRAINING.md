# Training SEDD for RNA-seq

This document describes how to train the discrete diffusion model (SEDD) for masked RNA-seq prediction.

## Overview

The training pipeline implements Score-Entropy Discrete Diffusion (SEDD) for single-cell RNA-seq data. The model learns to predict masked gene expression values, which can be used for:

- Imputation of missing values
- Denoising of gene expression
- Generation of synthetic cells

## Quick Start

### Local Training

For a quick sanity test:

```bash
./bash/train_quick.sh
```

For full local training:

```bash
./bash/train.sh
```

### Cluster Training (SLURM)

Submit a training job to the cluster:

```bash
./bash/train_submit.sh
```

Or directly with sbatch:

```bash
sbatch slurm/sb-ism.sbatch
```

## Training Scripts

### 1. Python Training Script (`scripts/train_rnaseq.py`)

Main training script with full argument support.

**Key arguments:**

- `--data_path`: Path to h5ad file (required)
- `--checkpoint_dir`: Where to save checkpoints (default: `experiments/rnaseq_diffusion`)
- `--num_epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--hidden_dim`: Model hidden dimension (default: 128)
- `--num_layers`: Number of transformer layers (default: 4)
- `--num_heads`: Number of attention heads (default: 4)
- `--mask_ratio`: Fraction of genes to mask (default: 0.15)
- `--resume`: Resume from checkpoint path

**Example:**

```bash
python scripts/train_rnaseq.py \
    --data_path /path/to/data.h5ad \
    --checkpoint_dir experiments/my_experiment \
    --num_epochs 200 \
    --batch_size 64 \
    --hidden_dim 256 \
    --num_layers 6
```

### 2. Bash Training Script (`bash/train.sh`)

Wrapper script that calls the Python training script with sensible defaults.

**Environment variables:**

- `DATA_PATH`: Path to h5ad file
- `CHECKPOINT_DIR`: Checkpoint directory
- `NUM_EPOCHS`: Number of epochs
- `BATCH_SIZE`: Batch size
- `LEARNING_RATE`: Learning rate
- `HIDDEN_DIM`: Model hidden dimension
- `NUM_LAYERS`: Number of layers
- `NUM_HEADS`: Number of heads

**Example:**

```bash
DATA_PATH=/path/to/data.h5ad \
NUM_EPOCHS=200 \
BATCH_SIZE=64 \
./bash/train.sh
```

### 3. Quick Test Script (`bash/train_quick.sh`)

Runs a small model for 5 epochs to verify the pipeline works.

```bash
./bash/train_quick.sh
```

### 4. SLURM Batch Script (`slurm/sb-ism.sbatch`)

SLURM job submission script for cluster training.

**SLURM configuration:**

- Job name: `SEDD-RNAseq`
- GPU: 1x RTX 6000
- Memory: 80GB per GPU
- Time limit: 24 hours
- QoS: embers

**Customization:**

You can override parameters via environment variables or by editing the script:

```bash
DATA_PATH=/path/to/data.h5ad \
NUM_EPOCHS=200 \
BATCH_SIZE=64 \
sbatch slurm/sb-ism.sbatch
```

### 5. Submit Script (`bash/train_submit.sh`)

Convenience wrapper for submitting SLURM jobs.

```bash
./bash/train_submit.sh
```

## Model Architecture

The default model configuration is:

- **Type**: Transformer-based discrete diffusion
- **Hidden dimension**: 128 (small), 256 (medium), 512 (large)
- **Layers**: 4 (small), 6 (medium), 12 (large)
- **Attention heads**: 4 (small), 8 (medium), 16 (large)
- **Dropout**: 0.1
- **Graph**: Absorbing graph (masking)
- **Noise schedule**: LogLinear with eps=1e-3

## Training Process

The training pipeline:

1. **Load data**: Reads h5ad file with RNA-seq expression data
2. **Discretize**: Expression values are already discretized in the data
3. **Split**: Creates train/validation split (default: 90/10)
4. **Train**: Trains model with masked prediction objective
5. **Validate**: Evaluates on held-out validation set each epoch
6. **Checkpoint**: Saves best model and periodic checkpoints

### Checkpointing

Checkpoints are saved to `{checkpoint_dir}/`:

- `best.pt`: Model with best validation loss
- `final.pt`: Model after final epoch
- `epoch_N.pt`: Periodic checkpoints every 10 epochs
- `args.json`: Training arguments for reproducibility

Each checkpoint contains:

- Model state dict
- Optimizer state dict
- Training step and epoch
- Best validation loss
- Training history

### Resuming Training

To resume from a checkpoint:

```bash
python scripts/train_rnaseq.py \
    --data_path /path/to/data.h5ad \
    --resume experiments/my_experiment/best.pt \
    --num_epochs 200
```

## Monitoring Training

### Local Training

Training progress is displayed with:

- Real-time loss values via tqdm progress bar
- Epoch summaries with train/val loss
- Checkpoint saves

### Cluster Training

Monitor SLURM job:

```bash
# Check job status
squeue -u $USER

# Watch output in real-time
tail -f slurm_out/sedd_rnaseq-*.out

# Check job details
scontrol show job JOBID
```

## Data Format

The training script expects h5ad files (AnnData format) with:

- `adata.X`: Expression matrix (cells × genes)
- `adata.var`: Gene information (optional, for logging)
- Expression values should be discretized integers (bin indices)

Example data preparation:

```python
import scanpy as sc
import numpy as np

# Load raw data
adata = sc.read_h5ad("raw_data.h5ad")

# Discretize expression (if needed)
# Your data may already be discretized
num_bins = 100
adata.X = np.digitize(adata.X, bins=np.linspace(0, adata.X.max(), num_bins))

# Save
adata.write_h5ad("discretized_data.h5ad")
```

## Hyperparameter Tuning

### Small-scale experiments

For testing or small datasets:

```bash
BATCH_SIZE=16 \
HIDDEN_DIM=64 \
NUM_LAYERS=2 \
NUM_HEADS=2 \
./bash/train.sh
```

### Medium-scale experiments

Balanced configuration:

```bash
BATCH_SIZE=32 \
HIDDEN_DIM=256 \
NUM_LAYERS=6 \
NUM_HEADS=8 \
./bash/train.sh
```

### Large-scale experiments

For large datasets and compute:

```bash
BATCH_SIZE=64 \
HIDDEN_DIM=512 \
NUM_LAYERS=12 \
NUM_HEADS=16 \
./bash/train.sh
```

## Troubleshooting

### Out of Memory

Reduce batch size or model size:

```bash
BATCH_SIZE=8 HIDDEN_DIM=64 ./bash/train.sh
```

### Slow Training

Increase batch size or reduce model size:

```bash
BATCH_SIZE=128 NUM_LAYERS=4 ./bash/train.sh
```

### Poor Performance

Try:

- Increase model capacity (hidden_dim, num_layers)
- Increase training epochs
- Adjust mask_ratio (try 0.10 - 0.20)
- Adjust learning rate (try 5e-5 to 5e-4)

## Reference

The training implementation is based on the methodology shown in `nbs/rnaseq.ipynb`.

Key components:

- **Model**: `sedd.model.SEDDTransformerSmall`
- **Graph**: `sedd.graph.AbsorbingGraph`
- **Noise**: `sedd.noise.LogLinearNoise`
- **Trainer**: `sedd.trainer.SEDDTrainer`
