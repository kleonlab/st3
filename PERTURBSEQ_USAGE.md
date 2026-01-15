# SEDD Perturbation Prediction - Usage Guide

This guide explains how to use the perturbation prediction training and inference scripts.

## Overview

The perturbation prediction model predicts the outcome of perturbing a cell with a specific gene/condition:
- **Input**: Control cell expression + Perturbation label
- **Output**: Predicted perturbed cell expression

## Files Created

### 1. Training Script
- **Location**: `scripts/train_perturbseq.py`
- **Description**: Python script for training the perturbation prediction model
- **Updates**: Modified to handle data like the notebook (includes gene names, proper data handling)

### 2. Inference Script
- **Location**: `scripts/inference_perturbseq.py`
- **Description**: Python script for running inference and evaluating predictions

### 3. Configuration File
- **Location**: `configs/perturbseq_dry_run.yaml`
- **Description**: YAML configuration with very small model size for quick dry-run testing
- **Model**: 8 hidden dim, 2 layers, 2 heads (3M parameters)
- **Training**: 2 epochs, batch size 32

### 4. Bash Scripts
- **Training**: `bash/train_pseq.sh`
- **Inference**: `bash/inf_pseq.sh`

## Quick Start - Dry Run

### Step 1: Update Config File

Edit `configs/perturbseq_dry_run.yaml` to set your data path:

```yaml
data:
  data_path: /path/to/your/data.h5ad  # UPDATE THIS
  pert_col: gene                       # Column with perturbation labels
  control_name: non-targeting          # Name of control condition
```

### Step 2: Run Training

```bash
# Option 1: Using bash script (recommended)
./bash/train_pseq.sh

# Option 2: With environment variables
CONFIG=configs/perturbseq_dry_run.yaml ./bash/train_pseq.sh

# Option 3: Override data path
./bash/train_pseq.sh --data_path /path/to/data.h5ad

# Option 4: Direct Python command
python scripts/train_perturbseq.py --config configs/perturbseq_dry_run.yaml
```

**Expected output:**
- Checkpoints saved to: `experiments/perturbseq_dry_run/`
- Training completes in ~5-10 minutes (depending on data size)

### Step 3: Run Inference

```bash
# Option 1: Using bash script (recommended)
./bash/inf_pseq.sh

# Option 2: With custom experiment directory
EXPERIMENT_DIR=experiments/perturbseq_dry_run ./bash/inf_pseq.sh

# Option 3: With custom output directory
./bash/inf_pseq.sh --output_dir my_inference_results

# Option 4: Direct Python command
python scripts/inference_perturbseq.py \
    --config configs/perturbseq_dry_run.yaml \
    --experiment_dir experiments/perturbseq_dry_run \
    --save_predictions
```

**Expected output:**
- Predictions saved to: `inference_results/`
- Visualizations: `inference_results/prediction_cell_*.png`
- Metrics printed to console

## Configuration Details

### Model Sizes

**Dry Run (perturbseq_dry_run.yaml):**
- hidden_dim: 8, layers: 2, heads: 2
- ~3M parameters
- Training time: ~5-10 min

**Small (perturbseq_small.yaml):**
- hidden_dim: 128, layers: 4, heads: 4
- ~50-100M parameters
- Training time: ~1-2 hours

**Medium (perturbseq_medium.yaml):**
- hidden_dim: 256, layers: 6, heads: 8
- ~200-300M parameters
- Training time: ~4-6 hours

### Key Parameters in Config

```yaml
# Model architecture
model:
  hidden_dim: 8        # Embedding dimension
  num_layers: 2        # Transformer layers
  num_heads: 2         # Attention heads
  dropout: 0.1

# Training
training:
  batch_size: 32       # Increase if you have more GPU memory
  num_epochs: 2        # Increase for better performance
  learning_rate: 1e-4
  mask_ratio: 0.15     # Fraction of genes masked during training

# Data
data:
  data_path: ...       # Path to your h5ad file
  pert_col: gene       # Column with perturbation labels
  control_name: ...    # Name of control/non-targeting condition
  val_fraction: 0.1    # Validation split

# Checkpointing
checkpointing:
  checkpoint_dir: ...  # Where to save checkpoints
  save_interval: 1     # Save every N epochs

# Inference
inference:
  batch_size: 64
  num_batches: 2       # null = all batches
  num_cells_visualize: 3
```

## Command Line Arguments

### Training (`train_perturbseq.py`)

```bash
python scripts/train_perturbseq.py \
    --config configs/perturbseq_dry_run.yaml \  # Config file
    --data_path /path/to/data.h5ad \            # Data file
    --pert_col gene \                            # Perturbation column
    --control_name non-targeting \               # Control name
    --hidden_dim 8 \                             # Model params
    --num_layers 2 \
    --num_heads 2 \
    --batch_size 32 \
    --num_epochs 2 \
    --learning_rate 1e-4
```

### Inference (`inference_perturbseq.py`)

```bash
python scripts/inference_perturbseq.py \
    --config configs/perturbseq_dry_run.yaml \           # Config file
    --experiment_dir experiments/perturbseq_dry_run \    # Checkpoint dir
    --data_path /path/to/data.h5ad \                     # Data file
    --batch_size 64 \
    --num_batches 2 \                                     # null = all
    --num_cells_visualize 3 \
    --output_dir inference_results \
    --save_predictions
```

## Data Format

Your h5ad file should have:
- **X**: Expression matrix (cells × genes), already discretized into bins
- **obs[pert_col]**: Column with perturbation labels (e.g., "gene")
- **obs values**: Should include control cells (e.g., "non-targeting")
- **var_names**: Gene names

Example structure:
```
adata.X                        # (5000, 8563) - expression bins
adata.obs["gene"]              # ["RPL3", "NCBP2", "non-targeting", ...]
adata.var_names                # ["GENE1", "GENE2", ...]
```

## Evaluation Metrics

The inference script reports:
- **Exact match accuracy**: % of genes with exact bin prediction
- **Mean Absolute Error (MAE)**: Average error in bin predictions
- **Within-k accuracy**: % of genes within k bins (k=1,3,5,10)
- **Per-gene correlation**: Average Pearson correlation across genes

## Outputs

### Training
```
experiments/perturbseq_dry_run/
├── args.json              # Training arguments
├── best.pt                # Best model checkpoint
├── final.pt               # Final model checkpoint
└── epoch_*.pt             # Epoch checkpoints
```

### Inference
```
inference_results/
├── predictions.pkl                          # Saved predictions
├── prediction_cell_0_<perturbation>.png     # Visualization
├── prediction_cell_1_<perturbation>.png
└── prediction_cell_2_<perturbation>.png
```

## Tips for Production Use

1. **Start with dry run** to verify everything works
2. **Use small config** (perturbseq_small.yaml) for initial experiments
3. **Monitor validation loss** - if it stops improving, training has converged
4. **Increase epochs** - 100 epochs is typical for good performance
5. **Tune batch size** - larger = faster, but needs more GPU memory
6. **Use multiple workers** - set `num_workers: 4` in config for faster data loading (only works with GPU)

## Troubleshooting

### Out of Memory (OOM)
- Reduce `batch_size` in config
- Use smaller model (reduce `hidden_dim`, `num_layers`)

### Training too slow
- Increase `batch_size` if you have GPU memory
- Use multiple workers: `num_workers: 4`
- Use GPU if available (automatically detected)

### No control cells found
- Check `control_name` matches your data
- Verify `pert_col` is correct
- Print unique values: `adata.obs[pert_col].unique()`

### Checkpoint not found
- Check `checkpoint_dir` in config
- Verify training completed successfully
- Use `--experiment_dir` to specify location

## Example Workflow

```bash
# 1. Update config with your data path
vim configs/perturbseq_dry_run.yaml

# 2. Run dry-run training (2 epochs, small model)
./bash/train_pseq.sh

# 3. Run inference to test
./bash/inf_pseq.sh

# 4. If successful, switch to production config
./bash/train_pseq.sh --config configs/perturbseq_small.yaml --num_epochs 100

# 5. Run inference on trained model
EXPERIMENT_DIR=experiments/psed_demo/perturbseq_small ./bash/inf_pseq.sh
```

## Questions?

Check the notebook `nbs/train_sedd_perturbation.ipynb` for more examples and visualizations.
