# Cell-Type Conditioned SEDD Perturbation Prediction

This module extends the SEDD (Score-Entropy Discrete Diffusion) perturbation prediction model to incorporate **cell-type conditioning**, enabling cell-type-specific perturbation predictions.

## Overview

### Key Features

1. **Cell-Type Conditioning**: The model learns cell-type-specific perturbation responses
2. **Combined Conditioning**: Uses time + perturbation + cell-type embeddings for adaptive layer normalization
3. **Flexible Inference**: Generate predictions for one cell type at a time via YAML config or CLI

### Architecture

```
Input: perturbed cell expression (noised) + perturbation label + cell-type label
                    ↓
         Token + Position Embeddings
                    ↓
         Time + Perturbation + Cell-Type Embeddings → Combined Conditioning
                    ↓
         Transformer Blocks (with Adaptive Layer Norm)
                    ↓
Output: Predicted perturbed cell expression
```

## Directory Structure

```
celltype_cond/
├── sedd/
│   ├── __init__.py
│   ├── model.py          # SEDDCellTypePerturbationTransformer models
│   ├── trainer.py        # CellTypePerturbationTrainer
│   └── sampling.py       # CellTypePerturbationEulerSampler
├── scripts/
│   ├── train_celltype_perturbseq.py    # Training script
│   └── inference_celltype_perturbseq.py # Inference script
├── configs/
│   ├── celltype_perturbseq_small.yaml  # Small model config
│   ├── celltype_perturbseq_medium.yaml # Medium model config
│   └── celltype_inference_example.yaml # Inference example config
├── bash/
│   ├── train_celltype.sh         # Training shell script
│   ├── inference_celltype.sh     # Inference shell script
│   └── inference_all_celltypes.sh # Batch inference for all cell types
├── slurm/
│   ├── sb-train-celltype.sbatch     # SLURM training script
│   └── sb-inference-celltype.sbatch # SLURM inference script
└── README.md
```

## Quick Start

### Training

1. **Prepare your data**: Ensure your h5ad file has:
   - Perturbation labels in `adata.obs['gene']` (or specify column via `--gene`)
   - Cell-type labels in `adata.obs['cell_type']` (or specify via `--cell_type_col`)

2. **Run training**:

```bash
# Using shell script
TRAIN_DATA_PATH=/path/to/data.h5ad \
CELL_TYPE_COL=cell_type \
./celltype_cond/bash/train_celltype.sh --config celltype_cond/configs/celltype_perturbseq_small.yaml

# Or directly
python celltype_cond/scripts/train_celltype_perturbseq.py \
    --config celltype_cond/configs/celltype_perturbseq_small.yaml \
    --train_data_path /path/to/data.h5ad \
    --cell_type_col cell_type
```

3. **Resume training**:

```bash
python celltype_cond/scripts/train_celltype_perturbseq.py \
    --config celltype_cond/configs/celltype_perturbseq_small.yaml \
    --resume auto  # Resumes from latest checkpoint
```

### Inference

The key feature is generating predictions for **one cell type at a time**.

1. **Generate for a specific cell type**:

```bash
# Via CLI
python celltype_cond/scripts/inference_celltype_perturbseq.py \
    --experiment_dir experiments/celltype_model \
    --cell_type "T cell" \
    --num_samples_per_condition 100

# Via shell script
CELL_TYPE="T cell" \
EXPERIMENT_DIR=experiments/celltype_model \
./celltype_cond/bash/inference_celltype.sh
```

2. **Generate for all cell types**:

```bash
./celltype_cond/bash/inference_all_celltypes.sh \
    --experiment_dir experiments/celltype_model \
    --cell_types "T cell" "B cell" "Monocyte" "NK cell"
```

3. **Configure via YAML**:

```yaml
# celltype_cond/configs/celltype_inference_example.yaml
inference:
  cell_type: "T cell"  # Set to null for all cell types
  num_samples_per_condition: 100
  num_steps: 50
  temperature: 1.0
```

### SLURM Jobs

```bash
# Training
CONFIG=celltype_cond/configs/celltype_perturbseq_medium.yaml \
TRAIN_DATA_PATH=/path/to/data.h5ad \
sbatch celltype_cond/slurm/sb-train-celltype.sbatch

# Inference for specific cell type
CELL_TYPE="T cell" \
EXPERIMENT_DIR=experiments/celltype_model \
sbatch celltype_cond/slurm/sb-inference-celltype.sbatch
```

## Configuration Options

### Key YAML Parameters

```yaml
# Data
data:
  train_data_path: /path/to/data.h5ad
  gene: gene                    # Perturbation column name
  cell_type_col: cell_type      # Cell-type column name
  cond_labels_pt_path: null     # Optional: protein embeddings
  celltype_labels_pt_path: null # Optional: cell-type embeddings

# Model
model:
  size: small  # small, medium, large
  hidden_dim: 128
  num_layers: 4
  num_heads: 4

# Inference
inference:
  cell_type: "T cell"  # KEY PARAMETER: specific cell type or null for all
  num_samples_per_condition: 100
  num_steps: 50
  temperature: 1.0
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `TRAIN_DATA_PATH` | Path to training h5ad file |
| `CELL_TYPE_COL` | Column name for cell types |
| `CELL_TYPE` | Specific cell type for inference |
| `COND_LABELS_PT_PATH` | Perturbation embeddings (.pt) |
| `CELLTYPE_LABELS_PT_PATH` | Cell-type embeddings (.pt) |
| `CHECKPOINT_DIR` | Directory for checkpoints |
| `RESUME` | Checkpoint to resume from |

## Model Variants

| Model | Hidden Dim | Layers | Heads | Parameters |
|-------|------------|--------|-------|------------|
| Small | 128 | 4 | 4 | ~1M |
| Medium | 256 | 6 | 8 | ~5M |
| Large | 512 | 8 | 8 | ~20M |

## Output Format

Generated cells are saved as h5ad files:

```
experiments/celltype_model/inference_results/
├── celltype_T_cell/
│   ├── generated_cells.h5ad
│   └── generation_config.json
├── celltype_B_cell/
│   ├── generated_cells.h5ad
│   └── generation_config.json
└── ...
```

The `generated_cells.h5ad` contains:
- `X`: Generated expression values (genes × bins)
- `obs['perturbation']`: Perturbation label for each cell
- `obs['cell_type']`: Cell type label for each cell

## Extending to Your Data

1. Ensure your h5ad file has cell-type annotations
2. Update config with your column names
3. (Optional) Provide pre-computed embeddings for perturbations/cell-types
4. Train and generate predictions per cell type

## Citation

Built upon the SEDD (Score-Entropy Discrete Diffusion) framework.
