# Cell Type Conditioning Feature

## Overview

The training pipeline now supports dual conditioning on both **perturbation labels** and **cell types**. This allows the model to generate cell states conditioned on:
1. The perturbation/gene being applied
2. The cell type context

## Changes Made

### 1. Training Script (`scripts/train_perturbseq.py`)

**Added:**
- Cell type extraction from `adata.obs['cell_type']`
- Cell type to index mapping (`cell_type_to_idx`)
- Cell type count (`NUM_CELL_TYPES`)
- Cell type metadata persistence in `args.json`
- Cell type lookup passed to trainer

**Key additions:**
```python
# Extract cell types
if 'cell_type' in adata.obs.columns:
    cell_types = adata.obs['cell_type'].unique()
    NUM_CELL_TYPES = len(cell_types)
    cell_type_to_idx = {ct: idx for idx, ct in enumerate(sorted(cell_types))}
```

### 2. Model (`sedd/model.py`)

**Modified `SEDDPerturbationTransformer`:**
- Added `num_cell_types` parameter to `__init__`
- Added cell type embedding layer (`self.cell_type_embed`)
- Added cell type projection network (`self.cell_type_proj`)
- Updated `forward()` to accept `cell_type_labels` parameter
- Updated conditioning to combine: `time + perturbation + cell_type`
- Updated `score()` and `get_loss()` methods to pass cell_type_labels

**Architecture:**
```
cond = t_emb + p_emb + ct_emb  # Combined conditioning
```

### 3. Trainer (`sedd/trainer.py`)

**Modified `PerturbationTrainer`:**
- Added `cell_type_lookup` parameter (dict mapping cell_type names to indices)
- Modified `train_step()` to extract cell_type from batch
- Modified `validate()` to extract cell_type from batch
- Updated `compute_loss()` to accept and pass `cell_type_labels`
- Cell type names are automatically converted to indices using the lookup

**Batch handling:**
```python
if 'cell_type' in batch and self.cell_type_lookup is not None:
    cell_type_names = batch['cell_type']
    cell_type_indices = [self.cell_type_lookup[ct_name] for ct_name in cell_type_names]
    cell_type_labels = torch.tensor(cell_type_indices, device=self.device)
```

### 4. Sampler (`sedd/sampling.py`)

**Modified `PerturbationEulerSampler`:**
- Updated `sample()` to accept `cell_type_labels` parameter
- Updated `step()` to pass cell_type_labels to model.score()
- Updated `denoise()` to accept and pass cell_type_labels

## Usage

### During Training

The training script automatically detects and uses cell_type information if available in the dataset:

```bash
python scripts/train_perturbseq.py --config configs/perturbseq_small.yaml
```

**Requirements:**
- Dataset must have `cell_type` column in `adata.obs`
- The dataloader should return cell_type information in batches

### During Inference

When generating samples, you can now specify both perturbation and cell type:

```python
from sedd.sampling import PerturbationEulerSampler

# Create sampler
sampler = PerturbationEulerSampler(
    model=model,
    graph=graph,
    noise=noise,
    num_steps=100,
    device=device
)

# Sample with dual conditioning
x_init = torch.full((batch_size, num_genes), mask_idx, device=device)
pert_labels = torch.tensor([pert_idx], device=device)  # Perturbation index
cell_type_labels = torch.tensor([cell_type_idx], device=device)  # Cell type index

samples = sampler.sample(
    x_init=x_init,
    pert_labels=pert_labels,
    cell_type_labels=cell_type_labels,  # NEW: Cell type conditioning
    show_progress=True
)
```

### Checkpoint Metadata

The `args.json` file now includes:
```json
{
  "num_cell_types": 3,
  "cell_type_to_idx": {
    "hepg2": 0,
    "jurkat": 1,
    "rpe1": 2
  }
}
```

This allows you to:
1. Know how many cell types the model was trained on
2. Map cell type names to indices for inference

## Example Workflow

### 1. Train on multi-cell-type dataset
```bash
python scripts/train_perturbseq.py \
    --config configs/perturbseq_small.yaml \
    --train_data_path datasets/20M/train.h5ad
```

The model will automatically condition on both perturbation and cell_type.

### 2. Load checkpoint and generate
```python
import json
import torch
from sedd.model import SEDDPerturbationTransformerSmall

# Load metadata
with open("experiments/perturbseq_diffusion/args.json", "r") as f:
    args = json.load(f)

cell_type_to_idx = args["cell_type_to_idx"]
num_cell_types = args["num_cell_types"]

# Create model
model = SEDDPerturbationTransformerSmall(
    num_genes=args["num_genes"],
    num_bins=args["num_bins"],
    num_perturbations=args["num_perturbations"],
    num_cell_types=num_cell_types  # Enable cell type conditioning
)

# Generate for specific perturbation + cell type
pert_name = "MYC"
cell_type = "hepg2"

pert_idx = perturbation_to_idx[pert_name]
cell_type_idx = cell_type_to_idx[cell_type]

samples = sampler.sample(
    x_init=x_init,
    pert_labels=torch.tensor([pert_idx]),
    cell_type_labels=torch.tensor([cell_type_idx])
)
```

## Backward Compatibility

The implementation maintains backward compatibility:
- If `num_cell_types=None` or `num_cell_types=0`, cell type conditioning is disabled
- Models trained without cell_type will work as before
- The `cell_type_labels` parameter is optional in all methods

## Data Requirements

For cell type conditioning to work, your dataset must:
1. Have a `cell_type` column in `adata.obs`
2. The dataloader must return `cell_type` in the batch dictionary

Example batch structure:
```python
{
    'pert_cell_emb': tensor([...]),  # Cell expression
    'pert_emb': tensor([...]),       # Perturbation embedding
    'cell_type': ['hepg2', 'jurkat', ...]  # Cell type names
}
```

## Benefits

1. **Context-aware generation**: Generate cell states specific to cell type
2. **Multi-cell-type learning**: Learn perturbation effects across different cell contexts
3. **Transfer learning**: Potentially generalize perturbations across cell types
4. **Experimental design**: Ask "What would this perturbation do in cell type X?"

## Notes

- Cell types are embedded and projected through a learned network (similar to perturbations)
- The conditioning is additive: `cond = time_emb + pert_emb + cell_type_emb`
- Missing cell types default to index 0 (first cell type)
- The feature is automatically enabled when cell_type data is available

