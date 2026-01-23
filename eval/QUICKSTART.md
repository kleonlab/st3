# Quick Start Guide

## Simplest Usage - Edit CONFIG and Run

1. **Open `eval/evaluate.py`** and edit the CONFIG section (lines 30-54):

```python
CONFIG = {
    # Set your file paths
    'real_data_path': 'data/real.h5ad',           # Your ground truth data
    'predicted_data_path': 'data/predicted.h5ad',  # Your model predictions

    # Choose evaluation mode
    'mode': 'both',  # 'reconstruction', 'generation', or 'both'

    # Where to save results (optional)
    'output_path': 'results/evaluation_results.json',
}
```

2. **Run the script**:

```bash
python eval/evaluate.py
```

That's it! Results will be printed to console and saved to `output_path`.

---

## Working with h5ad Files

### Basic h5ad Evaluation

```python
CONFIG = {
    'real_data_path': 'data/pbmc_real.h5ad',
    'predicted_data_path': 'data/pbmc_generated.h5ad',
    'mode': 'generation',
}
```

### Filter by Perturbation

Evaluate only cells with specific perturbation:

```python
CONFIG = {
    'real_data_path': 'data/perturbseq_real.h5ad',
    'predicted_data_path': 'data/perturbseq_generated.h5ad',
    'filter_by_perturbation': 'KRAS',  # Only evaluate KRAS-perturbed cells
    'mode': 'both',
}
```

### Filter by Gene

```python
CONFIG = {
    'filter_by_gene': 'TP53',  # Only evaluate cells with TP53 gene label
}
```

### Use Specific Layer

```python
CONFIG = {
    'use_layer': 'counts',  # Use raw counts instead of .X
}
```

### Complete h5ad Example

```python
CONFIG = {
    'real_data_path': 'data/perturbseq_real.h5ad',
    'predicted_data_path': 'data/perturbseq_generated.h5ad',
    'mode': 'both',
    'use_layer': None,  # Use .X (default)
    'filter_by_perturbation': 'KRAS',
    'output_path': 'results/kras_metrics.json',
    'output_format': 'json',
    'w2_projections': 1000,
    'mmd_subsample': 2000,
    'random_seed': 42,
    'verbose': True,
}
```

---

## Command Line Overrides (Optional)

You can override CONFIG settings without editing the file:

```bash
# Override input files
python eval/evaluate.py --real data/real.h5ad --pred data/pred.h5ad

# Filter by perturbation
python eval/evaluate.py --filter-perturbation KRAS

# Change mode
python eval/evaluate.py --mode generation

# Use specific layer
python eval/evaluate.py --use-layer counts

# Change output
python eval/evaluate.py --output results/my_results.json --format txt

# Combine multiple overrides
python eval/evaluate.py --real data/real.h5ad --filter-perturbation KRAS --mode generation
```

---

## Understanding h5ad File Structure

Your h5ad file should have:

```python
import scanpy as sc

adata = sc.read_h5ad('data.h5ad')

# Expression matrix
adata.X  # (n_cells, n_genes) - default matrix

# Cell metadata (labels)
adata.obs
# Example columns: 'perturbation', 'gene', 'cell_type', etc.

# Gene metadata
adata.var

# Alternative expression matrices
adata.layers['counts']      # Raw counts
adata.layers['normalized']  # Normalized
```

The evaluation script will:
- Automatically detect `perturbation` or `perturbation_label` columns
- Automatically detect `gene` or `gene_label` columns
- Show available columns if filters don't match
- Include metadata in results

---

## Evaluation Modes

### 1. Reconstruction Mode

For models that reconstruct input (autoencoders, VAEs):
- Requires same number of cells in both files
- Computes: RE ↓, PCC ↑, MSE ↓

```python
CONFIG = {
    'mode': 'reconstruction',
}
```

### 2. Generation Mode

For generative models (GANs, diffusion models):
- Cell counts can differ
- Computes: W2 ↓, MMD2 ↓, FD ↓

```python
CONFIG = {
    'mode': 'generation',
}
```

### 3. Both Modes

Compute all metrics (requires same cell count):

```python
CONFIG = {
    'mode': 'both',
}
```

---

## Common Use Cases

### Case 1: Compare Multiple Models

Create a script to evaluate multiple models:

```python
models = ['scLDM', 'CFGen', 'scVI', 'scDiffusion']

for model in models:
    CONFIG = {
        'real_data_path': 'data/real.h5ad',
        'predicted_data_path': f'data/{model}_generated.h5ad',
        'mode': 'generation',
        'output_path': f'results/{model}_metrics.json',
    }
    # Run evaluation...
```

Or use bash:

```bash
for model in scLDM CFGen scVI scDiffusion; do
    python eval/evaluate.py \
        --pred data/${model}_generated.h5ad \
        --output results/${model}_metrics.json
done
```

### Case 2: Evaluate Specific Perturbations

Test each perturbation separately:

```python
perturbations = ['KRAS', 'TP53', 'EGFR']

for pert in perturbations:
    CONFIG = {
        'filter_by_perturbation': pert,
        'output_path': f'results/{pert}_metrics.json',
    }
    # Run evaluation...
```

### Case 3: Quick Test (No File Saving)

```python
CONFIG = {
    'real_data_path': 'data/real.h5ad',
    'predicted_data_path': 'data/predicted.h5ad',
    'mode': 'generation',
    'output_path': None,  # Don't save, just print
    'verbose': True,
}
```

---

## Performance Tips

For large datasets, adjust these parameters:

```python
CONFIG = {
    'w2_projections': 500,     # Reduce from 1000 for faster W2
    'mmd_subsample': 1000,     # Reduce from 2000 for faster MMD
    'random_seed': 42,         # Keep for reproducibility
}
```

Or use command line:

```bash
python eval/evaluate.py --w2-projections 500 --mmd-subsample 1000
```

---

## Troubleshooting

### "FileNotFoundError: Data file not found"

Check your file paths in CONFIG or provide absolute paths:

```python
CONFIG = {
    'real_data_path': '/absolute/path/to/data/real.h5ad',
}
```

### "No 'perturbation' column found"

The script looks for columns named `perturbation`, `perturbation_label`, `gene`, or `gene_label`.

Check your h5ad file:

```python
import scanpy as sc
adata = sc.read_h5ad('data.h5ad')
print(adata.obs.columns)  # See available columns
```

Then use the correct name:

```python
# If your column is named 'treatment' instead of 'perturbation'
# You'll need to rename it or add it manually
```

### "Shape mismatch"

For reconstruction mode, both files must have the same shape. For generation mode, only the number of genes must match.

---

## Output Format

### JSON Output (default)

```json
{
  "config": {
    "real_data_path": "data/real.h5ad",
    "predicted_data_path": "data/predicted.h5ad",
    "mode": "both"
  },
  "metadata": {
    "real_shape": [1000, 2000],
    "pred_shape": [1000, 2000],
    "real_metadata": {
      "n_cells": 1000,
      "n_genes": 2000,
      "unique_perturbations": ["KRAS", "TP53", "EGFR"]
    }
  },
  "reconstruction": {
    "re": 0.123456,
    "pcc": 0.987654,
    "mse": 0.234567
  },
  "generation": {
    "w2": 0.345678,
    "mmd2_rbf": 0.456789,
    "fd": 0.567890
  }
}
```

### Text Output

Set `'output_format': 'txt'` for human-readable format:

```
Single-Cell RNA-seq Evaluation Results
==================================================

Reconstruction Metrics:
--------------------------------------------------
  RE (Reconstruction Error):  0.123456 ↓
  PCC (Pearson Correlation):  0.987654 ↑
  MSE (Mean Squared Error):   0.234567 ↓

Generation Metrics:
--------------------------------------------------
  W2 (Wasserstein-2):         0.345678 ↓
  MMD2 RBF:                   0.456789 ↓
  FD (Fréchet Distance):      0.567890 ↓

↓ = lower is better, ↑ = higher is better
```

---

## Next Steps

1. Set your file paths in CONFIG
2. Run `python eval/evaluate.py`
3. Check results in console and output file
4. Compare metrics across different models
5. See `eval/README.md` for detailed documentation
