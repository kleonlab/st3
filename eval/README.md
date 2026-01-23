# Single-Cell RNA-seq Evaluation Metrics

This directory contains evaluation scripts for comparing single-cell RNA-seq models, implementing metrics from the paper:

**"Scalable Single-Cell Gene Expression Generation with Latent Diffusion Models"**
https://arxiv.org/abs/2511.02986

## Metrics Implemented

### Reconstruction Metrics (`reconstruction_metrics.py`)

Used to evaluate how well a model reconstructs input data (for autoencoders, VAEs, etc.):

- **RE (Reconstruction Error)** ↓ - Measures L1 distance between original and reconstructed data
- **PCC (Pearson Correlation Coefficient)** ↑ - Measures linear correlation (ranges 0-1)
- **MSE (Mean Squared Error)** ↓ - Average squared difference between predictions and ground truth

### Generation Metrics (`generation_metrics.py`)

Used to evaluate distribution similarity between real and generated data:

- **W2 (Wasserstein-2 Distance)** ↓ - Earth mover's distance between distributions
- **MMD2 RBF (Maximum Mean Discrepancy)** ↓ - Kernel-based distribution comparison
- **FD (Fréchet Distance)** ↓ - Compares distributions via mean and covariance

**↓** = lower is better, **↑** = higher is better

## Installation

Required dependencies:

```bash
# Core dependencies
pip install numpy scipy

# Optional (for loading .h5ad files)
pip install scanpy
```

Or install from project root:

```bash
pip install -e .
```

## Usage

### Quick Start - Command Line

Evaluate a model's performance:

```bash
# Both reconstruction and generation metrics
python eval/evaluate.py --real data/real.npy --pred data/predicted.npy --mode both

# Only reconstruction metrics
python eval/evaluate.py --real data/real.npy --pred data/reconstructed.npy --mode reconstruction

# Only generation metrics
python eval/evaluate.py --real data/real.npy --pred data/generated.npy --mode generation

# Save results to file
python eval/evaluate.py --real data/real.npy --pred data/predicted.npy --output results.json

# With AnnData format
python eval/evaluate.py --real data/real.h5ad --pred data/predicted.h5ad --mode both
```

### Python API

#### Reconstruction Metrics

```python
import numpy as np
from eval.reconstruction_metrics import evaluate_reconstruction

# Load your data (n_cells, n_genes)
original = np.load('data/original.npy')
reconstructed = np.load('data/reconstructed.npy')

# Evaluate all metrics
results = evaluate_reconstruction(original, reconstructed, verbose=True)
# Output:
# Reconstruction Metrics:
#   RE (Reconstruction Error):  0.123456 ↓
#   PCC (Pearson Correlation):  0.987654 ↑
#   MSE (Mean Squared Error):   0.234567 ↓

# Access individual metrics
print(f"RE: {results['re']}")
print(f"PCC: {results['pcc']}")
print(f"MSE: {results['mse']}")
```

#### Generation Metrics

```python
import numpy as np
from eval.generation_metrics import evaluate_generation

# Load your data (n_cells, n_genes)
real_data = np.load('data/real.npy')
generated_data = np.load('data/generated.npy')

# Evaluate all metrics
results = evaluate_generation(real_data, generated_data, verbose=True)
# Output:
# Generation Metrics:
#   W2 (Wasserstein-2):        0.123456 ↓
#   MMD2 RBF:                  0.234567 ↓
#   FD (Fréchet Distance):     0.345678 ↓

# Access individual metrics
print(f"W2: {results['w2']}")
print(f"MMD2: {results['mmd2_rbf']}")
print(f"FD: {results['fd']}")
```

#### Individual Metrics

```python
from eval.reconstruction_metrics import reconstruction_error, pearson_correlation_coefficient, mean_squared_error
from eval.generation_metrics import wasserstein2_distance, mmd2_rbf, frechet_distance

# Reconstruction metrics
re = reconstruction_error(original, reconstructed)
pcc, pval = pearson_correlation_coefficient(original, reconstructed)
mse = mean_squared_error(original, reconstructed)

# Generation metrics
w2 = wasserstein2_distance(real, generated, num_projections=1000)
mmd = mmd2_rbf(real, generated, subsample=2000)
fd = frechet_distance(real, generated)
```

## Supported Data Formats

- **`.npy`** - NumPy array files
- **`.npz`** - Compressed NumPy files
- **`.h5ad`** - AnnData files (requires scanpy)
- **`.csv`** / **`.tsv`** - Text files (cells × genes)

All formats should contain a 2D matrix of shape `(n_cells, n_genes)`.

## Example Workflow

### Comparing Models

```python
import numpy as np
from eval import evaluate_reconstruction, evaluate_generation

# Load ground truth data
real_data = np.load('data/pbmc_real.npy')

# Evaluate different models
models = {
    'scLDM': np.load('results/scldm_generated.npy'),
    'CFGen': np.load('results/cfgen_generated.npy'),
    'scVI': np.load('results/scvi_generated.npy'),
    'scDiffusion': np.load('results/scdiffusion_generated.npy'),
}

results = {}
for model_name, generated_data in models.items():
    print(f"\nEvaluating {model_name}...")
    results[model_name] = evaluate_generation(real_data, generated_data)

# Compare results
import pandas as pd
df = pd.DataFrame(results).T
print("\nModel Comparison:")
print(df.to_string())
```

### Batch Evaluation

```bash
# Evaluate multiple models
for model in scldm cfgen scvi scdiffusion; do
    python eval/evaluate.py \
        --real data/real.npy \
        --pred results/${model}_generated.npy \
        --mode generation \
        --output results/${model}_metrics.json
done
```

## Computational Considerations

For large datasets, generation metrics can be computationally expensive:

**Wasserstein-2 Distance:**
- Uses sliced Wasserstein for efficiency
- Default: 1000 random projections
- Adjust with `--w2-projections` or `num_projections` parameter

**MMD2 RBF:**
- Can subsample for speed
- Default: 2000 samples max
- Adjust with `--mmd-subsample` or `subsample` parameter

**Fréchet Distance:**
- Computes full covariance matrices
- No subsampling option
- Consider downsampling input data for very large datasets

Example for faster computation on large datasets:

```bash
python eval/evaluate.py \
    --real data/large_real.npy \
    --pred data/large_generated.npy \
    --mode generation \
    --w2-projections 500 \
    --mmd-subsample 1000
```

## Testing

Run the example code in each module:

```bash
# Test reconstruction metrics
python eval/reconstruction_metrics.py

# Test generation metrics
python eval/generation_metrics.py

# Test full evaluation
python eval/evaluate.py --help
```

## File Structure

```
eval/
├── __init__.py                 # Package initialization
├── README.md                   # This file
├── reconstruction_metrics.py   # RE, PCC, MSE implementations
├── generation_metrics.py       # W2, MMD2, FD implementations
└── evaluate.py                 # Main evaluation script
```

## Citation

If you use these metrics in your research, please cite the scLDM paper:

```bibtex
@article{scldm2024,
    title={Scalable Single-Cell Gene Expression Generation with Latent Diffusion Models},
    author={...},
    journal={arXiv preprint arXiv:2511.02986},
    year={2024}
}
```

## Notes

- All metrics assume input data is in the form of gene expression matrices (cells × genes)
- For log-normalized data, metrics are computed in log-space
- Missing values (NaN) should be handled before evaluation
- Reconstruction metrics require matching cell counts; generation metrics do not
