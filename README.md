## documenting implementation 

- run the pyproject.toml file by running uv sync in the terminal. 
now all the requirements are downloaded and setup. 

- the demo_rnaseq is a complete running example. 

## Overview

This library adapts discrete diffusion models for gene expression prediction:

- **Discretization**: Continuous gene expression values are binned into discrete tokens
- **Absorbing Diffusion**: Tokens are progressively masked during the forward process
- **Score Matching**: A transformer learns to predict masked tokens from context
- **Sampling**: Reverse diffusion recovers the original gene expression

### Applications

- **Gene Expression Imputation**: Fill in missing or dropout values
- **Denoising**: Correct noisy measurements
- **Generation**: Create synthetic single-cell expression profiles

## Installation

```bash
# Clone the repository
git clone https://github.com/kleonlab/st3.git
cd st3

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

```python
import torch
from sedd import (
    SEDDTransformerSmall,
    AbsorbingGraph,
    LogLinearNoise,
    SEDDTrainer,
    RNASeqDataset,
    impute_masked
)

# 1. Prepare data
from sedd.data import create_synthetic_rnaseq
expression, cell_types, gene_names = create_synthetic_rnaseq(
    num_cells=1000,
    num_genes=200,
    seed=42
)

# 2. Create dataset (automatically discretizes expression)
dataset = RNASeqDataset(
    expression=expression,
    num_bins=100,
    discretization_method='log_uniform'
)
train_loader = dataset.get_dataloader(batch_size=32)

# 3. Create model
model = SEDDTransformerSmall(
    num_genes=dataset.num_genes,
    num_bins=dataset.num_bins
)
graph = AbsorbingGraph(num_states=dataset.num_bins + 1)
noise = LogLinearNoise()

# 4. Train
trainer = SEDDTrainer(model, graph, noise)
trainer.train(train_loader, num_epochs=50)

# 5. Impute missing values
x = dataset[0:10]  # Get some cells
mask = torch.rand_like(x.float()) < 0.2  # Mask 20% of genes
imputed = impute_masked(model, graph, noise, x, mask)
```

## Architecture

### Core Components

```
sedd/
├── noise.py      # Noise schedules (LogLinear, Geometric)
├── graph.py      # Transition graphs (Absorbing, Uniform)
├── model.py      # Transformer model with adaptive layer norm
├── sampling.py   # Sampling strategies (Euler, Analytic)
├── trainer.py    # Training loop and utilities
└── data.py       # RNA-seq data handling and discretization
```

### Model Sizes

| Size | Hidden Dim | Layers | Heads | Parameters |
|------|------------|--------|-------|------------|
| Small | 128 | 4 | 4 | ~1M |
| Medium | 256 | 6 | 8 | ~6M |
| Large | 512 | 8 | 8 | ~25M |

## Key Concepts

### Discrete Diffusion

Unlike continuous diffusion (e.g., DDPM), discrete diffusion works with categorical data:

1. **Forward Process**: Tokens are progressively corrupted towards a stationary distribution
2. **Absorbing Graph**: All tokens transition towards a special "mask" token
3. **Reverse Process**: The model learns to unmask tokens based on context

### Score Entropy Loss

The model is trained to match the "score" - the gradient of the log probability:

```
L = E[CrossEntropy(score(x_noised), x_clean) at masked positions]
```

This is equivalent to masked language modeling but with a principled diffusion framework.

## Examples

See [`examples/demo_rnaseq.ipynb`](examples/demo_rnaseq.ipynb) for a complete walkthrough including:

- Data loading and preprocessing
- Model training
- Gene expression imputation
- De novo cell generation
- Visualization

### Data

```python
# Discretize expression
discretized, metadata = discretize_expression(
    expression,
    num_bins=100,
    method='log_uniform'  # or 'uniform', 'quantile'
)

# Create dataset
dataset = RNASeqDataset(
    expression=expression,
    num_bins=100,
    gene_names=gene_names,
    cell_labels=cell_types
)

# Load 10x Genomics data
expression, genes, barcodes = load_10x_h5('filtered_feature_bc_matrix.h5')
```

### Model

```python
model = SEDDTransformer(
    num_genes=500,        # Number of genes (sequence length)
    num_bins=100,         # Expression bins (vocabulary size)
    hidden_dim=256,       # Transformer hidden dimension
    num_layers=6,         # Number of transformer blocks
    num_heads=8,          # Attention heads
    dropout=0.1           # Dropout rate
)
```

### Training

```python
trainer = SEDDTrainer(model, graph, noise)
history = trainer.train(
    train_loader,
    val_loader=val_loader,
    num_epochs=100,
    mask_ratio=0.15,      # Fraction of genes to mask
    checkpoint_dir='checkpoints'
)
```

### Sampling

```python
# Impute specific positions
imputed = impute_masked(
    model, graph, noise,
    x=data,
    mask=positions_to_impute,
    num_steps=100,
    temperature=1.0
)

# Generate from scratch
sampler = EulerSampler(model, graph, noise, num_steps=100)
x_init = graph.sample_limiting((batch_size, num_genes), device)
generated = sampler.sample(x_init)
```

## Citation

If you use this code, please cite the original SEDD paper:

```bibtex
@inproceedings{lou2024discrete,
  title={Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution},
  author={Lou, Aaron and Meng, Chenlin and Ermon, Stefano},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```

## License

MIT License
