# RNA-seq Model Configuration

This directory contains YAML configuration files for SEDD RNA-seq models. These configs centralize all model parameters, training hyperparameters, and inference settings in one place.

## Available Configs

### `rnaseq_small.yaml`
Small model configuration:
- **Model**: 128 hidden dims, 4 layers, 4 heads
- **Training**: batch_size=8, lr=1e-4
- **Use case**: Quick experiments, testing, smaller datasets

### `rnaseq_large.yaml`
Large model configuration:
- **Model**: 512 hidden dims, 12 layers, 8 heads
- **Training**: batch_size=4, lr=5e-5
- **Use case**: Full-scale training, larger datasets

## Usage

### Training

**Using bash scripts:**
```bash
# Small model
./bash/train_mlm.sh --config configs/rnaseq_small.yaml --data_path /path/to/data.h5ad

# Large model
CONFIG=configs/rnaseq_large.yaml DATA_PATH=/path/to/data.h5ad ./bash/train_mlm.sh
```

**Using sbatch (SLURM):**
```bash
# Small model
sbatch slurm/sb-train-mlm.sbatch --config configs/rnaseq_small.yaml --data_path /path/to/data.h5ad

# Large model
CONFIG=configs/rnaseq_large.yaml DATA_PATH=/path/to/data.h5ad sbatch slurm/sb-train-mlm.sbatch
```

**Direct Python:**
```bash
python scripts/train_rnaseq.py --config configs/rnaseq_small.yaml --data_path /path/to/data.h5ad
python scripts/train_rnaseq_large.py --config configs/rnaseq_large.yaml --data_path /path/to/data.h5ad
```

### Inference

**Using bash scripts:**
```bash
./bash/inf_imputation.sh \
    --config configs/rnaseq_small.yaml \
    --experiment_dir experiments/rnaseq_small \
    --data_path /path/to/test_data.h5ad
```

**Using sbatch:**
```bash
sbatch slurm/sb-inference-imputation.sbatch \
    --config configs/rnaseq_small.yaml \
    --experiment_dir experiments/rnaseq_small
```

**Direct Python:**
```bash
python scripts/inference_imputation.py \
    --config configs/rnaseq_small.yaml \
    --experiment_dir experiments/rnaseq_small \
    --data_path /path/to/test_data.h5ad
```

## Overriding Parameters

All config parameters can be overridden via command line:

```bash
# Override specific parameters
python scripts/train_rnaseq.py \
    --config configs/rnaseq_small.yaml \
    --data_path /path/to/data.h5ad \
    --num_epochs 50 \
    --batch_size 16 \
    --learning_rate 2e-4
```

## Config Structure

```yaml
# Model architecture
model:
  name: SEDDTransformerSmall
  hidden_dim: 128
  num_layers: 4
  num_heads: 4
  dropout: 0.1

# Data parameters
data:
  data_path: null  # Must specify via CLI
  val_fraction: 0.1

# Training hyperparameters
training:
  batch_size: 8
  num_epochs: 100
  learning_rate: 1.0e-4
  weight_decay: 0.01
  mask_ratio: 0.15
  gradient_clip: 1.0
  optimizer:
    type: AdamW
    betas: [0.9, 0.999]

# Checkpointing and logging
checkpointing:
  checkpoint_dir: experiments/rnaseq_small
  save_interval: 10
  resume: null

logging:
  log_interval: 50
  val_interval: 1

# Inference parameters
inference:
  mask_ratio: 0.2
  num_steps: 50
  temperature: 1.0
  sampler: euler
  batch_size: 32
  num_batches: null
  num_cells_visualize: 3

# Other parameters
other:
  seed: 42
  num_workers: 4

# Diffusion parameters
diffusion:
  noise_schedule: LogLinearNoise
  noise_eps: 1.0e-3
  graph_type: AbsorbingGraph
```

## Creating Custom Configs

To create a custom configuration:

1. Copy an existing config:
   ```bash
   cp configs/rnaseq_small.yaml configs/my_custom_config.yaml
   ```

2. Edit the parameters as needed

3. Use your custom config:
   ```bash
   python scripts/train_rnaseq.py --config configs/my_custom_config.yaml --data_path /path/to/data.h5ad
   ```

## Benefits

- **No duplication**: Parameters defined once, used across training and inference
- **Easy management**: Change parameters in one place
- **Version control**: Track configuration changes with git
- **Reproducibility**: Share exact configurations for experiments
- **Flexibility**: Still allows command-line overrides when needed
