# Inference with Trained SEDD Models

This document describes how to run inference with trained SEDD models for RNA-seq data. Two types of inference are supported:

1. **Imputation**: Mask some genes and predict their expression values
2. **Generation**: Generate entirely new synthetic cells from scratch

## Prerequisites

Make sure you have:
- A trained model checkpoint (from training, see [TRAINING.md](TRAINING.md))
- Test data (h5ad format)
- Configuration file (`config.toml`) with data paths set

## Configuration

The inference scripts read data paths from `config.toml`. Edit this file to set your data paths:

```toml
[data]
# Path to training/full dataset
train_data = "/path/to/your/data.h5ad"

# Path to test dataset (optional)
test_data = ""

# Validation fraction when splitting train_data
val_fraction = 0.1
```

## Quick Start

### Imputation Inference

Run imputation on a trained model:

```bash
# Local execution
python scripts/inference_imputation.py \
    --experiment_dir experiments/rnaseq_diffusion_20240115

# Cluster (SLURM)
sbatch slurm/sb-inference-imputation.sbatch --experiment_dir experiments/rnaseq_diffusion_20240115
```

### Generation Inference

Generate new synthetic cells:

```bash
# Local execution
python scripts/inference_generation.py \
    --experiment_dir experiments/rnaseq_diffusion_20240115

# Cluster (SLURM)
sbatch slurm/sb-inference-generation.sbatch --experiment_dir experiments/rnaseq_diffusion_20240115
```

## Imputation Inference

### Script: `scripts/inference_imputation.py`

Performs gene expression imputation by masking genes and predicting their values.

**Key Arguments:**

- `--experiment_dir`: Path to trained model directory (required)
- `--checkpoint`: Specific checkpoint file (optional, auto-finds best/final if not provided)
- `--data_path`: Path to test data (defaults to `config.toml`)
- `--mask_ratio`: Fraction of genes to mask (default: 0.2)
- `--num_steps`: Number of sampling steps (default: 50)
- `--temperature`: Sampling temperature (default: 1.0)
- `--batch_size`: Batch size for inference (default: 32)
- `--num_batches`: Number of batches to evaluate (default: all)
- `--num_cells_visualize`: Number of cells to visualize (default: 3)

**Example:**

```bash
python scripts/inference_imputation.py \
    --experiment_dir experiments/my_trained_model \
    --mask_ratio 0.3 \
    --num_steps 100 \
    --temperature 0.9
```

### SLURM Script: `slurm/sb-inference-imputation.sbatch`

**Environment Variables:**

- `EXPERIMENT_DIR`: Experiment directory (required)
- `MASK_RATIO`: Mask ratio (default: 0.2)
- `NUM_STEPS`: Sampling steps (default: 50)
- `TEMPERATURE`: Temperature (default: 1.0)
- `BATCH_SIZE`: Batch size (default: 32)

**Example:**

```bash
EXPERIMENT_DIR=experiments/my_model \
MASK_RATIO=0.3 \
NUM_STEPS=100 \
sbatch slurm/sb-inference-imputation.sbatch
```

### Outputs (saved to `experiment_dir/imputation_results/`)

**Metrics:**
- `metrics.json`: Comprehensive metrics including accuracy, MAE, within-k accuracy
- `accuracy.txt`: Exact match accuracy
- `mae.json`: Mean absolute error
- `within_k.csv`: Within-k bin accuracy (k=1,3,5,10)
- `cell_N_acc.txt`, `cell_N_mae.json`: Per-cell metrics

**Visualizations:**
- `imputation_scatter_hist.png`: Predicted vs true values scatter plot and error distribution
- `single_cell_N.png`: Individual cell expression profiles showing original and imputed values

**Example Output:**

```
Imputation Metrics
==================================================
Exact match accuracy: 47.13%
Mean Absolute Error (bins): 1.68
Within 1 bins: 79.33%
Within 3 bins: 94.62%
Within 5 bins: 97.58%
Within 10 bins: 98.89%
```

## Generation Inference

### Script: `scripts/inference_generation.py`

Generates new synthetic cells from scratch by sampling from the trained model.

**Key Arguments:**

- `--experiment_dir`: Path to trained model directory (required)
- `--checkpoint`: Specific checkpoint file (optional, auto-finds best/final)
- `--data_path`: Path to real data for comparison (defaults to `config.toml`)
- `--num_generate`: Number of cells to generate (default: 100)
- `--num_steps`: Number of sampling steps (default: 100)
- `--temperature`: Sampling temperature (default: 1.0)
- `--num_cells_visualize`: Number of generated cells to visualize (default: 3)
- `--num_real_visualize`: Number of real cells to show (default: 3)

**Example:**

```bash
python scripts/inference_generation.py \
    --experiment_dir experiments/my_trained_model \
    --num_generate 500 \
    --num_steps 100 \
    --temperature 1.0
```

### SLURM Script: `slurm/sb-inference-generation.sbatch`

**Environment Variables:**

- `EXPERIMENT_DIR`: Experiment directory (required)
- `NUM_GENERATE`: Number of cells to generate (default: 100)
- `NUM_STEPS`: Sampling steps (default: 100)
- `TEMPERATURE`: Temperature (default: 1.0)

**Example:**

```bash
EXPERIMENT_DIR=experiments/my_model \
NUM_GENERATE=500 \
NUM_STEPS=100 \
sbatch slurm/sb-inference-generation.sbatch
```

### Outputs (saved to `experiment_dir/generation_results/`)

**Generated Data:**
- `generated_cells.pt`: PyTorch tensor of generated cells
- `generated_cells.npy`: NumPy array of generated cells

**Metrics:**
- `metrics.json`: Comprehensive metrics including correlations and statistics
- `mean_corr.txt`: Mean expression correlation with real cells
- `std_corr.txt`: Standard deviation correlation with real cells

**Visualizations:**
- `real_vs_generated.png`: Side-by-side comparison of real and generated cell expression profiles
- `expression_stats.png`: Mean and variance comparison between real and generated
- `distribution_comparison.png`: Distribution analysis (overall, per-cell mean, sparsity, non-zero)

**Example Output:**

```
Generation Statistics
==================================================
Mean expression correlation: 0.9234
Std expression correlation: 0.8765

Real cells - Mean: 12.45, Std: 8.32
Generated cells - Mean: 12.38, Std: 8.41
```

## Checkpoint Auto-Discovery

Both inference scripts automatically find checkpoints in the experiment directory with the following priority:

1. `best.pt` - Best validation checkpoint
2. `final.pt` - Final training checkpoint
3. `epoch_N.pt` - Most recent epoch checkpoint

You can also specify a checkpoint explicitly:

```bash
python scripts/inference_imputation.py \
    --experiment_dir experiments/my_model \
    --checkpoint experiments/my_model/epoch_50.pt
```

## Model Parameters

The inference scripts automatically load model parameters from `args.json` saved during training. This ensures the model architecture matches exactly:

- `hidden_dim`: Transformer hidden dimension
- `num_layers`: Number of transformer layers
- `num_heads`: Number of attention heads
- `dropout`: Dropout rate

No need to specify these manually - they're loaded from the training configuration!

## Sampling Parameters

### Temperature

Controls randomness in sampling:
- `temperature=0.8`: More conservative, stays closer to training distribution
- `temperature=1.0`: (default) Balanced sampling
- `temperature=1.2`: More creative, higher diversity

### Number of Steps

Controls quality vs speed trade-off:
- `num_steps=50`: Fast inference (imputation default)
- `num_steps=100`: Balanced (generation default)
- `num_steps=200`: High quality, slower

### Mask Ratio (Imputation only)

Controls how much to mask:
- `mask_ratio=0.1`: Easy imputation task
- `mask_ratio=0.2`: (default) Moderate difficulty
- `mask_ratio=0.5`: Challenging imputation

## Monitoring Cluster Jobs

Check SLURM job status:

```bash
# View job queue
squeue -u $USER

# Watch output in real-time
tail -f slurm_out/sedd_imputation-*.out
tail -f slurm_out/sedd_generation-*.out

# Check completed job
sacct -j JOBID --format=JobID,JobName,State,ExitCode,Elapsed
```

## Tips and Best Practices

### For Imputation

1. **Start with low mask_ratio** (0.15-0.2) to see baseline performance
2. **Use more steps** (100+) for better accuracy on challenging masks
3. **Evaluate on held-out data** using `--use_train_split=False` (default)
4. **Check per-cell visualizations** to identify failure modes

### For Generation

1. **Generate many cells** (500-1000) for robust statistics
2. **Compare distributions** carefully - check sparsity and mean/std
3. **Use temperature tuning** if generated cells are too similar/different
4. **Verify with downstream tasks** (clustering, differential expression)

### Cluster Usage

1. **Imputation is faster** - typically 1-2 hours
2. **Generation takes longer** - allocate 3-4 hours for large batches
3. **Monitor GPU memory** - reduce batch_size if OOM errors occur
4. **Save generated cells** - use them for downstream analysis

## Troubleshooting

### Checkpoint not found

```
Error: No checkpoints found in experiments/my_model
```

**Solution**: Verify the experiment directory path and that training completed successfully.

### Out of memory

```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce batch size or number of cells to generate:

```bash
python scripts/inference_imputation.py \
    --experiment_dir experiments/my_model \
    --batch_size 16
```

### Poor imputation accuracy

Possible causes:
- Model undertrained - train for more epochs
- Mask ratio too high - try lower values
- Temperature too high - try 0.8-0.9
- Not enough sampling steps - increase to 100+

### Generated cells don't match real distribution

Try:
- Adjust temperature (0.8-1.2)
- Increase sampling steps
- Check if model was trained long enough
- Verify training data quality

## Example Workflow

Complete workflow from training to inference:

```bash
# 1. Set up configuration
cat > config.toml << EOF
[data]
train_data = "/path/to/data.h5ad"
val_fraction = 0.1
EOF

# 2. Train model
sbatch slurm/sb-ism.sbatch

# 3. Wait for training to complete, then run inference

# 4. Run imputation inference
sbatch slurm/sb-inference-imputation.sbatch \
    --experiment_dir experiments/rnaseq_diffusion_20240115

# 5. Run generation inference
sbatch slurm/sb-inference-generation.sbatch \
    --experiment_dir experiments/rnaseq_diffusion_20240115

# 6. Check results
ls experiments/rnaseq_diffusion_20240115/imputation_results/
ls experiments/rnaseq_diffusion_20240115/generation_results/
```

## Reference

The inference implementation is based on the methodology shown in `nbs/eval_sedd.ipynb`.

Key components:
- **Imputation**: `sedd.sampling.impute_masked`
- **Generation**: `sedd.sampling.EulerSampler`
- **Model**: `sedd.model.SEDDTransformerSmall`
- **Graph**: `sedd.graph.AbsorbingGraph`
- **Noise**: `sedd.noise.LogLinearNoise`
