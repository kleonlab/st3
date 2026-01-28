# Inference with Cell Type Conditioning

## Overview

The inference script now supports generating cell states conditioned on both perturbation and cell type. This allows you to specify which cell type context you want to generate for.

## Usage

### Command Line

Run inference with cell type conditioning:

```bash
python scripts/inference_conditional.py \
    --experiment_dir experiments/perturbseq_diffusion \
    --perturbations_file datasets/20M/test_perts.txt \
    --perturbations_all_file datasets/20M/all_perts.txt \
    --cell_type hepg2 \
    --num_samples_per_pert 100 \
    --num_steps 100 \
    --temperature 1.0
```

### Using Config File

Add cell_type to your config YAML:

```yaml
inference:
  perturbations_file: datasets/20M/test_perts.txt
  perturbations_all_file: datasets/20M/all_perts.txt
  num_samples_per_pert: 100
  num_steps: 100
  temperature: 1.0
  cell_type: hepg2  # Specify cell type here
```

Then run:

```bash
python scripts/inference_conditional.py \
    --config configs/inference_config.yaml \
    --experiment_dir experiments/perturbseq_diffusion
```

## Parameters

### Required
- `--experiment_dir`: Path to trained model directory
- `--perturbations_file`: File with perturbations to generate (one per line)
- `--perturbations_all_file`: File with all perturbations from training

### Optional
- `--cell_type`: Cell type to condition on (e.g., 'hepg2', 'jurkat', 'rpe1')
- `--num_samples_per_pert`: Number of samples per perturbation (default: 10)
- `--num_steps`: Sampling steps (default: 50)
- `--temperature`: Sampling temperature (default: 1.0)

## Cell Type Validation

The script will:
1. Load available cell types from the trained model's `args.json`
2. Validate that your specified cell type exists
3. Convert cell type name to index automatically
4. Generate samples conditioned on both perturbation + cell type

### Example Output

```
Model was trained with cell type conditioning: 3 cell types
Available cell types: ['hepg2', 'jurkat', 'rpe1']

Generating with cell type conditioning: 'hepg2' (index: 0)
```

## Output Files

Generated files will include cell type information:

### `generated_cells.h5ad`
AnnData object with:
- `X`: Generated expression profiles
- `obs['perturbation']`: Perturbation names
- `obs['cell_type']`: Cell type (if specified)
- `obs['cell_type_idx']`: Cell type index (if specified)

### `generation_metadata.json`
Includes:
```json
{
  "num_cells": 5000,
  "num_perturbations": 50,
  "cell_type": "hepg2",
  "cell_type_idx": 0,
  ...
}
```

## Example: Generate for Test Perturbations with hepg2

Your test file has 50 perturbations. To generate 100 samples each for hepg2:

```bash
python scripts/inference_conditional.py \
    --experiment_dir experiments/perturbseq_diffusion \
    --perturbations_file datasets/20M/test_perts.txt \
    --perturbations_all_file datasets/20M/all_perts.txt \
    --cell_type hepg2 \
    --num_samples_per_pert 100 \
    --num_steps 100
```

This will generate:
- 50 perturbations × 100 samples = 5,000 cells
- All conditioned on perturbation + hepg2 cell type

## Backward Compatibility

If you don't specify `--cell_type`:
- Models trained WITH cell type conditioning will generate without it (warning shown)
- Models trained WITHOUT cell type conditioning work as before

## Error Handling

### Cell type not found
```
ValueError: Cell type 'invalid' not found in training data.
Available cell types: ['hepg2', 'jurkat', 'rpe1']
```

### Model without cell type support
```
WARNING: --cell_type 'hepg2' specified but model was not trained with cell type conditioning
Ignoring cell type argument
```

## Multi-Cell Type Generation

To generate for multiple cell types, run the script multiple times:

```bash
# Generate for hepg2
python scripts/inference_conditional.py \
    --experiment_dir experiments/perturbseq_diffusion \
    --perturbations_file datasets/20M/test_perts.txt \
    --cell_type hepg2 \
    --num_samples_per_pert 100

# Generate for jurkat
python scripts/inference_conditional.py \
    --experiment_dir experiments/perturbseq_diffusion \
    --perturbations_file datasets/20M/test_perts.txt \
    --cell_type jurkat \
    --num_samples_per_pert 100

# Generate for rpe1
python scripts/inference_conditional.py \
    --experiment_dir experiments/perturbseq_diffusion \
    --perturbations_file datasets/20M/test_perts.txt \
    --cell_type rpe1 \
    --num_samples_per_pert 100
```

Each run will create separate output directories or you can specify different output paths.

## Integration with Evaluation

The generated AnnData will contain cell_type information, allowing you to:
1. Compare generated vs real cells within the same cell type
2. Analyze perturbation effects across cell types
3. Evaluate cell-type-specific perturbation predictions

