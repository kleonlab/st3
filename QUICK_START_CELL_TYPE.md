# Quick Start: Cell Type Conditioning

## What Changed?

Your model now supports conditioning on **both perturbation and cell type** during training and inference.

## 🎯 Quick Commands

### 1. Train with Cell Type Conditioning

Your training script automatically detects and uses cell_type from your data:

```bash
python scripts/train_perturbseq.py \
    --config configs/perturbseq_small.yaml \
    --train_data_path datasets/20M/train.h5ad
```

**Requirements:**
- Your data must have `cell_type` column in `adata.obs`
- Dataloader must return `cell_type` in batch dictionary

### 2. Inference for Specific Cell Type

Generate perturbed cells for hepg2:

```bash
python scripts/inference_conditional.py \
    --experiment_dir experiments/perturbseq_diffusion \
    --perturbations_file datasets/20M/test_perts.txt \
    --perturbations_all_file datasets/20M/all_perts.txt \
    --cell_type hepg2 \
    --num_samples_per_pert 100 \
    --num_steps 100
```

### 3. Using Bash Script

```bash
EXPERIMENT_DIR=experiments/perturbseq_diffusion \
PERTURBATIONS_FILE=datasets/20M/test_perts.txt \
CELL_TYPE=hepg2 \
NUM_SAMPLES_PER_PERT=100 \
bash/inf_conditional.sh
```

## 📊 Your Data

Based on your test data:

- **Available cell types**: hepg2, jurkat, rpe1
- **Test perturbations**: 50 genes (in `datasets/20M/test_perts.txt`)
- **Training data**: 650,686 cells across 3 cell types

## 🚀 Example Workflows

### Generate for Test Set (hepg2)

```bash
# Generate 100 samples per perturbation for all 50 test genes
python scripts/inference_conditional.py \
    --experiment_dir experiments/perturbseq_diffusion \
    --perturbations_file datasets/20M/test_perts.txt \
    --perturbations_all_file datasets/20M/all_perts.txt \
    --cell_type hepg2 \
    --num_samples_per_pert 100 \
    --num_steps 100

# Output: 50 perturbations × 100 samples = 5,000 cells
# Location: experiments/perturbseq_diffusion/inference_results/
```

### Generate for All Cell Types

```bash
for CELL_TYPE in hepg2 jurkat rpe1; do
    python scripts/inference_conditional.py \
        --experiment_dir experiments/perturbseq_diffusion \
        --perturbations_file datasets/20M/test_perts.txt \
        --perturbations_all_file datasets/20M/all_perts.txt \
        --cell_type ${CELL_TYPE} \
        --num_samples_per_pert 100 \
        --num_steps 100
    
    # Move results to cell-type-specific directory
    mv experiments/perturbseq_diffusion/inference_results \
       experiments/perturbseq_diffusion/inference_results_${CELL_TYPE}
done
```

## 📁 Output Files

After running inference, you'll get:

```
experiments/perturbseq_diffusion/inference_results/
├── generated_cells.h5ad          # AnnData with generated cells
├── generated_cells.npy           # Raw numpy array
├── generation_metadata.json       # Includes cell_type info
└── generation_summary.png         # Visualizations
```

### `generated_cells.h5ad` contents:

```python
import scanpy as sc

adata = sc.read_h5ad('generated_cells.h5ad')

# adata.obs contains:
# - 'perturbation': gene names
# - 'cell_type': 'hepg2' (or specified cell type)
# - 'cell_type_idx': 0 (index)
# - 'sample_idx': 0-99 (sample number)
```

## 🔍 Validation

Check if your model supports cell type:

```bash
# Look at training metadata
cat experiments/perturbseq_diffusion/args.json | grep -A 5 cell_type

# Should show:
# "num_cell_types": 3,
# "cell_type_to_idx": {
#   "hepg2": 0,
#   "jurkat": 1,
#   "rpe1": 2
# }
```

## ⚠️ Common Issues

### "Cell type not found"

```
ValueError: Cell type 'hepg2' not found in training data.
```

**Fix**: Check available cell types in `args.json` or training logs.

### "Model not trained with cell type"

```
WARNING: Model was not trained with cell type conditioning
```

**Fix**: Retrain model with data that has `cell_type` column.

### Wrong cell type predictions

**Fix**: Ensure cell_type was properly passed during training. Check:
1. `adata.obs` has `cell_type` column
2. Dataloader returns `cell_type` in batch
3. Training logs show "Cell type conditioning enabled"

## 📖 More Information

- **Training details**: See `CELL_TYPE_CONDITIONING.md`
- **Inference details**: See `INFERENCE_WITH_CELL_TYPE.md`
- **Example script**: Run `EXAMPLE_INFERENCE_CELL_TYPE.sh`

## 💡 Tips

1. **Start small**: Test with `--num_samples_per_pert 10` first
2. **Reduce steps for speed**: Use `--num_steps 50` for faster testing
3. **Check logs**: Training logs will show if cell type conditioning is active
4. **Validate outputs**: Generated cells should have cell_type in metadata

## Next Steps

1. ✅ Train model with cell type data
2. ✅ Verify cell type info in `args.json`
3. ✅ Run inference with `--cell_type` parameter
4. ✅ Compare generated vs real cells per cell type
5. ✅ Analyze cell-type-specific perturbation effects

