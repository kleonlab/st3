# scGPT Model Files

This directory should contain the following pre-trained scGPT model files:

## Required Files

1. **args.json** - Model configuration parameters
2. **best_model.pt** or **best_model.ckpt** - Trained model weights
3. **vocab.json** - Gene vocabulary mapping

## Optional Files

4. **var_dims.pkl** - Variable dimensions
5. **pert_one-hot-map.pt** - Perturbation one-hot mapping

## Usage

Once these files are placed in this directory, you can run scGPT imputation using:

### Python Script
```bash
bash/inf_scgpt_imputation.sh
```

### Jupyter Notebook
```bash
jupyter notebook nbs/scgpt_imputation.ipynb
```

## File Sources

These files should come from a pre-trained scGPT model checkpoint. Download them from your scGPT training run or from a provided model checkpoint.
