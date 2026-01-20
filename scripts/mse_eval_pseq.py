import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

print("\n" + "="*50)
print("Evaluating Gene-Gene Correlation MSE")
print("="*50)

import scanpy as sc
import numpy as np
import torch

output_dir = Path("/home/b5cc/sanjukta.b5cc/st3/experiments/5k_psed/evaluation")
output_dir.mkdir(parents=True, exist_ok=True)

test_data_path = "/home/b5cc/sanjukta.b5cc/st3/datasets/5k/k562_5k_test_split.h5ad"
predicted_data_path = "/home/b5cc/sanjukta.b5cc/st3/experiments/5k_psed/inference_results/generated_cells.h5ad"

# Load test data
adata_test = sc.read_h5ad(test_data_path)
test_expr = adata_test.X
if hasattr(test_expr, 'toarray'):
    test_expr = test_expr.toarray()
test_expr = torch.from_numpy(test_expr).float()
test_labels = adata_test.obs['gene'].values

print(f"Test data: {len(test_expr)} cells")
print(f"Test labels: {len(test_labels)} labels")

# Load predicted data
adata_pred = sc.read_h5ad(predicted_data_path)
pred_expr = adata_pred.X
if hasattr(pred_expr, 'toarray'):
    pred_expr = pred_expr.toarray()
pred_expr = torch.from_numpy(pred_expr).float()
pred_labels = adata_pred.obs['perturbation'].values

print(f"Predicted data: {len(pred_expr)} cells")
print(f"Predicted labels: {len(pred_labels)} labels")


def compute_gene_gene_correlation_mse(test_expr, test_labels, pred_expr, pred_labels):
    """
    Compute MSE of gene-gene correlation matrices between predicted and true cell states.

    For each perturbation:
    1. Compute gene-gene correlation matrix for real cells (Pearson correlation)
    2. Compute gene-gene correlation matrix for generated cells
    3. Calculate MSE between the two correlation matrices:
       MSE = (1/N_pairs) * sum((r_ij_real - r_ij_gen)^2)

    Where:
    - r_ij is the Pearson correlation coefficient between gene i and gene j
    - N_pairs is the number of gene pairs (including diagonal)

    Args:
        test_expr: tensor of shape [n_test_cells, n_genes] - real cell expressions
        test_labels: array of perturbation labels for test cells
        pred_expr: tensor of shape [n_pred_cells, n_genes] - generated cell expressions
        pred_labels: array of perturbation labels for predicted cells

    Returns:
        per_perturbation_results: dict with MSE and correlation info per perturbation
    """
    unique_genes = np.intersect1d(test_labels, pred_labels)
    print(f"\nEvaluating {len(unique_genes)} perturbations")

    per_perturbation_results = {}

    for gene in unique_genes:
        # Filter cells for this specific perturbation
        truth_mask = (test_labels == gene)
        pred_mask = (pred_labels == gene)

        # Get matrices (Cells x Genes)
        truth_cells = test_expr[truth_mask].float()  # [n_truth_cells, n_genes]
        pred_cells = pred_expr[pred_mask].float()    # [n_pred_cells, n_genes]

        if truth_cells.shape[0] == 0 or pred_cells.shape[0] == 0:
            print(f"Skipping {gene}: no cells found")
            continue

        # Need at least 2 cells to compute correlation
        if truth_cells.shape[0] < 2 or pred_cells.shape[0] < 2:
            print(f"Skipping {gene}: insufficient cells (truth={truth_cells.shape[0]}, pred={pred_cells.shape[0]})")
            continue

        # Convert to numpy for correlation computation
        truth_cells_np = truth_cells.numpy()  # [n_truth_cells, n_genes]
        pred_cells_np = pred_cells.numpy()    # [n_pred_cells, n_genes]

        # Compute gene-gene correlation matrices
        # np.corrcoef expects genes as rows, so we transpose
        # Result is [n_genes, n_genes] correlation matrix
        try:
            corr_real = np.corrcoef(truth_cells_np.T)  # [n_genes, n_genes]
            corr_gen = np.corrcoef(pred_cells_np.T)    # [n_genes, n_genes]

            # Handle NaN values that might arise from constant gene expression
            # Replace NaN with 0 (no correlation)
            corr_real = np.nan_to_num(corr_real, nan=0.0)
            corr_gen = np.nan_to_num(corr_gen, nan=0.0)

            # Compute MSE between correlation matrices
            # MSE = (1/N_pairs) * sum((r_ij_real - r_ij_gen)^2)
            diff = corr_real - corr_gen
            squared_diff = diff ** 2
            mse = np.mean(squared_diff)  # Average over all gene pairs

            # Additional statistics
            n_genes = corr_real.shape[0]
            n_pairs = n_genes * n_genes  # Total number of gene pairs (including diagonal)

            # Statistics about the differences
            abs_diff = np.abs(diff)
            max_diff = np.max(abs_diff)
            mean_abs_diff = np.mean(abs_diff)

            per_perturbation_results[gene] = {
                'mse': float(mse),
                'n_genes': int(n_genes),
                'n_pairs': int(n_pairs),
                'max_abs_diff': float(max_diff),
                'mean_abs_diff': float(mean_abs_diff),
                'n_truth_cells': int(truth_cells.shape[0]),
                'n_pred_cells': int(pred_cells.shape[0]),
            }

            print(f"{gene}: MSE={mse:.6f}, n_genes={n_genes}, n_pairs={n_pairs}")

        except Exception as e:
            print(f"Error computing correlation for {gene}: {e}")
            continue

    return per_perturbation_results


# Execute computation
print("\nComputing gene-gene correlation MSE...")
eval_results = compute_gene_gene_correlation_mse(test_expr, test_labels, pred_expr, pred_labels)

# Save evaluation results
with open(output_dir / "mse_gene_correlation.json", 'w') as f:
    json.dump(eval_results, f, indent=2)

print(f"\n" + "="*50)
print(f"Results saved to {output_dir / 'mse_gene_correlation.json'}")

# Print summary statistics
if eval_results:
    mse_values = [result['mse'] for result in eval_results.values()]
    print(f"\nSummary Statistics:")
    print(f"  Number of perturbations evaluated: {len(mse_values)}")
    print(f"  Mean MSE: {np.mean(mse_values):.6f}")
    print(f"  Median MSE: {np.median(mse_values):.6f}")
    print(f"  Min MSE: {np.min(mse_values):.6f}")
    print(f"  Max MSE: {np.max(mse_values):.6f}")
    print(f"  Std MSE: {np.std(mse_values):.6f}")
else:
    print("\nNo results to summarize")

print("="*50)
