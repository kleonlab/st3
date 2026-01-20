import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

print("\n" + "="*50)
print("Evaluating against ground truth")
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
test_expr = torch.from_numpy(test_expr).long()
test_labels = adata_test.obs['gene'].values

print(len(test_expr))
print(len(test_labels))

print(test_expr[20])
print(test_labels[1])

adata_pred = sc.read_h5ad(predicted_data_path)
pred_expr = adata_pred.X
if hasattr(pred_expr, 'toarray'):
    pred_expr = pred_expr.toarray()
pred_expr = torch.from_numpy(pred_expr).long()
pred_labels = adata_pred.obs['perturbation'].values

print(len(pred_expr))
print(len(pred_labels))

print(pred_expr[40])
print(pred_labels[1])

# Compute per-perturbation metrics
eval_results = {}

#unique errors 
def compute_l1_errors(test_expr, test_labels, pred_expr, pred_labels):
    """
    Compute L1 errors between predicted and true cell states.
    
    For each perturbation, computes pairwise L1 distances between 
    individual predicted cells and individual true cells, then averages.
    
    Returns:
        per_perturbation_results: dict with L1 error per perturbation
    """
    unique_genes = np.intersect1d(test_labels, pred_labels)
    print(f"Evaluating {len(unique_genes)} perturbations")
    
    per_perturbation_results = {}

    for gene in unique_genes:
        # Filter cells for this specific perturbation
        truth_mask = (test_labels == gene)
        pred_mask = (pred_labels == gene)
        
        # Get matrices (Cells x Genes)
        truth_cells = test_expr[truth_mask].float()  # [n_truth_cells, n_genes]
        pred_cells = pred_expr[pred_mask].float()    # [n_pred_cells, n_genes]
        
        if truth_cells.shape[0] == 0 or pred_cells.shape[0] == 0:
            continue

        # Compute pairwise L1 distances between all predicted and true cells
        # For each predicted cell, compute L1 to all true cells, then average
        n_pred = pred_cells.shape[0]
        n_truth = truth_cells.shape[0]
        
        # Method 1: Compute all pairwise distances and average
        # Shape: [n_pred, n_truth, n_genes]
        pairwise_diff = pred_cells.unsqueeze(1) - truth_cells.unsqueeze(0)
        print(pairwise_diff)
        pairwise_l1 = torch.abs(pairwise_diff).sum(dim=2)/(n_pred*n_truth)  # [n_pred, n_truth]
        print(pairwise_l1)
        
        # Average L1 distance across all pairs
        mean_l1 = pairwise_l1.mean().item()
        print(mean_l1)
        
        # Also store per-cell statistics
        # Average L1 for each predicted cell (averaged over all true cells)
        mean_l1_per_pred = pairwise_l1.mean(dim=1)  # [n_pred]
        
        # Average L1 for each true cell (averaged over all predicted cells)
        mean_l1_per_truth = pairwise_l1.mean(dim=0)  # [n_truth]
        
        per_perturbation_results[gene] = {
            'l1_mean': mean_l1,  # Overall average L1 across all pairs
            'l1_std': pairwise_l1.std().item(),  # Std of pairwise L1s
            'l1_per_pred_cell': mean_l1_per_pred.numpy(),  # [n_pred]
            'l1_per_truth_cell': mean_l1_per_truth.numpy(),  # [n_truth]
            'n_pred_cells': n_pred,
            'n_truth_cells': n_truth,
        }
    
    return per_perturbation_results

# Execute computation
eval_results = compute_l1_errors(test_expr, test_labels, pred_expr, pred_labels)

# Prepare a JSON-serializable version of eval_results
serializable_results = {}
for gene, data in eval_results.items():
    serializable_results[gene] = {
        'l1_mean': float(data['l1_mean']),
        'l1_std': float(data['l1_std']),
        'l1_per_pred_cell': data['l1_per_pred_cell'].tolist(),  # Convert array to list
        'l1_per_truth_cell': data['l1_per_truth_cell'].tolist(),  # Convert array to list
        'n_pred_cells': int(data['n_pred_cells']),
        'n_truth_cells': int(data['n_truth_cells']),
    }
# Save evaluation results
with open(output_dir / "l1_error.json", 'w') as f:
    json.dump(serializable_results, f, indent=2)

print(f"Results saved to {output_dir / 'l1_error.evals'}")