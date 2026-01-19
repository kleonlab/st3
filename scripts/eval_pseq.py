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

output_dir = Path("/home/b5cc/sanjukta.b5cc/st3/experiments/psed_demo/perturbseq_dry_run/evaluation")
output_dir.mkdir(parents=True, exist_ok=True)



test_data_path = "/home/b5cc/sanjukta.b5cc/st3/datasets/dataset/k562_5k_test_split.h5ad"
predicted_data_path = "/home/b5cc/sanjukta.b5cc/st3/experiments/psed_demo/perturbseq_dry_run/inference_results/generated_cells.h5ad"

# Load test data
adata_test = sc.read_h5ad(test_data_path)
test_expr = adata_test.X
if hasattr(test_expr, 'toarray'):
    test_expr = test_expr.toarray()
test_expr = torch.from_numpy(test_expr).long()
test_labels = adata_test.obs['gene'].values

print(len(test_expr))
print(len(test_labels))

print(test_expr[40])
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
    unique_genes = np.intersect1d(test_labels, pred_labels)
    print(unique_genes)
    print(test_labels)
    print(pred_labels)
    per_perturbation_results = {}

    for gene in unique_genes:
        # Filter cells for this specific perturbation
        truth_mask = (test_labels == gene)
        pred_mask = (pred_labels == gene)
        
        # Get matrices (Cells x Genes)
        truth_cells = test_expr[truth_mask].float()
        pred_cells = pred_expr[pred_mask].float()
        
        if truth_cells.shape[0] == 0 or pred_cells.shape[0] == 0:
            continue

        # Compute mean profiles for each gene transcript across all cells of this perturbation
        mean_truth = truth_cells.mean(dim=0)
        mean_pred = pred_cells.mean(dim=0)
        
        # L1 Error (Absolute Difference) per transcript
        l1_per_transcript = torch.abs(mean_truth - mean_pred)
        
        per_perturbation_results[gene] = {
            'l1_mean': l1_per_transcript.mean().item(),
            'l1_per_transcript': l1_per_transcript.numpy()
        }
    
    return per_perturbation_results

# Execute computation
eval_results = compute_l1_errors(test_expr, test_labels, pred_expr, pred_labels)

# 1. Plotting Average L1 Error per Perturbation
perturbations = list(eval_results.keys())
avg_errors = [eval_results[p]['l1_mean'] for p in perturbations]

plt.figure(figsize=(12, 5))
plt.bar(perturbations, avg_errors, color='skyblue')
plt.xticks(rotation=90)
plt.ylabel("Mean L1 Error")
plt.title("Average L1 Error per Perturbation (Across all Transcripts)")
plt.tight_layout()
plt.savefig(output_dir / "l1_error_per_perturbation.png", dpi=300)
plt.show()

# 2. Plotting Transcript-wise Error for a specific Perturbation (e.g., the first one)
sample_pert = perturbations[0]
transcript_errors = eval_results[sample_pert]['l1_per_transcript']

plt.figure(figsize=(12, 5))
plt.plot(transcript_errors, alpha=0.7, color='coral')
plt.xlabel("Gene Transcript Index")
plt.ylabel("L1 Error")
plt.title(f"Transcript-wise L1 Error for Perturbation: {sample_pert}")
plt.tight_layout()
plt.show()

import seaborn as sns

# Create a matrix of (Perturbations x Transcripts) for the first 100 genes to keep it readable
error_matrix = np.array([eval_results[p]['l1_per_transcript'][:100] for p in perturbations])

plt.figure(figsize=(12, 8))
sns.heatmap(error_matrix, xticklabels=50, yticklabels=perturbations, cmap="YlOrRd")
plt.title("L1 Error Heatmap (First 100 Transcripts)")
plt.xlabel("Transcript Index")
plt.ylabel("Perturbation")

# Save the heatmap
plt.savefig(output_dir / "l1_error_heatmap.png", dpi=300)
plt.show()


# Prepare a JSON-serializable version of eval_results
serializable_results = {}
for gene, data in eval_results.items():
    serializable_results[gene] = {
        'l1_mean': float(data['l1_mean']),
        'l1_per_transcript': data['l1_per_transcript'].tolist() # Convert array to list
    }

# Save evaluation results
with open(output_dir / "evaluation_results.json", 'w') as f:
    json.dump(serializable_results, f, indent=2)

print(f"Results saved to {output_dir / 'evaluation_results.evals'}")