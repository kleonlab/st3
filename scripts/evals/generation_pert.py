import json
from pathlib import Path
from typing import Optional
import pandas as pd

import numpy as np
import scanpy as sc
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist
import warnings

from generation_pseq import evaluate_generation


# [Keep all your existing metric functions: wasserstein2_distance, mmd2_rbf, frechet_distance, evaluate_generation]


def evaluate_generation_by_perturbation(
    adata_test: sc.AnnData,
    adata_pred: sc.AnnData,
    perturbation_col: str = 'perturbation',
    w2_projections: int = 1000,
    mmd_subsample: Optional[int] = 2000,
    seed: Optional[int] = 42,
    verbose: bool = True
) -> dict:
    """
    Evaluate generation metrics separately for each perturbation.

    Args:
        adata_test: Test data AnnData object with perturbation labels
        adata_pred: Predicted data AnnData object with perturbation labels
        perturbation_col: Column name in .obs containing perturbation labels
        w2_projections: Number of projections for sliced Wasserstein distance
        mmd_subsample: Maximum samples for MMD calculation (None for all)
        seed: Random seed for reproducibility
        verbose: If True, print results

    Returns:
        Dictionary with:
        - 'per_perturbation': Dict mapping perturbation -> metrics
        - 'aggregate': Mean metrics across all perturbations
        - 'overall': Global metrics (all cells combined)
    """
    # Check that perturbation column exists
    if perturbation_col not in adata_test.obs.columns:
        raise ValueError(f"Column '{perturbation_col}' not found in test data")
    if perturbation_col not in adata_pred.obs.columns:
        raise ValueError(f"Column '{perturbation_col}' not found in predicted data")

    # Get expression matrices
    test_expr = adata_test.X
    if hasattr(test_expr, "toarray"):
        test_expr = test_expr.toarray()
    
    pred_expr = adata_pred.X
    if hasattr(pred_expr, "toarray"):
        pred_expr = pred_expr.toarray()

    # Get perturbation labels
    test_perts = adata_test.obs[perturbation_col].values
    pred_perts = adata_pred.obs[perturbation_col].values

    # Find common perturbations
    common_perts = set(test_perts) & set(pred_perts)
    
    if len(common_perts) == 0:
        raise ValueError("No common perturbations found between test and predicted data")

    if verbose:
        print(f"Found {len(common_perts)} common perturbations")
        print(f"Test perturbations: {len(set(test_perts))}")
        print(f"Predicted perturbations: {len(set(pred_perts))}")
        print()

    # Calculate metrics per perturbation
    per_pert_results = {}
    
    for pert in sorted(common_perts):
        if verbose:
            print(f"\nEvaluating perturbation: {pert}")
        
        # Get cells for this perturbation
        test_mask = test_perts == pert
        pred_mask = pred_perts == pert
        
        test_pert_expr = test_expr[test_mask]
        pred_pert_expr = pred_expr[pred_mask]
        
        if verbose:
            print(f"  Test cells: {len(test_pert_expr)}")
            print(f"  Predicted cells: {len(pred_pert_expr)}")
        
        # Skip if too few cells
        if len(test_pert_expr) < 10 or len(pred_pert_expr) < 10:
            if verbose:
                print(f"  ⚠️  Skipping (too few cells)")
            continue
        
        # Calculate metrics for this perturbation
        try:
            pert_metrics = evaluate_generation(
                test_pert_expr,
                pred_pert_expr,
                w2_projections=w2_projections,
                mmd_subsample=mmd_subsample,
                seed=seed,
                verbose=False
            )
            per_pert_results[pert] = pert_metrics
            
            if verbose:
                print(f"  W2:    {pert_metrics['w2']:.6f}")
                print(f"  MMD2:  {pert_metrics['mmd2_rbf']:.6f}")
                print(f"  FD:    {pert_metrics['fd']:.6f}")
        except Exception as e:
            if verbose:
                print(f"  ⚠️  Error: {e}")
            continue

    # Calculate aggregate metrics (mean across perturbations)
    if per_pert_results:
        aggregate = {
            'w2': np.mean([m['w2'] for m in per_pert_results.values()]),
            'mmd2_rbf': np.mean([m['mmd2_rbf'] for m in per_pert_results.values()]),
            'fd': np.mean([m['fd'] for m in per_pert_results.values()]),
        }
        
        # Also calculate median
        aggregate_median = {
            'w2_median': np.median([m['w2'] for m in per_pert_results.values()]),
            'mmd2_rbf_median': np.median([m['mmd2_rbf'] for m in per_pert_results.values()]),
            'fd_median': np.median([m['fd'] for m in per_pert_results.values()]),
        }
        aggregate.update(aggregate_median)
    else:
        aggregate = None

    # Calculate overall metrics (all cells combined)
    if verbose:
        print("\n" + "="*50)
        print("Computing OVERALL metrics (all cells combined)...")
    
    overall = evaluate_generation(
        test_expr,
        pred_expr,
        w2_projections=w2_projections,
        mmd_subsample=mmd_subsample,
        seed=seed,
        verbose=verbose
    )

    # Print summary
    if verbose and aggregate:
        print("\n" + "="*50)
        print("AGGREGATE metrics (mean across perturbations):")
        print(f"  W2 (mean):    {aggregate['w2']:.6f} (median: {aggregate['w2_median']:.6f})")
        print(f"  MMD2 (mean):  {aggregate['mmd2_rbf']:.6f} (median: {aggregate['mmd2_rbf_median']:.6f})")
        print(f"  FD (mean):    {aggregate['fd']:.6f} (median: {aggregate['fd_median']:.6f})")

    results = {
        'per_perturbation': per_pert_results,
        'aggregate': aggregate,
        'overall': overall,
        'num_perturbations_evaluated': len(per_pert_results)
    }

    return results


# ========== MAIN EVALUATION ==========

print("\n" + "=" * 50)
print("Evaluating Generation Metrics")
print("=" * 50)

output_dir = Path("/home/b5cc/sanjukta.b5cc/st3/experiments/30k2/evaluation")
output_dir.mkdir(parents=True, exist_ok=True)

test_data_path = "/home/b5cc/sanjukta.b5cc/st3/datasets/30k/k562_test_split.h5ad"
predicted_data_path = "/home/b5cc/sanjukta.b5cc/st3/experiments/30k/inference_results/generated_cells_5000.h5ad"

# Load data as AnnData objects (not just expression)
adata_test = sc.read_h5ad(test_data_path)
adata_pred = sc.read_h5ad(predicted_data_path)

print(f"Test data: {len(adata_test)} cells")
print(f"Predicted data: {len(adata_pred)} cells")

# Evaluate per-perturbation metrics
print("\nComputing per-perturbation metrics...")
results = evaluate_generation_by_perturbation(
    adata_test,
    adata_pred,
    perturbation_col='gene',  # Change this to match your column name
    verbose=True
)

# Save results
output_file = output_dir / "generation_metrics_detailed.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

# Also save as CSV for easy viewing
if results['per_perturbation']:
    df = pd.DataFrame(results['per_perturbation']).T
    df.index.name = 'perturbation'
    df.to_csv(output_dir / "generation_metrics_per_perturbation.csv")
    print(f"\nPer-perturbation results saved to {output_dir / 'generation_metrics_per_perturbation.csv'}")

print("\n" + "=" * 50)
print(f"Detailed results saved to {output_file}")
print("=" * 50)