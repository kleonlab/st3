import json
from pathlib import Path
from typing import Optional
import pandas as pd

import numpy as np
import scanpy as sc
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist
import warnings

import numpy as np
import scanpy as sc
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist
import warnings


def wasserstein2_distance(
    real: np.ndarray,
    generated: np.ndarray,
    num_projections: int = 1000,
    seed: Optional[int] = None
) -> float:
    """
    Calculate Wasserstein-2 (W2) distance between real and generated distributions.

    Uses sliced Wasserstein distance for computational efficiency with high-dimensional data.
    The sliced version projects data onto random 1D lines and computes the average
    Wasserstein distance across projections.

    Args:
        real: Real gene expression matrix (n_real_cells, n_genes)
        generated: Generated gene expression matrix (n_generated_cells, n_genes)
        num_projections: Number of random projections for sliced Wasserstein
        seed: Random seed for reproducibility

    Returns:
        W2 distance (lower is better)

    Lower values indicate generated samples are more similar to real data.
    """
    if real.shape[1] != generated.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: real {real.shape[1]} vs generated {generated.shape[1]}"
        )

    if seed is not None:
        np.random.seed(seed)

    n_features = real.shape[1]
    distances = []

    # Compute sliced Wasserstein distance
    for _ in range(num_projections):
        # Random projection direction
        direction = np.random.randn(n_features)
        direction = direction / np.linalg.norm(direction)

        # Project data onto this direction
        real_proj = real @ direction
        gen_proj = generated @ direction

        # Sort projections
        real_sorted = np.sort(real_proj)
        gen_sorted = np.sort(gen_proj)

        # Align lengths by resampling if needed
        if len(real_sorted) != len(gen_sorted):
            min_len = min(len(real_sorted), len(gen_sorted))
            real_sorted = np.interp(
                np.linspace(0, len(real_sorted) - 1, min_len),
                np.arange(len(real_sorted)),
                real_sorted
            )
            gen_sorted = np.interp(
                np.linspace(0, len(gen_sorted) - 1, min_len),
                np.arange(len(gen_sorted)),
                gen_sorted
            )

        # Wasserstein-2 distance for 1D distributions
        w2 = np.sqrt(np.mean((real_sorted - gen_sorted) ** 2))
        distances.append(w2)

    return float(np.mean(distances))


def mmd2_rbf(
    real: np.ndarray,
    generated: np.ndarray,
    gamma: Optional[float] = None,
    subsample: Optional[int] = None,
    seed: Optional[int] = None
) -> float:
    """
    Calculate Maximum Mean Discrepancy with RBF (Gaussian) kernel.

    MMD2 measures the distance between two distributions by comparing their mean
    embeddings in a reproducing kernel Hilbert space (RKHS).

    Args:
        real: Real gene expression matrix (n_real_cells, n_genes)
        generated: Generated gene expression matrix (n_generated_cells, n_genes)
        gamma: RBF kernel bandwidth parameter. If None, uses median heuristic
        subsample: Maximum number of samples to use (for computational efficiency)
        seed: Random seed for subsampling

    Returns:
        MMD2 value (lower is better)

    Lower values indicate better match between distributions.
    """
    if real.shape[1] != generated.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: real {real.shape[1]} vs generated {generated.shape[1]}"
        )

    if seed is not None:
        np.random.seed(seed)

    # Subsample for computational efficiency
    if subsample is not None:
        if len(real) > subsample:
            idx = np.random.choice(len(real), subsample, replace=False)
            real = real[idx]
        if len(generated) > subsample:
            idx = np.random.choice(len(generated), subsample, replace=False)
            generated = generated[idx]

    # Auto-select gamma using median heuristic if not provided
    if gamma is None:
        # Sample a subset for gamma estimation
        sample_size = min(100, len(real), len(generated))
        sample_real = real[np.random.choice(len(real), sample_size, replace=False)]
        sample_gen = generated[np.random.choice(len(generated), sample_size, replace=False)]
        combined = np.vstack([sample_real, sample_gen])

        # Compute pairwise distances
        pairwise_dists = cdist(combined, combined, metric='euclidean')
        median_dist = np.median(pairwise_dists[pairwise_dists > 0])
        gamma = 1.0 / (2 * median_dist ** 2) if median_dist > 0 else 1.0

    def rbf_kernel(X, Y, gamma):
        """RBF (Gaussian) kernel: k(x,y) = exp(-gamma * ||x-y||^2)"""
        pairwise_dists = cdist(X, Y, metric='sqeuclidean')
        return np.exp(-gamma * pairwise_dists)

    # Compute kernel matrices
    K_real_real = rbf_kernel(real, real, gamma)
    K_gen_gen = rbf_kernel(generated, generated, gamma)
    K_real_gen = rbf_kernel(real, generated, gamma)

    # MMD^2 = E[k(x,x')] - 2*E[k(x,y)] + E[k(y,y')]
    # Using unbiased estimator (excluding diagonal for same-distribution terms)
    n_real = len(real)
    n_gen = len(generated)

    # Remove diagonal for unbiased estimator
    np.fill_diagonal(K_real_real, 0)
    np.fill_diagonal(K_gen_gen, 0)

    term1 = (K_real_real.sum() / (n_real * (n_real - 1))) if n_real > 1 else 0
    term2 = K_real_gen.mean()
    term3 = (K_gen_gen.sum() / (n_gen * (n_gen - 1))) if n_gen > 1 else 0

    mmd2 = term1 - 2 * term2 + term3

    return float(max(0, mmd2))  # MMD^2 should be non-negative

import numpy as np
import warnings
from scipy.linalg import sqrtm
from typing import Optional

def frechet_distance(
    real: np.ndarray,
    generated: np.ndarray,
    eps: float = 1e-6,
    use_pca: bool = True,
    n_components: Optional[int] = None
) -> float:
    """
    Calculate Fréchet Distance (FD) between real and generated distributions.

    Also known as Fréchet Inception Distance (FID) in image generation literature.
    FD measures the distance between two Gaussian distributions by comparing
    both their means and covariances.

    Formula: FD = ||mu_real - mu_gen||^2 + Tr(Sigma_real + Sigma_gen - 2*sqrt(Sigma_real @ Sigma_gen))

    Args:
        real: Real gene expression matrix (n_real_cells, n_genes)
        generated: Generated gene expression matrix (n_generated_cells, n_genes)
        eps: Small constant for numerical stability (default: 1e-6)
        use_pca: Whether to apply PCA dimensionality reduction (default: True)
        n_components: Number of PCA components. If None, uses min(50, n_samples-1, n_features)

    Returns:
        Fréchet distance (lower is better)

    Lower values indicate generated samples better match statistical properties of real data.
    """
    if real.shape[1] != generated.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: real {real.shape[1]} vs generated {generated.shape[1]}"
        )
    
    n_real, n_features = real.shape
    n_gen = generated.shape[0]
    
    # Apply PCA if requested or if undersampled
    if use_pca or n_real < n_features or n_gen < n_features:
        from sklearn.decomposition import PCA
        
        if n_real < n_features or n_gen < n_features:
            warnings.warn(
                f"Small sample size relative to features: "
                f"real={n_real}, gen={n_gen}, features={n_features}. "
                f"Applying PCA for numerical stability.",
                UserWarning
            )
        
        # Determine number of components
        if n_components is None:
            n_components = min(50, n_real - 1, n_gen - 1, n_features)
        else:
            n_components = min(n_components, n_real - 1, n_gen - 1, n_features)
        
        # Fit PCA on real data and transform both datasets
        pca = PCA(n_components=n_components)
        real = pca.fit_transform(real)
        generated = pca.transform(generated)
        
        n_features = real.shape[1]  # Update feature count

    # Calculate means
    mu_real = np.mean(real, axis=0)
    mu_gen = np.mean(generated, axis=0)

    # Calculate covariances with bias correction
    sigma_real = np.cov(real, rowvar=False, bias=False)
    sigma_gen = np.cov(generated, rowvar=False, bias=False)

    # Ensure covariances are 2D
    if sigma_real.ndim == 0:
        sigma_real = sigma_real.reshape(1, 1)
    if sigma_gen.ndim == 0:
        sigma_gen = sigma_gen.reshape(1, 1)
    
    # Add regularization to diagonal for numerical stability
    # This prevents singular matrices with small sample sizes
    sigma_real = sigma_real + np.eye(sigma_real.shape[0]) * eps
    sigma_gen = sigma_gen + np.eye(sigma_gen.shape[0]) * eps

    # Calculate mean difference
    diff = mu_real - mu_gen
    mean_dist = np.dot(diff, diff)

    # Calculate sqrt of product of covariances
    # Use symmetric square root for better numerical stability
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # Method 1: Direct square root
            covmean = sqrtm(sigma_real @ sigma_gen)
            
            # Handle complex results from numerical errors
            if np.iscomplexobj(covmean):
                if not np.allclose(covmean.imag, 0, atol=1e-3):
                    warnings.warn(
                        "Complex component in covariance sqrt is significant. "
                        "Results may be unreliable.",
                        UserWarning
                    )
                covmean = covmean.real
                
        except Exception as e:
            # Fallback: use eigenvalue decomposition (more stable)
            warnings.warn(
                f"sqrtm failed ({e}), using eigenvalue decomposition",
                UserWarning
            )
            product = sigma_real @ sigma_gen
            eigvals, eigvecs = np.linalg.eigh(product)
            eigvals = np.maximum(eigvals, 0)  # Ensure non-negative
            covmean = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    # Calculate trace
    covmean_trace = np.trace(covmean)

    # Calculate FD
    fd = mean_dist + np.trace(sigma_real) + np.trace(sigma_gen) - 2 * covmean_trace

    # Ensure non-negative (numerical errors can cause small negative values)
    fd = float(max(0, fd))
    
    return fd

# [Keep all your existing metric functions: wasserstein2_distance, mmd2_rbf, frechet_distance, evaluate_generation]
#

def evaluate_generation(
    real: np.ndarray,
    generated: np.ndarray,
    w2_projections: int = 1000,
    mmd_subsample: Optional[int] = 2000,
    seed: Optional[int] = 42,
    verbose: bool = True
) -> dict:
    """
    Evaluate all generation metrics at once.

    Args:
        real: Real gene expression matrix (n_real_cells, n_genes)
        generated: Generated gene expression matrix (n_generated_cells, n_genes)
        w2_projections: Number of projections for sliced Wasserstein distance
        mmd_subsample: Maximum samples for MMD calculation (None for all)
        seed: Random seed for reproducibility
        verbose: If True, print results

    Returns:
        Dictionary containing all generation metrics:
        - 'w2': Wasserstein-2 Distance (lower is better)
        - 'mmd2_rbf': MMD2 with RBF kernel (lower is better)
        - 'fd': Fréchet Distance (lower is better)
    """
    results = {}

    # Calculate W2
    if verbose:
        print("Calculating Wasserstein-2 Distance...")
    results['w2'] = wasserstein2_distance(
        real, generated, num_projections=w2_projections, seed=seed
    )

    # Calculate MMD2 RBF
    if verbose:
        print("Calculating MMD2 RBF...")
    results['mmd2_rbf'] = mmd2_rbf(
        real, generated, subsample=mmd_subsample, seed=seed
    )

    # Calculate FD
    if verbose:
        print("Calculating Fréchet Distance...")
    results['fd'] = frechet_distance(real, generated)

    if verbose:
        print("\nGeneration Metrics:")
        print(f"  W2 (Wasserstein-2):        {results['w2']:.6f} ↓")
        print(f"  MMD2 RBF:                  {results['mmd2_rbf']:.6f} ↓")
        print(f"  FD (Fréchet Distance):     {results['fd']:.6f} ↓")
        print()
        print("↓ = lower is better")

    return results

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