import json
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
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


def evaluate_generation(
    real: np.ndarray,
    generated: np.ndarray,
    w2_projections: int = 1000,
    mmd_subsample: Optional[int] = 2000,
    seed: Optional[int] = 42,
    verbose: bool = False
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
        print("  Calculating Wasserstein-2 Distance...")
    results['w2'] = wasserstein2_distance(
        real, generated, num_projections=w2_projections, seed=seed
    )

    # Calculate MMD2 RBF
    if verbose:
        print("  Calculating MMD2 RBF...")
    results['mmd2_rbf'] = mmd2_rbf(
        real, generated, subsample=mmd_subsample, seed=seed
    )

    # Calculate FD
    if verbose:
        print("  Calculating Fréchet Distance...")
    results['fd'] = frechet_distance(real, generated)

    if verbose:
        print(f"  W2:        {results['w2']:.6f}")
        print(f"  MMD2 RBF:  {results['mmd2_rbf']:.6f}")
        print(f"  FD:        {results['fd']:.6f}")

    return results


def evaluate_per_perturbation(
    real_adata: sc.AnnData,
    generated_adata: sc.AnnData,
    real_pert_col: str = "gene",
    gen_pert_col: str = "perturbation",
    w2_projections: int = 1000,
    mmd_subsample: Optional[int] = 2000,
    seed: Optional[int] = 42,
    verbose: bool = True
) -> Dict[str, dict]:
    """
    Evaluate generation metrics for each perturbation separately.

    Args:
        real_adata: AnnData with real cells
        generated_adata: AnnData with generated cells
        real_pert_col: Column name in real_adata.obs containing perturbation names
        gen_pert_col: Column name in generated_adata.obs containing perturbation names
        w2_projections: Number of projections for sliced Wasserstein distance
        mmd_subsample: Maximum samples for MMD calculation (None for all)
        seed: Random seed for reproducibility
        verbose: If True, print progress

    Returns:
        Dictionary mapping perturbation names to their metrics
    """
    # Extract expression matrices
    real_expr = real_adata.X
    if hasattr(real_expr, "toarray"):
        real_expr = real_expr.toarray()
    
    gen_expr = generated_adata.X
    if hasattr(gen_expr, "toarray"):
        gen_expr = gen_expr.toarray()
    
    # Get perturbations present in both datasets
    real_perts = set(real_adata.obs[real_pert_col].unique())
    gen_perts = set(generated_adata.obs[gen_pert_col].unique())
    common_perts = sorted(real_perts & gen_perts)
    
    if verbose:
        print(f"\nFound {len(common_perts)} common perturbations")
        print(f"Real perturbations: {len(real_perts)}")
        print(f"Generated perturbations: {len(gen_perts)}")
        if len(real_perts - gen_perts) > 0:
            print(f"Missing in generated: {real_perts - gen_perts}")
        if len(gen_perts - real_perts) > 0:
            print(f"Extra in generated: {gen_perts - real_perts}")
        print()
    
    results = {}
    
    for pert in common_perts:
        if verbose:
            print(f"Processing: {pert}")
        
        # Filter data for this perturbation
        real_mask = real_adata.obs[real_pert_col] == pert
        gen_mask = generated_adata.obs[gen_pert_col] == pert
        
        real_pert_expr = real_expr[real_mask]
        gen_pert_expr = gen_expr[gen_mask]
        
        if verbose:
            print(f"  Real cells: {len(real_pert_expr)}, Generated cells: {len(gen_pert_expr)}")
        
        # Skip if too few samples
        if len(real_pert_expr) < 2 or len(gen_pert_expr) < 2:
            print(f"  Warning: Too few samples for {pert}, skipping")
            continue
        
        # Evaluate metrics
        pert_metrics = evaluate_generation(
            real_pert_expr,
            gen_pert_expr,
            w2_projections=w2_projections,
            mmd_subsample=mmd_subsample,
            seed=seed,
            verbose=verbose
        )
        
        # Add sample counts
        pert_metrics['n_real'] = int(len(real_pert_expr))
        pert_metrics['n_generated'] = int(len(gen_pert_expr))
        
        results[pert] = pert_metrics
        
        if verbose:
            print()
    
    return results


def compute_aggregate_metrics(per_pert_results: Dict[str, dict]) -> dict:
    """
    Compute aggregate metrics across all perturbations.

    Args:
        per_pert_results: Dictionary mapping perturbation names to their metrics

    Returns:
        Dictionary with mean, median, std for each metric
    """
    if not per_pert_results:
        return {}
    
    # Collect metrics
    w2_values = [m['w2'] for m in per_pert_results.values()]
    mmd2_values = [m['mmd2_rbf'] for m in per_pert_results.values()]
    fd_values = [m['fd'] for m in per_pert_results.values()]
    
    aggregate = {
        'w2': {
            'mean': float(np.mean(w2_values)),
            'median': float(np.median(w2_values)),
            'std': float(np.std(w2_values)),
            'min': float(np.min(w2_values)),
            'max': float(np.max(w2_values)),
        },
        'mmd2_rbf': {
            'mean': float(np.mean(mmd2_values)),
            'median': float(np.median(mmd2_values)),
            'std': float(np.std(mmd2_values)),
            'min': float(np.min(mmd2_values)),
            'max': float(np.max(mmd2_values)),
        },
        'fd': {
            'mean': float(np.mean(fd_values)),
            'median': float(np.median(fd_values)),
            'std': float(np.std(fd_values)),
            'min': float(np.min(fd_values)),
            'max': float(np.max(fd_values)),
        },
        'num_perturbations': len(per_pert_results)
    }
    
    return aggregate


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Evaluating Per-Perturbation Generation Metrics")
    print("=" * 60)

    output_dir = Path("/home/b5cc/sanjukta.b5cc/st3/experiments/30k3")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_data_path = "/home/b5cc/sanjukta.b5cc/st3/datasets/30k/k562_test_split.h5ad"
    predicted_data_path = "/home/b5cc/sanjukta.b5cc/st3/experiments/30k3/inference_results/generated_cells.h5ad"

    print(f"\nLoading data...")
    print(f"  Real: {test_data_path}")
    print(f"  Generated: {predicted_data_path}")

    # Load test data
    adata_test = sc.read_h5ad(test_data_path)
    print(f"\nReal data: {adata_test.n_obs} cells, {adata_test.n_vars} genes")
    print(f"  Perturbation column: 'gene'")
    print(f"  Unique perturbations: {len(adata_test.obs['gene'].unique())}")

    # Load predicted data
    adata_pred = sc.read_h5ad(predicted_data_path)
    print(f"\nGenerated data: {adata_pred.n_obs} cells, {adata_pred.n_vars} genes")
    print(f"  Perturbation column: 'perturbation'")
    print(f"  Unique perturbations: {len(adata_pred.obs['perturbation'].unique())}")

    # Compute per-perturbation metrics
    print("\n" + "=" * 60)
    print("Computing per-perturbation metrics...")
    print("=" * 60)
    
    per_pert_metrics = evaluate_per_perturbation(
        real_adata=adata_test,
        generated_adata=adata_pred,
        real_pert_col="gene",
        gen_pert_col="perturbation",
        w2_projections=1000,
        mmd_subsample=100,
        seed=42,
        verbose=True
    )

    # Compute aggregate statistics
    print("\n" + "=" * 60)
    print("Aggregate Statistics Across Perturbations")
    print("=" * 60)
    
    aggregate_metrics = compute_aggregate_metrics(per_pert_metrics)
    
    print(f"\nNumber of perturbations evaluated: {aggregate_metrics['num_perturbations']}")
    print("\nWasserstein-2 Distance:")
    print(f"  Mean:   {aggregate_metrics['w2']['mean']:.6f}")
    print(f"  Median: {aggregate_metrics['w2']['median']:.6f}")
    print(f"  Std:    {aggregate_metrics['w2']['std']:.6f}")
    print(f"  Range:  [{aggregate_metrics['w2']['min']:.6f}, {aggregate_metrics['w2']['max']:.6f}]")
    
    print("\nMMD2 RBF:")
    print(f"  Mean:   {aggregate_metrics['mmd2_rbf']['mean']:.6f}")
    print(f"  Median: {aggregate_metrics['mmd2_rbf']['median']:.6f}")
    print(f"  Std:    {aggregate_metrics['mmd2_rbf']['std']:.6f}")
    print(f"  Range:  [{aggregate_metrics['mmd2_rbf']['min']:.6f}, {aggregate_metrics['mmd2_rbf']['max']:.6f}]")
    
    print("\nFréchet Distance:")
    print(f"  Mean:   {aggregate_metrics['fd']['mean']:.6f}")
    print(f"  Median: {aggregate_metrics['fd']['median']:.6f}")
    print(f"  Std:    {aggregate_metrics['fd']['std']:.6f}")
    print(f"  Range:  [{aggregate_metrics['fd']['min']:.6f}, {aggregate_metrics['fd']['max']:.6f}]")

    # Save results
    output_file_detailed = output_dir / "generation_metrics_per_perturbation.json"
    output_file_aggregate = output_dir / "generation_metrics_aggregate.json"
    
    with open(output_file_detailed, "w") as f:
        json.dump(per_pert_metrics, f, indent=2)
    
    with open(output_file_aggregate, "w") as f:
        json.dump(aggregate_metrics, f, indent=2)

    # Create a summary CSV for easy viewing
    df_results = pd.DataFrame.from_dict(per_pert_metrics, orient='index')
    df_results.index.name = 'perturbation'
    df_results = df_results.sort_values('w2')
    df_results.to_csv(output_dir / "generation_metrics_per_perturbation.csv")

    print("\n" + "=" * 60)
    print("Results saved:")
    print(f"  {output_file_detailed}")
    print(f"  {output_file_aggregate}")
    print(f"  {output_dir / 'generation_metrics_per_perturbation.csv'}")
    print("=" * 60)

