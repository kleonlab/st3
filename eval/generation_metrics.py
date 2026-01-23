"""
Generation Metrics for Single-Cell RNA-seq Data Evaluation

This module implements generation/distribution comparison metrics used in the scLDM paper:
- W2 (Wasserstein-2 Distance): Measures distribution distance
- MMD2 RBF (Maximum Mean Discrepancy): Kernel-based distribution comparison
- FD (Fréchet Distance): Compares distributions via mean and covariance

Reference: "Scalable Single-Cell Gene Expression Generation with Latent Diffusion Models"
https://arxiv.org/abs/2511.02986
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import sqrtm
from typing import Optional, Union
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
    eps: float = 1e-6
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
        eps: Small constant for numerical stability

    Returns:
        Fréchet distance (lower is better)

    Lower values indicate generated samples better match statistical properties of real data.
    """
    if real.shape[1] != generated.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: real {real.shape[1]} vs generated {generated.shape[1]}"
        )

    # Calculate means
    mu_real = np.mean(real, axis=0)
    mu_gen = np.mean(generated, axis=0)

    # Calculate covariances
    sigma_real = np.cov(real, rowvar=False)
    sigma_gen = np.cov(generated, rowvar=False)

    # Ensure covariances are 2D
    if sigma_real.ndim == 0:
        sigma_real = sigma_real.reshape(1, 1)
    if sigma_gen.ndim == 0:
        sigma_gen = sigma_gen.reshape(1, 1)

    # Calculate mean difference
    diff = mu_real - mu_gen
    mean_dist = np.dot(diff, diff)

    # Calculate sqrt of product of covariances
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        covmean = sqrtm(sigma_real @ sigma_gen)

    # Check for numerical errors
    if np.iscomplexobj(covmean):
        # If complex, take real part (can happen due to numerical errors)
        covmean = covmean.real

    # Add small constant for numerical stability
    covmean_trace = np.trace(covmean)

    # Calculate FD
    fd = mean_dist + np.trace(sigma_real) + np.trace(sigma_gen) - 2 * covmean_trace

    return float(max(0, fd))  # FD should be non-negative


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


if __name__ == "__main__":
    # Example usage
    print("Example: Generation Metrics Calculation\n")

    # Generate synthetic test data
    np.random.seed(42)
    n_real, n_gen, n_genes = 200, 200, 50

    # Real data from one distribution
    real = np.random.randn(n_real, n_genes)

    # Generated data from similar but slightly different distribution
    generated = np.random.randn(n_gen, n_genes) * 1.1 + 0.2

    # Evaluate
    results = evaluate_generation(real, generated, verbose=True)
