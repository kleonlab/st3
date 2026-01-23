"""
Evaluation metrics for single-cell RNA-seq models.

This package provides metrics for evaluating:
1. Reconstruction quality (RE, PCC, MSE)
2. Generation quality (W2, MMD2, FD)

Reference: "Scalable Single-Cell Gene Expression Generation with Latent Diffusion Models"
https://arxiv.org/abs/2511.02986
"""

from .reconstruction_metrics import (
    reconstruction_error,
    pearson_correlation_coefficient,
    mean_squared_error,
    evaluate_reconstruction
)

from .generation_metrics import (
    wasserstein2_distance,
    mmd2_rbf,
    frechet_distance,
    evaluate_generation
)

__all__ = [
    # Reconstruction metrics
    'reconstruction_error',
    'pearson_correlation_coefficient',
    'mean_squared_error',
    'evaluate_reconstruction',
    # Generation metrics
    'wasserstein2_distance',
    'mmd2_rbf',
    'frechet_distance',
    'evaluate_generation',
]

__version__ = '0.1.0'
