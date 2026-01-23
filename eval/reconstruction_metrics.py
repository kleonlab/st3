"""
Reconstruction Metrics for Single-Cell RNA-seq Data Evaluation

This module implements reconstruction metrics used in the scLDM paper:
- RE (Reconstruction Error): Measures reconstruction quality
- PCC (Pearson Correlation Coefficient): Measures correlation between original and reconstructed
- MSE (Mean Squared Error): Standard L2 prediction error

Reference: "Scalable Single-Cell Gene Expression Generation with Latent Diffusion Models"
https://arxiv.org/abs/2511.02986
"""

import numpy as np
from scipy.stats import pearsonr
from typing import Union, Tuple
import warnings


def reconstruction_error(
    original: np.ndarray,
    reconstructed: np.ndarray,
    reduction: str = "mean"
) -> Union[float, np.ndarray]:
    """
    Calculate Reconstruction Error (RE) between original and reconstructed data.

    RE is typically computed as the L1 norm (mean absolute error) or L2 norm between
    original and reconstructed gene expression values.

    Args:
        original: Original gene expression matrix (n_cells, n_genes)
        reconstructed: Reconstructed gene expression matrix (n_cells, n_genes)
        reduction: How to reduce the error ('mean', 'sum', or 'none')
                  'mean' returns average error per cell
                  'sum' returns total error
                  'none' returns error per cell (n_cells,)

    Returns:
        Reconstruction error value(s)

    Lower values indicate better reconstruction quality.
    """
    if original.shape != reconstructed.shape:
        raise ValueError(
            f"Shape mismatch: original {original.shape} vs reconstructed {reconstructed.shape}"
        )

    # Calculate L1 norm (mean absolute error) per cell
    error_per_cell = np.abs(original - reconstructed).mean(axis=1)

    if reduction == "mean":
        return float(np.mean(error_per_cell))
    elif reduction == "sum":
        return float(np.sum(error_per_cell))
    elif reduction == "none":
        return error_per_cell
    else:
        raise ValueError(f"Unknown reduction: {reduction}. Use 'mean', 'sum', or 'none'")


def pearson_correlation_coefficient(
    original: np.ndarray,
    reconstructed: np.ndarray,
    per_cell: bool = False
) -> Union[float, Tuple[float, float], np.ndarray]:
    """
    Calculate Pearson Correlation Coefficient (PCC) between original and reconstructed data.

    PCC measures the linear correlation between predicted and actual gene expression values.

    Args:
        original: Original gene expression matrix (n_cells, n_genes)
        reconstructed: Reconstructed gene expression matrix (n_cells, n_genes)
        per_cell: If True, compute correlation per cell and return array
                 If False, compute global correlation across all values

    Returns:
        If per_cell=False: (correlation, p_value) tuple
        If per_cell=True: array of correlations per cell (n_cells,)

    Higher values (closer to 1) indicate better correlation.
    """
    if original.shape != reconstructed.shape:
        raise ValueError(
            f"Shape mismatch: original {original.shape} vs reconstructed {reconstructed.shape}"
        )

    if per_cell:
        # Calculate correlation for each cell across genes
        correlations = np.zeros(original.shape[0])
        for i in range(original.shape[0]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                corr, _ = pearsonr(original[i], reconstructed[i])
                correlations[i] = corr if not np.isnan(corr) else 0.0
        return correlations
    else:
        # Calculate global correlation across all values
        original_flat = original.flatten()
        reconstructed_flat = reconstructed.flatten()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr, pval = pearsonr(original_flat, reconstructed_flat)

        return (float(corr), float(pval))


def mean_squared_error(
    original: np.ndarray,
    reconstructed: np.ndarray,
    reduction: str = "mean"
) -> Union[float, np.ndarray]:
    """
    Calculate Mean Squared Error (MSE) between original and reconstructed data.

    MSE is the average of squared differences between predictions and ground truth.

    Args:
        original: Original gene expression matrix (n_cells, n_genes)
        reconstructed: Reconstructed gene expression matrix (n_cells, n_genes)
        reduction: How to reduce the error ('mean', 'sum', or 'none')
                  'mean' returns average MSE per cell
                  'sum' returns total squared error
                  'none' returns MSE per cell (n_cells,)

    Returns:
        MSE value(s)

    Lower values indicate better reconstruction quality.
    """
    if original.shape != reconstructed.shape:
        raise ValueError(
            f"Shape mismatch: original {original.shape} vs reconstructed {reconstructed.shape}"
        )

    # Calculate MSE per cell
    mse_per_cell = ((original - reconstructed) ** 2).mean(axis=1)

    if reduction == "mean":
        return float(np.mean(mse_per_cell))
    elif reduction == "sum":
        return float(np.sum(mse_per_cell))
    elif reduction == "none":
        return mse_per_cell
    else:
        raise ValueError(f"Unknown reduction: {reduction}. Use 'mean', 'sum', or 'none'")


def evaluate_reconstruction(
    original: np.ndarray,
    reconstructed: np.ndarray,
    verbose: bool = True
) -> dict:
    """
    Evaluate all reconstruction metrics at once.

    Args:
        original: Original gene expression matrix (n_cells, n_genes)
        reconstructed: Reconstructed gene expression matrix (n_cells, n_genes)
        verbose: If True, print results

    Returns:
        Dictionary containing all reconstruction metrics:
        - 're': Reconstruction Error (lower is better)
        - 'pcc': Pearson Correlation Coefficient (higher is better)
        - 'pcc_pvalue': P-value for PCC
        - 'mse': Mean Squared Error (lower is better)
    """
    results = {}

    # Calculate RE
    results['re'] = reconstruction_error(original, reconstructed)

    # Calculate PCC
    pcc, pval = pearson_correlation_coefficient(original, reconstructed)
    results['pcc'] = pcc
    results['pcc_pvalue'] = pval

    # Calculate MSE
    results['mse'] = mean_squared_error(original, reconstructed)

    if verbose:
        print("Reconstruction Metrics:")
        print(f"  RE (Reconstruction Error):  {results['re']:.6f} ↓")
        print(f"  PCC (Pearson Correlation):  {results['pcc']:.6f} ↑")
        print(f"  MSE (Mean Squared Error):   {results['mse']:.6f} ↓")
        print()
        print("↓ = lower is better, ↑ = higher is better")

    return results


if __name__ == "__main__":
    # Example usage
    print("Example: Reconstruction Metrics Calculation\n")

    # Generate synthetic test data
    np.random.seed(42)
    n_cells, n_genes = 100, 50

    original = np.random.randn(n_cells, n_genes)
    # Add some noise to create reconstructed data
    reconstructed = original + np.random.randn(n_cells, n_genes) * 0.1

    # Evaluate
    results = evaluate_reconstruction(original, reconstructed, verbose=True)
