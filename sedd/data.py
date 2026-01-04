"""
Data utilities for single-cell RNA-seq data.

Provides functions for loading, preprocessing, and discretizing
gene expression data for use with SEDD.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Union, List
import warnings

Tensor = torch.Tensor
Array = np.ndarray


def discretize_expression(
    expression: Union[Tensor, Array],
    num_bins: int = 100,
    method: str = "log_uniform",
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> Tuple[Tensor, dict]:
    """Discretize continuous gene expression values into bins.

    Args:
        expression: Continuous expression values [num_cells, num_genes]
        num_bins: Number of discrete bins
        method: Discretization method
            - "uniform": Equal-width bins
            - "quantile": Equal-count bins
            - "log_uniform": Log-transform then uniform bins
        min_val: Minimum value for binning (default: data min)
        max_val: Maximum value for binning (default: data max)

    Returns:
        Tuple of:
            - Discretized expression as LongTensor [num_cells, num_genes]
            - Metadata dict with binning parameters
    """
    if isinstance(expression, Tensor):
        expression = expression.numpy()

    expression = expression.astype(np.float32)

    if method == "log_uniform":
        # Log transform (add pseudocount for zeros)
        expr_transformed = np.log1p(expression)
    else:
        expr_transformed = expression.copy()

    # Compute bin edges
    if min_val is None:
        min_val = float(np.min(expr_transformed))
    if max_val is None:
        max_val = float(np.max(expr_transformed))

    if method == "quantile":
        # Use quantiles for bin edges
        percentiles = np.linspace(0, 100, num_bins + 1)
        bin_edges = np.percentile(expr_transformed.flatten(), percentiles)
        # Make edges unique
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < num_bins + 1:
            # Fall back to uniform if not enough unique quantiles
            bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    else:
        # Uniform bins
        bin_edges = np.linspace(min_val, max_val, num_bins + 1)

    # Digitize (bin indices from 0 to num_bins-1)
    discretized = np.digitize(expr_transformed, bin_edges[1:-1])
    discretized = np.clip(discretized, 0, num_bins - 1)

    metadata = {
        "num_bins": num_bins,
        "method": method,
        "bin_edges": bin_edges,
        "min_val": min_val,
        "max_val": max_val,
    }

    return torch.from_numpy(discretized).long(), metadata


def undiscretize_expression(
    discretized: Tensor,
    metadata: dict,
) -> Tensor:
    """Convert discretized values back to continuous expression.

    Uses bin centers for reconstruction.

    Args:
        discretized: Discretized expression [num_cells, num_genes]
        metadata: Metadata dict from discretize_expression

    Returns:
        Reconstructed continuous expression
    """
    bin_edges = metadata["bin_edges"]
    method = metadata["method"]

    # Compute bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Map discretized values to bin centers
    discretized_np = discretized.numpy()
    continuous = bin_centers[discretized_np]

    if method == "log_uniform":
        # Reverse log transform
        continuous = np.expm1(continuous)

    return torch.from_numpy(continuous).float()


class RNASeqDataset(Dataset):
    """Dataset for single-cell RNA-seq expression data.

    Handles discretization and provides samples for training.
    """

    def __init__(
        self,
        expression: Union[Tensor, Array],
        gene_names: Optional[List[str]] = None,
        cell_labels: Optional[Union[Tensor, Array]] = None,
        num_bins: int = 100,
        discretization_method: str = "log_uniform",
        precomputed_discrete: bool = False,
    ):
        """
        Args:
            expression: Gene expression matrix [num_cells, num_genes]
                If precomputed_discrete=True, should be discretized LongTensor
            gene_names: Optional list of gene names
            cell_labels: Optional cell type labels
            num_bins: Number of discrete bins
            discretization_method: Method for discretization
            precomputed_discrete: Whether expression is already discretized
        """
        if precomputed_discrete:
            self.expression = expression if isinstance(expression, Tensor) else torch.from_numpy(expression).long()
            self.metadata = {"num_bins": num_bins, "method": "precomputed"}
        else:
            self.expression, self.metadata = discretize_expression(
                expression,
                num_bins=num_bins,
                method=discretization_method,
            )

        self.num_cells, self.num_genes = self.expression.shape
        self.num_bins = num_bins
        self.gene_names = gene_names

        if cell_labels is not None:
            if isinstance(cell_labels, np.ndarray):
                cell_labels = torch.from_numpy(cell_labels)
            self.cell_labels = cell_labels
        else:
            self.cell_labels = None

    def __len__(self) -> int:
        return self.num_cells

    def __getitem__(self, idx: int) -> Tensor:
        """Get a single cell's expression profile."""
        return self.expression[idx]

    def get_with_label(self, idx: int) -> Tuple[Tensor, Optional[Tensor]]:
        """Get expression with cell label."""
        expr = self.expression[idx]
        label = self.cell_labels[idx] if self.cell_labels is not None else None
        return expr, label

    def get_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        **kwargs
    ) -> DataLoader:
        """Create a DataLoader for this dataset.

        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of data loading workers
            **kwargs: Additional DataLoader arguments

        Returns:
            DataLoader instance
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs
        )


def create_synthetic_rnaseq(
    num_cells: int = 1000,
    num_genes: int = 500,
    num_cell_types: int = 5,
    sparsity: float = 0.7,
    seed: Optional[int] = None,
) -> Tuple[Array, Array, List[str]]:
    """Create synthetic single-cell RNA-seq data for testing.

    Generates data with:
    - Log-normal expression distribution
    - High sparsity (many zeros)
    - Cell type-specific expression patterns

    Args:
        num_cells: Number of cells
        num_genes: Number of genes
        num_cell_types: Number of cell types
        sparsity: Fraction of zero entries
        seed: Random seed

    Returns:
        Tuple of:
            - Expression matrix [num_cells, num_genes]
            - Cell type labels [num_cells]
            - Gene names
    """
    if seed is not None:
        np.random.seed(seed)

    # Assign cell types
    cell_types = np.random.randint(0, num_cell_types, size=num_cells)

    # Create cell type-specific expression profiles
    type_profiles = np.random.lognormal(mean=1, sigma=1, size=(num_cell_types, num_genes))

    # Generate expression
    expression = np.zeros((num_cells, num_genes), dtype=np.float32)
    for i in range(num_cells):
        ct = cell_types[i]
        # Base expression from cell type profile
        base = type_profiles[ct]
        # Add noise
        noise = np.random.lognormal(mean=0, sigma=0.5, size=num_genes)
        expression[i] = base * noise

    # Apply sparsity (dropout)
    dropout_mask = np.random.random((num_cells, num_genes)) < sparsity
    expression[dropout_mask] = 0

    # Gene names
    gene_names = [f"Gene_{i}" for i in range(num_genes)]

    return expression, cell_types, gene_names


def load_10x_h5(
    filepath: str,
    min_genes: int = 200,
    min_cells: int = 3,
) -> Tuple[Array, List[str], List[str]]:
    """Load 10x Genomics HDF5 file.

    Args:
        filepath: Path to .h5 file
        min_genes: Minimum genes per cell
        min_cells: Minimum cells per gene

    Returns:
        Tuple of:
            - Expression matrix [num_cells, num_genes]
            - Gene names
            - Cell barcodes
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required for loading 10x files: pip install h5py")

    with h5py.File(filepath, "r") as f:
        # Handle both v2 and v3 formats
        if "matrix" in f:
            # v3 format
            grp = f["matrix"]
            data = np.array(grp["data"])
            indices = np.array(grp["indices"])
            indptr = np.array(grp["indptr"])
            shape = tuple(grp["shape"])
            genes = [g.decode() for g in grp["features/name"]]
            barcodes = [b.decode() for b in grp["barcodes"]]
        else:
            # v2 format
            grp = f
            data = np.array(grp["data"])
            indices = np.array(grp["indices"])
            indptr = np.array(grp["indptr"])
            shape = tuple(grp["shape"])
            genes = [g.decode() for g in grp["gene_names"]]
            barcodes = [b.decode() for b in grp["barcodes"]]

    # Construct sparse matrix
    try:
        from scipy import sparse
        matrix = sparse.csc_matrix((data, indices, indptr), shape=shape)
        matrix = matrix.T.toarray()  # [cells, genes]
    except ImportError:
        raise ImportError("scipy required for sparse matrix: pip install scipy")

    # Filter cells and genes
    gene_counts = (matrix > 0).sum(axis=1)
    cell_counts = (matrix > 0).sum(axis=0)

    cell_mask = gene_counts >= min_genes
    gene_mask = cell_counts >= min_cells

    matrix = matrix[cell_mask][:, gene_mask]
    barcodes = [b for i, b in enumerate(barcodes) if cell_mask[i]]
    genes = [g for i, g in enumerate(genes) if gene_mask[i]]

    return matrix.astype(np.float32), genes, barcodes


def load_csv(
    filepath: str,
    transpose: bool = False,
    sep: str = ",",
) -> Tuple[Array, List[str], List[str]]:
    """Load expression data from CSV file.

    Args:
        filepath: Path to CSV file
        transpose: Whether to transpose (if genes are rows)
        sep: Separator character

    Returns:
        Tuple of:
            - Expression matrix [num_cells, num_genes]
            - Gene names (column headers)
            - Cell IDs (row indices)
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required for CSV loading: pip install pandas")

    df = pd.read_csv(filepath, sep=sep, index_col=0)

    if transpose:
        df = df.T

    expression = df.values.astype(np.float32)
    gene_names = list(df.columns)
    cell_ids = list(df.index)

    return expression, gene_names, cell_ids


def train_val_split(
    dataset: RNASeqDataset,
    val_fraction: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[Dataset, Dataset]:
    """Split dataset into training and validation sets.

    Args:
        dataset: RNASeqDataset to split
        val_fraction: Fraction for validation
        seed: Random seed

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    from torch.utils.data import Subset

    num_samples = len(dataset)
    num_val = int(num_samples * val_fraction)
    num_train = num_samples - num_val

    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None

    indices = torch.randperm(num_samples, generator=generator).tolist()
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return train_dataset, val_dataset


def normalize_expression(
    expression: Union[Tensor, Array],
    method: str = "log1p",
    target_sum: Optional[float] = 10000,
) -> Array:
    """Normalize gene expression data.

    Args:
        expression: Raw expression [num_cells, num_genes]
        method: Normalization method
            - "log1p": Log(1 + x) transform
            - "cpm": Counts per million
            - "log_cpm": Log(1 + CPM)
        target_sum: Target sum for normalization

    Returns:
        Normalized expression
    """
    if isinstance(expression, Tensor):
        expression = expression.numpy()

    expression = expression.astype(np.float32)

    if method in ["cpm", "log_cpm"]:
        # Normalize to counts per million (or target_sum)
        row_sums = expression.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        expression = expression / row_sums * target_sum

    if method in ["log1p", "log_cpm"]:
        expression = np.log1p(expression)

    return expression
