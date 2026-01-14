import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Union, List
import warnings

Tensor = torch.Tensor
Array = np.ndarray


class RNASeqDataset(Dataset):

    def __init__(
        self,
        expression: Union[Tensor, Array],
        gene_names: Optional[List[str]] = None,
        cell_labels: Optional[Union[Tensor, Array]] = None,
        num_bins: int = 100,
    ):
        self.expression = expression if isinstance(expression, Tensor) else torch.from_numpy(expression).long()
        self.metadata = {"num_bins": num_bins, "method": "precomputed"}

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

        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs
        )


def train_val_split(
    dataset: RNASeqDataset,
    val_fraction: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[Dataset, Dataset]:

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


def discretize_expression(
    expression: Union[Tensor, Array],
    num_bins: int = 100,
    method: str = "linear",
) -> Tuple[Tensor, dict]:
    """Discretize continuous expression values into bins.
    
    Args:
        expression: Continuous expression values [num_cells, num_genes]
        num_bins: Number of discrete bins
        method: Discretization method ('linear', 'log', 'quantile')
    
    Returns:
        Tuple of (discretized tensor, metadata dict for undiscretization)
    """
    if isinstance(expression, np.ndarray):
        expression = torch.from_numpy(expression).float()
    
    min_val = expression.min().item()
    max_val = expression.max().item()
    
    if method == "log":
        # Log transform first (add 1 to handle zeros)
        expression = torch.log1p(expression)
        min_val = expression.min().item()
        max_val = expression.max().item()
    
    # Linear binning
    if max_val > min_val:
        normalized = (expression - min_val) / (max_val - min_val)
        discretized = (normalized * (num_bins - 1)).long()
        discretized = discretized.clamp(0, num_bins - 1)
    else:
        discretized = torch.zeros_like(expression).long()
    
    metadata = {
        "num_bins": num_bins,
        "min_val": min_val,
        "max_val": max_val,
        "method": method,
    }
    
    return discretized, metadata


def undiscretize_expression(
    discretized: Tensor,
    metadata: dict,
) -> Tensor:
    """Convert discretized expression back to continuous values.
    
    Args:
        discretized: Discretized expression tensor
        metadata: Metadata from discretize_expression
    
    Returns:
        Continuous expression tensor
    """
    num_bins = metadata["num_bins"]
    min_val = metadata["min_val"]
    max_val = metadata["max_val"]
    method = metadata.get("method", "linear")
    
    # Convert to float and normalize to [0, 1]
    normalized = discretized.float() / (num_bins - 1)
    
    # Scale back to original range
    continuous = normalized * (max_val - min_val) + min_val
    
    if method == "log":
        # Inverse log transform
        continuous = torch.expm1(continuous)
    
    return continuous


def normalize_expression(
    expression: Union[Tensor, Array],
    method: str = "log1p",
) -> Tensor:
    """Normalize expression values.
    
    Args:
        expression: Raw expression values
        method: Normalization method ('log1p', 'zscore', 'minmax')
    
    Returns:
        Normalized expression tensor
    """
    if isinstance(expression, np.ndarray):
        expression = torch.from_numpy(expression).float()
    else:
        expression = expression.float()
    
    if method == "log1p":
        return torch.log1p(expression)
    elif method == "zscore":
        mean = expression.mean(dim=0, keepdim=True)
        std = expression.std(dim=0, keepdim=True) + 1e-8
        return (expression - mean) / std
    elif method == "minmax":
        min_val = expression.min(dim=0, keepdim=True)[0]
        max_val = expression.max(dim=0, keepdim=True)[0]
        return (expression - min_val) / (max_val - min_val + 1e-8)
    else:
        return expression


def create_synthetic_rnaseq(
    num_cells: int = 1000,
    num_genes: int = 2000,
    num_bins: int = 100,
    sparsity: float = 0.7,
    seed: Optional[int] = None,
) -> Tuple[Tensor, List[str]]:
    """Create synthetic RNA-seq data for testing.
    
    Args:
        num_cells: Number of cells to generate
        num_genes: Number of genes
        num_bins: Number of expression bins
        sparsity: Fraction of zero values
        seed: Random seed
    
    Returns:
        Tuple of (expression tensor, gene names list)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Generate expression with specified sparsity
    expression = torch.randint(0, num_bins, (num_cells, num_genes))
    
    # Apply sparsity mask
    mask = torch.rand(num_cells, num_genes) < sparsity
    expression[mask] = 0
    
    # Generate gene names
    gene_names = [f"Gene_{i}" for i in range(num_genes)]
    
    return expression, gene_names

