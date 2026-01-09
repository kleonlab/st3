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

