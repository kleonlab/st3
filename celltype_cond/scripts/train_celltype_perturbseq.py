#!/usr/bin/env python3
"""
Training script for cell-type conditioned SEDD perturbation prediction.

This script trains a model that conditions on both perturbation labels
AND cell-type labels for cell-type-specific perturbation predictions.

Task: perturbation label + cell-type label -> perturbed cell expression
"""

import argparse
import os
from pathlib import Path
import sys

import torch
import scanpy as sc
import numpy as np
from torch.utils.data import DataLoader, Dataset
from cell_load.data_modules import PerturbationDataModule
import yaml
import json

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from celltype_cond.sedd.model import (
    SEDDCellTypePerturbationTransformerSmall,
    SEDDCellTypePerturbationTransformerMedium,
    SEDDCellTypePerturbationTransformerLarge,
)
from celltype_cond.sedd.trainer import CellTypePerturbationTrainer
from sedd.graph import AbsorbingGraph
from sedd.noise import LogLinearNoise


def find_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in a checkpoint directory."""
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return None

    # Find most recent epoch checkpoint
    checkpoints = list(checkpoint_dir.glob("epoch_*.pt"))
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.stem.split("_")[1]))
        latest = checkpoints[-1]
        print(f"Found latest epoch checkpoint: {latest}")
        return latest

    # Check for final checkpoint
    final_ckpt = checkpoint_dir / "final.pt"
    if final_ckpt.exists():
        print(f"Found final checkpoint: {final_ckpt}")
        return final_ckpt

    # Check for best checkpoint
    best_ckpt = checkpoint_dir / "best.pt"
    if best_ckpt.exists():
        print(f"Found best checkpoint: {best_ckpt}")
        return best_ckpt

    return None


def load_yaml_config(config_path):
    config_file = Path(config_path)
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def load_conditional_labels(pt_path, names_list):
    """Load conditional labels from .pt file and create lookup mapping."""
    if pt_path is None:
        return None, []

    print(f"\nLoading conditional labels from: {pt_path}")
    pt_data = torch.load(pt_path, map_location="cpu")

    if not isinstance(pt_data, dict):
        raise TypeError(f"Expected .pt file to contain a dictionary, got {type(pt_data)}")

    pt_keys = set(pt_data.keys())
    print(f"Loaded {len(pt_keys)} items from .pt file")

    # Check coverage
    present = [n for n in names_list if n in pt_keys]
    missing = [n for n in names_list if n not in pt_keys]

    print(f"Total items in dataset: {len(names_list)}")
    print(f"Present in .pt file: {len(present)}")
    print(f"Missing from .pt file: {len(missing)}")

    if missing:
        print(f"WARNING: Missing items (first 10): {missing[:10]}")

    # Create lookup tensor
    label_lookup = []
    for name in names_list:
        if name in pt_data:
            label_val = pt_data[name]
            if isinstance(label_val, torch.Tensor):
                label_lookup.append(label_val.cpu())
            else:
                label_lookup.append(torch.tensor(label_val))
        else:
            label_lookup.append(torch.tensor(-1))

    if len(label_lookup) > 0:
        if label_lookup[0].dim() == 0:
            label_lookup = torch.stack(label_lookup)
        else:
            label_lookup = torch.stack(label_lookup)
    else:
        label_lookup = None

    print(
        f"Created label lookup tensor of shape: {label_lookup.shape if label_lookup is not None else 'None'}"
    )

    return label_lookup, missing


class CellTypePerturbSeqDataset(Dataset):
    """Dataset for cell-type conditioned perturbation prediction.

    Returns (pert_label, celltype_label, perturbed_expression) tuples.
    """

    def __init__(
        self,
        expression: torch.Tensor,
        pert_labels: np.ndarray,
        celltype_labels: np.ndarray,
        num_bins: int,
    ):
        """
        Args:
            expression: Expression matrix [num_cells, num_genes]
            pert_labels: Perturbation labels for each cell
            celltype_labels: Cell-type labels for each cell
            num_bins: Number of expression bins
        """
        self.expression = expression
        self.num_bins = num_bins

        # Create label mappings
        self.pert_names = sorted(list(set(pert_labels)))
        self.pert_to_idx = {p: i for i, p in enumerate(self.pert_names)}
        self.pert_indices = torch.tensor([self.pert_to_idx[p] for p in pert_labels])

        self.celltype_names = sorted(list(set(celltype_labels)))
        self.celltype_to_idx = {ct: i for i, ct in enumerate(self.celltype_names)}
        self.celltype_indices = torch.tensor([self.celltype_to_idx[ct] for ct in celltype_labels])

        self.num_perturbations = len(self.pert_names)
        self.num_cell_types = len(self.celltype_names)

        print(f"Dataset: {len(self.expression)} cells")
        print(f"Perturbations: {self.num_perturbations}")
        print(f"Cell types: {self.num_cell_types}: {self.celltype_names}")

    def __len__(self):
        return len(self.expression)

    def __getitem__(self, idx):
        return (
            self.pert_indices[idx],
            self.celltype_indices[idx],
            self.expression[idx],
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train cell-type conditioned SEDD for perturbation prediction"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="celltype_cond/configs/celltype_perturbseq_small.yaml",
        help="Path to YAML config file"
    )

    args, remaining = parser.parse_known_args()
    config = load_yaml_config(args.config)
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    training_config = config.get("training", {})
    checkpoint_config = config.get("checkpointing", {})
    logging_config = config.get("logging", {})
    other_config = config.get("other", {})

    # Data arguments
    parser.add_argument(
        "--train_data_path",
        type=str,
        default=data_config.get("train_data_path", None),
        help="Path to h5ad file containing perturbation-seq data"
    )
    parser.add_argument(
        "--gene",
        type=str,
        default=data_config.get("gene", "gene"),
        help="Column name in adata.obs containing perturbation labels"
    )
    parser.add_argument(
        "--cell_type_col",
        type=str,
        default=data_config.get("cell_type_col", "cell_type"),
        help="Column name in adata.obs containing cell-type labels"
    )
    parser.add_argument(
        "--control_name",
        type=str,
        default=data_config.get("control_name", "control"),
        help="Name of control perturbation in pert_col"
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=data_config.get("val_fraction", 0.1),
        help="Fraction of data to use for validation"
    )
    parser.add_argument(
        "--loader_path",
        type=str,
        default=data_config.get("loader_path", None),
        help="Path to cell_load TOML config file"
    )
    parser.add_argument(
        "--cond_labels_pt_path",
        type=str,
        default=data_config.get("cond_labels_pt_path", None),
        help="Path to .pt file containing conditional labels for perturbations"
    )
    parser.add_argument(
        "--celltype_labels_pt_path",
        type=str,
        default=data_config.get("celltype_labels_pt_path", None),
        help="Path to .pt file containing conditional labels for cell types"
    )

    # Model arguments
    parser.add_argument(
        "--model_size",
        type=str,
        default=model_config.get("size", "small"),
        choices=["small", "medium", "large"],
        help="Model size (small, medium, large)"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=model_config.get("hidden_dim", 128),
        help="Hidden dimension of the transformer"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=model_config.get("num_layers", 4),
        help="Number of transformer layers"
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=model_config.get("num_heads", 4),
        help="Number of attention heads"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=model_config.get("dropout", 0.1),
        help="Dropout rate"
    )

    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=training_config.get("batch_size", 8),
        help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=training_config.get("num_epochs", 10),
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=training_config.get("learning_rate", 1e-4),
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=training_config.get("weight_decay", 0.01),
        help="Weight decay"
    )
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=training_config.get("mask_ratio", 0.15),
        help="Fraction of genes to mask during training"
    )
    parser.add_argument(
        "--gradient_clip",
        type=float,
        default=training_config.get("gradient_clip", 1.0),
        help="Gradient clipping value"
    )

    # Checkpointing arguments
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=checkpoint_config.get("checkpoint_dir", "experiments/celltype_perturbseq"),
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=checkpoint_config.get("save_interval", 2),
        help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=checkpoint_config.get("resume", None),
        help="Path to checkpoint to resume from ('auto' for latest)"
    )

    # Logging arguments
    parser.add_argument(
        "--log_interval",
        type=int,
        default=logging_config.get("log_interval", 5),
        help="Log training metrics every N steps"
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=logging_config.get("val_interval", 1),
        help="Run validation every N epochs"
    )

    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=other_config.get("seed", 42),
        help="Random seed"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=other_config.get("num_workers", 4),
        help="Number of data loading workers"
    )

    return parser.parse_args()


def print_values(NUM_GENES, NUM_BINS, VOCAB_SIZE, NUM_PERTURBATIONS, NUM_CELL_TYPES):
    print(f"\nData statistics:")
    print(f"Number of genes: {NUM_GENES}")
    print(f"Number of bins: {NUM_BINS}")
    print(f"Vocabulary size: {VOCAB_SIZE}")
    print(f"Number of perturbations: {NUM_PERTURBATIONS}")
    print(f"Number of cell types: {NUM_CELL_TYPES}")


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading perturbation-seq data from {args.train_data_path}")
    adata = sc.read_h5ad(args.train_data_path)
    print(f"Loaded {len(adata)} cells with {adata.n_vars} genes")

    # Get perturbation labels
    if args.gene not in adata.obs.columns:
        raise ValueError(f"Perturbation column '{args.gene}' not found. Available: {list(adata.obs.columns)}")
    pert_labels = adata.obs[args.gene].values
    print(f"Found {len(np.unique(pert_labels))} unique perturbations")

    # Get cell-type labels
    if args.cell_type_col not in adata.obs.columns:
        raise ValueError(f"Cell-type column '{args.cell_type_col}' not found. Available: {list(adata.obs.columns)}")
    celltype_labels = adata.obs[args.cell_type_col].values
    unique_celltypes = np.unique(celltype_labels)
    print(f"Found {len(unique_celltypes)} unique cell types: {list(unique_celltypes)}")

    # Get expression
    expression = adata.X
    if hasattr(expression, 'toarray'):
        expression = expression.toarray()
    expression = torch.from_numpy(expression).long()

    # Calculate dimensions
    NUM_BINS = int(expression.max().item())
    NUM_GENES = expression.shape[1]
    VOCAB_SIZE = NUM_BINS + 1
    perturbations = adata.obs[args.gene].unique()
    NUM_PERTURBATIONS = len(perturbations)
    NUM_CELL_TYPES = len(unique_celltypes)

    print_values(NUM_GENES, NUM_BINS, VOCAB_SIZE, NUM_PERTURBATIONS, NUM_CELL_TYPES)
    print(f"Sparsity: {(expression == 0).sum().item() / expression.numel():.2%}")

    # Save configuration
    args_payload = dict(vars(args))
    args_payload.update({
        "num_genes": NUM_GENES,
        "num_bins": NUM_BINS,
        "num_perturbations": NUM_PERTURBATIONS,
        "num_cell_types": NUM_CELL_TYPES,
        "vocab_size": VOCAB_SIZE,
        "cell_types": list(unique_celltypes),
        "perturbations": list(perturbations),
    })
    with open(checkpoint_dir / "args.json", "w") as f:
        json.dump(args_payload, f, indent=2, default=str)

    # Load conditional labels if provided
    all_genes = list(adata.obs[args.gene].unique())
    cond_label_lookup, missing_perts = load_conditional_labels(
        args.cond_labels_pt_path,
        all_genes
    )

    celltype_label_lookup = None
    if args.celltype_labels_pt_path:
        celltype_label_lookup, missing_celltypes = load_conditional_labels(
            args.celltype_labels_pt_path,
            list(unique_celltypes)
        )

    # Use cell_load if loader_path is provided
    if args.loader_path:
        print(f"\nUsing cell_load data module from: {args.loader_path}")
        dm = PerturbationDataModule(
            toml_config_path=args.loader_path,
            embed_key="X_hvg",
            num_workers=0,
            batch_size=args.batch_size,
            pert_col=args.gene,
            control_pert=args.control_name,
            perturbations_to_use=all_genes,
            batch_col="gem_group",
            cell_sentence_len=1,
            cell_type_key=args.cell_type_col,
        )
        dm.setup()
        print(f'DataModule setup complete!')

        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
    else:
        # Create custom dataset
        print("\nCreating custom dataset...")
        dataset = CellTypePerturbSeqDataset(
            expression=expression,
            pert_labels=pert_labels,
            celltype_labels=celltype_labels,
            num_bins=NUM_BINS,
        )

        # Split into train/val
        train_size = int(len(dataset) * (1 - args.val_fraction))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        NUM_PERTURBATIONS = dataset.num_perturbations
        NUM_CELL_TYPES = dataset.num_cell_types

    print(f"Number of training batches: {len(train_loader)}")

    # Infer dimensions from dataloader
    try:
        first_batch = next(iter(train_loader))
        if isinstance(first_batch, dict):
            pert_emb = first_batch["pert_emb"]
            if pert_emb.dim() == 2 and pert_emb.shape[1] > 1:
                inferred_num_perts = int(pert_emb.shape[1])
            else:
                inferred_num_perts = int(pert_emb.squeeze(-1).max().item()) + 1

            if 'cell_type_emb' in first_batch:
                celltype_emb = first_batch['cell_type_emb']
                if celltype_emb.dim() == 2 and celltype_emb.shape[1] > 1:
                    inferred_num_celltypes = int(celltype_emb.shape[1])
                else:
                    inferred_num_celltypes = int(celltype_emb.squeeze(-1).max().item()) + 1
            else:
                inferred_num_celltypes = NUM_CELL_TYPES

            if inferred_num_perts != NUM_PERTURBATIONS:
                print(f"Overriding NUM_PERTURBATIONS {NUM_PERTURBATIONS} -> {inferred_num_perts}")
                NUM_PERTURBATIONS = inferred_num_perts

            if inferred_num_celltypes != NUM_CELL_TYPES:
                print(f"Overriding NUM_CELL_TYPES {NUM_CELL_TYPES} -> {inferred_num_celltypes}")
                NUM_CELL_TYPES = inferred_num_celltypes
    except Exception as exc:
        print(f"WARNING: Could not infer dimensions from dataloader: {exc}")

    # Infer precomputed embedding dimensions
    precomputed_pert_emb_dim = None
    if cond_label_lookup is not None and cond_label_lookup.dim() == 2:
        precomputed_pert_emb_dim = cond_label_lookup.shape[1]
        print(f"Detected precomputed perturbation embedding dimension: {precomputed_pert_emb_dim}")

    precomputed_celltype_emb_dim = None
    if celltype_label_lookup is not None and celltype_label_lookup.dim() == 2:
        precomputed_celltype_emb_dim = celltype_label_lookup.shape[1]
        print(f"Detected precomputed cell-type embedding dimension: {precomputed_celltype_emb_dim}")

    # Create model
    print("\nCreating cell-type conditioned perturbation prediction model...")
    model_classes = {
        "small": SEDDCellTypePerturbationTransformerSmall,
        "medium": SEDDCellTypePerturbationTransformerMedium,
        "large": SEDDCellTypePerturbationTransformerLarge,
    }
    ModelClass = model_classes[args.model_size]

    model = ModelClass(
        num_genes=NUM_GENES,
        num_bins=NUM_BINS,
        num_perturbations=NUM_PERTURBATIONS,
        num_cell_types=NUM_CELL_TYPES,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_seq_len=NUM_GENES,
        precomputed_pert_emb_dim=precomputed_pert_emb_dim,
        precomputed_celltype_emb_dim=precomputed_celltype_emb_dim,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create graph and noise schedule
    graph = AbsorbingGraph(num_states=VOCAB_SIZE)
    noise = LogLinearNoise(eps=1e-3)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    # Create trainer
    trainer = CellTypePerturbationTrainer(
        model=model,
        graph=graph,
        noise=noise,
        optimizer=optimizer,
        device=device,
        gradient_clip=args.gradient_clip,
        cond_label_lookup=cond_label_lookup,
        celltype_label_lookup=celltype_label_lookup,
    )

    # Resume from checkpoint if specified
    if args.resume:
        if args.resume.lower() in ["auto", "latest", "last"]:
            print(f"\nAuto-resuming from latest checkpoint in {checkpoint_dir}")
            checkpoint_path = find_checkpoint(checkpoint_dir)
            if checkpoint_path:
                trainer.load_checkpoint(checkpoint_path)
                print(f"Resumed from epoch {trainer.epoch + 1}")
            else:
                print("No checkpoint found, starting from scratch")
        else:
            print(f"\nResuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)
            print(f"Resumed from epoch {trainer.epoch + 1}")

    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        mask_ratio=args.mask_ratio,
        log_interval=args.log_interval,
        val_interval=args.val_interval,
        checkpoint_dir=checkpoint_dir,
        save_interval=args.save_interval
    )

    print("\nTraining complete!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    if history['val_loss']:
        print(f"Final val loss: {history['val_loss'][-1]:.4f}")
        print(f"Best val loss: {trainer.best_loss:.4f}")

    # Save final checkpoint
    final_checkpoint = checkpoint_dir / "final.pt"
    trainer.save_checkpoint(final_checkpoint)
    print(f"\nFinal checkpoint saved to: {final_checkpoint}")


if __name__ == "__main__":
    main()
