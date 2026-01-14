#!/usr/bin/env python3
"""
Training script for perturbation prediction using discrete diffusion.

This script trains a SEDD model for predicting perturbed cell states from
control cells and perturbation labels.

Task: control cell + perturbation label -> perturbed cell
"""

import argparse
import os
from pathlib import Path
import sys

import torch
import scanpy as sc
import numpy as np
from torch.utils.data import DataLoader
import yaml

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sedd.model import SEDDPerturbationTransformerSmall
from sedd.graph import AbsorbingGraph
from sedd.noise import LogLinearNoise
from sedd.trainer import PerturbationTrainer
from sedd.data import PerturbSeqDataset, train_val_split


def load_yaml_config(config_path):
    """Load configuration from YAML file."""
    if config_path is None:
        return {}

    config_file = Path(config_path)
    if not config_file.exists():
        raise ValueError(f"Config file not found: {config_file}")

    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SEDD for perturbation prediction"
    )

    # Config file argument (parsed first)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (e.g., configs/perturbseq_small.yaml)"
    )

    # Parse args once to get config file
    args, remaining = parser.parse_known_args()

    # Load config file
    config = load_yaml_config(args.config)
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    training_config = config.get("training", {})
    checkpoint_config = config.get("checkpointing", {})
    logging_config = config.get("logging", {})
    other_config = config.get("other", {})

    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default=data_config.get("data_path", None),
        help="Path to h5ad file containing perturbation-seq data"
    )
    parser.add_argument(
        "--pert_col",
        type=str,
        default=data_config.get("pert_col", "perturbation"),
        help="Column name in adata.obs containing perturbation labels"
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

    # Model arguments
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
        default=training_config.get("num_epochs", 100),
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

    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=checkpoint_config.get("checkpoint_dir", "experiments/perturbseq_diffusion"),
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=checkpoint_config.get("save_interval", 10),
        help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=checkpoint_config.get("resume", None),
        help="Path to checkpoint to resume from"
    )

    # Logging arguments
    parser.add_argument(
        "--log_interval",
        type=int,
        default=logging_config.get("log_interval", 50),
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


def main():
    args = parse_args()

    # Validate required arguments
    if not args.data_path:
        raise ValueError(
            "No data_path provided. Either set it in config or pass --data_path argument"
        )

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save arguments
    import json
    with open(checkpoint_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load data
    print(f"Loading perturbation-seq data from {args.data_path}")
    adata = sc.read_h5ad(args.data_path)
    print(f"Loaded {len(adata)} cells with {adata.n_vars} genes")

    # Check for perturbation column
    if args.pert_col not in adata.obs.columns:
        raise ValueError(
            f"Perturbation column '{args.pert_col}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    # Get perturbation labels
    pert_labels = adata.obs[args.pert_col].values
    print(f"Found {len(np.unique(pert_labels))} unique perturbations")
    print(f"Perturbation distribution:")
    unique, counts = np.unique(pert_labels, return_counts=True)
    for u, c in zip(unique[:10], counts[:10]):  # Show first 10
        print(f"  {u}: {c} cells")
    if len(unique) > 10:
        print(f"  ... and {len(unique) - 10} more")

    # Convert expression to tensor
    expression = adata.X
    if hasattr(expression, 'toarray'):
        expression = expression.toarray()
    expression = torch.from_numpy(expression).long()

    # Calculate vocab size from data
    NUM_BINS = int(expression.max().item())
    NUM_GENES = expression.shape[1]
    VOCAB_SIZE = NUM_BINS + 1  # +1 for mask token

    print(f"\nData statistics:")
    print(f"Number of genes: {NUM_GENES}")
    print(f"Number of bins: {NUM_BINS}")
    print(f"Vocabulary size: {VOCAB_SIZE}")
    print(f"Sparsity: {(expression == 0).sum().item() / expression.numel():.2%}")

    # Create dataset
    dataset = PerturbSeqDataset(
        expression=expression,
        pert_labels=pert_labels,
        num_bins=NUM_BINS,
        control_pert_name=args.control_name,
    )

    NUM_PERTURBATIONS = dataset.num_perturbations

    # Split into train/val
    train_dataset, val_dataset = train_val_split(
        dataset,
        val_fraction=args.val_fraction,
        seed=args.seed
    )
    print(f"\nTrain size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # Create model
    print("\nCreating perturbation prediction model...")
    model = SEDDPerturbationTransformerSmall(
        num_genes=NUM_GENES,
        num_bins=NUM_BINS,
        num_perturbations=NUM_PERTURBATIONS,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_seq_len=NUM_GENES
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
    trainer = PerturbationTrainer(
        model=model,
        graph=graph,
        noise=noise,
        optimizer=optimizer,
        device=device,
        gradient_clip=args.gradient_clip
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
        print(f"Resumed from epoch {trainer.epoch + 1}")

    # Train
    print("\nStarting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        mask_ratio=args.mask_ratio,
        log_interval=args.log_interval,
        val_interval=args.val_interval,
        checkpoint_dir=checkpoint_dir
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
