#!/usr/bin/env python3
"""
Inference script for imputation using trained SEDD discrete diffusion model.

This script loads a trained model and performs gene expression imputation
by masking some genes and predicting their values.
"""

import argparse
import json
import os
from pathlib import Path
import sys

import torch
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sedd.model import SEDDTransformerSmall
from sedd.graph import AbsorbingGraph
from sedd.noise import LogLinearNoise
from sedd.trainer import SEDDTrainer
from sedd.data import train_val_split
from sedd.sampling import impute_masked

# Import TOML library
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def load_config(config_path="config.toml"):
    """Load configuration from TOML file."""
    config_file = Path(__file__).parent.parent / config_path
    if not config_file.exists():
        print(f"Warning: Config file not found at {config_file}, using defaults")
        return {}

    with open(config_file, "rb") as f:
        return tomllib.load(f)


def find_checkpoint(experiment_dir):
    """
    Find the best or final checkpoint in an experiment directory.

    Priority:
    1. best.pt (best validation checkpoint)
    2. final.pt (final checkpoint)
    3. Most recent epoch checkpoint
    """
    experiment_dir = Path(experiment_dir)

    if not experiment_dir.exists():
        raise ValueError(f"Experiment directory not found: {experiment_dir}")

    # Check for best checkpoint
    best_ckpt = experiment_dir / "best.pt"
    if best_ckpt.exists():
        print(f"Found best checkpoint: {best_ckpt}")
        return best_ckpt

    # Check for final checkpoint
    final_ckpt = experiment_dir / "final.pt"
    if final_ckpt.exists():
        print(f"Found final checkpoint: {final_ckpt}")
        return final_ckpt

    # Find most recent epoch checkpoint
    checkpoints = list(experiment_dir.glob("epoch_*.pt"))
    if checkpoints:
        # Sort by epoch number
        checkpoints.sort(key=lambda x: int(x.stem.split("_")[1]))
        latest = checkpoints[-1]
        print(f"Found latest epoch checkpoint: {latest}")
        return latest

    raise ValueError(f"No checkpoints found in {experiment_dir}")


def parse_args():
    # Load config file
    config = load_config()
    data_config = config.get("data", {})

    parser = argparse.ArgumentParser(description="Run imputation inference with trained SEDD model")

    # Experiment/checkpoint arguments
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Path to experiment directory containing trained model checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific checkpoint file (if not provided, will auto-find best/final)"
    )

    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default=data_config.get("test_data", None) or data_config.get("train_data", None),
        help="Path to h5ad file containing test data (defaults to config.toml)"
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=data_config.get("val_fraction", 0.1),
        help="Validation fraction (used if test_data not separate)"
    )
    parser.add_argument(
        "--use_train_split",
        action="store_true",
        help="Use training split instead of validation split"
    )

    # Imputation arguments
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.2,
        help="Fraction of genes to mask for imputation"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="Number of sampling steps for imputation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="euler",
        choices=["euler"],
        help="Sampler to use for imputation"
    )

    # Evaluation arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=None,
        help="Number of batches to evaluate (None = all)"
    )
    parser.add_argument(
        "--num_cells_visualize",
        type=int,
        default=3,
        help="Number of individual cells to visualize"
    )

    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Find checkpoint
    experiment_dir = Path(args.experiment_dir)
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = find_checkpoint(experiment_dir)

    # Create output directory for results
    output_dir = experiment_dir / "imputation_results"
    output_dir.mkdir(exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # Load training configuration
    args_file = experiment_dir / "args.json"
    if not args_file.exists():
        raise ValueError(f"Training config not found: {args_file}")

    with open(args_file, "r") as f:
        train_config = json.load(f)

    print(f"\nLoaded training configuration from {args_file}")
    print(f"Model: hidden_dim={train_config.get('hidden_dim')}, "
          f"layers={train_config.get('num_layers')}, "
          f"heads={train_config.get('num_heads')}")

    # Load data
    if not args.data_path:
        raise ValueError("No data_path provided. Set it in config.toml or pass --data_path")

    print(f"\nLoading data from {args.data_path}")
    adata = sc.read_h5ad(args.data_path)
    print(f"Loaded {len(adata)} cells with {adata.n_vars} genes")

    # Convert to tensor
    expression = adata.X
    dataset = torch.tensor(expression).long()

    # Calculate vocab size
    NUM_BINS = int(dataset.max().item())
    NUM_GENES = dataset.shape[1]
    VOCAB_SIZE = NUM_BINS + 1

    print(f"Number of genes: {NUM_GENES}")
    print(f"Number of bins: {NUM_BINS}")
    print(f"Vocabulary size: {VOCAB_SIZE}")

    # Split into train/val (use val for testing)
    train_dataset, test_dataset = train_val_split(
        dataset,
        val_fraction=args.val_fraction,
        seed=args.seed
    )

    # Choose which split to use
    eval_dataset = train_dataset if args.use_train_split else test_dataset
    split_name = "train" if args.use_train_split else "val"
    print(f"Using {split_name} split: {len(eval_dataset)} cells")

    # Create data loader
    test_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    print(f"Number of batches: {len(test_loader)}")

    # Create model
    print("\nCreating model...")
    model = SEDDTransformerSmall(
        num_genes=NUM_GENES,
        num_bins=NUM_BINS,
        hidden_dim=train_config.get("hidden_dim", 128),
        num_layers=train_config.get("num_layers", 4),
        num_heads=train_config.get("num_heads", 4),
        dropout=train_config.get("dropout", 0.1),
        max_seq_len=NUM_GENES
    ).to(device)

    # Create graph and noise schedule
    graph = AbsorbingGraph(num_states=VOCAB_SIZE)
    noise = LogLinearNoise(eps=1e-3)

    # Create trainer and load checkpoint
    trainer = SEDDTrainer(
        model=model,
        graph=graph,
        noise=noise,
        device=device
    )

    print(f"\nLoading checkpoint from {checkpoint_path}")
    trainer.load_checkpoint(checkpoint_path)
    print(f"Model loaded! Trained for {trainer.epoch + 1} epochs.")

    # Run imputation
    print(f"\nRunning imputation with mask_ratio={args.mask_ratio}, "
          f"num_steps={args.num_steps}, temperature={args.temperature}")

    all_original_masked = []
    all_predicted_masked = []
    sample_cells_original = []
    sample_cells_imputed = []
    sample_cells_masks = []

    num_batches = args.num_batches or len(test_loader)

    for i, batch in enumerate(tqdm(test_loader, total=num_batches, desc="Imputing")):
        if i >= num_batches:
            break

        batch = batch.to(device)

        # Create random mask
        mask = torch.rand_like(batch.float()) < args.mask_ratio

        # Impute masked positions
        imputed = impute_masked(
            model=model,
            graph=graph,
            noise=noise,
            x=batch,
            mask=mask,
            sampler=args.sampler,
            num_steps=args.num_steps,
            temperature=args.temperature,
            show_progress=False
        )

        # Collect results
        original_masked = batch[mask]
        predicted_masked = imputed[mask]

        all_original_masked.append(original_masked.cpu())
        all_predicted_masked.append(predicted_masked.cpu())

        # Save some cells for visualization
        if len(sample_cells_original) < args.num_cells_visualize:
            for j in range(min(args.num_cells_visualize - len(sample_cells_original), batch.size(0))):
                sample_cells_original.append(batch[j].cpu())
                sample_cells_imputed.append(imputed[j].cpu())
                sample_cells_masks.append(mask[j].cpu())

    # Concatenate all results
    all_original_masked = torch.cat(all_original_masked)
    all_predicted_masked = torch.cat(all_predicted_masked)

    print(f"\nTotal masked positions evaluated: {len(all_original_masked)}")

    # Calculate metrics
    print("\n" + "="*50)
    print("Imputation Metrics")
    print("="*50)

    accuracy = (all_original_masked == all_predicted_masked).float().mean().item()
    print(f"Exact match accuracy: {accuracy:.2%}")

    mae = (all_original_masked - all_predicted_masked).abs().float().mean().item()
    print(f"Mean Absolute Error (bins): {mae:.2f}")

    within_k_metrics = {}
    for k in [1, 3, 5, 10]:
        within_k = ((all_original_masked - all_predicted_masked).abs() <= k).float().mean().item()
        within_k_metrics[k] = within_k
        print(f"Within {k} bins: {within_k:.2%}")

    # Save metrics
    print(f"\nSaving metrics to {output_dir}")
    (output_dir / "accuracy.txt").write_text(f"{accuracy:.6f}\n")
    (output_dir / "mae.json").write_text(f'{{"mae_bins": {mae:.6f}}}\n')
    for k, v in within_k_metrics.items():
        (output_dir / f"within_{k}.csv").write_text(f"within_{k},{v:.6f}\n")

    # Save comprehensive metrics
    metrics = {
        "accuracy": accuracy,
        "mae_bins": mae,
        "within_k": within_k_metrics,
        "mask_ratio": args.mask_ratio,
        "num_steps": args.num_steps,
        "temperature": args.temperature,
        "num_masked_positions": len(all_original_masked),
        "checkpoint": str(checkpoint_path)
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Visualizations
    print("\nGenerating visualizations...")

    # 1. Scatter plot and error distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(
        all_original_masked.numpy(),
        all_predicted_masked.numpy(),
        alpha=0.3,
        s=10
    )
    axes[0].plot([0, NUM_BINS], [0, NUM_BINS], 'r--', label='Perfect prediction')
    axes[0].set_xlabel('True Bin')
    axes[0].set_ylabel('Predicted Bin')
    axes[0].set_title('Imputation: Predicted vs True')
    axes[0].legend()

    errors = (all_predicted_masked - all_original_masked).numpy()
    axes[1].hist(errors, bins=50, alpha=0.7)
    axes[1].axvline(0, color='r', linestyle='--', label='Zero error')
    axes[1].set_xlabel('Prediction Error (bins)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Prediction Error Distribution')
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(output_dir / "imputation_scatter_hist.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Individual cell visualizations
    for cell_idx in range(len(sample_cells_original)):
        original_cell = sample_cells_original[cell_idx].numpy()
        imputed_cell = sample_cells_imputed[cell_idx].numpy()
        cell_mask = sample_cells_masks[cell_idx].numpy()

        fig, axes = plt.subplots(2, 1, figsize=(14, 6))

        gene_indices = np.arange(len(original_cell))

        # Original
        axes[0].bar(gene_indices, original_cell, alpha=0.7, label='Original', width=1.0)
        axes[0].scatter(gene_indices[cell_mask], original_cell[cell_mask],
                        c='red', s=20, zorder=5, label='Masked positions')
        axes[0].set_xlabel('Gene Index')
        axes[0].set_ylabel('Expression Bin')
        axes[0].set_title('Original Expression (masked positions highlighted)')
        axes[0].legend()

        # Imputed
        axes[1].bar(gene_indices, imputed_cell, alpha=0.7, label='Imputed', width=1.0)
        axes[1].scatter(gene_indices[cell_mask], imputed_cell[cell_mask],
                        c='green', s=20, zorder=5, label='Imputed positions')
        axes[1].set_xlabel('Gene Index')
        axes[1].set_ylabel('Expression Bin')
        axes[1].set_title('Imputed Expression')
        axes[1].legend()

        plt.tight_layout()
        fig.savefig(output_dir / f"single_cell_{cell_idx}.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Cell-specific metrics
        cell_acc = (original_cell[cell_mask] == imputed_cell[cell_mask]).mean()
        cell_mae = np.abs(original_cell[cell_mask] - imputed_cell[cell_mask]).mean()
        print(f"Cell {cell_idx} - Accuracy: {cell_acc:.2%}, MAE: {cell_mae:.2f} bins")

        (output_dir / f"cell_{cell_idx}_acc.txt").write_text(f"{cell_acc:.6f}\n")
        (output_dir / f"cell_{cell_idx}_mae.json").write_text(f'{{"mae_bins": {cell_mae:.6f}}}\n')

    print("\n" + "="*50)
    print(f"Imputation complete! Results saved to {output_dir}")
    print("="*50)


if __name__ == "__main__":
    main()
