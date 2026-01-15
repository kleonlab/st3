#!/usr/bin/env python3
"""
Inference script for perturbation prediction using trained SEDD model.

This script loads a trained perturbation prediction model and generates
predictions for control cells with specified perturbations.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import pickle

import torch
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import yaml

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sedd.model import (
    SEDDPerturbationTransformerSmall,
    SEDDPerturbationTransformerMedium,
    SEDDPerturbationTransformerLarge,
)
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
    parser = argparse.ArgumentParser(
        description="Run inference with trained SEDD perturbation prediction model"
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
    data_config = config.get("data", {})
    inference_config = config.get("inference", {})
    checkpoint_config = config.get("checkpointing", {})
    model_config = config.get("model", {})

    # Experiment/checkpoint arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint file (.pt)"
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default=checkpoint_config.get("checkpoint_dir", None),
        help="Path to experiment directory (will auto-find checkpoint)"
    )

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

    # Inference arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=inference_config.get("batch_size", 32),
        help="Batch size for inference"
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=inference_config.get("num_batches", None),
        help="Number of batches to run inference on (None = all)"
    )
    parser.add_argument(
        "--num_cells_visualize",
        type=int,
        default=inference_config.get("num_cells_visualize", 3),
        help="Number of example cells to visualize"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_results",
        help="Directory to save inference results"
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save predictions to file"
    )

    # Model architecture (needed to reconstruct model)
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
        default=0,
        help="Number of data loading workers"
    )

    return parser.parse_args()


def evaluate_predictions(predicted, true_perturbed, dataset, num_cells_viz=3):
    """Evaluate and visualize predictions."""

    # Calculate metrics
    with torch.no_grad():
        # Exact match accuracy
        accuracy = (predicted == true_perturbed).float().mean().item()
        print(f"\nEvaluation metrics:")
        print(f"  Exact match accuracy: {accuracy:.2%}")

        # Mean absolute error
        mae = (predicted - true_perturbed).abs().float().mean().item()
        print(f"  Mean Absolute Error: {mae:.2f} bins")

        # Within-k accuracy
        for k in [1, 3, 5, 10]:
            within_k = ((predicted - true_perturbed).abs() <= k).float().mean().item()
            print(f"  Within {k} bins: {within_k:.2%}")

        # Per-gene correlation
        correlations = []
        num_genes = predicted.shape[1]
        for gene_idx in range(num_genes):
            true_gene = true_perturbed[:, gene_idx].float().cpu().numpy()
            pred_gene = predicted[:, gene_idx].float().cpu().numpy()
            if true_gene.std() > 0 and pred_gene.std() > 0:
                corr = np.corrcoef(true_gene, pred_gene)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

        if correlations:
            print(f"  Average per-gene correlation: {np.mean(correlations):.3f}")

    return {
        "accuracy": accuracy,
        "mae": mae,
        "correlations": correlations
    }


def visualize_predictions(control, true_perturbed, predicted, pert_labels,
                         dataset, output_dir, num_cells=3):
    """Visualize example predictions."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_genes = predicted.shape[1]
    n_genes_viz = min(200, num_genes)

    for cell_idx in range(min(num_cells, control.shape[0])):
        control_cell = control[cell_idx].cpu().numpy()
        true_cell = true_perturbed[cell_idx].cpu().numpy()
        pred_cell = predicted[cell_idx].cpu().numpy()
        pert_label = pert_labels[cell_idx].item()

        # Get perturbation name if available
        if dataset.idx_to_pert is not None:
            pert_name = dataset.idx_to_pert[pert_label]
        else:
            pert_name = f"Perturbation {pert_label}"

        gene_indices = np.arange(n_genes_viz)

        fig, axes = plt.subplots(4, 1, figsize=(15, 12))

        # Control
        axes[0].bar(gene_indices, control_cell[:n_genes_viz], alpha=0.7, width=1.0, color='blue')
        axes[0].set_ylabel('Expression (bin)', fontsize=11)
        axes[0].set_title(f'Control Cell Expression (first {n_genes_viz} genes)', fontsize=12)
        axes[0].grid(True, alpha=0.3, axis='y')

        # True perturbed
        axes[1].bar(gene_indices, true_cell[:n_genes_viz], alpha=0.7, width=1.0, color='red')
        axes[1].set_ylabel('Expression (bin)', fontsize=11)
        axes[1].set_title(f'True Perturbed Cell - {pert_name}', fontsize=12)
        axes[1].grid(True, alpha=0.3, axis='y')

        # Predicted
        axes[2].bar(gene_indices, pred_cell[:n_genes_viz], alpha=0.7, width=1.0, color='green')
        axes[2].set_ylabel('Expression (bin)', fontsize=11)
        axes[2].set_title(f'Predicted Perturbed Cell', fontsize=12)
        axes[2].grid(True, alpha=0.3, axis='y')

        # Error
        error = (pred_cell - true_cell)[:n_genes_viz]
        colors = ['green' if abs(e) <= 5 else 'orange' if abs(e) <= 10 else 'red' for e in error]
        axes[3].bar(gene_indices, error, alpha=0.7, width=1.0, color=colors)
        axes[3].axhline(0, color='black', linestyle='--', linewidth=1)
        axes[3].set_xlabel('Gene Index', fontsize=11)
        axes[3].set_ylabel('Prediction Error', fontsize=11)
        axes[3].set_title('Prediction Error (Green: ≤5, Orange: ≤10, Red: >10 bins)', fontsize=12)
        axes[3].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_dir / f'prediction_cell_{cell_idx}_{pert_name}.png', dpi=150)
        plt.close()

        # Cell metrics
        cell_acc = (true_cell == pred_cell).mean()
        cell_mae = np.abs(pred_cell - true_cell).mean()
        print(f'\nCell {cell_idx} ({pert_name}) metrics:')
        print(f'  Accuracy: {cell_acc:.2%}')
        print(f'  MAE: {cell_mae:.2f} bins')


def main():
    args = parse_args()

    # Validate arguments
    if not args.checkpoint and not args.experiment_dir:
        raise ValueError("Must provide either --checkpoint or --experiment_dir")

    if not args.data_path:
        raise ValueError("Must provide --data_path")

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = find_checkpoint(args.experiment_dir)

    print(f"\nLoading checkpoint from: {checkpoint_path}")

    # Load data
    print(f"\nLoading perturbation-seq data from {args.data_path}")
    adata = sc.read_h5ad(args.data_path)
    print(f"Loaded {len(adata)} cells with {adata.n_vars} genes")

    # Check for perturbation column
    if args.pert_col not in adata.obs.columns:
        raise ValueError(
            f"Perturbation column '{args.pert_col}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    # Get perturbation labels and gene names
    pert_labels = adata.obs[args.pert_col].astype(str).to_numpy()
    gene_names = adata.var_names.tolist()

    print(f"Found {len(np.unique(pert_labels))} unique perturbations")

    # Convert expression to numpy array
    expression = adata.X
    if hasattr(expression, 'toarray'):
        expression = expression.toarray()
    else:
        expression = np.asarray(expression)

    # Calculate vocab size from data
    NUM_BINS = int(expression.max())
    NUM_GENES = expression.shape[1]
    VOCAB_SIZE = NUM_BINS + 1  # +1 for mask token

    print(f"\nData statistics:")
    print(f"  Number of genes: {NUM_GENES}")
    print(f"  Number of bins: {NUM_BINS}")
    print(f"  Vocabulary size: {VOCAB_SIZE}")

    # Create dataset
    dataset = PerturbSeqDataset(
        expression=expression,
        pert_labels=pert_labels,
        gene_names=gene_names,
        num_bins=NUM_BINS,
        control_pert_name=args.control_name,
    )

    NUM_PERTURBATIONS = dataset.num_perturbations

    # Split into train/val (we'll use val for inference)
    train_dataset, val_dataset = train_val_split(
        dataset,
        val_fraction=args.val_fraction,
        seed=args.seed
    )
    print(f"\nUsing validation set for inference: {len(val_dataset)} samples")

    # Create data loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    print(f"Number of inference batches: {len(val_loader)}")

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

    # Create optimizer (not used for inference, but needed for trainer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Create trainer and load checkpoint
    trainer = PerturbationTrainer(
        model=model,
        graph=graph,
        noise=noise,
        optimizer=optimizer,
        device=device,
        gradient_clip=1.0
    )

    print(f"Loading checkpoint...")
    trainer.load_checkpoint(checkpoint_path)
    print(f"Loaded model from epoch {trainer.epoch + 1}")

    # Set model to eval mode
    model.eval()

    # Run inference
    print("\nRunning inference...")
    all_predictions = []
    all_true = []
    all_controls = []
    all_pert_labels = []

    num_batches = args.num_batches if args.num_batches else len(val_loader)

    with torch.no_grad():
        for batch_idx, (control, pert_label, perturbed) in enumerate(val_loader):
            if batch_idx >= num_batches:
                break

            control = control.to(device)
            pert_label = pert_label.to(device)
            perturbed = perturbed.to(device)

            batch_size = control.shape[0]

            # Start from all masked
            x_init = torch.full((batch_size, NUM_GENES), NUM_BINS, device=device, dtype=torch.long)

            # Forward pass with low noise for simple inference
            sigma = torch.ones(batch_size, device=device) * 0.01
            logits = model(x_init, sigma, pert_label)
            predicted = logits.argmax(dim=-1)

            all_predictions.append(predicted.cpu())
            all_true.append(perturbed.cpu())
            all_controls.append(control.cpu())
            all_pert_labels.append(pert_label.cpu())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{num_batches} batches")

    # Concatenate all results
    all_predictions = torch.cat(all_predictions, dim=0)
    all_true = torch.cat(all_true, dim=0)
    all_controls = torch.cat(all_controls, dim=0)
    all_pert_labels = torch.cat(all_pert_labels, dim=0)

    print(f"\nInference complete! Processed {len(all_predictions)} cells")

    # Evaluate
    metrics = evaluate_predictions(all_predictions, all_true, dataset, args.num_cells_visualize)

    # Visualize examples
    print(f"\nVisualizing {args.num_cells_visualize} example predictions...")
    visualize_predictions(
        all_controls,
        all_true,
        all_predictions,
        all_pert_labels,
        dataset,
        args.output_dir,
        args.num_cells_visualize
    )

    # Save predictions if requested
    if args.save_predictions:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "predictions": all_predictions.numpy(),
            "true": all_true.numpy(),
            "controls": all_controls.numpy(),
            "pert_labels": all_pert_labels.numpy(),
            "metrics": metrics,
            "args": vars(args)
        }

        results_path = output_dir / "predictions.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nSaved predictions to {results_path}")

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
