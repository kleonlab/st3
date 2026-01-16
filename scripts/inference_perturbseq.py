#!/usr/bin/env python3
"""
Inference script for perturbation prediction using trained SEDD model.

This script loads a trained perturbation prediction model and evaluates it
on test data, predicting perturbed cell states from control cells + perturbation labels.

Task: control cell + perturbation label -> perturbed cell
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

from sedd.model import SEDDPerturbationTransformerSmall
from sedd.graph import AbsorbingGraph
from sedd.noise import LogLinearNoise
from sedd.trainer import PerturbationTrainer
from sedd.data import PerturbSeqDataset, train_val_split
from sedd.sampling import PerturbationEulerSampler

import yaml


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
        description="Run perturbation prediction inference with trained SEDD model"
    )

    # Config file argument (parsed first)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (e.g., configs/perturbseq_dry_run.yaml)"
    )

    # Parse args once to get config file
    args, remaining = parser.parse_known_args()

    # Load config file
    config = load_yaml_config(args.config)
    data_config = config.get("data", {})
    inference_config = config.get("inference", {})
    other_config = config.get("other", {})

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
        default=data_config.get("test_data_path", data_config.get("data_path", None)),
        help="Path to h5ad file containing test perturbation-seq data"
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
        help="Validation fraction (used if test_data not separate)"
    )
    parser.add_argument(
        "--use_train_split",
        action="store_true",
        help="Use training split instead of validation split"
    )

    # Inference arguments
    parser.add_argument(
        "--num_steps",
        type=int,
        default=inference_config.get("num_steps", 50),
        help="Number of sampling steps for prediction"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=inference_config.get("temperature", 1.0),
        help="Sampling temperature"
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default=inference_config.get("sampler", "euler"),
        choices=["euler"],
        help="Sampler to use for prediction"
    )

    # Evaluation arguments
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
        help="Number of batches to evaluate (None = all)"
    )
    parser.add_argument(
        "--num_cells_visualize",
        type=int,
        default=inference_config.get("num_cells_visualize", 3),
        help="Number of individual cells to visualize"
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
    output_dir = experiment_dir / "perturbseq_inference_results"
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

    # Load test data
    if not args.data_path:
        raise ValueError(
            "No data_path provided. Set it in config or pass --data_path argument"
        )

    print(f"\nLoading test data from {args.data_path}")
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
    print(f"Found {len(np.unique(pert_labels))} unique perturbations in test data")

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

    # Create dataset
    dataset = PerturbSeqDataset(
        expression=expression,
        pert_labels=pert_labels,
        num_bins=NUM_BINS,
        control_pert_name=args.control_name,
    )

    NUM_PERTURBATIONS = dataset.num_perturbations
    print(f"Number of perturbations: {NUM_PERTURBATIONS}")

    # Create data loader (use all test data)
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    print(f"Number of test batches: {len(test_loader)}")

    # Create model
    print("\nCreating perturbation prediction model...")
    model = SEDDPerturbationTransformerSmall(
        num_genes=NUM_GENES,
        num_bins=NUM_BINS,
        num_perturbations=NUM_PERTURBATIONS,
        hidden_dim=train_config.get("hidden_dim", 128),
        num_layers=train_config.get("num_layers", 4),
        num_heads=train_config.get("num_heads", 4),
        dropout=train_config.get("dropout", 0.1),
        max_seq_len=NUM_GENES
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create graph and noise schedule
    graph = AbsorbingGraph(num_states=VOCAB_SIZE)
    noise = LogLinearNoise(eps=1e-3)

    # Create trainer and load checkpoint
    trainer = PerturbationTrainer(
        model=model,
        graph=graph,
        noise=noise,
        device=device
    )

    print(f"\nLoading checkpoint from {checkpoint_path}")
    trainer.load_checkpoint(checkpoint_path)
    print(f"Model loaded! Trained for {trainer.epoch + 1} epochs.")

    # Run inference
    print(f"\nRunning perturbation prediction with num_steps={args.num_steps}, "
          f"temperature={args.temperature}")

    model.eval()

    all_predictions = []
    all_targets = []
    sample_controls = []
    sample_predictions = []
    sample_targets = []
    sample_pert_labels = []

    num_batches = args.num_batches or len(test_loader)

    # Create sampler with perturbation conditioning
    sampler = PerturbationEulerSampler(
        model=model,
        graph=graph,
        noise=noise,
        num_steps=args.num_steps,
        temperature=args.temperature,
        device=device
    )

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, total=num_batches, desc="Predicting")):
            if i >= num_batches:
                break

            control, pert_label, target = batch
            control = control.to(device)
            pert_label = pert_label.to(device)
            target = target.to(device)

            # Note: control cells are loaded but not currently used for conditioning.
            # Future enhancement: could incorporate control cell state into the generation process.

            # Initialize with fully masked sequence
            batch_size = control.size(0)
            mask_idx = graph.mask_index
            x_init = torch.full_like(target, mask_idx)

            # Sample from the model WITH perturbation conditioning
            predicted = sampler.sample(
                x_init,
                pert_labels=pert_label,
                show_progress=False
            )

            # Collect results
            all_predictions.append(predicted.cpu())
            all_targets.append(target.cpu())

            # Save some cells for visualization
            if len(sample_controls) < args.num_cells_visualize:
                for j in range(min(args.num_cells_visualize - len(sample_controls), batch_size)):
                    sample_controls.append(control[j].cpu())
                    sample_predictions.append(predicted[j].cpu())
                    sample_targets.append(target[j].cpu())
                    sample_pert_labels.append(pert_label[j].cpu().item())

    # Concatenate all results
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)

    print(f"\nTotal cells evaluated: {len(all_predictions)}")

    # Calculate metrics
    print("\n" + "="*50)
    print("Perturbation Prediction Metrics")
    print("="*50)

    # Overall accuracy (exact match across all genes)
    exact_match = (all_predictions == all_targets).all(dim=1).float().mean().item()
    print(f"Exact cell match: {exact_match:.2%}")

    # Per-gene accuracy
    per_gene_acc = (all_predictions == all_targets).float().mean().item()
    print(f"Per-gene accuracy: {per_gene_acc:.2%}")

    # MAE
    mae = (all_predictions - all_targets).abs().float().mean().item()
    print(f"Mean Absolute Error (bins): {mae:.2f}")

    # Within k bins
    within_k_metrics = {}
    for k in [1, 3, 5, 10]:
        within_k = ((all_predictions - all_targets).abs() <= k).float().mean().item()
        within_k_metrics[k] = within_k
        print(f"Within {k} bins: {within_k:.2%}")

    # Save metrics
    print(f"\nSaving metrics to {output_dir}")
    metrics = {
        "exact_cell_match": exact_match,
        "per_gene_accuracy": per_gene_acc,
        "mae_bins": mae,
        "within_k": within_k_metrics,
        "num_steps": args.num_steps,
        "temperature": args.temperature,
        "num_cells": len(all_predictions),
        "checkpoint": str(checkpoint_path)
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Visualizations
    print("\nGenerating visualizations...")

    # 1. Overall scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(
        all_targets.flatten().numpy(),
        all_predictions.flatten().numpy(),
        alpha=0.1,
        s=5
    )
    axes[0].plot([0, NUM_BINS], [0, NUM_BINS], 'r--', label='Perfect prediction')
    axes[0].set_xlabel('True Expression Bin')
    axes[0].set_ylabel('Predicted Expression Bin')
    axes[0].set_title('Perturbation Prediction: Predicted vs True')
    axes[0].legend()

    errors = (all_predictions - all_targets).flatten().numpy()
    axes[1].hist(errors, bins=50, alpha=0.7)
    axes[1].axvline(0, color='r', linestyle='--', label='Zero error')
    axes[1].set_xlabel('Prediction Error (bins)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Prediction Error Distribution')
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(output_dir / "prediction_scatter_hist.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Individual cell visualizations
    for cell_idx in range(len(sample_controls)):
        control_cell = sample_controls[cell_idx].numpy()
        predicted_cell = sample_predictions[cell_idx].numpy()
        target_cell = sample_targets[cell_idx].numpy()
        pert_label = sample_pert_labels[cell_idx]

        fig, axes = plt.subplots(3, 1, figsize=(14, 9))

        gene_indices = np.arange(len(control_cell))

        # Control
        axes[0].bar(gene_indices, control_cell, alpha=0.7, label='Control', width=1.0)
        axes[0].set_xlabel('Gene Index')
        axes[0].set_ylabel('Expression Bin')
        axes[0].set_title(f'Control Cell (Perturbation: {pert_label})')
        axes[0].legend()

        # Predicted
        axes[1].bar(gene_indices, predicted_cell, alpha=0.7, label='Predicted', width=1.0, color='orange')
        axes[1].set_xlabel('Gene Index')
        axes[1].set_ylabel('Expression Bin')
        axes[1].set_title('Predicted Perturbed Cell')
        axes[1].legend()

        # Target
        axes[2].bar(gene_indices, target_cell, alpha=0.7, label='Target', width=1.0, color='green')
        axes[2].set_xlabel('Gene Index')
        axes[2].set_ylabel('Expression Bin')
        axes[2].set_title('True Perturbed Cell')
        axes[2].legend()

        plt.tight_layout()
        fig.savefig(output_dir / f"prediction_cell_{cell_idx}.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Cell-specific metrics
        cell_acc = (predicted_cell == target_cell).mean()
        cell_mae = np.abs(predicted_cell - target_cell).mean()
        print(f"Cell {cell_idx} (pert={pert_label}) - Accuracy: {cell_acc:.2%}, MAE: {cell_mae:.2f} bins")

    print("\n" + "="*50)
    print(f"Perturbation prediction complete! Results saved to {output_dir}")
    print("="*50)


if __name__ == "__main__":
    main()
