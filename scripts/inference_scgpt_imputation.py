#!/usr/bin/env python3
"""
Inference script for imputation using scGPT model.

This script loads a pre-trained scGPT model and performs gene expression imputation
by masking some genes and predicting their values.

Note: The models/ directory is typically a symlink to GPU storage. Place your
scGPT model files in models/scGPT/ on the GPU node where they are accessible.
"""

import argparse
import json
import os
import pickle
from pathlib import Path
import sys

import torch
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

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


def load_scgpt_model(model_dir, device):
    """
    Load scGPT model from directory containing model files.

    Expected files in model_dir:
    - args.json: model configuration
    - best_model.pt or best_model.ckpt: model weights
    - vocab.json: vocabulary
    - var_dims.pkl: variable dimensions (optional)
    - pert_one-hot-map.pt: perturbation mapping (optional)
    """
    model_dir = Path(model_dir)

    # Load configuration
    args_path = model_dir / "args.json"
    if not args_path.exists():
        raise ValueError(f"args.json not found in {model_dir}")

    with open(args_path, "r") as f:
        model_args = json.load(f)
    print(f"Loaded model configuration from {args_path}")

    # Load vocabulary
    vocab_path = model_dir / "vocab.json"
    if not vocab_path.exists():
        raise ValueError(f"vocab.json not found in {model_dir}")

    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    print(f"Loaded vocabulary: {len(vocab)} genes")

    # Load variable dimensions if available
    var_dims_path = model_dir / "var_dims.pkl"
    if var_dims_path.exists():
        with open(var_dims_path, "rb") as f:
            var_dims = pickle.load(f)
        print(f"Loaded variable dimensions")
    else:
        var_dims = None

    # Try to import scGPT
    try:
        from scgpt.model import TransformerModel
        from scgpt.tokenizer import tokenize_and_pad_batch
        from scgpt.preprocess import Preprocessor
    except ImportError:
        raise ImportError(
            "scGPT not installed. Please install it with: pip install scgpt"
        )

    # Load model checkpoint
    checkpoint_path = model_dir / "best_model.pt"
    if not checkpoint_path.exists():
        checkpoint_path = model_dir / "best_model.ckpt"
    if not checkpoint_path.exists():
        raise ValueError(f"No model checkpoint found in {model_dir}")

    print(f"Loading model from {checkpoint_path}")

    # Create model based on args
    model = TransformerModel(
        ntoken=model_args.get("ntoken", len(vocab) + 1),
        d_model=model_args.get("d_model", 512),
        nhead=model_args.get("nhead", 8),
        d_hid=model_args.get("d_hid", 512),
        nlayers=model_args.get("nlayers", 12),
        dropout=model_args.get("dropout", 0.2),
        pad_token=model_args.get("pad_token", "<pad>"),
        pad_value=model_args.get("pad_value", 0),
        do_mvc=model_args.get("do_mvc", True),
        do_dab=model_args.get("do_dab", False),
        use_batch_labels=model_args.get("use_batch_labels", False),
        domain_spec_batchnorm=model_args.get("domain_spec_batchnorm", False),
        n_input_bins=model_args.get("n_input_bins", 51),
    )

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully!")

    return model, model_args, vocab


def impute_with_scgpt(model, batch, mask, device, num_steps=1):
    """
    Perform imputation using scGPT model.

    Args:
        model: scGPT model
        batch: Input data tensor [batch_size, num_genes]
        mask: Boolean mask indicating positions to impute [batch_size, num_genes]
        device: torch device
        num_steps: Number of refinement steps (scGPT typically uses 1)

    Returns:
        Imputed data tensor [batch_size, num_genes]
    """
    with torch.no_grad():
        # Create masked input
        masked_batch = batch.clone()
        masked_batch[mask] = 0  # Mask token is typically 0

        # Forward pass through model
        # scGPT returns predictions for masked positions
        output = model(
            masked_batch.to(device),
            src_key_padding_mask=None,
            batch_labels=None
        )

        # Get predictions (output is typically a dict with 'mlm_output' for masked language modeling)
        if isinstance(output, dict):
            predictions = output.get('mlm_output', output.get('pred', output))
        else:
            predictions = output

        # Take argmax to get discrete predictions
        if predictions.dim() == 3:  # [batch, seq, vocab]
            predictions = predictions.argmax(dim=-1)

        # Combine original and imputed values
        result = batch.clone()
        result[mask] = predictions[mask]

        return result


def parse_args():
    # Load config file
    config = load_config()
    data_config = config.get("data", {})

    parser = argparse.ArgumentParser(description="Run imputation inference with scGPT model")

    # Model arguments
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/scGPT",
        help="Path to directory containing scGPT model files"
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
        default=1,
        help="Number of refinement steps (scGPT typically uses 1)"
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

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (defaults to model_dir/imputation_results)"
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

    # Load scGPT model
    print(f"\nLoading scGPT model from {args.model_dir}")
    model, model_args, vocab = load_scgpt_model(args.model_dir, device)

    # Create output directory for results
    model_dir = Path(args.model_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = model_dir / "imputation_results"
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Results will be saved to: {output_dir}")

    # Load data
    if not args.data_path:
        raise ValueError("No data_path provided. Set it in config.toml or pass --data_path")

    print(f"\nLoading data from {args.data_path}")
    adata = sc.read_h5ad(args.data_path)
    print(f"Loaded {len(adata)} cells with {adata.n_vars} genes")

    # Convert to tensor
    if hasattr(adata.X, 'toarray'):
        expression = adata.X.toarray()
    else:
        expression = adata.X

    dataset = torch.tensor(expression).long()

    # Calculate vocab size
    NUM_BINS = int(dataset.max().item())
    NUM_GENES = dataset.shape[1]
    VOCAB_SIZE = NUM_BINS + 1

    print(f"Number of genes: {NUM_GENES}")
    print(f"Number of bins: {NUM_BINS}")
    print(f"Vocabulary size: {VOCAB_SIZE}")

    # Split into train/val
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=args.val_fraction,
        random_state=args.seed
    )

    # Choose which split to use
    if args.use_train_split:
        eval_idx = train_idx
        split_name = "train"
    else:
        eval_idx = val_idx
        split_name = "val"

    eval_dataset = dataset[eval_idx]
    print(f"Using {split_name} split: {len(eval_dataset)} cells")

    # Create data loader
    test_loader = DataLoader(
        TensorDataset(eval_dataset),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    print(f"Number of batches: {len(test_loader)}")

    # Run imputation
    print(f"\nRunning imputation with mask_ratio={args.mask_ratio}")

    all_original_masked = []
    all_predicted_masked = []
    sample_cells_original = []
    sample_cells_imputed = []
    sample_cells_masks = []

    num_batches = args.num_batches or len(test_loader)

    for i, (batch,) in enumerate(tqdm(test_loader, total=num_batches, desc="Imputing")):
        if i >= num_batches:
            break

        batch = batch.to(device)

        # Create random mask
        mask = torch.rand_like(batch.float()) < args.mask_ratio

        # Impute masked positions
        imputed = impute_with_scgpt(
            model=model,
            batch=batch,
            mask=mask,
            device=device,
            num_steps=args.num_steps
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
    print("Imputation Metrics (scGPT)")
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
        "model": "scGPT",
        "accuracy": accuracy,
        "mae_bins": mae,
        "within_k": within_k_metrics,
        "mask_ratio": args.mask_ratio,
        "num_steps": args.num_steps,
        "num_masked_positions": len(all_original_masked),
        "model_dir": str(args.model_dir)
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
    axes[0].set_title('scGPT Imputation: Predicted vs True')
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
        axes[1].set_title('scGPT Imputed Expression')
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
    print(f"scGPT Imputation complete! Results saved to {output_dir}")
    print("="*50)


if __name__ == "__main__":
    main()
