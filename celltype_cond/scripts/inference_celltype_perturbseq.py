#!/usr/bin/env python3
"""
Inference script for cell-type conditioned perturbation prediction.

This script loads a trained cell-type conditioned model and generates
perturbed cell states for a specific cell type.

Key feature: Specify cell_type in YAML config to generate perturbed cells
for one cell type at a time. This allows systematic generation of
cell-type-specific perturbation responses.

Usage:
    python inference_celltype_perturbseq.py --config configs/celltype_inference.yaml \\
        --experiment_dir experiments/celltype_model \\
        --cell_type "T cell"
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import yaml

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from celltype_cond.sedd.model import (
    SEDDCellTypePerturbationTransformerSmall,
    SEDDCellTypePerturbationTransformerMedium,
    SEDDCellTypePerturbationTransformerLarge,
)
from celltype_cond.sedd.sampling import CellTypePerturbationEulerSampler
from sedd.graph import AbsorbingGraph
from sedd.noise import LogLinearNoise


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
    """Find the best or final checkpoint in an experiment directory."""
    experiment_dir = Path(experiment_dir)

    if not experiment_dir.exists():
        raise ValueError(f"Experiment directory not found: {experiment_dir}")

    # Priority: best > final > latest epoch
    best_ckpt = experiment_dir / "best.pt"
    if best_ckpt.exists():
        print(f"Found best checkpoint: {best_ckpt}")
        return best_ckpt

    final_ckpt = experiment_dir / "final.pt"
    if final_ckpt.exists():
        print(f"Found final checkpoint: {final_ckpt}")
        return final_ckpt

    checkpoints = list(experiment_dir.glob("epoch_*.pt"))
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.stem.split("_")[1]))
        latest = checkpoints[-1]
        print(f"Found latest epoch checkpoint: {latest}")
        return latest

    raise ValueError(f"No checkpoints found in {experiment_dir}")


def load_conditional_labels(pt_path, names_list):
    """Load conditional labels from .pt file."""
    if pt_path is None:
        return None, []

    print(f"\nLoading conditional labels from: {pt_path}")
    pt_data = torch.load(pt_path, map_location="cpu")

    if not isinstance(pt_data, dict):
        raise TypeError(f"Expected .pt file to contain a dictionary, got {type(pt_data)}")

    label_lookup = []
    missing = []
    for name in names_list:
        if name in pt_data:
            label_val = pt_data[name]
            if isinstance(label_val, torch.Tensor):
                label_lookup.append(label_val.cpu())
            else:
                label_lookup.append(torch.tensor(label_val))
        else:
            label_lookup.append(torch.tensor(-1))
            missing.append(name)

    if len(label_lookup) > 0:
        label_lookup = torch.stack(label_lookup)
    else:
        label_lookup = None

    return label_lookup, missing


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run cell-type conditioned perturbation prediction inference"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file"
    )

    args, remaining = parser.parse_known_args()
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
        default=data_config.get("data_path", data_config.get("test_data_path", None)),
        help="Path to h5ad file containing test data"
    )
    parser.add_argument(
        "--gene",
        type=str,
        default=data_config.get("gene", "gene"),
        help="Column name for perturbation labels"
    )
    parser.add_argument(
        "--cell_type_col",
        type=str,
        default=data_config.get("cell_type_col", "cell_type"),
        help="Column name for cell-type labels"
    )
    parser.add_argument(
        "--control_name",
        type=str,
        default=data_config.get("control_name", "non-targeting"),
        help="Name of control perturbation"
    )

    # Cell-type filtering - THE KEY FEATURE
    parser.add_argument(
        "--cell_type",
        type=str,
        default=inference_config.get("cell_type", None),
        help="Specific cell type to generate predictions for. If not specified, generates for all cell types."
    )

    # Perturbation filtering
    parser.add_argument(
        "--perturbations",
        type=str,
        nargs="+",
        default=inference_config.get("perturbations", None),
        help="Specific perturbations to generate. If not specified, uses all."
    )
    parser.add_argument(
        "--perturbations_file",
        type=str,
        default=inference_config.get("perturbations_file", None),
        help="File containing perturbation names (one per line)"
    )

    # Inference arguments
    parser.add_argument(
        "--num_samples_per_condition",
        type=int,
        default=inference_config.get("num_samples_per_condition", 100),
        help="Number of samples to generate per (perturbation, cell_type) combination"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=inference_config.get("num_steps", 50),
        help="Number of sampling steps"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=inference_config.get("temperature", 1.0),
        help="Sampling temperature"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=inference_config.get("batch_size", 32),
        help="Batch size for inference"
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default=inference_config.get("output_dir", None),
        help="Output directory (default: experiment_dir/inference_results)"
    )
    parser.add_argument(
        "--save_h5ad",
        action="store_true",
        default=inference_config.get("save_h5ad", True),
        help="Save results as h5ad file"
    )

    # Conditional labels
    parser.add_argument(
        "--cond_labels_pt_path",
        type=str,
        default=data_config.get("cond_labels_pt_path", None),
        help="Path to perturbation conditional labels .pt file"
    )
    parser.add_argument(
        "--celltype_labels_pt_path",
        type=str,
        default=data_config.get("celltype_labels_pt_path", None),
        help="Path to cell-type conditional labels .pt file"
    )

    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=other_config.get("seed", 42),
        help="Random seed"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Find checkpoint
    experiment_dir = Path(args.experiment_dir)
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = find_checkpoint(experiment_dir)

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = experiment_dir / "inference_results"

    # Include cell type in output directory if specified
    if args.cell_type:
        safe_celltype = args.cell_type.replace(" ", "_").replace("/", "_")
        output_dir = output_dir / f"celltype_{safe_celltype}"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # Load training configuration
    args_file = experiment_dir / "args.json"
    if not args_file.exists():
        raise ValueError(f"Training config not found: {args_file}")

    with open(args_file, "r") as f:
        train_config = json.load(f)

    print(f"\nLoaded training configuration from {args_file}")
    print(f"Model size: {train_config.get('model_size', 'unknown')}")
    print(f"Trained cell types: {train_config.get('cell_types', 'unknown')}")

    # Get model dimensions from training config
    NUM_GENES = train_config["num_genes"]
    NUM_BINS = train_config["num_bins"]
    NUM_PERTURBATIONS = train_config["num_perturbations"]
    NUM_CELL_TYPES = train_config["num_cell_types"]
    VOCAB_SIZE = NUM_BINS + 1

    print(f"\nModel dimensions:")
    print(f"  Genes: {NUM_GENES}")
    print(f"  Bins: {NUM_BINS}")
    print(f"  Perturbations: {NUM_PERTURBATIONS}")
    print(f"  Cell types: {NUM_CELL_TYPES}")

    # Get cell type and perturbation mappings from training config
    trained_cell_types = train_config.get("cell_types", [])
    trained_perturbations = train_config.get("perturbations", [])

    celltype_to_idx = {ct: i for i, ct in enumerate(trained_cell_types)}
    pert_to_idx = {p: i for i, p in enumerate(trained_perturbations)}

    # Validate requested cell type
    if args.cell_type:
        if args.cell_type not in celltype_to_idx:
            print(f"\nERROR: Requested cell type '{args.cell_type}' not found in trained cell types.")
            print(f"Available cell types: {trained_cell_types}")
            return
        target_cell_types = [args.cell_type]
        print(f"\nGenerating predictions for cell type: {args.cell_type}")
    else:
        target_cell_types = trained_cell_types
        print(f"\nGenerating predictions for all {len(target_cell_types)} cell types")

    # Get perturbations to generate
    if args.perturbations_file:
        with open(args.perturbations_file, "r") as f:
            target_perturbations = [line.strip() for line in f if line.strip()]
    elif args.perturbations:
        target_perturbations = args.perturbations
    else:
        target_perturbations = trained_perturbations

    # Filter to valid perturbations
    valid_perturbations = [p for p in target_perturbations if p in pert_to_idx]
    if len(valid_perturbations) < len(target_perturbations):
        missing = set(target_perturbations) - set(valid_perturbations)
        print(f"\nWARNING: {len(missing)} perturbations not in trained model: {list(missing)[:5]}...")
    target_perturbations = valid_perturbations

    print(f"Generating for {len(target_perturbations)} perturbations")

    # Load conditional labels
    cond_label_lookup = None
    if args.cond_labels_pt_path:
        cond_label_lookup, _ = load_conditional_labels(
            args.cond_labels_pt_path,
            trained_perturbations
        )
        if cond_label_lookup is not None:
            cond_label_lookup = cond_label_lookup.to(device)

    celltype_label_lookup = None
    if args.celltype_labels_pt_path:
        celltype_label_lookup, _ = load_conditional_labels(
            args.celltype_labels_pt_path,
            trained_cell_types
        )
        if celltype_label_lookup is not None:
            celltype_label_lookup = celltype_label_lookup.to(device)

    # Create model
    print("\nCreating model...")
    model_size = train_config.get("model_size", "small")
    model_classes = {
        "small": SEDDCellTypePerturbationTransformerSmall,
        "medium": SEDDCellTypePerturbationTransformerMedium,
        "large": SEDDCellTypePerturbationTransformerLarge,
    }
    ModelClass = model_classes.get(model_size, SEDDCellTypePerturbationTransformerSmall)

    # Infer precomputed embedding dimensions
    precomputed_pert_emb_dim = None
    if cond_label_lookup is not None and cond_label_lookup.dim() == 2:
        precomputed_pert_emb_dim = cond_label_lookup.shape[1]

    precomputed_celltype_emb_dim = None
    if celltype_label_lookup is not None and celltype_label_lookup.dim() == 2:
        precomputed_celltype_emb_dim = celltype_label_lookup.shape[1]

    model = ModelClass(
        num_genes=NUM_GENES,
        num_bins=NUM_BINS,
        num_perturbations=NUM_PERTURBATIONS,
        num_cell_types=NUM_CELL_TYPES,
        hidden_dim=train_config.get("hidden_dim", 128),
        num_layers=train_config.get("num_layers", 4),
        num_heads=train_config.get("num_heads", 4),
        dropout=train_config.get("dropout", 0.1),
        max_seq_len=NUM_GENES,
        precomputed_pert_emb_dim=precomputed_pert_emb_dim,
        precomputed_celltype_emb_dim=precomputed_celltype_emb_dim,
    ).to(device)

    # Load checkpoint
    print(f"\nLoading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model loaded! Trained for {checkpoint.get('epoch', 'unknown') + 1} epochs.")

    # Create graph, noise, and sampler
    graph = AbsorbingGraph(num_states=VOCAB_SIZE)
    noise = LogLinearNoise(eps=1e-3)

    sampler = CellTypePerturbationEulerSampler(
        model=model,
        graph=graph,
        noise=noise,
        num_steps=args.num_steps,
        temperature=args.temperature,
        device=device,
    )

    # Generate predictions
    print(f"\nGenerating {args.num_samples_per_condition} samples per condition...")
    print(f"Total conditions: {len(target_cell_types)} cell types x {len(target_perturbations)} perturbations")

    all_generated = []
    all_metadata = []

    model.eval()
    mask_idx = graph.mask_index

    for cell_type in tqdm(target_cell_types, desc="Cell types"):
        celltype_idx = celltype_to_idx[cell_type]

        for pert in tqdm(target_perturbations, desc=f"Perturbations ({cell_type})", leave=False):
            pert_idx = pert_to_idx[pert]

            # Generate samples in batches
            num_remaining = args.num_samples_per_condition
            generated_for_condition = []

            while num_remaining > 0:
                batch_size = min(args.batch_size, num_remaining)

                # Create initial masked sequence
                x_init = torch.full((batch_size, NUM_GENES), mask_idx, dtype=torch.long, device=device)

                # Create perturbation and cell-type labels
                pert_labels = torch.full((batch_size,), pert_idx, dtype=torch.long, device=device)
                celltype_labels = torch.full((batch_size,), celltype_idx, dtype=torch.long, device=device)

                # Apply conditional label lookups if provided
                if cond_label_lookup is not None:
                    pert_labels = cond_label_lookup[pert_labels]
                if celltype_label_lookup is not None:
                    celltype_labels = celltype_label_lookup[celltype_labels]

                # Sample
                with torch.no_grad():
                    generated = sampler.sample(
                        x_init,
                        pert_labels=pert_labels,
                        celltype_labels=celltype_labels,
                        show_progress=False,
                    )

                generated_for_condition.append(generated.cpu().numpy())
                num_remaining -= batch_size

            # Combine all samples for this condition
            generated_for_condition = np.concatenate(generated_for_condition, axis=0)
            all_generated.append(generated_for_condition)

            # Store metadata
            for _ in range(args.num_samples_per_condition):
                all_metadata.append({
                    "perturbation": pert,
                    "cell_type": cell_type,
                    "pert_idx": pert_idx,
                    "celltype_idx": celltype_idx,
                })

    # Combine all generated samples
    all_generated = np.concatenate(all_generated, axis=0)
    print(f"\nGenerated {len(all_generated)} total samples")

    # Save results
    if args.save_h5ad:
        import pandas as pd

        # Create AnnData object
        adata_gen = sc.AnnData(
            X=all_generated,
            obs=pd.DataFrame(all_metadata),
        )
        adata_gen.obs["generated"] = True

        # Save
        h5ad_path = output_dir / "generated_cells.h5ad"
        adata_gen.write_h5ad(h5ad_path)
        print(f"Saved generated cells to: {h5ad_path}")

    # Save generation config
    gen_config = {
        "experiment_dir": str(experiment_dir),
        "checkpoint": str(checkpoint_path),
        "cell_type": args.cell_type,
        "target_cell_types": target_cell_types,
        "num_perturbations": len(target_perturbations),
        "num_samples_per_condition": args.num_samples_per_condition,
        "num_steps": args.num_steps,
        "temperature": args.temperature,
        "total_samples": len(all_generated),
        "seed": args.seed,
    }
    with open(output_dir / "generation_config.json", "w") as f:
        json.dump(gen_config, f, indent=2)

    # Summary statistics
    print("\n" + "=" * 50)
    print("Generation Summary")
    print("=" * 50)
    print(f"Cell types: {len(target_cell_types)}")
    print(f"Perturbations: {len(target_perturbations)}")
    print(f"Samples per condition: {args.num_samples_per_condition}")
    print(f"Total samples: {len(all_generated)}")
    print(f"Output directory: {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
