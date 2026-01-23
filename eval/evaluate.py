"""
Main Evaluation Script for Single-Cell RNA-seq Model Comparison

This script provides a unified interface to evaluate both reconstruction and generation
metrics for single-cell RNA-seq models (scLDM, CFGen, scVI, scDiffusion, etc.).

Usage:
    python eval/evaluate.py --real data/real.npy --pred data/predicted.npy --mode both
    python eval/evaluate.py --real data/real.h5ad --pred data/predicted.h5ad --mode reconstruction

Reference: "Scalable Single-Cell Gene Expression Generation with Latent Diffusion Models"
https://arxiv.org/abs/2511.02986
"""

import argparse
import numpy as np
import json
from pathlib import Path
from typing import Optional, Union
import sys

# Import metric modules
from reconstruction_metrics import evaluate_reconstruction
from generation_metrics import evaluate_generation


def load_data(filepath: str) -> np.ndarray:
    """
    Load gene expression data from various formats.

    Supports:
    - .npy: NumPy array files
    - .npz: Compressed NumPy files (loads first array)
    - .h5ad: AnnData files (requires scanpy)
    - .csv/.tsv: Text files (cells x genes)

    Args:
        filepath: Path to data file

    Returns:
        NumPy array of shape (n_cells, n_genes)
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    # Load based on extension
    if filepath.suffix == '.npy':
        data = np.load(filepath)
    elif filepath.suffix == '.npz':
        with np.load(filepath) as npz_file:
            # Load first array in npz file
            key = list(npz_file.keys())[0]
            data = npz_file[key]
    elif filepath.suffix == '.h5ad':
        try:
            import scanpy as sc
            adata = sc.read_h5ad(filepath)
            # Get expression matrix
            data = adata.X
            # Convert sparse to dense if needed
            if hasattr(data, 'toarray'):
                data = data.toarray()
        except ImportError:
            raise ImportError("scanpy is required to load .h5ad files. Install with: pip install scanpy")
    elif filepath.suffix in ['.csv', '.tsv']:
        delimiter = ',' if filepath.suffix == '.csv' else '\t'
        data = np.loadtxt(filepath, delimiter=delimiter, skiprows=1)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

    # Ensure 2D array
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {data.shape}")

    return data


def save_results(results: dict, output_path: str, format: str = 'json'):
    """
    Save evaluation results to file.

    Args:
        results: Dictionary of metric results
        output_path: Path to save results
        format: Output format ('json' or 'txt')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'json':
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    elif format == 'txt':
        with open(output_path, 'w') as f:
            f.write("Single-Cell RNA-seq Evaluation Results\n")
            f.write("=" * 50 + "\n\n")

            if 'reconstruction' in results:
                f.write("Reconstruction Metrics:\n")
                f.write("-" * 50 + "\n")
                f.write(f"  RE (Reconstruction Error):  {results['reconstruction']['re']:.6f} ↓\n")
                f.write(f"  PCC (Pearson Correlation):  {results['reconstruction']['pcc']:.6f} ↑\n")
                f.write(f"  MSE (Mean Squared Error):   {results['reconstruction']['mse']:.6f} ↓\n")
                f.write("\n")

            if 'generation' in results:
                f.write("Generation Metrics:\n")
                f.write("-" * 50 + "\n")
                f.write(f"  W2 (Wasserstein-2):         {results['generation']['w2']:.6f} ↓\n")
                f.write(f"  MMD2 RBF:                   {results['generation']['mmd2_rbf']:.6f} ↓\n")
                f.write(f"  FD (Fréchet Distance):      {results['generation']['fd']:.6f} ↓\n")
                f.write("\n")

            f.write("↓ = lower is better, ↑ = higher is better\n")
    else:
        raise ValueError(f"Unsupported format: {format}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate single-cell RNA-seq model performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate both reconstruction and generation metrics
  python eval/evaluate.py --real data/real.npy --pred data/predicted.npy --mode both

  # Only evaluate reconstruction metrics (for autoencoders)
  python eval/evaluate.py --real data/real.npy --pred data/reconstructed.npy --mode reconstruction

  # Only evaluate generation metrics (for generative models)
  python eval/evaluate.py --real data/real.npy --pred data/generated.npy --mode generation

  # Load from AnnData format
  python eval/evaluate.py --real data/real.h5ad --pred data/predicted.h5ad --mode both

  # Save results to file
  python eval/evaluate.py --real data/real.npy --pred data/predicted.npy --output results.json
        """
    )

    # Required arguments
    parser.add_argument('--real', type=str, required=True,
                       help='Path to real/ground truth data')
    parser.add_argument('--pred', type=str, required=True,
                       help='Path to predicted/generated data')

    # Optional arguments
    parser.add_argument('--mode', type=str, default='both',
                       choices=['reconstruction', 'generation', 'both'],
                       help='Which metrics to compute (default: both)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save results (json or txt format)')
    parser.add_argument('--format', type=str, default='json',
                       choices=['json', 'txt'],
                       help='Output format (default: json)')

    # Generation metric parameters
    parser.add_argument('--w2-projections', type=int, default=1000,
                       help='Number of projections for W2 distance (default: 1000)')
    parser.add_argument('--mmd-subsample', type=int, default=2000,
                       help='Max samples for MMD calculation (default: 2000)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')

    # Display options
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')

    args = parser.parse_args()

    # Load data
    print(f"Loading real data from: {args.real}")
    real_data = load_data(args.real)
    print(f"  Shape: {real_data.shape}")

    print(f"Loading predicted data from: {args.pred}")
    pred_data = load_data(args.pred)
    print(f"  Shape: {pred_data.shape}")

    # Validate shapes
    if real_data.shape != pred_data.shape:
        print(f"\nWarning: Shape mismatch!")
        print(f"  Real: {real_data.shape}")
        print(f"  Predicted: {pred_data.shape}")

        if args.mode == 'reconstruction':
            raise ValueError("Reconstruction mode requires identical shapes")

        if real_data.shape[1] != pred_data.shape[1]:
            raise ValueError(f"Feature dimension mismatch: {real_data.shape[1]} vs {pred_data.shape[1]}")

        print(f"  Proceeding with generation metrics (different number of cells is OK)")

    print()

    # Store results
    results = {}

    # Evaluate reconstruction metrics
    if args.mode in ['reconstruction', 'both']:
        if real_data.shape != pred_data.shape:
            print("Skipping reconstruction metrics due to shape mismatch")
        else:
            print("=" * 60)
            print("RECONSTRUCTION METRICS")
            print("=" * 60)
            recon_results = evaluate_reconstruction(
                real_data, pred_data, verbose=not args.quiet
            )
            results['reconstruction'] = recon_results

    # Evaluate generation metrics
    if args.mode in ['generation', 'both']:
        print("=" * 60)
        print("GENERATION METRICS")
        print("=" * 60)
        gen_results = evaluate_generation(
            real_data, pred_data,
            w2_projections=args.w2_projections,
            mmd_subsample=args.mmd_subsample,
            seed=args.seed,
            verbose=not args.quiet
        )
        results['generation'] = gen_results

    # Save results if output path specified
    if args.output:
        save_results(results, args.output, args.format)
        print(f"\nResults saved to: {args.output}")

    return results


if __name__ == "__main__":
    main()
