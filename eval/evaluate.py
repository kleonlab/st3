"""
Main Evaluation Script for Single-Cell RNA-seq Model Comparison

This script evaluates both reconstruction and generation metrics for
single-cell RNA-seq models (scLDM, CFGen, scVI, scDiffusion, etc.).

Simply set the file paths and parameters in the CONFIG section below and run:
    python eval/evaluate.py

Reference: "Scalable Single-Cell Gene Expression Generation with Latent Diffusion Models"
https://arxiv.org/abs/2511.02986
"""

import argparse
import numpy as np
import json
from pathlib import Path
from typing import Optional, Union, Tuple, Dict
import sys

# Import metric modules
from reconstruction_metrics import evaluate_reconstruction
from generation_metrics import evaluate_generation


# ============================================================================
# CONFIGURATION - Set your file paths and parameters here
# ============================================================================

CONFIG = {
    # Input files (h5ad, npy, npz, csv formats supported)
    'real_data_path': 'data/real.h5ad',
    'predicted_data_path': 'data/predicted.h5ad',

    # Evaluation mode: 'reconstruction', 'generation', or 'both'
    'mode': 'both',

    # Output settings (None to skip saving)
    'output_path': 'results/evaluation_results.json',
    'output_format': 'json',  # 'json' or 'txt'

    # AnnData specific settings (for h5ad files)
    'use_layer': None,  # Use specific layer (e.g., 'counts'), None for .X
    'filter_by_perturbation': None,  # Filter specific perturbation (e.g., 'gene1'), None for all
    'filter_by_gene': None,  # Filter specific gene label, None for all

    # Generation metric parameters
    'w2_projections': 1000,  # Number of projections for Wasserstein distance
    'mmd_subsample': 2000,   # Max samples for MMD calculation (None for all)
    'random_seed': 42,        # Random seed for reproducibility

    # Display options
    'verbose': True,  # Print detailed results
}

# ============================================================================


def load_h5ad_with_labels(
    filepath: str,
    use_layer: Optional[str] = None,
    filter_by_perturbation: Optional[str] = None,
    filter_by_gene: Optional[str] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Load AnnData file with support for gene/perturbation labels.

    Args:
        filepath: Path to .h5ad file
        use_layer: Specific layer to use (None for .X)
        filter_by_perturbation: Filter cells by perturbation label
        filter_by_gene: Filter cells by gene label

    Returns:
        Tuple of (expression_matrix, metadata_dict)
    """
    try:
        import scanpy as sc
    except ImportError:
        raise ImportError("scanpy is required to load .h5ad files. Install with: pip install scanpy")

    # Load AnnData
    adata = sc.read_h5ad(filepath)

    # Store metadata
    metadata = {
        'n_cells': adata.n_obs,
        'n_genes': adata.n_vars,
        'obs_keys': list(adata.obs.columns),
        'var_keys': list(adata.var.columns),
        'layers': list(adata.layers.keys()) if len(adata.layers) > 0 else []
    }

    # Apply filters if specified
    if filter_by_perturbation is not None:
        if 'perturbation' in adata.obs.columns:
            adata = adata[adata.obs['perturbation'] == filter_by_perturbation].copy()
            print(f"  Filtered by perturbation '{filter_by_perturbation}': {adata.n_obs} cells")
        elif 'perturbation_label' in adata.obs.columns:
            adata = adata[adata.obs['perturbation_label'] == filter_by_perturbation].copy()
            print(f"  Filtered by perturbation '{filter_by_perturbation}': {adata.n_obs} cells")
        else:
            print(f"  Warning: No 'perturbation' column found in obs. Available columns: {list(adata.obs.columns)}")

    if filter_by_gene is not None:
        if 'gene' in adata.obs.columns:
            adata = adata[adata.obs['gene'] == filter_by_gene].copy()
            print(f"  Filtered by gene '{filter_by_gene}': {adata.n_obs} cells")
        elif 'gene_label' in adata.obs.columns:
            adata = adata[adata.obs['gene_label'] == filter_by_gene].copy()
            print(f"  Filtered by gene '{filter_by_gene}': {adata.n_obs} cells")
        else:
            print(f"  Warning: No 'gene' column found in obs. Available columns: {list(adata.obs.columns)}")

    # Get expression matrix
    if use_layer is not None and use_layer in adata.layers:
        print(f"  Using layer: {use_layer}")
        data = adata.layers[use_layer]
    else:
        data = adata.X

    # Convert sparse to dense if needed
    if hasattr(data, 'toarray'):
        data = data.toarray()

    # Update metadata after filtering
    metadata['final_n_cells'] = adata.n_obs
    metadata['final_n_genes'] = adata.n_vars

    # Include perturbation/gene info if available
    if 'perturbation' in adata.obs.columns:
        metadata['unique_perturbations'] = list(adata.obs['perturbation'].unique())
    if 'gene' in adata.obs.columns:
        metadata['unique_genes'] = list(adata.obs['gene'].unique())

    return data, metadata


def load_data(
    filepath: str,
    use_layer: Optional[str] = None,
    filter_by_perturbation: Optional[str] = None,
    filter_by_gene: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[Dict]]:
    """
    Load gene expression data from various formats.

    Supports:
    - .npy: NumPy array files
    - .npz: Compressed NumPy files (loads first array)
    - .h5ad: AnnData files (requires scanpy) with gene/perturbation labels
    - .csv/.tsv: Text files (cells x genes)

    Args:
        filepath: Path to data file
        use_layer: For h5ad, specific layer to use (None for .X)
        filter_by_perturbation: For h5ad, filter by perturbation label
        filter_by_gene: For h5ad, filter by gene label

    Returns:
        Tuple of (data_matrix, metadata_dict or None)
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    # Load based on extension
    if filepath.suffix == '.npy':
        data = np.load(filepath)
        metadata = None
    elif filepath.suffix == '.npz':
        with np.load(filepath) as npz_file:
            # Load first array in npz file
            key = list(npz_file.keys())[0]
            data = npz_file[key]
        metadata = None
    elif filepath.suffix == '.h5ad':
        data, metadata = load_h5ad_with_labels(
            filepath, use_layer, filter_by_perturbation, filter_by_gene
        )
    elif filepath.suffix in ['.csv', '.tsv']:
        delimiter = ',' if filepath.suffix == '.csv' else '\t'
        data = np.loadtxt(filepath, delimiter=delimiter, skiprows=1)
        metadata = None
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

    # Ensure 2D array
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {data.shape}")

    return data, metadata


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

            # Write metadata if available
            if 'metadata' in results:
                f.write("Dataset Information:\n")
                f.write("-" * 50 + "\n")
                for key, value in results['metadata'].items():
                    if not isinstance(value, (list, dict)):
                        f.write(f"  {key}: {value}\n")
                f.write("\n")

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


def run_evaluation(config: dict) -> dict:
    """
    Run evaluation with given configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Results dictionary
    """
    # Load data
    print(f"Loading real data from: {config['real_data_path']}")
    real_data, real_metadata = load_data(
        config['real_data_path'],
        use_layer=config['use_layer'],
        filter_by_perturbation=config['filter_by_perturbation'],
        filter_by_gene=config['filter_by_gene'],
    )
    print(f"  Shape: {real_data.shape}")

    print(f"\nLoading predicted data from: {config['predicted_data_path']}")
    pred_data, pred_metadata = load_data(
        config['predicted_data_path'],
        use_layer=config['use_layer'],
        filter_by_perturbation=config['filter_by_perturbation'],
        filter_by_gene=config['filter_by_gene'],
    )
    print(f"  Shape: {pred_data.shape}")

    # Validate shapes
    if real_data.shape != pred_data.shape:
        print(f"\nWarning: Shape mismatch!")
        print(f"  Real: {real_data.shape}")
        print(f"  Predicted: {pred_data.shape}")

        if config['mode'] == 'reconstruction':
            raise ValueError("Reconstruction mode requires identical shapes")

        if real_data.shape[1] != pred_data.shape[1]:
            raise ValueError(f"Feature dimension mismatch: {real_data.shape[1]} vs {pred_data.shape[1]}")

        print(f"  Proceeding with generation metrics (different number of cells is OK)")

    print()

    # Store results
    results = {
        'config': {
            'real_data_path': config['real_data_path'],
            'predicted_data_path': config['predicted_data_path'],
            'mode': config['mode'],
        },
        'metadata': {
            'real_shape': real_data.shape,
            'pred_shape': pred_data.shape,
        }
    }

    # Add metadata from h5ad files if available
    if real_metadata:
        results['metadata']['real_metadata'] = real_metadata
    if pred_metadata:
        results['metadata']['pred_metadata'] = pred_metadata

    # Evaluate reconstruction metrics
    if config['mode'] in ['reconstruction', 'both']:
        if real_data.shape != pred_data.shape:
            print("Skipping reconstruction metrics due to shape mismatch\n")
        else:
            print("=" * 60)
            print("RECONSTRUCTION METRICS")
            print("=" * 60)
            recon_results = evaluate_reconstruction(
                real_data, pred_data, verbose=config['verbose']
            )
            results['reconstruction'] = recon_results

    # Evaluate generation metrics
    if config['mode'] in ['generation', 'both']:
        print("=" * 60)
        print("GENERATION METRICS")
        print("=" * 60)
        gen_results = evaluate_generation(
            real_data, pred_data,
            w2_projections=config['w2_projections'],
            mmd_subsample=config['mmd_subsample'],
            seed=config['random_seed'],
            verbose=config['verbose']
        )
        results['generation'] = gen_results

    # Save results if output path specified
    if config['output_path']:
        save_results(results, config['output_path'], config['output_format'])
        print(f"\nResults saved to: {config['output_path']}")

    return results


def main():
    """Main function with optional CLI support."""
    parser = argparse.ArgumentParser(
        description='Evaluate single-cell RNA-seq model performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config (edit CONFIG in script)
  python eval/evaluate.py

  # Override config with command line
  python eval/evaluate.py --real data/real.h5ad --pred data/predicted.h5ad

  # Evaluate specific perturbation
  python eval/evaluate.py --filter-perturbation gene1

  # Generation metrics only
  python eval/evaluate.py --mode generation --output results/gen_metrics.json
        """
    )

    # Optional arguments (override CONFIG)
    parser.add_argument('--real', type=str, default=None,
                       help='Path to real/ground truth data (overrides CONFIG)')
    parser.add_argument('--pred', type=str, default=None,
                       help='Path to predicted/generated data (overrides CONFIG)')
    parser.add_argument('--mode', type=str, default=None,
                       choices=['reconstruction', 'generation', 'both'],
                       help='Which metrics to compute (overrides CONFIG)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save results (overrides CONFIG)')
    parser.add_argument('--format', type=str, default=None,
                       choices=['json', 'txt'],
                       help='Output format (overrides CONFIG)')

    # H5ad specific options
    parser.add_argument('--use-layer', type=str, default=None,
                       help='Use specific layer from h5ad file (overrides CONFIG)')
    parser.add_argument('--filter-perturbation', type=str, default=None,
                       help='Filter by perturbation label (overrides CONFIG)')
    parser.add_argument('--filter-gene', type=str, default=None,
                       help='Filter by gene label (overrides CONFIG)')

    # Generation metric parameters
    parser.add_argument('--w2-projections', type=int, default=None,
                       help='Number of projections for W2 distance (overrides CONFIG)')
    parser.add_argument('--mmd-subsample', type=int, default=None,
                       help='Max samples for MMD calculation (overrides CONFIG)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (overrides CONFIG)')

    # Display options
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')

    args = parser.parse_args()

    # Create config from defaults and override with CLI args if provided
    config = CONFIG.copy()

    if args.real is not None:
        config['real_data_path'] = args.real
    if args.pred is not None:
        config['predicted_data_path'] = args.pred
    if args.mode is not None:
        config['mode'] = args.mode
    if args.output is not None:
        config['output_path'] = args.output
    if args.format is not None:
        config['output_format'] = args.format
    if args.use_layer is not None:
        config['use_layer'] = args.use_layer
    if args.filter_perturbation is not None:
        config['filter_by_perturbation'] = args.filter_perturbation
    if args.filter_gene is not None:
        config['filter_by_gene'] = args.filter_gene
    if args.w2_projections is not None:
        config['w2_projections'] = args.w2_projections
    if args.mmd_subsample is not None:
        config['mmd_subsample'] = args.mmd_subsample
    if args.seed is not None:
        config['random_seed'] = args.seed
    if args.quiet:
        config['verbose'] = False

    # Run evaluation
    print("=" * 60)
    print("Single-Cell RNA-seq Evaluation")
    print("=" * 60)
    print()

    results = run_evaluation(config)

    return results


if __name__ == "__main__":
    main()
