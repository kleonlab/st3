#!/usr/bin/env python3
"""
Inference script for cell generation using trained SEDD discrete diffusion model.

This script loads a trained model and generates new synthetic cells from scratch
by starting from an all-masked state and sampling. Outputs a clean h5ad file.
"""

import argparse
import json
import os
from pathlib import Path
import sys

import torch
import numpy as np
import scanpy as sc
import yaml
from scipy.sparse import csr_matrix

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sedd.model import SEDDTransformerSmall
from sedd.graph import AbsorbingGraph
from sedd.noise import LogLinearNoise
from sedd.trainer import SEDDTrainer
from sedd.sampling import EulerSampler


def load_config(config_path):
    """Load YAML configuration file."""
    if config_path is None or not Path(config_path).exists():
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config if config else {}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic cells using trained SEDD model",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Path to experiment directory containing trained model"
    )
    
    # Optional arguments
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (optional)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific checkpoint path (if not provided, auto-finds best/final/latest)"
    )
    parser.add_argument(
        "--reference_data",
        type=str,
        default=None,
        help="Path to reference h5ad file for gene structure"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path for generated cells (default: experiment_dir/inference/generated_cells.h5ad)"
    )
    parser.add_argument(
        "--num_generate",
        type=int,
        default=None,
        help="Number of cells to generate"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="Number of sampling steps"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed"
    )
    parser.add_argument(
        "--keep_sparse",
        action="store_true",
        help="Save output as sparse matrix"
    )
    
    return parser.parse_args()


def merge_configs(args, config):
    """
    Merge command-line arguments with YAML config.
    Priority: CLI args > YAML config > defaults
    """
    # Default values
    defaults = {
        'num_generate': 100,
        'num_steps': 50,
        'temperature': 1.0,
        'seed': 42,
        'keep_sparse': False,
    }
    
    # Get inference section from config if it exists
    inference_config = config.get('inference', {})
    
    # Merge with priority: args > config > defaults
    merged = {}
    
    # Required argument
    merged['experiment_dir'] = args.experiment_dir
    
    # Optional arguments with fallback chain
    merged['checkpoint'] = args.checkpoint
    merged['config'] = args.config
    
    merged['reference_data'] = (
        args.reference_data 
        or inference_config.get('reference_data')
        or None
    )
    
    merged['output_path'] = (
        args.output_path
        or inference_config.get('output_path')
        or None
    )
    
    merged['num_generate'] = (
        args.num_generate
        if args.num_generate is not None
        else inference_config.get('num_generate', defaults['num_generate'])
    )
    
    merged['num_steps'] = (
        args.num_steps
        if args.num_steps is not None
        else inference_config.get('num_steps', defaults['num_steps'])
    )
    
    merged['temperature'] = (
        args.temperature
        if args.temperature is not None
        else inference_config.get('temperature', defaults['temperature'])
    )
    
    merged['seed'] = (
        args.seed
        if args.seed is not None
        else inference_config.get('seed', defaults['seed'])
    )
    
    merged['keep_sparse'] = (
        args.keep_sparse
        or inference_config.get('keep_sparse', defaults['keep_sparse'])
    )
    
    return merged


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


def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Load YAML config if provided
    config = {}
    if args.config:
        print(f"Loading configuration from {args.config}")
        config = load_config(args.config)
    
    # Merge configurations (CLI args > config file > defaults)
    params = merge_configs(args, config)
    
    # Extract parameters
    experiment_dir = params['experiment_dir']
    checkpoint = params['checkpoint']
    reference_data = params['reference_data']
    output_path = params['output_path']
    num_generate = params['num_generate']
    num_steps = params['num_steps']
    temperature = params['temperature']
    seed = params['seed']
    keep_sparse = params['keep_sparse']
    
    # Validate required parameters
    if reference_data is None:
        raise ValueError(
            "reference_data must be provided either via --reference_data argument "
            "or in the config file under inference.reference_data"
        )
    
    # Print configuration
    print("="*60)
    print("Generation Configuration")
    print("="*60)
    print(f"Experiment dir : {experiment_dir}")
    print(f"Checkpoint     : {checkpoint if checkpoint else 'auto (best/final/latest)'}")
    print(f"Reference data : {reference_data}")
    print(f"Output path    : {output_path if output_path else 'auto (experiment_dir/inference/generated_cells.h5ad)'}")
    print(f"Num generate   : {num_generate}")
    print(f"Num steps      : {num_steps}")
    print(f"Temperature    : {temperature}")
    print(f"Seed           : {seed}")
    print(f"Keep sparse    : {keep_sparse}")
    print("="*60)
    print()
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Find checkpoint
    experiment_dir = Path(experiment_dir)
    if checkpoint:
        checkpoint_path = Path(checkpoint)
    else:
        checkpoint_path = find_checkpoint(experiment_dir)

    # Determine output path
    if output_path:
        output_path = Path(output_path)
    else:
        output_path = experiment_dir / "inference" / "generated_cells.h5ad"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    # Load reference data to get gene names and structure
    print(f"\nLoading reference data from {reference_data}")
    adata_ref = sc.read_h5ad(reference_data)
    print(f"Reference: {len(adata_ref)} cells with {adata_ref.n_vars} genes")

    # Get gene names and metadata structure
    gene_names = adata_ref.var_names.tolist()
    
    # Get model parameters from reference data
    expression = adata_ref.X
    if hasattr(expression, 'toarray'):
        expression = expression.toarray()
    dataset = torch.tensor(expression).long()

    NUM_BINS = int(dataset.max().item())
    NUM_GENES = dataset.shape[1]
    VOCAB_SIZE = NUM_BINS + 1

    print(f"Number of genes: {NUM_GENES}")
    print(f"Number of bins: {NUM_BINS}")
    print(f"Vocabulary size: {VOCAB_SIZE}")

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

    # Create sampler
    print(f"\nGenerating {num_generate} cells with {num_steps} steps, "
          f"temperature={temperature}")

    # Enable BF16 for faster inference
    use_amp = True
    amp_dtype = torch.bfloat16
    print(f"Using AMP for inference: {use_amp}, dtype: bfloat16\n")
    
    sampler = EulerSampler(
        model=model,
        graph=graph,
        noise=noise,
        num_steps=num_steps,
        device=device,
        temperature=temperature,
        use_amp=use_amp,
        amp_dtype=amp_dtype
    )

    # ========== ONE CELL AT A TIME GENERATION ==========
    print(f"Generating cells one at a time (mask token: {VOCAB_SIZE - 1})")
    print("="*60)
    
    all_generated = []
    
    for cell_idx in range(num_generate):
        print(f"Generating cell {cell_idx + 1}/{num_generate}...", end=' ')
        
        # Generate ONE cell from all-masked starting point
        x_init = graph.sample_limiting((1, NUM_GENES), device)  # batch_size=1
        
        # Sample this single cell (show progress only for first cell)
        generated_cell = sampler.sample(x_init, show_progress=(cell_idx == 0))
        
        # Move to CPU and store
        all_generated.append(generated_cell.cpu())
        
        print("✓")
        
        # Clear GPU cache periodically (every 10 cells)
        if device.type == 'cuda' and (cell_idx + 1) % 10 == 0:
            torch.cuda.empty_cache()
    
    # Concatenate all cells
    generated = torch.cat(all_generated, dim=0)
    print("="*60)
    print(f"✓ Generation complete! Total cells generated: {generated.shape[0]}")
    # ========== END ONE-AT-A-TIME GENERATION ==========

    # Convert to numpy
    generated_np = generated.cpu().numpy()

    # Create AnnData object with same structure as reference
    print(f"\nCreating h5ad file with {num_generate} generated cells...")
    
    # Create expression matrix (optionally sparse)
    if keep_sparse:
        X = csr_matrix(generated_np)
        print("Saving as sparse matrix")
    else:
        X = generated_np
        print("Saving as dense matrix")
    
    # Create AnnData object
    adata_generated = sc.AnnData(
        X=X,
        var=adata_ref.var.copy(),  # Copy gene metadata
    )
    
    # Add observation (cell) metadata
    adata_generated.obs['cell_id'] = [f"generated_cell_{i}" for i in range(num_generate)]
    adata_generated.obs['source'] = 'sedd_generated'
    
    # Add generation metadata to uns
    adata_generated.uns['generation_params'] = {
        'num_steps': num_steps,
        'temperature': temperature,
        'seed': seed,
        'checkpoint': str(checkpoint_path),
        'model_config': train_config,
        'generation_mode': 'one_cell_at_a_time'
    }

    # Save to h5ad
    print(f"Saving generated cells to {output_path}")
    adata_generated.write_h5ad(output_path, compression='gzip')
    
    # Report file size
    file_size = output_path.stat().st_size / (1024**2)  # MB
    print(f"Saved {num_generate} cells to {output_path}")
    print(f"File size: {file_size:.2f} MB")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("Generation Summary")
    print("="*50)
    print(f"Generated cells: {adata_generated.n_obs}")
    print(f"Genes: {adata_generated.n_vars}")
    print(f"Mean expression: {generated_np.mean():.2f}")
    print(f"Std expression: {generated_np.std():.2f}")
    print(f"Sparsity (zeros): {(generated_np == 0).mean()*100:.1f}%")
    print(f"Output: {output_path}")
    print("="*50)


if __name__ == "__main__":
    main()