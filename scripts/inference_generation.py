#!/usr/bin/env python3
"""
Inference script for cell generation using trained SEDD discrete diffusion model.

This script loads a trained model and generates new synthetic cells from scratch
by starting from an all-masked state and sampling. Outputs a clean h5ad file.
"""

import json
import os
from pathlib import Path
import sys

import torch
import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sedd.model import SEDDTransformerSmall
from sedd.graph import AbsorbingGraph
from sedd.noise import LogLinearNoise
from sedd.trainer import SEDDTrainer
from sedd.sampling import EulerSampler


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
    # ========== HARDCODED CONFIGURATION ==========
    # Experiment settings
    experiment_dir = "experiments/mlm_demo/rnaseq_small"
    checkpoint = None  # None = auto-find best/final, or specify path like "experiments/.../best.pt"
    
    # Data settings
    reference_data = "/home/b5cc/sanjukta.b5cc/aracneseq/datasets/k562_5k.h5ad"
    output_path = None  # None = save to experiment_dir/generated_cells.h5ad, or specify custom path
    
    # Generation settings
    num_generate = 100  # Number of cells to generate
    num_steps = 10      # Number of sampling steps
    temperature = 1.0   # Sampling temperature
    seed = 42           # Random seed
    keep_sparse = False # Save as sparse matrix
    # ========== END CONFIGURATION ==========

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
        output_path = experiment_dir / "generated_cells.h5ad"
    
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

    sampler = EulerSampler(
        model=model,
        graph=graph,
        noise=noise,
        num_steps=num_steps,
        device=device,
        temperature=temperature
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