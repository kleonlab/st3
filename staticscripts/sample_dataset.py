import scanpy as sc
import numpy as np
import argparse
from pathlib import Path

def sample_cells_from_h5ad(input_path, output_path, n_cells=5000, random_seed=42):
    """
    Randomly sample cells from an h5ad file and save to a new h5ad file.
    
    Parameters:
    -----------
    input_path : str
        Path to input h5ad file
    output_path : str
        Path to output h5ad file
    n_cells : int
        Number of cells to sample (default: 5000)
    random_seed : int
        Random seed for reproducibility (default: 42)
    """
    
    print(f"Loading data from {input_path}...")
    adata = sc.read_h5ad(input_path)
    
    total_cells = adata.n_obs
    print(f"Total cells in dataset: {total_cells}")
    
    if n_cells > total_cells:
        print(f"Warning: Requested {n_cells} cells but only {total_cells} available.")
        print(f"Using all {total_cells} cells.")
        n_cells = total_cells
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Randomly sample cell indices
    sampled_indices = np.random.choice(total_cells, size=n_cells, replace=False)
    
    print(f"Sampling {n_cells} cells randomly...")
    # Subset the AnnData object
    adata_sampled = adata[sampled_indices, :].copy()
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving sampled data to {output_path}...")
    adata_sampled.write_h5ad(output_path)
    
    print(f"Successfully saved {n_cells} cells to {output_path}")
    print(f"Output shape: {adata_sampled.shape} (cells x genes)")

if __name__ == "__main__":
    sample_cells_from_h5ad(
    input_path="/path/to/input.h5ad",
    output_path="/path/to/output.h5ad",
    n_cells=5000,
    random_seed=42)