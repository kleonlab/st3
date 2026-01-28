import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path

def load_filtered_perturbations(filepath):
    """Load the list of perturbations to keep from a text file."""
    with open(filepath, 'r') as f:
        # Read lines and strip whitespace, filter out empty lines
        perturbations = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(perturbations)} perturbations from {filepath}")
    return set(perturbations)

def filter_h5ad_by_perturbations(input_path, output_path, allowed_perturbations, perturb_column='gene', control_name='non-targeting'):
    """
    Filter an h5ad file to only include cells with perturbations in the allowed list.
    Control cells (non-targeting) are always kept regardless of the allowed list.
    
    Parameters:
    -----------
    input_path : str
        Path to input h5ad file
    output_path : str
        Path to save filtered h5ad file
    allowed_perturbations : set
        Set of perturbation names to keep
    perturb_column : str
        Name of the column in adata.obs that contains perturbation labels
    control_name : str
        Name of the control/non-targeting perturbation to always keep
    """
    print(f"\nProcessing: {input_path}")
    
    # Load the h5ad file
    adata = sc.read_h5ad(input_path)
    print(f"  Original shape: {adata.n_obs} cells × {adata.n_vars} genes")
    
    # Check if perturbation column exists
    if perturb_column not in adata.obs.columns:
        print(f"  ERROR: Column '{perturb_column}' not found in adata.obs")
        print(f"  Available columns: {adata.obs.columns.tolist()}")
        return
    
    # Get unique perturbations in the dataset
    unique_perts_in_data = set(adata.obs[perturb_column].unique())
    print(f"  Unique perturbations in data: {len(unique_perts_in_data)}")
    
    # Check if control cells exist
    has_control = control_name in unique_perts_in_data
    n_control_cells = (adata.obs[perturb_column] == control_name).sum() if has_control else 0
    if has_control:
        print(f"  Control cells ('{control_name}'): {n_control_cells} (will be kept)")
    
    # Find perturbations that will be kept (excluding control which is always kept)
    non_control_perts = unique_perts_in_data - {control_name}
    perts_to_keep = non_control_perts.intersection(allowed_perturbations)
    perts_to_remove = non_control_perts - allowed_perturbations
    
    print(f"  Perturbations to keep: {len(perts_to_keep)}")
    print(f"  Perturbations to remove: {len(perts_to_remove)}")
    
    if perts_to_remove:
        print(f"  Removed perturbations: {sorted(list(perts_to_remove))}")
    
    # Filter the data: keep cells that are either control OR in the allowed list
    mask = (adata.obs[perturb_column] == control_name) | adata.obs[perturb_column].isin(allowed_perturbations)
    adata_filtered = adata[mask].copy()
    
    print(f"  Filtered shape: {adata_filtered.n_obs} cells × {adata_filtered.n_vars} genes")
    print(f"  Cells removed: {adata.n_obs - adata_filtered.n_obs}")
    
    # Save the filtered data
    adata_filtered.write_h5ad(output_path)
    print(f"  Saved to: {output_path}")
    
    return adata_filtered

def main():
    # ========== CONFIGURATION ==========
    # Path to the filtered perturbations list
    filtered_perts_file = '/home/b5cc/sanjukta.b5cc/st3/datasets/20M/filtered_perts.txt'
    
    # Input h5ad files
    input_files = [
        '/home/b5cc/sanjukta.b5cc/st3/datasets/20M/hepg2.h5ad',
        '/home/b5cc/sanjukta.b5cc/st3/datasets/20M/jurkat.h5ad',
        '/home/b5cc/sanjukta.b5cc/st3/datasets/20M/rpe1.h5ad'
    ]
    
    # Output directory for filtered files
    output_dir = '/home/b5cc/sanjukta.b5cc/st3/datasets/20M/filtered'
    
    
    # Column name containing perturbation labels
    perturb_column = 'gene'
    # ====================================
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load the list of allowed perturbations
    print("\n" + "=" * 70)
    print("Loading filtered perturbations list")
    print("=" * 70)
    allowed_perturbations = load_filtered_perturbations(filtered_perts_file)
    
    # Process each h5ad file
    print("\n" + "=" * 70)
    print("Filtering h5ad files")
    print("=" * 70)
    
    summary = []
    for input_path in input_files:
        # Get the filename
        filename = Path(input_path).name
        output_path = Path(output_dir) / filename
        
        # Filter the file
        adata_filtered = filter_h5ad_by_perturbations(
            input_path=input_path,
            output_path=str(output_path),
            allowed_perturbations=allowed_perturbations,
            perturb_column=perturb_column,
            control_name='non-targeting'
        )
        
        if adata_filtered is not None:
            summary.append({
                'file': filename,
                'original_cells': sc.read_h5ad(input_path).n_obs,
                'filtered_cells': adata_filtered.n_obs,
                'cells_removed': sc.read_h5ad(input_path).n_obs - adata_filtered.n_obs
            })
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nFiltered perturbations: {len(allowed_perturbations)}")
    print(f"Output directory: {output_dir}")
    print("\nPer-file statistics:")
    for item in summary:
        print(f"\n  {item['file']}:")
        print(f"    Original cells:  {item['original_cells']:,}")
        print(f"    Filtered cells:  {item['filtered_cells']:,}")
        print(f"    Cells removed:   {item['cells_removed']:,}")
        print(f"    Retention rate:  {100 * item['filtered_cells'] / item['original_cells']:.2f}%")
    
    total_original = sum(item['original_cells'] for item in summary)
    total_filtered = sum(item['filtered_cells'] for item in summary)
    total_removed = sum(item['cells_removed'] for item in summary)
    
    print(f"\n  TOTAL:")
    print(f"    Original cells:  {total_original:,}")
    print(f"    Filtered cells:  {total_filtered:,}")
    print(f"    Cells removed:   {total_removed:,}")
    print(f"    Retention rate:  {100 * total_filtered / total_original:.2f}%")
    
    print("\n" + "=" * 70)
    print("Filtering complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()

