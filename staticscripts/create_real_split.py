import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path

def load_test_perturbations(filepath):
    """Load the list of test perturbations from a text file."""
    with open(filepath, 'r') as f:
        # Read lines and strip whitespace, filter out empty lines
        perturbations = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(perturbations)} test perturbations from {filepath}")
    return set(perturbations)

def create_train_test_split(input_path, output_dir, test_perturbations, perturb_column='gene', 
                            control_name='non-targeting', control_fraction=0.01, seed=42):
    """
    Create both train and test splits from an h5ad file.
    
    Test split contains:
    - All cells with perturbations in the test list
    - A small fraction (default 1%) of control cells
    
    Train split contains:
    - All remaining cells (perturbations NOT in test list + remaining control cells)
    
    Parameters:
    -----------
    input_path : str
        Path to input h5ad file
    output_dir : str
        Directory to save train and test split h5ad files
    test_perturbations : set
        Set of perturbation names for test set
    perturb_column : str
        Name of the column in adata.obs that contains perturbation labels
    control_name : str
        Name of the control/non-targeting perturbation
    control_fraction : float
        Fraction of control cells to keep in test set (default 0.01 = 1%)
    seed : int
        Random seed for reproducibility
    """
    print(f"\nProcessing: {input_path}")
    
    # Load the h5ad file
    adata = sc.read_h5ad(input_path)
    print(f"  Original shape: {adata.n_obs} cells × {adata.n_vars} genes")
    
    # Check if perturbation column exists
    if perturb_column not in adata.obs.columns:
        print(f"  ERROR: Column '{perturb_column}' not found in adata.obs")
        print(f"  Available columns: {adata.obs.columns.tolist()}")
        return None, None
    
    # Get unique perturbations in the dataset
    unique_perts_in_data = set(adata.obs[perturb_column].unique())
    print(f"  Unique perturbations in data: {len(unique_perts_in_data)}")
    
    # Separate control and perturbed cells
    is_control = adata.obs[perturb_column] == control_name
    control_indices = adata.obs_names[is_control]
    perturbed_indices = adata.obs_names[~is_control]
    
    n_control_total = len(control_indices)
    n_perturbed_total = len(perturbed_indices)
    
    print(f"\n  Cell breakdown:")
    print(f"    Control cells ('{control_name}'): {n_control_total}")
    print(f"    Perturbed cells: {n_perturbed_total}")
    
    # Sample control cells for test set (1%)
    n_control_test = max(1, int(n_control_total * control_fraction))
    np.random.seed(seed)
    test_control_indices = np.random.choice(control_indices, size=n_control_test, replace=False)
    
    # Remaining control cells go to train
    train_control_indices = np.setdiff1d(control_indices, test_control_indices)
    n_control_train = len(train_control_indices)
    
    print(f"\n  Control cell split:")
    print(f"    Test control cells: {n_control_test} ({control_fraction*100}%)")
    print(f"    Train control cells: {n_control_train} ({100 - control_fraction*100}%)")
    
    # Find test perturbations that exist in this dataset
    non_control_perts = unique_perts_in_data - {control_name}
    test_perts_in_data = non_control_perts.intersection(test_perturbations)
    test_perts_missing = test_perturbations - unique_perts_in_data
    train_perts_in_data = non_control_perts - test_perturbations
    
    print(f"\n  Perturbation split:")
    print(f"    Test perturbations in data: {len(test_perts_in_data)}")
    print(f"    Train perturbations in data: {len(train_perts_in_data)}")
    if test_perts_missing:
        print(f"    Test perturbations not in data: {len(test_perts_missing)}")
        print(f"      Missing: {sorted(list(test_perts_missing))[:10]}")  # Show first 10
    
    # Get indices of cells with test perturbations
    test_perturbed_mask = adata.obs[perturb_column].isin(test_perturbations)
    test_perturbed_indices = adata.obs_names[test_perturbed_mask]
    n_perturbed_test = len(test_perturbed_indices)
    
    # Get indices of cells with train perturbations (NOT in test set)
    train_perturbed_mask = ~test_perturbed_mask & ~is_control  # Not test, not control
    train_perturbed_indices = adata.obs_names[train_perturbed_mask]
    n_perturbed_train = len(train_perturbed_indices)
    
    print(f"\n  Perturbed cells split:")
    print(f"    Test perturbed cells: {n_perturbed_test}")
    print(f"    Train perturbed cells: {n_perturbed_train}")
    
    # Count cells per test perturbation
    if n_perturbed_test > 0:
        test_pert_counts = adata[test_perturbed_indices].obs[perturb_column].value_counts()
        print(f"\n  Top 10 test perturbations by cell count:")
        for pert, count in test_pert_counts.head(10).items():
            print(f"    {pert}: {count}")
        if len(test_pert_counts) > 10:
            print(f"    ... and {len(test_pert_counts) - 10} more test perturbations")
    
    # Create TEST split: test perturbations + test control cells
    test_indices = np.concatenate([test_control_indices, test_perturbed_indices])
    adata_test = adata[test_indices].copy()
    
    # Create TRAIN split: train perturbations + train control cells
    train_indices = np.concatenate([train_control_indices, train_perturbed_indices])
    adata_train = adata[train_indices].copy()
    
    total_test = len(test_indices)
    total_train = len(train_indices)
    
    print(f"\n  Final splits:")
    print(f"    TEST:  {total_test:,} cells ({n_control_test} control + {n_perturbed_test} perturbed)")
    print(f"    TRAIN: {total_train:,} cells ({n_control_train} control + {n_perturbed_train} perturbed)")
    print(f"    Test retention rate: {100 * total_test / adata.n_obs:.2f}%")
    print(f"    Train retention rate: {100 * total_train / adata.n_obs:.2f}%")
    
    # Verify no overlap
    assert total_test + total_train == adata.n_obs, "ERROR: Train + Test cells don't equal total!"
    assert len(set(test_indices) & set(train_indices)) == 0, "ERROR: Overlap between train and test!"
    print(f"  ✓ Verified: No overlap between train and test")
    
    # Get base filename
    filename = Path(input_path).stem  # e.g., 'hepg2' from 'hepg2.h5ad'
    
    # Save the test split
    test_output_path = Path(output_dir) / f"{filename}_test.h5ad"
    adata_test.write_h5ad(test_output_path)
    print(f"\n  Saved TEST to: {test_output_path}")
    
    # Save the train split
    train_output_path = Path(output_dir) / f"{filename}_train.h5ad"
    adata_train.write_h5ad(train_output_path)
    print(f"  Saved TRAIN to: {train_output_path}")
    
    return adata_train, adata_test

def main():
    # ========== CONFIGURATION ==========
    # Path to the test perturbations list
    test_perts_file = '/home/b5cc/sanjukta.b5cc/st3/datasets/20M/test_perts.txt'
    
    # Input h5ad files
    input_files = [
        '/home/b5cc/sanjukta.b5cc/st3/datasets/20M/filtered/hepg2.h5ad',
        '/home/b5cc/sanjukta.b5cc/st3/datasets/20M/filtered/jurkat.h5ad',
        '/home/b5cc/sanjukta.b5cc/st3/datasets/20M/filtered/rpe1.h5ad'
    ]
    
    # Output directory for train and test split files
    output_dir = '/home/b5cc/sanjukta.b5cc/st3/datasets/20M/splits'
    
    # Column name containing perturbation labels
    perturb_column = 'gene'
    
    # Control settings
    control_name = 'non-targeting'
    control_fraction = 0.01  # 1% of control cells go to test
    
    # Random seed for reproducibility
    seed = 42
    # ====================================
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load the list of test perturbations
    print("\n" + "=" * 70)
    print("Loading test perturbations list")
    print("=" * 70)
    test_perturbations = load_test_perturbations(test_perts_file)
    
    # Process each h5ad file
    print("\n" + "=" * 70)
    print("Creating train/test splits")
    print("=" * 70)
    
    summary = []
    for input_path in input_files:
        # Create train and test splits
        adata_train, adata_test = create_train_test_split(
            input_path=input_path,
            output_dir=output_dir,
            test_perturbations=test_perturbations,
            perturb_column=perturb_column,
            control_name=control_name,
            control_fraction=control_fraction,
            seed=seed
        )
        
        if adata_train is not None and adata_test is not None:
            # Read original file to get original cell count
            adata_original = sc.read_h5ad(input_path)
            filename = Path(input_path).name
            
            n_control_original = (adata_original.obs[perturb_column] == control_name).sum()
            n_control_test = (adata_test.obs[perturb_column] == control_name).sum()
            n_control_train = (adata_train.obs[perturb_column] == control_name).sum()
            
            n_perturbed_test = adata_test.n_obs - n_control_test
            n_perturbed_train = adata_train.n_obs - n_control_train
            
            summary.append({
                'file': filename,
                'original_cells': adata_original.n_obs,
                'train_cells': adata_train.n_obs,
                'test_cells': adata_test.n_obs,
                'control_cells_original': n_control_original,
                'control_cells_train': n_control_train,
                'control_cells_test': n_control_test,
                'perturbed_cells_train': n_perturbed_train,
                'perturbed_cells_test': n_perturbed_test,
            })
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTest perturbations: {len(test_perturbations)}")
    print(f"Control fraction (test): {control_fraction*100}%")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {seed}")
    
    print("\nPer-file statistics:")
    for item in summary:
        base_name = Path(item['file']).stem
        print(f"\n  {item['file']}:")
        print(f"    Original cells:        {item['original_cells']:,}")
        print(f"    ├─ TRAIN split:        {item['train_cells']:,} ({100 * item['train_cells'] / item['original_cells']:.1f}%)")
        print(f"    │   ├─ Control:        {item['control_cells_train']:,}")
        print(f"    │   └─ Perturbed:      {item['perturbed_cells_train']:,}")
        print(f"    └─ TEST split:         {item['test_cells']:,} ({100 * item['test_cells'] / item['original_cells']:.1f}%)")
        print(f"        ├─ Control:        {item['control_cells_test']:,}")
        print(f"        └─ Perturbed:      {item['perturbed_cells_test']:,}")
        print(f"    Output files:")
        print(f"      - {base_name}_train.h5ad")
        print(f"      - {base_name}_test.h5ad")
    


if __name__ == "__main__":
    main()
