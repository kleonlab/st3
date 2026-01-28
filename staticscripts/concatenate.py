#!/usr/bin/env python3
"""
Concatenate multiple h5ad files into a single combined dataset.
"""
import scanpy as sc
import pandas as pd
from pathlib import Path


def concatenate_h5ad_files(input_files, output_path, add_source_column=True, 
                           source_column_name='dataset_source', dataset_names=None,
                           join_type='outer', make_cell_ids_unique=True):
    """
    Concatenate multiple h5ad files into one.
    
    Parameters:
    -----------
    input_files : list
        List of paths to h5ad files to concatenate
    output_path : str
        Path to save the combined h5ad file
    add_source_column : bool
        Whether to add a column tracking which file each cell came from
    source_column_name : str
        Name of the source tracking column
    dataset_names : list or None
        Optional custom names for each dataset. If None, uses filenames.
    join_type : str
        'outer' to keep all genes, 'inner' to keep only common genes
    make_cell_ids_unique : bool
        Whether to prefix cell IDs with dataset name to avoid collisions
    """
    
    print("="*70)
    print("CONCATENATING H5AD FILES")
    print("="*70)
    
    # Validate inputs
    if not input_files:
        raise ValueError("No input files provided!")
    
    # Generate dataset names if not provided
    if dataset_names is None:
        dataset_names = [Path(f).stem for f in input_files]
    elif len(dataset_names) != len(input_files):
        raise ValueError(f"Number of dataset names ({len(dataset_names)}) must match number of input files ({len(input_files)})")
    
    # Load all datasets
    print(f"\nLoading {len(input_files)} datasets...")
    adatas = []
    
    for i, (filepath, name) in enumerate(zip(input_files, dataset_names), 1):
        print(f"\n[{i}/{len(input_files)}] Loading: {filepath}")
        adata = sc.read_h5ad(filepath)
        print(f"  Dataset: {name}")
        print(f"  Shape: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
        
        # Add source column if requested
        if add_source_column:
            adata.obs[source_column_name] = name
        
        # Make cell IDs unique if requested
        if make_cell_ids_unique:
            adata.obs_names = [f"{name}_{cell_id}" for cell_id in adata.obs_names]
        
        adatas.append(adata)
    
    # Analyze gene overlap
    print("\n" + "="*70)
    print("GENE OVERLAP ANALYSIS")
    print("="*70)
    
    gene_sets = [set(adata.var_names) for adata in adatas]
    all_genes = set.union(*gene_sets)
    common_genes = set.intersection(*gene_sets)
    
    print(f"\nGenes per dataset:")
    for name, gene_set in zip(dataset_names, gene_sets):
        print(f"  {name}: {len(gene_set):,} genes")
    
    print(f"\nGene overlap:")
    print(f"  Total unique genes: {len(all_genes):,}")
    print(f"  Common to all datasets: {len(common_genes):,}")
    
    if join_type == 'outer':
        print(f"  Using 'outer' join: keeping all {len(all_genes):,} genes")
    else:
        print(f"  Using 'inner' join: keeping only {len(common_genes):,} common genes")
    
    # Concatenate datasets
    print("\n" + "="*70)
    print("CONCATENATING DATASETS")
    print("="*70)
    print(f"Join type: {join_type}")
    
    combined_adata = sc.concat(
        adatas,
        axis=0,  # Concatenate along cells (rows)
        join=join_type,
        label=None,
        keys=None,
        index_unique=None,
        fill_value=0  # Fill missing values with 0 for 'outer' join
    )
    
    print(f"\n✓ Concatenation complete!")
    print(f"  Combined shape: {combined_adata.n_obs:,} cells × {combined_adata.n_vars:,} genes")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    print(f"\nCells per dataset:")
    for name, adata in zip(dataset_names, adatas):
        print(f"  {name}: {adata.n_obs:,} cells")
    print(f"  TOTAL: {combined_adata.n_obs:,} cells")
    
    if add_source_column:
        print(f"\nSource column '{source_column_name}' breakdown:")
        source_counts = combined_adata.obs[source_column_name].value_counts()
        for dataset, count in source_counts.items():
            print(f"  {dataset}: {count:,} cells")
    
    # Print observation columns
    print(f"\nObservation columns (adata.obs):")
    for col in combined_adata.obs.columns:
        print(f"  - {col}")
    
    # Save the combined file
    print("\n" + "="*70)
    print("SAVING COMBINED DATASET")
    print("="*70)
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving to: {output_path}")
    combined_adata.write_h5ad(output_path)
    
    file_size_gb = output_file.stat().st_size / (1024**3)
    print(f"\n✓ Successfully saved combined dataset!")
    print(f"  Location: {output_path}")
    print(f"  Size: {file_size_gb:.2f} GB")
    
    return combined_adata


def filter_to_common_genes(input_h5ad, reference_h5ad, output_path):
    """
    Filter an h5ad file to keep only genes present in a reference dataset.
    
    This is useful when you have a combined dataset and want to ensure
    other datasets have the same gene set for compatibility.
    
    Parameters:
    -----------
    input_h5ad : str
        Path to the h5ad file to filter
    reference_h5ad : str
        Path to the reference h5ad file (e.g., the combined dataset)
    output_path : str
        Path to save the filtered h5ad file
    
    Returns:
    --------
    filtered_adata : AnnData
        The filtered dataset containing only common genes
    """
    
    print("="*70)
    print("FILTERING TO COMMON GENES")
    print("="*70)
    
    # Load the datasets
    print(f"\nLoading input dataset: {input_h5ad}")
    input_adata = sc.read_h5ad(input_h5ad)
    print(f"  Shape: {input_adata.n_obs:,} cells × {input_adata.n_vars:,} genes")
    
    print(f"\nLoading reference dataset: {reference_h5ad}")
    reference_adata = sc.read_h5ad(reference_h5ad)
    print(f"  Shape: {reference_adata.n_obs:,} cells × {reference_adata.n_vars:,} genes")
    
    # Find common genes
    input_genes = set(input_adata.var_names)
    reference_genes = set(reference_adata.var_names)
    common_genes = input_genes.intersection(reference_genes)
    
    print("\n" + "="*70)
    print("GENE ANALYSIS")
    print("="*70)
    print(f"\nInput genes: {len(input_genes):,}")
    print(f"Reference genes: {len(reference_genes):,}")
    print(f"Common genes: {len(common_genes):,}")
    print(f"Genes only in input: {len(input_genes - reference_genes):,}")
    print(f"Genes only in reference: {len(reference_genes - input_genes):,}")
    
    if len(common_genes) == 0:
        raise ValueError("No common genes found between input and reference datasets!")
    
    # Filter to common genes (keeping order from reference)
    common_genes_ordered = [g for g in reference_adata.var_names if g in common_genes]
    filtered_adata = input_adata[:, common_genes_ordered].copy()
    
    print("\n" + "="*70)
    print("FILTERING RESULTS")
    print("="*70)
    print(f"\nFiltered dataset shape: {filtered_adata.n_obs:,} cells × {filtered_adata.n_vars:,} genes")
    print(f"Genes retained: {len(common_genes):,} / {len(input_genes):,} ({100*len(common_genes)/len(input_genes):.1f}%)")
    
    # Save the filtered dataset
    print("\n" + "="*70)
    print("SAVING FILTERED DATASET")
    print("="*70)
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving to: {output_path}")
    filtered_adata.write_h5ad(output_path)
    
    file_size_gb = output_file.stat().st_size / (1024**3)
    print(f"\n✓ Successfully saved filtered dataset!")
    print(f"  Location: {output_path}")
    print(f"  Size: {file_size_gb:.2f} GB")
    
    return filtered_adata


def main():
    # ========== CONFIGURATION - EDIT THESE VARIABLES ==========
    
    # List all h5ad files to concatenate
    input_files = [
        '/home/b5cc/sanjukta.b5cc/st3/datasets/20M/splits/hepg2_train_processed.h5ad',
        '/home/b5cc/sanjukta.b5cc/st3/datasets/20M/splits/jurkat_train_processed.h5ad',
        '/home/b5cc/sanjukta.b5cc/st3/datasets/20M/splits/rpe1_train_processed.h5ad',
        '/home/b5cc/sanjukta.b5cc/st3/datasets/20M/splits/jurkat_test_processed.h5ad',
        '/home/b5cc/sanjukta.b5cc/st3/datasets/20M/splits/rpe1_test_processed.h5ad',
    ]
    
    # Output path for combined file
    output_path = '/home/b5cc/sanjukta.b5cc/st3/datasets/20M/combined_train.h5ad'
    
    # Optional: Custom names for each dataset (if None, uses filenames)
    dataset_names = None  # Set to None to auto-generate from filenames
    
    # Add a column to track which file each cell came from
    add_source_column = True
    source_column_name = 'dataset_source'
    
    # Join type: 'outer' keeps all genes, 'inner' keeps only common genes
    join_type = 'inner'
    
    # Make cell IDs unique by prefixing with dataset name
    make_cell_ids_unique = True
    
    # ===== FILTER CONFIGURATION (Optional) =====
    # Set to True to filter additional datasets to match the combined gene set
    run_filtering = True
    
    # List of datasets to filter (will be filtered to match combined dataset genes)
    files_to_filter = ["/home/b5cc/sanjukta.b5cc/st3/datasets/20M/splits/hepg2_test_processed.h5ad",
        # Example: '/home/b5cc/sanjukta.b5cc/st3/datasets/20M/splits/hepg2_test_processed.h5ad',
    ]
    
    # Output paths for filtered files
    filtered_output_paths = ["/home/b5cc/sanjukta.b5cc/st3/datasets/20M/splits/hepg2_test_filtered.h5ad",
        # Example: '/home/b5cc/sanjukta.b5cc/st3/datasets/20M/splits/hepg2_test_filtered.h5ad',
    ]
    
    # ===========================================================
    
    # Concatenate the files
    combined_adata = concatenate_h5ad_files(
        input_files=input_files,
        output_path=output_path,
        add_source_column=add_source_column,
        source_column_name=source_column_name,
        dataset_names=dataset_names,
        join_type=join_type,
        make_cell_ids_unique=make_cell_ids_unique
    )
    
    print("\n" + "="*70)
    print("CONCATENATION COMPLETE!")
    print("="*70)
    
    # Filter additional datasets if configured
    if run_filtering:
        if len(files_to_filter) != len(filtered_output_paths):
            raise ValueError(f"Number of files_to_filter ({len(files_to_filter)}) must match filtered_output_paths ({len(filtered_output_paths)})")
        
        print("\n" + "="*70)
        print(f"FILTERING {len(files_to_filter)} ADDITIONAL DATASET(S)")
        print("="*70)
        
        for i, (input_file, output_file) in enumerate(zip(files_to_filter, filtered_output_paths), 1):
            print(f"\n[{i}/{len(files_to_filter)}] Filtering: {input_file}")
            filter_to_common_genes(
                input_h5ad=input_file,
                reference_h5ad=output_path,
                output_path=output_file
            )
        
        print("\n" + "="*70)
        print("ALL FILTERING COMPLETE!")
        print("="*70)


if __name__ == "__main__":
    main()