import os

import scanpy as sc
from pathlib import Path

def get_perturbed_genes(adata, perturb_column):

    perturbed_genes = set()
    
    for perturb in adata.obs[perturb_column].unique():
        perturb_str = str(perturb)
        
        # Skip control conditions
        if perturb_str.lower() in ['non-targeting']:
            continue
        
        # Handle multiple genes separated by '+' or other delimiters
        genes = perturb_str.replace('+', ',').replace('|', ',').split(',')
        
        for gene in genes:
            gene = gene.strip()
            if gene and gene.lower() not in ['control', 'ctrl', 'non-targeting', 'nt']:
                perturbed_genes.add(gene)
    
    return perturbed_genes


def main():
    # ========== CONFIGURATION - EDIT THESE VARIABLES ==========
    k562_file = '/home/b5cc/sanjukta.b5cc/st3/datasets/20M/hepg2.h5ad'
    other_file = '/home/b5cc/sanjukta.b5cc/st3/datasets/20M/jurkat.h5ad'
    k562_column = 'gene'  # Column name in K562 file
    other_column = 'gene'  # Column name in other file
    output_file = 'datasets/20M/genes_difference_sanity_2.txt'
    reverse = False  # Set to True to find genes in K562 but NOT in other file
    # ===========================================================
    
    print(f"Loading K562 file: {k562_file}")
    k562_adata = sc.read_h5ad(k562_file)
    
    print(f"Loading other file: {other_file}")
    other_adata = sc.read_h5ad(other_file)
    
    # Check if columns exist
    if k562_column not in k562_adata.obs.columns:
        print(f"\nAvailable columns in K562 file:")
        print(k562_adata.obs.columns.tolist())
        raise ValueError(f"Column '{k562_column}' not found in K562 file")
    
    if other_column not in other_adata.obs.columns:
        print(f"\nAvailable columns in other file:")
        print(other_adata.obs.columns.tolist())
        raise ValueError(f"Column '{other_column}' not found in other file")
    
    # Extract perturbed genes
    print(f"\nExtracting genes from K562 perturbation column: '{k562_column}'")
    k562_genes = get_perturbed_genes(k562_adata, k562_column)
    print(f"Found {len(k562_genes)} unique genes in K562 perturbations")
    
    print(f"\nExtracting genes from other file perturbation column: '{other_column}'")
    other_genes = get_perturbed_genes(other_adata, other_column)
    print(f"Found {len(other_genes)} unique genes in other file perturbations")
    
    # Calculate difference
    if reverse:
        difference = k562_genes - other_genes
        print(f"\nGenes in K562 but NOT in other file: {len(difference)}")
    else:
        difference = other_genes - k562_genes
        print(f"\nGenes in other file but NOT in K562: {len(difference)}")
    
    # Sort for consistent output
    sorted_genes = sorted(difference)
    
    # Save to file
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        for gene in sorted_genes:
            f.write(f"{gene}\n")
    
    print(f"\nGene list saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"K562 unique genes: {len(k562_genes)}")
    print(f"Other file unique genes: {len(other_genes)}")
    print(f"Genes in both: {len(k562_genes & other_genes)}")
    if reverse:
        print(f"Genes ONLY in K562: {len(difference)}")
    else:
        print(f"Genes ONLY in other file: {len(difference)}")
    
    # Show first few genes as preview
    if sorted_genes:
        print(f"\nFirst 10 genes in difference list:")
        for gene in sorted_genes[:10]:
            print(f"  - {gene}")
        if len(sorted_genes) > 10:
            print(f"  ... and {len(sorted_genes) - 10} more")


if __name__ == "__main__":
    main()