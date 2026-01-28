import os
import scanpy as sc
from pathlib import Path

def get_perturbed_genes(adata, perturb_column):
    """
    Extract unique perturbed genes from the perturbation column.
    Handles various formats like 'gene1', 'gene1+gene2', 'control', etc.
    """
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
    file1_path = '/home/b5cc/sanjukta.b5cc/st3/datasets/20M/hepg2.h5ad'
    file2_path = '/home/b5cc/sanjukta.b5cc/st3/datasets/20M/jurkat.h5ad'
    file3_path = '/home/b5cc/sanjukta.b5cc/st3/datasets/20M/rpe1.h5ad'
    
    file1_column = 'gene'  # Column name in file 1
    file2_column = 'gene'  # Column name in file 2
    file3_column = 'gene'  # Column name in file 3
    
    # Optional: give names to your files for better output readability
    file1_name = 'HepG2'
    file2_name = 'Jurkat'
    file3_name = 'Rpe1'
    
    output_file = 'datasets/20M/all_perturbations.txt'
    # ===========================================================
    
    print(f"Loading {file1_name} file: {file1_path}")
    adata1 = sc.read_h5ad(file1_path)
    
    print(f"Loading {file2_name} file: {file2_path}")
    adata2 = sc.read_h5ad(file2_path)
    
    print(f"Loading {file3_name} file: {file3_path}")
    adata3 = sc.read_h5ad(file3_path)
    
    # Check if columns exist
    if file1_column not in adata1.obs.columns:
        print(f"\nAvailable columns in {file1_name} file:")
        print(adata1.obs.columns.tolist())
        raise ValueError(f"Column '{file1_column}' not found in {file1_name} file")
    
    if file2_column not in adata2.obs.columns:
        print(f"\nAvailable columns in {file2_name} file:")
        print(adata2.obs.columns.tolist())
        raise ValueError(f"Column '{file2_column}' not found in {file2_name} file")
    
    if file3_column not in adata3.obs.columns:
        print(f"\nAvailable columns in {file3_name} file:")
        print(adata3.obs.columns.tolist())
        raise ValueError(f"Column '{file3_column}' not found in {file3_name} file")
    
    # Extract perturbed genes from each file
    print(f"\nExtracting genes from {file1_name} perturbation column: '{file1_column}'")
    genes1 = get_perturbed_genes(adata1, file1_column)
    print(f"Found {len(genes1)} unique genes in {file1_name} perturbations")
    
    print(f"\nExtracting genes from {file2_name} perturbation column: '{file2_column}'")
    genes2 = get_perturbed_genes(adata2, file2_column)
    print(f"Found {len(genes2)} unique genes in {file2_name} perturbations")
    
    print(f"\nExtracting genes from {file3_name} perturbation column: '{file3_column}'")
    genes3 = get_perturbed_genes(adata3, file3_column)
    print(f"Found {len(genes3)} unique genes in {file3_name} perturbations")
    
    # Combine all unique genes using set union
    all_unique_genes = genes1 | genes2 | genes3
    print(f"\nTotal unique genes across ALL three files: {len(all_unique_genes)}")
    
    # Sort for consistent output
    sorted_all_genes = sorted(all_unique_genes)
    
    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for gene in sorted_all_genes:
            f.write(f"{gene}\n")
    
    print(f"\nAll unique genes list saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{file1_name} unique genes: {len(genes1)}")
    print(f"{file2_name} unique genes: {len(genes2)}")
    print(f"{file3_name} unique genes: {len(genes3)}")
    print(f"\nTotal unique genes (union): {len(all_unique_genes)}")
    
    # Show overlap statistics
    print(f"\nGenes common to ALL THREE: {len(genes1 & genes2 & genes3)}")
    print(f"Genes in {file1_name} AND {file2_name} only: {len((genes1 & genes2) - genes3)}")
    print(f"Genes in {file1_name} AND {file3_name} only: {len((genes1 & genes3) - genes2)}")
    print(f"Genes in {file2_name} AND {file3_name} only: {len((genes2 & genes3) - genes1)}")
    print(f"\nGenes ONLY in {file1_name}: {len(genes1 - genes2 - genes3)}")
    print(f"Genes ONLY in {file2_name}: {len(genes2 - genes1 - genes3)}")
    print(f"Genes ONLY in {file3_name}: {len(genes3 - genes1 - genes2)}")
    
    # Show first few genes as preview
    if sorted_all_genes:
        print(f"\nFirst 10 genes in combined list:")
        for gene in sorted_all_genes[:10]:
            print(f"  - {gene}")
        if len(sorted_all_genes) > 10:
            print(f"  ... and {len(sorted_all_genes) - 10} more")


if __name__ == "__main__":
    main()