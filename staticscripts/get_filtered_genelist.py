from pathlib import Path


def load_genes(file_path: str) -> set:
    """Load gene names from a text file (one per line)."""
    with open(file_path, 'r') as f:
        return {line.strip() for line in f if line.strip()}


def main():
    # ========== CONFIGURATION - EDIT THESE VARIABLES ==========
    file1_path = "/home/b5cc/sanjukta.b5cc/st3/datasets/20M/all_perturbations.txt"
    file2_path = "/home/b5cc/sanjukta.b5cc/st3/datasets/20M/genes_difference.txt"
    
    output_file = "/home/b5cc/sanjukta.b5cc/st3/datasets/20M/filtered_perts.txt"
    
    # Set to True to find genes in file1 but NOT in file2
    # Set to False to find genes in file2 but NOT in file1
    reverse = True
    # ===========================================================
    
    print(f"Loading genes from file 1: {file1_path}")
    genes1 = load_genes(file1_path)
    print(f"  Found {len(genes1)} genes")
    
    print(f"\nLoading genes from file 2: {file2_path}")
    genes2 = load_genes(file2_path)
    print(f"  Found {len(genes2)} genes")
    
    # Calculate difference
    if reverse:
        difference = genes1 - genes2
        print(f"\nGenes in file1 but NOT in file2: {len(difference)}")
    else:
        difference = genes2 - genes1
        print(f"\nGenes in file2 but NOT in file1: {len(difference)}")
    
    # Sort for consistent output
    sorted_genes = sorted(difference)
    
    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for gene in sorted_genes:
            f.write(f"{gene}\n")
    
    print(f"\nGene difference saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"File 1 genes: {len(genes1)}")
    print(f"File 2 genes: {len(genes2)}")
    print(f"Genes in both: {len(genes1 & genes2)}")
    if reverse:
        print(f"Genes ONLY in file1: {len(difference)}")
    else:
        print(f"Genes ONLY in file2: {len(difference)}")
    
    # Show first few genes as preview
    if sorted_genes:
        print(f"\nFirst 10 genes in difference:")
        for gene in sorted_genes[:10]:
            print(f"  - {gene}")
        if len(sorted_genes) > 10:
            print(f"  ... and {len(sorted_genes) - 10} more")
    else:
        print("\nNo difference found - both files have the same genes!")


if __name__ == "__main__":
    main()