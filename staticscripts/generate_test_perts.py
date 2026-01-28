import random
from pathlib import Path


def load_genes(file_path: str) -> list:
    """Load gene names from a text file (one per line)."""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def main():
    # ========== CONFIGURATION - EDIT THESE VARIABLES ==========
    input_file = '/home/b5cc/sanjukta.b5cc/st3/datasets/20M/filtered_perts.txt'
    output_file = 'datasets/20M/test_perts.txt'
    
    num_genes = 49  # Number of genes to randomly select
    random_seed = 42  # Set to None for different results each time, or use a number for reproducibility
    # ===========================================================
    
    print(f"Loading genes from: {input_file}")
    all_genes = load_genes(input_file)
    print(f"  Found {len(all_genes)} total genes")
    
    # Check if we have enough genes
    if len(all_genes) < num_genes:
        print(f"\n⚠️  Warning: Only {len(all_genes)} genes available, but {num_genes} requested!")
        print(f"  Will select all {len(all_genes)} genes instead.")
        num_genes = len(all_genes)
        selected_genes = all_genes
    else:
        # Set random seed for reproducibility (if specified)
        if random_seed is not None:
            random.seed(random_seed)
            print(f"  Using random seed: {random_seed}")
        
        # Randomly select genes
        selected_genes = random.sample(all_genes, num_genes)
        print(f"\n✓ Randomly selected {num_genes} genes")
    
    # Sort for easier reading (optional - comment out if you want random order)
    sorted_genes = sorted(selected_genes)
    
    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for gene in sorted_genes:
            f.write(f"{gene}\n")
    
    print(f"\n✓ Saved {len(sorted_genes)} genes to: {output_path}")
    


if __name__ == "__main__":
    main()