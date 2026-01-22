
#h5_path = "/home/b5cc/sanjukta.b5cc/aracneseq/datasets/k562.h5ad"
#list_genes = [ALG1L, PRPF4B, RPS10-NUDT3, GCOM1]

import scanpy as sc
import numpy as np

# Load the h5ad file
h5_path = "/home/b5cc/sanjukta.b5cc/aracneseq/datasets/k562.h5ad"
adata = sc.read_h5ad(h5_path)

# Genes to remove
list_genes = ['ALG1L', 'PRPF4B', 'RPS10-NUDT3', 'GCOM1']

print(f"Original dataset shape: {adata.shape}")
print(f"Original number of cells: {adata.n_obs}")

# Check the perturbation column name (it might be 'perturbation', 'gene', 'guide', etc.)
print(f"\nAvailable obs columns: {adata.obs.columns.tolist()}")

# Assuming the column is named 'perturbation' - adjust if different
perturbation_col = 'gene'  # Change this if needed

# Check how many cells have these perturbations
mask = adata.obs[perturbation_col].isin(list_genes)
n_cells_to_remove = mask.sum()
print(f"\nCells to remove: {n_cells_to_remove}")

# Show breakdown by gene
for gene in list_genes:
    count = (adata.obs[perturbation_col] == gene).sum()
    print(f"  {gene}: {count} cells")

# Filter out cells with these perturbations
adata_filtered = adata[~mask].copy()

print(f"\nFiltered dataset shape: {adata_filtered.shape}")
print(f"Filtered number of cells: {adata_filtered.n_obs}")
print(f"Cells removed: {adata.n_obs - adata_filtered.n_obs}")

# Save the filtered dataset
output_path = "/home/b5cc/sanjukta.b5cc/st3/datasets/30k/k562_filtered.h5ad"
adata_filtered.write_h5ad(output_path)
print(f"\nFiltered dataset saved to: {output_path}")

# Verify the genes are gone
remaining_perturbations = adata_filtered.obs[perturbation_col].unique()
genes_still_present = [g for g in list_genes if g in remaining_perturbations]

if genes_still_present:
    print(f"\n⚠️  WARNING: These genes still present: {genes_still_present}")
else:
    print(f"\n✓ Successfully removed all {len(list_genes)} genes from the dataset")