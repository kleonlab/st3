import scanpy as sc
import os

# Load the data
adata = sc.read_h5ad("/home/b5cc/sanjukta.b5cc/st3/experiments/30k/inference_results/generated_cells_5000.h5ad")

print("Before renaming:")
print(f"Observation columns: {list(adata.obs.columns)}")

# Rename 'perturbation' column to 'gene'
adata.obs.rename(columns={'perturbation': 'gene'}, inplace=True)

print("\nAfter renaming:")
print(f"Observation columns: {list(adata.obs.columns)}")

# Overwrite the original file
adata.write_h5ad("/home/b5cc/sanjukta.b5cc/st3/experiments/30k/inference_results/generated_cells_5000.h5ad")
print("\n✓ File updated successfully")