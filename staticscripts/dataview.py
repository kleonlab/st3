import scanpy as sc
import os
#adata = sc.read_h5ad("../datasets/competition_support_set/k562_gwps.h5")
adata = sc.read_h5ad("/home/b5cc/sanjukta.b5cc/st3/datasets/20M/combined_train.h5ad")

print(adata)
print(len(adata))
print(list(adata.obs.columns))

print(adata.X[0])
print(adata.X[2])

print(adata.shape)

#print(adata.obs['gene'].value_counts())
print(f"Number of unique values in 'gene' column: {adata.obs['gene'].nunique()}")

#print(adata.obs['characteristics: cell cluster'].value_counts())
#print(adata.obs['characteristics: cell cluster'].unique())


print(adata.obs['cell_type'].value_counts())
print(adata.obs['cell_type'].unique())

# Save as a simple text file with two columns
#counts = adata.obs['gene'].value_counts()
#output_dir = "/home/b5cc/sanjukta.b5cc/st3/datasets/k562"
#output_path = os.path.join(output_dir, "percount.gc")

#counts.to_csv(output_path, sep="\t", header=True)

