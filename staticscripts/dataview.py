import scanpy as sc
import os
#adata = sc.read_h5ad("../datasets/competition_support_set/k562_gwps.h5")
adata = sc.read_h5ad("/home/b5cc/sanjukta.b5cc/st3/datasets/dentate/dentate.h5ad")

print(adata)
print(len(adata))
print(list(adata.obs.columns))

print(adata.obs['molecule'].value_counts())
# Save as a simple text file with two columns
counts = adata.obs['molecule'].value_counts()
output_dir = "/home/b5cc/sanjukta.b5cc/st3/datasets/dentate"
output_path = os.path.join(output_dir, "percount.gc")

counts.to_csv(output_path, sep="\t", header=True)

