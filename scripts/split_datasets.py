import scanpy as sc

adata_path = "/home/b5cc/sanjukta.b5cc/aracneseq/datasets/k562.h5ad"

adata = sc.read_h5ad(adata_path)
expression = adata.X
NUM_BINS = int(expression.max())

gene_names = adata.var['gene_name'].tolist()
num_genes = len(gene_names)

from sedd.data import train_val_split
import torch
dataset = torch.tensor(expression).long() 

train_dataset, test_dataset = train_val_split(dataset, val_fraction=0.1, seed=42)
print(f'Train size: {len(train_dataset)}, Val size: {len(test_dataset)}')

# Save train and test datasets as new h5ad files
train_adata = adata[train_dataset.indices].copy() if hasattr(train_dataset, 'indices') else adata[:len(train_dataset)].copy()
test_adata = adata[test_dataset.indices].copy() if hasattr(test_dataset, 'indices') else adata[-len(test_dataset):].copy()

train_adata.write_h5ad("datasets/k562_train.h5ad")
test_adata.write_h5ad("datasets/k562_test.h5ad")
print("Train and test AnnData objects saved as 'k562_5k_train.h5ad' and 'test_dataset.h5ad'.")

