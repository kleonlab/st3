import scanpy as sc

adata_path = "/home/b5cc/sanjukta.b5cc/aracneseq/datasets/k562_5k.h5ad"

adata = sc.read_h5ad(adata_path)
expression = adata.X
NUM_BINS = int(expression.max())

gene_names = adata.var['gene_name'].tolist()
num_genes = len(gene_names)

from sedd.data import train_val_split
import torch
dataset = torch.tensor(expression).long() 

train_dataset, test_dataset = train_val_split(dataset, test_fraction=0.1, seed=42)
print(f'Train size: {len(train_dataset)}, Val size: {len(test_dataset)}')