import scanpy as sc
import pandas as pd
import numpy as np
from typing import Tuple, Optional

import os

def stratified_label_split(adata_path: str, seed: int = 42) -> Tuple[sc.AnnData, sc.AnnData]:
    # 1. Load data
    adata = sc.read_h5ad(adata_path)
    
    # 2. Separate non-targeting controls from the rest
    is_control = adata.obs['gene'] == 'non-targeting'
    control_indices = adata.obs_names[is_control]
    perturbed_indices = adata.obs_names[~is_control]
    
    # --- Part A: Handle Non-Targeting Split (1% for Test) ---
    n_control = len(control_indices)
    n_test_control = int(n_control * 0.01 + 1)
    
    np.random.seed(seed)
    test_control_idx = np.random.choice(control_indices, size=n_test_control, replace=False)
    train_control_idx = np.setdiff1d(control_indices, test_control_idx)
    
    # --- Part B: Sample Test Genes (50 genes) ---
    # We only sample from genes that are NOT 'non-targeting'
    label_counts = adata.obs.loc[perturbed_indices, 'gene'].value_counts().reset_index()
    label_counts.columns = ['label_name', 'cell_count']

    high = label_counts[label_counts['cell_count'] > 50]
    med = label_counts[(label_counts['cell_count'] >= 10) & (label_counts['cell_count'] <= 50)]
    low = label_counts[label_counts['cell_count'] < 10]

    def get_sample(df, n_requested, bucket_name):
        if len(df) == 0: return []
        if len(df) < n_requested:
            print(f"Warning: {bucket_name} only has {len(df)} labels. Taking all.")
            return df['label_name'].tolist()
        return df.sample(n=n_requested, random_state=seed)['label_name'].tolist()

    test_gene_labels = []
    test_gene_labels.extend(get_sample(high, 45, "High"))
    test_gene_labels.extend(get_sample(med, 3, "Med"))
    test_gene_labels.extend(get_sample(low, 2, "Low"))

    # --- Part C: Complete Separation - ALL cells of test genes go to test ---
    # Test set: ALL cells from the 50 sampled genes + 1% of control cells
    # Train set: ALL cells from remaining genes + 99% of control cells
    
    test_perturbed_idx = []
    
    # For each selected test gene, take ALL its cells
    for gene_label in test_gene_labels:
        gene_cells = adata.obs_names[adata.obs['gene'] == gene_label]
        test_perturbed_idx.extend(gene_cells)
    
    test_perturbed_idx = np.array(test_perturbed_idx)
    
    # Train set: All cells from genes NOT in test gene list
    train_perturbed_idx = adata.obs_names[
        (~is_control) & (~adata.obs['gene'].isin(test_gene_labels))
    ]
    
    # Combine indices
    final_test_idx = np.concatenate([test_perturbed_idx, test_control_idx])
    final_train_idx = np.concatenate([train_perturbed_idx, train_control_idx])
    
    # Slice the AnnData
    test_adata = adata[final_test_idx].copy()
    train_adata = adata[final_train_idx].copy()

    # Verification: ensure no gene overlap (except non-targeting)
    train_genes = set(train_adata.obs['gene'].unique())
    test_genes = set(test_adata.obs['gene'].unique())
    overlap = train_genes.intersection(test_genes) - {'non-targeting'}
    
    print(f"--- Final Split Stats ---")
    print(f"Test Genes sampled: {len(test_gene_labels)}")
    print(f"  - {len(test_perturbed_idx)} perturbed cells in Test")
    print(f"Train Genes: {len(train_genes) - 1} (excluding non-targeting)")  # -1 for non-targeting
    print(f"  - {len(train_perturbed_idx)} perturbed cells in Train")
    print(f"Non-targeting in Test: {len(test_control_idx)} cells")
    print(f"Non-targeting in Train: {len(train_control_idx)} cells")
    print(f"Total: Train={train_adata.n_obs} cells, Test={test_adata.n_obs} cells")
    print(f"Gene overlap (should be empty): {overlap}")

    train_adata.write_h5ad("/home/b5cc/sanjukta.b5cc/st3/datasets/5k/k562_5k_train_split.h5ad")
    test_adata.write_h5ad("/home/b5cc/sanjukta.b5cc/st3/datasets/5k/k562_5k_test_split.h5ad")

    return train_adata, test_adata


# Example usage:
if __name__ == "__main__":
    adata = sc.read_h5ad("/home/b5cc/sanjukta.b5cc/aracneseq/datasets/k562_5k.h5ad")
    perturbation_list = adata.obs['gene'].value_counts()      
    print(perturbation_list)
    
    train, test = stratified_label_split("/home/b5cc/sanjukta.b5cc/aracneseq/datasets/k562_5k.h5ad")
    
    print("\n--- Train Gene Counts ---")
    print(train.obs['gene'].value_counts())
    print("\n--- Test Gene Counts ---")
    print(test.obs['gene'].value_counts())