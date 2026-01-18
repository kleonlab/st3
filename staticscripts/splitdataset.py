import scanpy as sc
import pandas as pd
from typing import Tuple

import os
#adata = sc.read_h5ad("../datasets/competition_support_set/k562_gwps.h5")
adata = sc.read_h5ad("/home/b5cc/sanjukta.b5cc/aracneseq/datasets/k562_5k.h5ad")
perturbation_list = adata.obs['gene'].value_counts()      

print(perturbation_list) 

gene_names = adata.var['gene_name'].tolist()
num_genes = len(gene_names)

import scanpy as sc
import numpy as np
import pandas as pd
from typing import Tuple, Optional

def stratified_gene_split_old(
    adata_path: str,
    seed: Optional[int] = 42
) -> Tuple[sc.AnnData, sc.AnnData]:
    """
    Splits an AnnData object into Train/Test based on gene expression density.
    """
    # 1. Load the dataset
    adata = sc.read_h5ad(adata_path)
    np.random.seed(seed)
    
    # 2. Calculate cell counts per gene (density)
    # We sum the boolean mask where expression > 0
    gene_counts = np.array((adata.X > 0).sum(axis=0)).flatten()
    gene_names = adata.var_names.tolist()
    
    df_genes = pd.DataFrame({
        'gene': gene_names,
        'count': gene_counts
    })

    # 3. Define the buckets
    high_density = df_genes[df_genes['count'] > 50]
    med_density = df_genes[(df_genes['count'] >= 10) & (df_genes['count'] <= 50)]
    low_density = df_genes[df_genes['count'] < 10]

    print(f"Bucket sizes found:")
    print(f"High (>50): {len(high_density)}")
    print(f"Med (10-50): {len(med_density)}")
    print(f"Low (<10): {len(low_density)}")

    # 4. Perform the sampling (Check if enough genes exist in each bucket)
    try:
        test_genes_high = high_density.sample(n=45, random_state=seed)['gene'].tolist()
        test_genes_med = med_density.sample(n=3, random_state=seed)['gene'].tolist()
        test_genes_low = low_density.sample(n=1, random_state=seed)['gene'].tolist()
    except ValueError as e:
        raise ValueError(f"One of your density buckets doesn't have enough genes: {e}")

    test_gene_list = test_genes_high + test_genes_med + test_genes_low
    
    # 5. Create the Splits
    # Test split: only the 50 selected genes
    test_adata = adata[:, test_gene_list].copy()
    
    # Train split: all other genes
    train_genes = [g for g in gene_names if g not in test_gene_list]
    train_adata = adata[:, train_genes].copy()

    print(f"Split complete. Train genes: {len(train_genes)}, Test genes: {len(test_gene_list)}")
    return train_adata, test_adata


def stratified_label_split_2(adata_path: str, seed: int = 42) -> Tuple[sc.AnnData, sc.AnnData]:
    # 1. Load data
    adata = sc.read_h5ad(adata_path)
    
    # 2. Count cells per perturbation label
    # This identifies how many cells each 'gene' label (perturbation) has
    label_counts = adata.obs['gene'].value_counts().reset_index()
    label_counts.columns = ['label_name', 'cell_count']

    # 3. Define the buckets
    high = label_counts[label_counts['cell_count'] > 50]
    med = label_counts[(label_counts['cell_count'] >= 10) & (label_counts['cell_count'] <= 50)]
    low = label_counts[label_counts['cell_count'] < 10]

    print(f"--- Bucket Inventory ---")
    print(f"High (>50 cells): {len(high)} labels available")
    print(f"Med (10-50 cells): {len(med)} labels available")
    print(f"Low (<10 cells): {len(low)} labels available")

    # Helper function to prevent the "larger sample than population" error
    def get_sample(df, n_requested, bucket_name):
        if len(df) == 0:
            print(f"Warning: Bucket {bucket_name} is empty!")
            return []
        if len(df) < n_requested:
            print(f"Warning: Requested {n_requested} from {bucket_name}, but only {len(df)} available. Taking all.")
            return df['label_name'].tolist()
        return df.sample(n=n_requested, random_state=seed)['label_name'].tolist()

    # 4. Stratified Sampling with safety checks
    test_labels = []
    test_labels.extend(get_sample(high, 45, "High"))
    test_labels.extend(get_sample(med, 3, "Med"))
    test_labels.extend(get_sample(low, 2, "Low"))

    print(f"Total unique labels selected for Test: {len(test_labels)}")

    # 5. Create the splits
    # Test set: All cells matching the sampled labels
    test_adata = adata[adata.obs['gene'].isin(test_labels)].copy()
    
    # Train set: All other cells
    train_adata = adata[~adata.obs['gene'].isin(test_labels)].copy()

    print(f"Final Split: Train={train_adata.n_obs} cells, Test={test_adata.n_obs} cells")
    return train_adata, test_adata


def stratified_label_split(adata_path: str, seed: int = 42) -> Tuple[sc.AnnData, sc.AnnData]:
    # 1. Load data
    adata = sc.read_h5ad(adata_path)
    
    # 2. Separate non-targeting controls from the rest
    is_control = adata.obs['gene'] == 'non-targeting'
    control_indices = adata.obs_names[is_control]
    perturbed_indices = adata.obs_names[~is_control]
    
    # --- Part A: Handle Non-Targeting Split (1% for Test) ---
    n_control = len(control_indices)
    n_test_control = int(n_control * 0.01+1)
    
    np.random.seed(seed)
    test_control_idx = np.random.choice(control_indices, size=n_test_control, replace=False)
    train_control_idx = np.setdiff1d(control_indices, test_control_idx)
    
    # --- Part B: Handle Perturbed Gene Sampling (The 50 Genes) ---
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

    # --- Part C: Final Assembly ---
    # Test set: All cells of the 50 sampled genes + 1% of control cells
    test_perturbed_idx = adata.obs_names[adata.obs['gene'].isin(test_gene_labels)]
    
    # Combine indices
    final_test_idx = np.concatenate([test_perturbed_idx, test_control_idx])
    
    # Train set: All cells of genes NOT in test list + 99% of control cells
    # (Excluding the test_perturbed genes and the test_control cells)
    train_perturbed_idx = np.setdiff1d(perturbed_indices, test_perturbed_idx)
    final_train_idx = np.concatenate([train_perturbed_idx, train_control_idx])

    # Slice the AnnData
    test_adata = adata[final_test_idx].copy()
    train_adata = adata[final_train_idx].copy()

    print(f"--- Final Split Stats ---")
    print(f"Test Genes sampled: {len(test_gene_labels)}")
    print(f"Non-targeting in Test: {len(test_control_idx)} cells")
    print(f"Non-targeting in Train: {len(train_control_idx)} cells")
    print(f"Total: Train={train_adata.n_obs} cells, Test={test_adata.n_obs} cells")

    train.write_h5ad("train_split.h5ad")
    test.write_h5ad("test_split.h5ad")

    return train_adata, test_adata


# Example usage:
train, test = stratified_label_split("/home/b5cc/sanjukta.b5cc/aracneseq/datasets/k562_5k.h5ad")

print(train.obs['gene'].value_counts())
print(test.obs['gene'].value_counts())

