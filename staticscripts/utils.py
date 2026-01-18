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

