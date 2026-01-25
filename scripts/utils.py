import os
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc 

# Get the repository root (parent of scripts folder)
#REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#ASSETS_DIR = os.path.join(REPO_ROOT, "assets")


def plot_cellstate(cellstate, output_path=None, title="Cell State Expression"):
    """
    Plot a cell state vector (RNA-seq expression counts) to a PNG.
    
    Args:
        cellstate: Array-like, discrete expression counts for each transcript (~18k)
        output_path: Path to save the PNG (default: assets/cellstate_plot.png in repo root)
        title: Title for the plots
    """
    if output_path is None:
        output_path = os.path.join(ASSETS_DIR, "cellstate_plot.png")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    cellstate = np.asarray(cellstate).flatten()
    n_transcripts = len(cellstate)
    max_expr = int(np.max(cellstate))
    
    # Compute expressed transcripts
    expressed_mask = cellstate > 0
    expressed_values = cellstate[expressed_mask]
    n_expressed = len(expressed_values)
    
    # Create combined figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot 1: Bar plot of expression across all transcripts
    axes[0].bar(range(n_transcripts), cellstate, width=1.0, color='steelblue', edgecolor='none')
    axes[0].set_xlabel('Transcript Index')
    axes[0].set_ylabel('Expression Count')
    axes[0].set_title(f'{title} - Expression per Transcript (n={n_transcripts})')
    axes[0].set_xlim(0, n_transcripts)
    
    # Plot 2: Histogram of expression value distribution
    bins = np.arange(0, max_expr + 2) - 0.5
    axes[1].hist(cellstate, bins=bins, color='coral', edgecolor='black', alpha=0.8)
    axes[1].set_xlabel('Expression Count')
    axes[1].set_ylabel('Number of Transcripts')
    axes[1].set_title(f'{title} - Distribution of Expression Values (max={max_expr})')
    axes[1].set_xticks(range(0, min(max_expr + 1, 50), max(1, max_expr // 20)))
    
    # Plot 3: Only expressed transcripts (count > 0)
    axes[2].bar(range(n_expressed), expressed_values, width=1.0, color='forestgreen', edgecolor='none')
    axes[2].set_xlabel('Expressed Transcript Index')
    axes[2].set_ylabel('Expression Count')
    axes[2].set_title(f'{title} - Expressed Transcripts Only (n={n_expressed} / {n_transcripts})')
    axes[2].set_xlim(0, n_expressed)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
        
    print(f"Saved cell state plot to: {output_path}")
    print(f"  Total transcripts: {n_transcripts}")
    print(f"  Expressed (>0): {n_expressed}")
    print(f"  Max expression: {max_expr}")
    
    return output_path

def calc_sparsity(cellstate):
    positive = 0
    total = len(cellstate)

    for eachcount in cellstate:
        if eachcount > 0:
            positive = positive + 1
    
    sparsity = (total - positive)/total

    print(sparsity)
    #return (sparsity)


def prepare_dataset(h5ad_path):
    adata = sc.read(h5ad_path)
    print(adata.obs['gene'].unique())

    sc.pp.highly_variable_genes(adata,n_top_genes=2000, subset=False, flavor="seurat_v3")
    adata.obsm["X_hvg"] = adata[:, adata.var.highly_variable].X.copy()
    print("x_hvg done")

    adata.obs["cell_type"] = "single_cell_type"
    adata.obs["cell_type"].value_counts()

    adata.write("/home/b5cc/sanjukta.b5cc/st3/datasets/rpe1_processed.h5ad")


    print("all done, ready for dataloader from cell-load")


def main():
    prepare_dataset("/home/b5cc/sanjukta.b5cc/st3/datasets/rpe1.h5ad")



if __name__ == "__main__":
    main()