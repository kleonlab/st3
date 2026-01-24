import pandas as pd
import scanpy as sc
from pathlib import Path

# Paths
DATA_DIR = Path("/home/b5cc/sanjukta.b5cc/st3/datasets/dentate")
COUNTS_TAB = DATA_DIR / "GSE104323_10X_expression_data_V2.tab"
META_TAB   = DATA_DIR / "GSE104323_metadata_barcodes_24185cells.txt"
OUT_H5AD   = DATA_DIR / "dentate.h5ad"

# 1. Load count matrix (genes × cells)
counts = pd.read_csv(
    COUNTS_TAB,
    sep="\t",
    index_col=0
)

print("Counts loaded:", counts.shape)

# 2. Load metadata
meta = pd.read_csv(
    META_TAB,
    sep="\t",
    index_col=0
)

print("Metadata loaded:", meta.shape)

# 3. Build AnnData (cells × genes)
adata = sc.AnnData(counts.T)
print("Metadata loaded:", meta.shape)

# Align metadata to cells
adata.obs = meta.loc[adata.obs_names]

# Optional: store gene names explicitly
adata.var_names = counts.index
print(len(adata))
# 4. Write proper HDF5-backed AnnData
adata.write_h5ad(
    OUT_H5AD,
    compression="gzip"
)

print("Saved:", OUT_H5AD)
# Align metadat
