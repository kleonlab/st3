import scanpy as sc
import pandas as pd
import numpy as np
from typing import Tuple, Optional

import os
#adata = sc.read_h5ad("../datasets/competition_support_set/k562_gwps.h5")


h5_path = "/home/b5cc/sanjukta.b5cc/st3/datasets/30k/k562_test_split.h5ad"
adata = sc.read_h5ad(h5_path)

perturbation_list = adata.obs['gene'].value_counts()      

print(perturbation_list) 

# 1. Get the list of unique perturbation names (the index of the value_counts)
perturbations = perturbation_list.index.tolist()

# 2. Define the output path
output_file = "/home/b5cc/sanjukta.b5cc/st3/datasets/30k/perturbation_names_test.txt"

# 3. Save to a text file (one perturbation per line)
with open(output_file, "w") as f:
    for p in perturbations:
        f.write(f"{p}\n")

print(f"Successfully saved {len(perturbations)} perturbation names to {output_file}")