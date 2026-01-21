# Protein Embeddings Generation Pipeline

This pipeline generates protein embeddings from gene names using the ESM2 model from NVIDIA/HuggingFace.

## Overview

The pipeline consists of three scripts:

1. `fetch_protein_sequences.py` - Fetches protein sequences from UniProt
2. `download_esm2_model.py` - Downloads the ESM2 model from HuggingFace
3. `generate_protemb.py` - Generates embeddings using the model

## Prerequisites

Install required dependencies:

```bash
pip install torch transformers requests
```

## Usage

### Step 1: Prepare your gene list

Create a text file with one gene name per line:

```
CSE1L
PSMC2
GPS1
ICE1
TRRAP
ANAPC15
RPS19
ESPN
```

### Step 2: Fetch protein sequences from UniProt

```bash
python scripts/fetch_protein_sequences.py \
    --input /path/to/your/gene_names.txt \
    --output /path/to/save/protein_sequences.tsv \
    --organism 9606  # 9606 = Homo sapiens
```

This will query UniProt's API and save the sequences to a TSV file.

### Step 3: Download the ESM2 model

```bash
python scripts/download_esm2_model.py \
    --output_dir /path/to/save/esm2_model
```

This will download the `nvidia/esm2_t6_8M_UR50D` model (~8M parameters) from HuggingFace.

### Step 4: Generate embeddings

```bash
python scripts/generate_protemb.py \
    --gene_file /path/to/your/gene_names.txt \
    --sequence_file /path/to/protein_sequences.tsv \
    --model_dir /path/to/esm2_model \
    --output /path/to/save/embeddings.pt \
    --device cuda  # or 'cpu' if no GPU available
    --batch_size 8
```

## Output Format

The output is a `.pt` file containing a dictionary:
- Keys: Gene names (strings)
- Values: Embedding tensors (torch.Tensor of shape [embedding_dim])

To load the embeddings:

```python
import torch

embeddings = torch.load('/path/to/embeddings.pt')

# Access embedding for a specific gene
gene_embedding = embeddings['CSE1L']
print(f"Embedding shape: {gene_embedding.shape}")  # e.g., torch.Size([320])
```

## Notes

- **Organism**: Default is human (taxonomy ID: 9606). Change with `--organism` flag.
- **Rate limiting**: The UniProt fetching script includes a 0.5s delay between requests by default.
- **GPU**: The embedding generation will automatically use GPU if available, or CPU otherwise.
- **Batch size**: Adjust based on your GPU memory. Larger batches are faster but use more memory.
- **Embedding dimension**: The ESM2-t6-8M model produces 320-dimensional embeddings.

## Troubleshooting

**Gene not found in UniProt**: Some gene names might not be found. Check:
- Ensure the gene name is correct
- Try alternative gene symbols
- Check if the organism ID is correct

**Out of memory**: Reduce the `--batch_size` parameter.

**Model download fails**: Ensure you have internet connection and sufficient disk space (~100MB for the model).
