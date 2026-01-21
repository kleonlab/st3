#!/usr/bin/env python3
"""
Generate protein embeddings using ESM2 model from HuggingFace.
"""
import argparse
from pathlib import Path
from typing import Dict
import torch
from transformers import AutoTokenizer, AutoModel


def load_gene_names(file_path: str) -> list[str]:
    """Load gene names from a text file (one per line)."""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def load_protein_sequences(file_path: str) -> Dict[str, str]:
    """
    Load protein sequences from TSV file.

    Expected format: gene_name\tprotein_sequence
    """
    sequences = {}
    with open(file_path, 'r') as f:
        # Skip header
        next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                gene_name, sequence = parts
                sequences[gene_name] = sequence
    return sequences


def generate_embeddings(
    sequences: Dict[str, str],
    model_dir: str,
    device: str = None,
    batch_size: int = 8
) -> Dict[str, torch.Tensor]:
    """
    Generate mean-pooled embeddings for protein sequences.

    Args:
        sequences: Dictionary mapping gene names to protein sequences
        model_dir: Path to the downloaded ESM2 model directory
        device: Device to run on ('cuda', 'cpu', or None for auto)
        batch_size: Number of sequences to process at once

    Returns:
        Dictionary mapping gene names to embedding tensors
    """
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model and tokenizer
    print(f"Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    model = model.to(device)
    model.eval()
    print("✓ Model loaded")

    embeddings = {}
    gene_names = list(sequences.keys())
    total = len(gene_names)

    print(f"\nGenerating embeddings for {total} proteins...")

    with torch.no_grad():
        for i in range(0, total, batch_size):
            batch_genes = gene_names[i:i + batch_size]
            batch_sequences = [sequences[gene] for gene in batch_genes]

            # Tokenize
            inputs = tokenizer(
                batch_sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate embeddings
            outputs = model(**inputs)

            # Mean pooling over sequence length (excluding padding)
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state

            # Mask out padding tokens
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask

            # Store embeddings
            for j, gene_name in enumerate(batch_genes):
                embeddings[gene_name] = mean_embeddings[j].cpu()

            print(f"  Processed {min(i + batch_size, total)}/{total}", end='\r')

    print(f"\n✓ Generated embeddings for {len(embeddings)} proteins")

    # Print embedding info
    first_emb = next(iter(embeddings.values()))
    print(f"  Embedding dimension: {first_emb.shape[0]}")

    return embeddings


def save_embeddings(embeddings: Dict[str, torch.Tensor], output_path: str):
    """Save embeddings as a .pt file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert to a dictionary that can be saved
    embedding_dict = {gene: emb for gene, emb in embeddings.items()}

    torch.save(embedding_dict, output_file)
    print(f"\n✓ Saved embeddings to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate protein embeddings using ESM2 model"
    )
    parser.add_argument(
        "--gene_file",
        type=str,
        required=True,
        help="Path to text file with gene names (one per line)"
    )
    parser.add_argument(
        "--sequence_file",
        type=str,
        required=True,
        help="Path to TSV file with protein sequences (output from fetch_protein_sequences.py)"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to downloaded ESM2 model directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output .pt file for embeddings"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help="Device to run on (default: auto-detect)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference (default: 8)"
    )

    args = parser.parse_args()

    # Load gene names from input file
    gene_names = load_gene_names(args.gene_file)
    print(f"Loaded {len(gene_names)} gene names from {args.gene_file}")

    # Load protein sequences
    sequences = load_protein_sequences(args.sequence_file)
    print(f"Loaded {len(sequences)} protein sequences from {args.sequence_file}")

    # Check for missing sequences
    missing = [gene for gene in gene_names if gene not in sequences]
    if missing:
        print(f"\n⚠️  Warning: {len(missing)} genes from input not found in sequence file:")
        for gene in missing[:10]:
            print(f"  - {gene}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

        # Filter to only genes with sequences
        gene_names = [gene for gene in gene_names if gene in sequences]
        sequences = {gene: sequences[gene] for gene in gene_names}
        print(f"\nProceeding with {len(sequences)} genes that have sequences")

    if not sequences:
        print("✗ No sequences to process. Exiting.")
        return

    # Generate embeddings
    embeddings = generate_embeddings(
        sequences,
        args.model_dir,
        args.device,
        args.batch_size
    )

    # Save embeddings
    save_embeddings(embeddings, args.output)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
