#!/usr/bin/env python3
"""
Fetch protein sequences from UniProt given a list of gene names.
"""
import argparse
import time
from pathlib import Path
from typing import Dict, Optional
import requests


def fetch_uniprot_sequence(gene_name: str, organism: str = "9606") -> Optional[str]:
    """
    Fetch the canonical protein sequence for a gene from UniProt.

    Args:
        gene_name: Gene symbol (e.g., 'TP53', 'BRCA1')
        organism: Taxonomy ID (default: 9606 for Homo sapiens)

    Returns:
        Protein sequence as string, or None if not found
    """
    # Query UniProt API
    url = "https://rest.uniprot.org/uniprotkb/search"
    query = f"(gene:{gene_name}) AND (organism_id:{organism}) AND (reviewed:true)"

    params = {
        "query": query,
        "format": "fasta",
        "size": 1  # Get only the first (canonical) result
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        # Parse FASTA format
        fasta_text = response.text.strip()
        if not fasta_text or fasta_text.startswith("<!DOCTYPE"):
            return None

        # Extract sequence (skip header line)
        lines = fasta_text.split('\n')
        if len(lines) < 2:
            return None

        sequence = ''.join(lines[1:])  # Join all lines after header
        return sequence if sequence else None

    except requests.RequestException as e:
        print(f"Error fetching {gene_name}: {e}")
        return None


def load_gene_names(file_path: str) -> list[str]:
    """Load gene names from a text file (one per line)."""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def fetch_all_sequences(
    gene_names: list[str],
    organism: str = "9606",
    delay: float = 0.5
) -> Dict[str, str]:
    """
    Fetch sequences for all genes with rate limiting.

    Args:
        gene_names: List of gene symbols
        organism: Taxonomy ID (default: 9606 for Homo sapiens)
        delay: Delay between requests in seconds

    Returns:
        Dictionary mapping gene names to protein sequences
    """
    sequences = {}
    failed = []

    total = len(gene_names)
    print(f"Fetching sequences for {total} genes...")

    for i, gene_name in enumerate(gene_names, 1):
        print(f"[{i}/{total}] Fetching {gene_name}...", end=' ')

        sequence = fetch_uniprot_sequence(gene_name, organism)

        if sequence:
            sequences[gene_name] = sequence
            print(f"✓ ({len(sequence)} aa)")
        else:
            failed.append(gene_name)
            print("✗ Not found")

        # Rate limiting to be respectful to UniProt servers
        if i < total:
            time.sleep(delay)

    if failed:
        print(f"\n⚠️  Failed to fetch {len(failed)} genes:")
        for gene in failed:
            print(f"  - {gene}")

    print(f"\n✓ Successfully fetched {len(sequences)}/{total} sequences")

    return sequences


def save_sequences(sequences: Dict[str, str], output_path: str):
    """Save sequences to a TSV file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("gene_name\tprotein_sequence\n")
        for gene_name, sequence in sequences.items():
            f.write(f"{gene_name}\t{sequence}\n")

    print(f"\n✓ Saved sequences to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch protein sequences from UniProt for a list of gene names"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input text file with gene names (one per line)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output TSV file for sequences"
    )
    parser.add_argument(
        "--organism",
        type=str,
        default="9606",
        help="Organism taxonomy ID (default: 9606 for Homo sapiens)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between API requests in seconds (default: 0.5)"
    )

    args = parser.parse_args()

    # Load gene names
    gene_names = load_gene_names(args.input)
    print(f"Loaded {len(gene_names)} gene names from {args.input}")

    # Fetch sequences
    sequences = fetch_all_sequences(gene_names, args.organism, args.delay)

    if not sequences:
        print("⚠️  No sequences fetched. Exiting.")
        return

    # Save results
    save_sequences(sequences, args.output)


if __name__ == "__main__":
    main()
