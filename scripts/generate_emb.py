#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch


def load_perturbations(path: str) -> list[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def dedupe_preserve_order(items: list[str]) -> tuple[list[str], list[str]]:
    seen = set()
    unique = []
    duplicates = [] 
    for item in items:
        if item in seen:
            duplicates.append(item)
            continue
        seen.add(item)
        unique.append(item)
    return unique, duplicates


def build_mapping(perturbations: list[str], control_name: str | None) -> dict[str, int]:
    if control_name and control_name in perturbations:
        others = [p for p in perturbations if p != control_name]
        ordered = [control_name] + others
    else:
        ordered = perturbations
    return {p: i for i, p in enumerate(ordered)}


def main():
    parser = argparse.ArgumentParser(
        description="Generate perturbation-to-index mapping and save as .pt"
    )
    parser.add_argument(
        "--perturbations_file",
        type=str,
        default="/home/b5cc/sanjukta.b5cc/st3/datasets/30k/perturbation_names_all.txt",
        help="Path to text file with one perturbation label per line",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/home/b5cc/sanjukta.b5cc/st3/datasets/30k/perturbation_mapping.pt",
        help="Output .pt path",
    )
    parser.add_argument(
        "--control_name",
        type=str,
        default="non-targeting",
        help="Optional control perturbation name to force index 0",
    )
    args = parser.parse_args()

    perturbations = load_perturbations(args.perturbations_file)
    if not perturbations:
        raise ValueError(f"No perturbations found in {args.perturbations_file}")

    perturbations, duplicates = dedupe_preserve_order(perturbations)
    if duplicates:
        print(f"WARNING: Found {len(duplicates)} duplicate names; deduplicated list.")
        print("Duplicate examples:", duplicates[:5])

    mapping = build_mapping(perturbations, args.control_name)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mapping, output_path)

    print(f"Saved {len(mapping)} perturbations to {output_path}")
    print("Example entries:", list(mapping.items())[:5])


if __name__ == "__main__":
    main()

