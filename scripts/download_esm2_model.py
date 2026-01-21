#!/usr/bin/env python3
"""
Download ESM2 model from HuggingFace and save to a local directory.
"""
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModel


def download_esm2_model(output_dir: str, model_name: str = "nvidia/esm2_t6_8M_UR50D"):
    """
    Download ESM2 model and tokenizer from HuggingFace.

    Args:
        output_dir: Directory to save the model
        model_name: HuggingFace model identifier
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading ESM2 model: {model_name}")
    print(f"Saving to: {output_path}")

    # Download tokenizer
    print("\n[1/2] Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_path)
    print("✓ Tokenizer saved")

    # Download model
    print("\n[2/2] Downloading model weights...")
    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(output_path)
    print("✓ Model weights saved")

    print(f"\n✓ Successfully downloaded model to {output_path}")
    print(f"\nModel info:")
    print(f"  - Parameters: ~8M")
    print(f"  - Embedding dimension: {model.config.hidden_size}")
    print(f"  - Number of layers: {model.config.num_hidden_layers}")


def main():
    parser = argparse.ArgumentParser(
        description="Download ESM2 model from HuggingFace"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the model"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="nvidia/esm2_t6_8M_UR50D",
        help="HuggingFace model identifier (default: nvidia/esm2_t6_8M_UR50D)"
    )

    args = parser.parse_args()

    try:
        download_esm2_model(args.output_dir, args.model_name)
    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        raise


if __name__ == "__main__":
    main()
