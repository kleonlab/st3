#!/usr/bin/env python3
"""
Download ESM2 model from HuggingFace and save to a local directory.
"""
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import snapshot_download


def download_esm2_model(
    output_dir: str,
    model_name: str = "facebook/esm2_t6_8M_UR50D",
    trust_remote_code: bool = False,
):
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
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    tokenizer.save_pretrained(output_path)
    print("✓ Tokenizer saved")

    # Download model
    print("\n[2/2] Downloading model weights...")
    try:
        model = AutoModel.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        model.save_pretrained(output_path)
        print("✓ Model weights saved")
        print(f"\nModel info:")
        print(f"  - Parameters: ~8M")
        print(f"  - Embedding dimension: {model.config.hidden_size}")
        print(f"  - Number of layers: {model.config.num_hidden_layers}")
    except Exception as e:
        print(f"✗ Model load failed ({e}). Falling back to snapshot download.")
        snapshot_download(
            repo_id=model_name,
            local_dir=str(output_path),
            local_dir_use_symlinks=False,
            allow_patterns=[
                "*.bin",
                "*.safetensors",
                "*.json",
                "*.txt",
                "*.model",
                "*.py",
            ],
        )
        print("✓ Snapshot downloaded (weights/config files)")

    print(f"\n✓ Successfully downloaded model to {output_path}")


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
        default="facebook/esm2_t6_8M_UR50D",
        help="HuggingFace model identifier (default: facebook/esm2_t6_8M_UR50D)"
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow execution of custom code from model repository",
    )

    args = parser.parse_args()

    try:
        download_esm2_model(
            args.output_dir,
            args.model_name,
            trust_remote_code=args.trust_remote_code,
        )
    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        raise


if __name__ == "__main__":
    main()
