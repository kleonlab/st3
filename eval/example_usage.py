"""
Example usage of single-cell RNA-seq evaluation metrics.

This script demonstrates how to use the evaluation metrics to compare
different single-cell RNA-seq models (scLDM, CFGen, scVI, scDiffusion, etc.).
"""

import numpy as np
from reconstruction_metrics import evaluate_reconstruction
from generation_metrics import evaluate_generation


def example_reconstruction_evaluation():
    """Example: Evaluating reconstruction quality (autoencoders, VAEs)."""
    print("=" * 70)
    print("EXAMPLE 1: Reconstruction Evaluation")
    print("=" * 70)
    print("Scenario: Testing an autoencoder's ability to reconstruct cells\n")

    # Generate synthetic data
    np.random.seed(42)
    n_cells, n_genes = 500, 2000

    print(f"Creating synthetic dataset:")
    print(f"  - {n_cells} cells")
    print(f"  - {n_genes} genes\n")

    # Original data
    original = np.random.randn(n_cells, n_genes)

    # Simulate reconstruction with some error
    noise_level = 0.1
    reconstructed = original + np.random.randn(n_cells, n_genes) * noise_level

    print(f"Simulating reconstruction with {noise_level} noise level\n")

    # Evaluate
    results = evaluate_reconstruction(original, reconstructed, verbose=True)

    print("\nInterpretation:")
    print("  - High PCC (>0.95) indicates good correlation preservation")
    print("  - Low MSE and RE indicate accurate reconstruction")
    print()

    return results


def example_generation_evaluation():
    """Example: Evaluating generation quality (GANs, diffusion models)."""
    print("=" * 70)
    print("EXAMPLE 2: Generation Evaluation")
    print("=" * 70)
    print("Scenario: Testing a generative model's synthetic cell quality\n")

    # Generate synthetic data
    np.random.seed(42)
    n_real_cells = 1000
    n_gen_cells = 1000
    n_genes = 2000

    print(f"Creating synthetic datasets:")
    print(f"  - Real data: {n_real_cells} cells × {n_genes} genes")
    print(f"  - Generated data: {n_gen_cells} cells × {n_genes} genes\n")

    # Real data from one distribution
    real = np.random.randn(n_real_cells, n_genes)

    # Generated data from slightly different distribution
    # Simulating a model that's pretty good but not perfect
    generated = np.random.randn(n_gen_cells, n_genes) * 1.05 + 0.1

    print("Simulating generated data with slight distribution shift\n")

    # Evaluate
    results = evaluate_generation(
        real, generated,
        w2_projections=1000,
        mmd_subsample=2000,
        verbose=True
    )

    print("\nInterpretation:")
    print("  - Lower W2 means generated distribution is closer to real")
    print("  - Lower MMD2 indicates better statistical match")
    print("  - Lower FD shows similar mean and covariance structure")
    print()

    return results


def example_model_comparison():
    """Example: Comparing multiple models."""
    print("=" * 70)
    print("EXAMPLE 3: Multi-Model Comparison")
    print("=" * 70)
    print("Scenario: Comparing 4 different generative models\n")

    # Generate synthetic data
    np.random.seed(42)
    n_cells = 800
    n_genes = 1500

    print(f"Dataset: {n_cells} cells × {n_genes} genes\n")

    # Real data
    real = np.random.randn(n_cells, n_genes)

    # Simulate different models with varying quality
    models = {
        'scLDM': real + np.random.randn(n_cells, n_genes) * 0.05,  # Best
        'CFGen': real + np.random.randn(n_cells, n_genes) * 0.15,  # Good
        'scVI': real + np.random.randn(n_cells, n_genes) * 0.25,   # OK
        'scDiffusion': real + np.random.randn(n_cells, n_genes) * 0.20,  # Better than scVI
    }

    print("Models to compare: scLDM, CFGen, scVI, scDiffusion\n")

    # Evaluate each model
    all_results = {}
    for model_name, generated in models.items():
        print(f"\nEvaluating {model_name}...")
        print("-" * 70)
        results = evaluate_generation(
            real, generated,
            w2_projections=500,  # Reduced for speed in example
            mmd_subsample=1000,
            verbose=False
        )
        all_results[model_name] = results

        print(f"  W2:       {results['w2']:.6f}")
        print(f"  MMD2:     {results['mmd2_rbf']:.6f}")
        print(f"  FD:       {results['fd']:.6f}")

    # Create comparison table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Model':<15} {'W2 ↓':<12} {'MMD2 ↓':<12} {'FD ↓':<12}")
    print("-" * 70)

    for model_name, results in all_results.items():
        print(f"{model_name:<15} {results['w2']:<12.6f} {results['mmd2_rbf']:<12.6f} {results['fd']:<12.6f}")

    print("\n↓ = lower is better\n")

    # Identify best model
    best_w2 = min(all_results.items(), key=lambda x: x[1]['w2'])
    best_mmd = min(all_results.items(), key=lambda x: x[1]['mmd2_rbf'])
    best_fd = min(all_results.items(), key=lambda x: x[1]['fd'])

    print("Best performing models:")
    print(f"  Best W2:   {best_w2[0]}")
    print(f"  Best MMD2: {best_mmd[0]}")
    print(f"  Best FD:   {best_fd[0]}")
    print()

    return all_results


def example_with_real_data_format():
    """Example: Working with different data formats."""
    print("=" * 70)
    print("EXAMPLE 4: Data Loading Examples")
    print("=" * 70)
    print()

    # Example data
    np.random.seed(42)
    n_cells, n_genes = 100, 50
    example_data = np.random.randn(n_cells, n_genes)

    # Save in different formats
    print("Saving example data in different formats:")

    # NumPy format
    np.save('example_data.npy', example_data)
    print("  ✓ example_data.npy")

    # Compressed NumPy format
    np.savez_compressed('example_data.npz', expression=example_data)
    print("  ✓ example_data.npz")

    # CSV format
    np.savetxt('example_data.csv', example_data, delimiter=',')
    print("  ✓ example_data.csv")

    print("\nLoading examples:")
    print("""
    # NumPy
    data = np.load('example_data.npy')

    # Compressed NumPy
    with np.load('example_data.npz') as f:
        data = f['expression']

    # CSV
    data = np.loadtxt('example_data.csv', delimiter=',')

    # AnnData (requires scanpy)
    import scanpy as sc
    adata = sc.read_h5ad('data.h5ad')
    data = adata.X
    if hasattr(data, 'toarray'):
        data = data.toarray()
    """)

    print("\nCommand-line evaluation:")
    print("""
    # NumPy files
    python eval/evaluate.py --real real.npy --pred generated.npy --mode generation

    # AnnData files
    python eval/evaluate.py --real real.h5ad --pred generated.h5ad --mode both

    # Save results
    python eval/evaluate.py --real real.npy --pred generated.npy --output results.json
    """)

    # Cleanup
    import os
    for f in ['example_data.npy', 'example_data.npz', 'example_data.csv']:
        if os.path.exists(f):
            os.remove(f)

    print("\nExample files cleaned up.")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("*" * 70)
    print("  Single-Cell RNA-seq Evaluation Metrics - Examples")
    print("*" * 70)
    print()

    # Run examples
    example_reconstruction_evaluation()
    input("\nPress Enter to continue to next example...\n")

    example_generation_evaluation()
    input("\nPress Enter to continue to next example...\n")

    example_model_comparison()
    input("\nPress Enter to continue to next example...\n")

    example_with_real_data_format()

    print("=" * 70)
    print("All examples completed!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Prepare your real and generated data")
    print("  2. Use evaluate.py to compute metrics")
    print("  3. Compare results across models")
    print("\nSee eval/README.md for more details.")
    print()


if __name__ == "__main__":
    main()
