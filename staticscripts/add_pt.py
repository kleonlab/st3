import torch

# Load the existing protein embeddings
pt_file = "/home/b5cc/sanjukta.b5cc/st3/datasets/30k/protein_embeddings.pt"
protein_embeddings = torch.load(pt_file)

print(f"Loaded protein embeddings from: {pt_file}")
print(f"Number of genes: {len(protein_embeddings)}")

# Check the structure and get embedding dimension
sample_gene = list(protein_embeddings.keys())[0]
sample_embedding = protein_embeddings[sample_gene]
embedding_dim = sample_embedding.shape[0]

print(f"\nSample gene: {sample_gene}")
print(f"Embedding shape: {sample_embedding.shape}")
print(f"Embedding dimension: {embedding_dim}")

# Check if 'non-targeting' already exists
if 'non-targeting' in protein_embeddings:
    print(f"\n⚠️  'non-targeting' already exists in the dictionary")
else:
    # Create null embedding (zeros) with the same dimension
    null_embedding = torch.zeros(embedding_dim)
    
    # Add it to the dictionary
    protein_embeddings['non-targeting'] = null_embedding
    
    print(f"\n✓ Added 'non-targeting' with null embedding of shape {null_embedding.shape}")

# Save the updated embeddings
output_file = pt_file  # Overwrite the original file
# Or use a new file: output_file = "/home/b5cc/sanjukta.b5cc/st3/datasets/30k/protein_embeddings_updated.pt"

torch.save(protein_embeddings, output_file)
print(f"\nSaved updated embeddings to: {output_file}")
print(f"Total genes now: {len(protein_embeddings)}")

# Verify the addition
print("\n--- Verification ---")
print(f"'non-targeting' in embeddings: {'non-targeting' in protein_embeddings}")
if 'non-targeting' in protein_embeddings:
    nt_embedding = protein_embeddings['non-targeting']
    print(f"'non-targeting' embedding shape: {nt_embedding.shape}")
    print(f"'non-targeting' embedding sum: {nt_embedding.sum().item()}")
    print(f"All zeros: {torch.all(nt_embedding == 0).item()}")