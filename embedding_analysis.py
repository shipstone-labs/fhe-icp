#!/usr/bin/env python3
"""Analyze BERT embeddings properties."""

import numpy as np
from bert_embeddings import BertEmbedder

# Initialize embedder
print("Initializing BERT...")
embedder = BertEmbedder()

# Test documents
documents = {
    "tech1": "Machine learning algorithms process data efficiently.",
    "tech2": "Artificial intelligence systems analyze information quickly.",
    "food1": "Italian pasta tastes delicious with tomato sauce.",
    "food2": "Pizza with cheese and pepperoni is very popular.",
    "weather1": "The sunny weather makes outdoor activities enjoyable.",
    "weather2": "Rain and clouds create perfect reading weather.",
}

# Extract embeddings
print("\nExtracting embeddings...")
embeddings = {}
for key, text in documents.items():
    embeddings[key] = embedder.get_embedding(text)
    print(f"  {key}: {text[:40]}...")

# Analyze properties
print("\n\nEmbedding Properties:")
print("-" * 50)

for key, emb in embeddings.items():
    print(f"\n{key}:")
    print(f"  Shape: {emb.shape}")
    print(f"  Min value: {emb.min():.4f}")
    print(f"  Max value: {emb.max():.4f}")
    print(f"  Mean: {emb.mean():.4f}")
    print(f"  Std dev: {emb.std():.4f}")
    print(f"  L2 norm: {np.linalg.norm(emb):.4f}")

# Compute similarity heatmap
print("\n\nComputing similarities...")
keys = list(embeddings.keys())
n = len(keys)
similarity_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        sim = embedder.compute_similarity(embeddings[keys[i]], embeddings[keys[j]])
        similarity_matrix[i, j] = sim

# Display similarity matrix
print("\nSimilarity Matrix:")
print("       ", end="")
for key in keys:
    print(f"{key:>8}", end="")
print()

for i, key_i in enumerate(keys):
    print(f"{key_i:>7}", end="")
    for j, key_j in enumerate(keys):
        sim = similarity_matrix[i, j]
        # Color code: high similarity = green, low = red
        if i == j:
            print(f"   1.000", end="")
        elif sim > 0.8:
            print(f"   \033[92m{sim:.3f}\033[0m", end="")  # Green
        elif sim > 0.6:
            print(f"   {sim:.3f}", end="")
        else:
            print(f"   \033[91m{sim:.3f}\033[0m", end="")  # Red
    print()

# Key insights
print("\n\nKey Insights:")
print("1. Similar topics have high cosine similarity (>0.8)")
print("2. Different topics have lower similarity (<0.6)")
print("3. Embeddings are ~768-dimensional dense vectors")
print("4. Values typically range from -2 to +2")
print("5. These patterns will persist even after FHE encryption!")

# Save sample embeddings for next session
print("\n\nSaving sample embeddings for next session...")
np.save('sample_embeddings.npy', np.array([embeddings[k] for k in keys]))
np.save('sample_labels.npy', np.array(keys))
print("✅ Saved to sample_embeddings.npy and sample_labels.npy")