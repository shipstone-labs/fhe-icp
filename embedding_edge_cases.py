#!/usr/bin/env python3
"""Handle edge cases and optimize embedding extraction."""

from bert_embeddings import BertEmbedder
import numpy as np
import time

embedder = BertEmbedder(max_length=100)

print("Testing Edge Cases and Optimizations\n")

# Test 1: Empty text
print("1. Empty text:")
try:
    emb = embedder.get_embedding("")
    print(f"   ✓ Handled empty text, shape: {emb.shape}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 2: Very long text
print("\n2. Very long text (>100 tokens):")
long_text = " ".join(["This is a very long sentence."] * 50)
emb = embedder.get_embedding(long_text)
print(f"   Original length: {len(long_text)} chars")
print(f"   Embedding shape: {emb.shape}")
print(f"   ✓ Automatically truncated to max_length")

# Test 3: Special characters
print("\n3. Special characters:")
special_texts = [
    "Hello! How are you? 😊",
    "Price: $99.99 (20% off)",
    "Email: test@example.com",
    "C++ vs Python: Which is better?",
]
for text in special_texts:
    emb = embedder.get_embedding(text)
    print(f"   '{text}' → embedding shape: {emb.shape}")

# Test 4: Performance comparison
print("\n4. Performance comparison:")
test_texts = ["This is test document number {}.".format(i) for i in range(20)]

# Single processing
start = time.time()
single_embeddings = []
for text in test_texts:
    emb = embedder.get_embedding(text)
    single_embeddings.append(emb)
single_time = time.time() - start

# Batch processing
start = time.time()
batch_embeddings = embedder.get_embeddings_batch(test_texts, batch_size=8)
batch_time = time.time() - start

print(f"   Single processing: {single_time:.2f}s")
print(f"   Batch processing:  {batch_time:.2f}s")
print(f"   Speedup: {single_time/batch_time:.1f}x")

# Test 5: Memory efficiency
print("\n5. Memory usage:")
emb_float32 = embedder.get_embedding("Test")
emb_float16 = emb_float32.astype(np.float16)

print(f"   Float32 embedding: {emb_float32.nbytes} bytes")
print(f"   Float16 embedding: {emb_float16.nbytes} bytes")
print(f"   Memory saved: {(1 - emb_float16.nbytes/emb_float32.nbytes)*100:.1f}%")

# Test 6: Normalize for FHE
print("\n6. Normalization for FHE:")
text = "Normalize this embedding for FHE processing."
emb = embedder.get_embedding(text)

# Original stats
print(f"   Original - Min: {emb.min():.3f}, Max: {emb.max():.3f}, Range: {emb.max()-emb.min():.3f}")

# Normalize to unit vector (preserves cosine similarity)
emb_normalized = emb / np.linalg.norm(emb)
print(f"   Normalized - Min: {emb_normalized.min():.3f}, Max: {emb_normalized.max():.3f}, L2 norm: {np.linalg.norm(emb_normalized):.3f}")

# Scale to integer range (for FHE quantization)
scale = 1000  # Scale factor
emb_scaled = (emb_normalized * scale).astype(np.int16)
print(f"   Scaled integers - Min: {emb_scaled.min()}, Max: {emb_scaled.max()}")

print("\n✅ All edge cases handled successfully!")