#!/usr/bin/env python3
"""Understanding BERT embeddings basics."""

print("BERT Embeddings Explained:\n")

print("1. What is BERT?")
print("   - Bidirectional Encoder Representations from Transformers")
print("   - Pre-trained on millions of documents")
print("   - Understands context and word relationships")

print("\n2. Text → Numbers Pipeline:")
print("   'Hello world' → [Tokens] → [Token IDs] → [Embeddings]")
print("   'Hello world' → ['Hello', 'world'] → [7592, 2088] → [768D vectors]")

print("\n3. Why 768 dimensions?")
print("   - BERT-base uses 768-dimensional vectors")
print("   - Each dimension captures different linguistic features")
print("   - Similar texts have similar vectors")

print("\n4. For our use case:")
print("   - Document → BERT → 768D vector")
print("   - Encrypt this vector with FHE")
print("   - Compare encrypted vs plain vectors")