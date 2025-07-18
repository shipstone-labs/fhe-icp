#!/usr/bin/env python3
"""Diagnostic test to identify the similarity scoring issue."""

import numpy as np
from fhe_similarity import FHESimilarityModel

def test_random_vs_proper_training():
    """Compare model trained on random data vs proper similarity data."""
    
    print("="*60)
    print("SIMILARITY MODEL TRAINING COMPARISON TEST")
    print("="*60)
    
    # Test 1: Model trained on random data (current bug)
    print("\n1. Testing model trained on RANDOM data (current bug):")
    model_random = FHESimilarityModel(input_dim=256, n_bits=8)
    
    # Train with random data (as in the bug)
    X_random = np.random.randn(100, 256).astype(np.float32)
    y_random = np.random.randn(100)  # Random targets!
    model_random.train(X_random, y_random, n_samples=100)
    
    # Test 2: Model trained on proper similarity data
    print("\n2. Testing model trained on PROPER similarity data:")
    model_proper = FHESimilarityModel(input_dim=256, n_bits=8)
    
    # Train with proper data (using internal method)
    X_proper, y_proper = model_proper.train()  # Uses _prepare_training_data internally
    
    # Create test cases
    print("\n3. Creating test embedding pairs:")
    
    # Similar embeddings (should have high similarity)
    emb1a = np.random.randn(128).astype(np.float32)
    emb1a = emb1a / np.linalg.norm(emb1a)
    emb1b = emb1a + 0.1 * np.random.randn(128)  # Small perturbation
    emb1b = emb1b / np.linalg.norm(emb1b)
    X_similar = np.hstack([emb1a, emb1b]).reshape(1, -1)
    true_similarity_similar = np.dot(emb1a, emb1b)
    
    # Different embeddings (should have low similarity)
    emb2a = np.random.randn(128).astype(np.float32)
    emb2a = emb2a / np.linalg.norm(emb2a)
    emb2b = np.random.randn(128).astype(np.float32)  # Completely different
    emb2b = emb2b / np.linalg.norm(emb2b)
    X_different = np.hstack([emb2a, emb2b]).reshape(1, -1)
    true_similarity_different = np.dot(emb2a, emb2b)
    
    # Identical embeddings (should have similarity ~1.0)
    emb3a = np.random.randn(128).astype(np.float32)
    emb3a = emb3a / np.linalg.norm(emb3a)
    X_identical = np.hstack([emb3a, emb3a]).reshape(1, -1)
    true_similarity_identical = 1.0
    
    # Test predictions
    print("\n4. Comparing predictions:")
    print("   True cosine similarities:")
    print(f"   - Similar pair: {true_similarity_similar:.3f}")
    print(f"   - Different pair: {true_similarity_different:.3f}")
    print(f"   - Identical pair: {true_similarity_identical:.3f}")
    
    print("\n   Model trained on RANDOM data predictions:")
    pred_similar_random = model_random.predict_clear(X_similar)[0]
    pred_different_random = model_random.predict_clear(X_different)[0]
    pred_identical_random = model_random.predict_clear(X_identical)[0]
    print(f"   - Similar pair: {pred_similar_random:.3f}")
    print(f"   - Different pair: {pred_different_random:.3f}")
    print(f"   - Identical pair: {pred_identical_random:.3f}")
    
    print("\n   Model trained on PROPER data predictions:")
    pred_similar_proper = model_proper.predict_clear(X_similar)[0]
    pred_different_proper = model_proper.predict_clear(X_different)[0]
    pred_identical_proper = model_proper.predict_clear(X_identical)[0]
    print(f"   - Similar pair: {pred_similar_proper:.3f}")
    print(f"   - Different pair: {pred_different_proper:.3f}")
    print(f"   - Identical pair: {pred_identical_proper:.3f}")
    
    # Analyze training data
    print("\n5. Training data analysis:")
    print(f"   Random y_train: mean={y_random.mean():.3f}, std={y_random.std():.3f}, "
          f"min={y_random.min():.3f}, max={y_random.max():.3f}")
    print(f"   Proper y_train: mean={y_proper.mean():.3f}, std={y_proper.std():.3f}, "
          f"min={y_proper.min():.3f}, max={y_proper.max():.3f}")
    
    # Diagnosis
    print("\n6. DIAGNOSIS:")
    if abs(pred_identical_random - 1.0) > 0.3:
        print("   ❌ Model trained on random data FAILS to recognize identical embeddings")
    if abs(pred_identical_proper - 1.0) < 0.1:
        print("   ✅ Model trained on proper data correctly recognizes identical embeddings")
        
    print("\n   This confirms the bug: training on random data produces nonsensical similarities!")
    

if __name__ == "__main__":
    test_random_vs_proper_training()
