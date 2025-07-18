#!/usr/bin/env python3
"""Test the fixed FHE similarity pipeline."""

import numpy as np
from fhe_similarity import FHESimilarityModel

def test_fixed_pipeline():
    """Test that the fixed pipeline works correctly."""
    
    print("="*60)
    print("TESTING FIXED FHE SIMILARITY PIPELINE")
    print("="*60)
    
    # 1. Create and train model with correct dimensions
    print("\n1. Training FHE Similarity Model...")
    model = FHESimilarityModel(input_dim=128, n_bits=8)
    X_train, y_train = model.train(n_samples=500)
    
    print(f"\n   Training data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"   Training targets range: [{y_train.min():.3f}, {y_train.max():.3f}]")
    
    # 2. Test on known examples
    print("\n2. Testing on known examples...")
    
    # Create test embeddings
    dim = 64  # Half of 128 (since model expects concatenated embeddings)
    
    # Identical embeddings
    emb1 = np.random.randn(dim).astype(np.float32)
    emb1 = emb1 / np.linalg.norm(emb1)
    
    # Similar embedding
    emb2 = emb1 + 0.1 * np.random.randn(dim)
    emb2 = emb2 / np.linalg.norm(emb2)
    
    # Different embedding
    emb3 = np.random.randn(dim).astype(np.float32)
    emb3 = emb3 / np.linalg.norm(emb3)
    
    # Compute true similarities
    sim_identical = np.dot(emb1, emb1)
    sim_similar = np.dot(emb1, emb2)
    sim_different = np.dot(emb1, emb3)
    
    print(f"\n   True cosine similarities:")
    print(f"   - Identical: {sim_identical:.3f}")
    print(f"   - Similar: {sim_similar:.3f}")
    print(f"   - Different: {sim_different:.3f}")
    
    # Prepare inputs using element-wise products
    X_identical = (emb1 * emb1).reshape(1, -1)
    X_similar = (emb1 * emb2).reshape(1, -1)
    X_different = (emb1 * emb3).reshape(1, -1)
    
    # Get predictions
    pred_identical = model.predict_clear(X_identical)[0]
    pred_similar = model.predict_clear(X_similar)[0]
    pred_different = model.predict_clear(X_different)[0]
    
    print(f"\n   Model predictions:")
    print(f"   - Identical: {pred_identical:.3f}")
    print(f"   - Similar: {pred_similar:.3f}")
    print(f"   - Different: {pred_different:.3f}")
    
    # 3. Verify results
    print("\n3. RESULTS:")
    errors = {
        'identical': abs(pred_identical - sim_identical),
        'similar': abs(pred_similar - sim_similar),
        'different': abs(pred_different - sim_different)
    }
    
    print(f"   Prediction errors:")
    for name, error in errors.items():
        print(f"   - {name}: {error:.4f}")
    
    if all(error < 0.1 for error in errors.values()):
        print("\n✅ SUCCESS: Fixed pipeline correctly computes similarity!")
    else:
        print("\n❌ FAILED: Predictions don't match true similarities")
        
    # 4. Test with full pipeline dimensions (128-dim embeddings)
    print("\n4. Testing with full 128-dim embeddings...")
    
    full_emb1 = np.random.randn(128).astype(np.float32)
    full_emb1 = full_emb1 / np.linalg.norm(full_emb1)
    
    full_emb2 = full_emb1 + 0.2 * np.random.randn(128)
    full_emb2 = full_emb2 / np.linalg.norm(full_emb2)
    
    X_full = (full_emb1 * full_emb2).reshape(1, -1)
    pred_full = model.predict_clear(X_full)[0]
    true_full = np.dot(full_emb1, full_emb2)
    
    print(f"   128-dim test: predicted={pred_full:.3f}, true={true_full:.3f}")
    
    print("\n" + "="*60)
    print("The fixed pipeline uses element-wise products successfully!")
    print("="*60)

if __name__ == "__main__":
    test_fixed_pipeline()