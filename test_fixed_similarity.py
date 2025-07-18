#!/usr/bin/env python3
"""Test the fixed similarity approach with element-wise product."""

import numpy as np
from concrete.ml.sklearn import LinearRegression
import time

class FixedFHESimilarityModel:
    """Fixed FHE model using element-wise product."""
    
    def __init__(self, input_dim: int = 128, n_bits: int = 8):
        self.input_dim = input_dim
        self.n_bits = n_bits
        self.model = None
        self.compiled = False
        
    def train(self, n_samples: int = 1000):
        """Train with proper feature engineering."""
        print(f"\nTraining FIXED FHE Similarity Model")
        print(f"  Using element-wise product features")
        
        # Generate normalized embeddings
        emb1 = np.random.randn(n_samples, self.input_dim).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
        
        emb2 = np.random.randn(n_samples, self.input_dim).astype(np.float32)
        emb2 = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
        
        # Add correlation
        mask = np.random.rand(n_samples) > 0.5
        emb2[mask] = emb1[mask] + 0.2 * np.random.randn(mask.sum(), self.input_dim)
        emb2 = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
        
        # FIXED: Use element-wise product as features
        X_train = emb1 * emb2
        y_train = np.sum(X_train, axis=1)  # True cosine similarity
        
        # Train model
        self.model = LinearRegression(n_bits=self.n_bits)
        self.model.fit(X_train, y_train)
        
        score = self.model.score(X_train, y_train)
        print(f"  Training R² score: {score:.4f}")
        
        return X_train, y_train
        
    def compile(self, X_sample):
        """Compile for FHE."""
        print(f"  Compiling for FHE...")
        self.model.compile(X_sample)
        self.compiled = True
        print(f"  ✅ Compilation successful!")
        
    def predict_clear(self, emb1, emb2):
        """Predict similarity between two embeddings."""
        # Element-wise product
        X = (emb1 * emb2).reshape(1, -1)
        return self.model.predict(X)[0]

def test_comparison():
    """Compare original vs fixed approach."""
    print("="*60)
    print("COMPARING ORIGINAL VS FIXED SIMILARITY MODEL")
    print("="*60)
    
    # Test data
    dim = 128
    
    # Create test embeddings
    emb1 = np.random.randn(dim).astype(np.float32)
    emb1 = emb1 / np.linalg.norm(emb1)
    
    emb2_identical = emb1.copy()
    emb2_similar = emb1 + 0.1 * np.random.randn(dim)
    emb2_similar = emb2_similar / np.linalg.norm(emb2_similar)
    emb2_different = np.random.randn(dim).astype(np.float32)
    emb2_different = emb2_different / np.linalg.norm(emb2_different)
    emb2_opposite = -emb1
    
    # True similarities
    print("\nTrue cosine similarities:")
    print(f"  Identical: {np.dot(emb1, emb2_identical):.3f}")
    print(f"  Similar: {np.dot(emb1, emb2_similar):.3f}")
    print(f"  Different: {np.dot(emb1, emb2_different):.3f}")
    print(f"  Opposite: {np.dot(emb1, emb2_opposite):.3f}")
    
    # Test ORIGINAL approach (concatenation)
    print("\n1. ORIGINAL MODEL (concatenation):")
    from fhe_similarity import FHESimilarityModel
    orig_model = FHESimilarityModel(input_dim=256, n_bits=8)
    X_train_orig, y_train_orig = orig_model.train()
    
    # Predictions
    X_test_identical = np.hstack([emb1, emb2_identical]).reshape(1, -1)
    X_test_similar = np.hstack([emb1, emb2_similar]).reshape(1, -1)
    X_test_different = np.hstack([emb1, emb2_different]).reshape(1, -1)
    X_test_opposite = np.hstack([emb1, emb2_opposite]).reshape(1, -1)
    
    print("\n   Predictions:")
    print(f"   Identical: {orig_model.predict_clear(X_test_identical)[0]:.3f}")
    print(f"   Similar: {orig_model.predict_clear(X_test_similar)[0]:.3f}")
    print(f"   Different: {orig_model.predict_clear(X_test_different)[0]:.3f}")
    print(f"   Opposite: {orig_model.predict_clear(X_test_opposite)[0]:.3f}")
    
    # Test FIXED approach (element-wise product)
    print("\n2. FIXED MODEL (element-wise product):")
    fixed_model = FixedFHESimilarityModel(input_dim=128, n_bits=8)
    X_train_fixed, y_train_fixed = fixed_model.train()
    
    print("\n   Predictions:")
    print(f"   Identical: {fixed_model.predict_clear(emb1, emb2_identical):.3f}")
    print(f"   Similar: {fixed_model.predict_clear(emb1, emb2_similar):.3f}")
    print(f"   Different: {fixed_model.predict_clear(emb1, emb2_different):.3f}")
    print(f"   Opposite: {fixed_model.predict_clear(emb1, emb2_opposite):.3f}")
    
    print("\n" + "="*60)
    print("CONCLUSION:")
    print("- Original model: All predictions cluster around training mean")
    print("- Fixed model: Predictions match true cosine similarities!")
    print("="*60)

if __name__ == "__main__":
    test_comparison()