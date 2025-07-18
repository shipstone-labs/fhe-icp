#!/usr/bin/env python3
"""Test if polynomial features can capture cosine similarity."""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from concrete.ml.sklearn import LinearRegression as ConcreteLinearRegression

def test_polynomial_approach():
    """Test if polynomial features can help LinearRegression learn cosine similarity."""
    
    print("="*60)
    print("TESTING POLYNOMIAL FEATURES FOR COSINE SIMILARITY")
    print("="*60)
    
    # Generate test data
    n_samples = 500
    dim = 128
    
    # Create normalized embeddings
    emb1 = np.random.randn(n_samples, dim).astype(np.float32)
    emb1 = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
    
    emb2 = np.random.randn(n_samples, dim).astype(np.float32)
    emb2 = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
    
    # Add correlation
    mask = np.random.rand(n_samples) > 0.5
    emb2[mask] = emb1[mask] + 0.2 * np.random.randn(mask.sum(), dim)
    emb2 = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
    
    # True cosine similarities
    y_true = np.sum(emb1 * emb2, axis=1)
    
    print(f"\n1. Testing different approaches:")
    
    # Approach 1: Direct concatenation (baseline)
    print(f"\n   a) Direct concatenation [a, b]:")
    X_concat = np.hstack([emb1, emb2])
    lr1 = LinearRegression()
    lr1.fit(X_concat, y_true)
    score1 = lr1.score(X_concat, y_true)
    print(f"      R² score: {score1:.4f}")
    
    # Test on identical embeddings
    test_emb = np.random.randn(dim).astype(np.float32)
    test_emb = test_emb / np.linalg.norm(test_emb)
    X_test = np.hstack([test_emb, test_emb]).reshape(1, -1)
    pred = lr1.predict(X_test)[0]
    print(f"      Identical embeddings: predicted={pred:.3f}, true=1.000")
    
    # Approach 2: Element-wise product features
    print(f"\n   b) Element-wise product features [a.*b]:")
    X_product = emb1 * emb2  # Element-wise product
    lr2 = LinearRegression()
    lr2.fit(X_product, y_true)
    score2 = lr2.score(X_product, y_true)
    print(f"      R² score: {score2:.4f}")
    
    # The coefficients should all be close to 1.0
    print(f"      Coefficient mean: {lr2.coef_.mean():.3f}, std: {lr2.coef_.std():.3f}")
    
    # Test on identical embeddings
    X_test_prod = (test_emb * test_emb).reshape(1, -1)
    pred2 = lr2.predict(X_test_prod)[0]
    print(f"      Identical embeddings: predicted={pred2:.3f}, true=1.000")
    
    # Approach 3: Combined features [a, b, a.*b]
    print(f"\n   c) Combined features [a, b, a.*b]:")
    X_combined = np.hstack([emb1, emb2, emb1 * emb2])
    lr3 = LinearRegression()
    lr3.fit(X_combined, y_true)
    score3 = lr3.score(X_combined, y_true)
    print(f"      R² score: {score3:.4f}")
    
    # Approach 4: Polynomial features (degree 2, only interactions)
    print(f"\n   d) Polynomial features (degree 2, interaction only):")
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X_concat)
    print(f"      Feature shape: {X_concat.shape} -> {X_poly.shape}")
    # This is too large for FHE!
    
    print(f"\n2. KEY FINDINGS:")
    print(f"   - Direct concatenation CANNOT learn cosine similarity (R²={score1:.4f})")
    print(f"   - Element-wise product features learn it PERFECTLY (R²={score2:.4f})")
    print(f"   - But element-wise product requires preprocessing outside FHE")
    print(f"   - Polynomial features explode dimensionality (not FHE-friendly)")
    
    print(f"\n3. PROPOSED SOLUTION:")
    print(f"   - Compute element-wise product BEFORE encryption")
    print(f"   - Feed product features to LinearRegression in FHE")
    print(f"   - This gives perfect cosine similarity computation!")
    
    # Test with Concrete ML
    print(f"\n4. Testing with Concrete ML:")
    X_product_subset = X_product[:100]
    y_subset = y_true[:100]
    
    concrete_lr = ConcreteLinearRegression(n_bits=8)
    concrete_lr.fit(X_product_subset, y_subset)
    score_concrete = concrete_lr.score(X_product_subset, y_subset)
    print(f"   Concrete ML R² score: {score_concrete:.4f}")
    
    # The model should just sum all features with coefficient ~1
    print(f"   Expected behavior: sum of all product features")
    print(f"   Actual coefficient mean: {concrete_lr.coef_.mean():.3f}")
    
    return X_product, y_true

if __name__ == "__main__":
    test_polynomial_approach()