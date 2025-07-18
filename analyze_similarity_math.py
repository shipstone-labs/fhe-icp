#!/usr/bin/env python3
"""Analyze the mathematical relationship in similarity computation."""

import numpy as np
from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt

def analyze_linear_relationship():
    """Analyze what LinearRegression learns for cosine similarity."""
    
    print("="*60)
    print("MATHEMATICAL ANALYSIS: LinearRegression for Cosine Similarity")
    print("="*60)
    
    # Create test data
    n_samples = 1000
    dim = 128
    
    # Generate normalized embeddings
    emb1 = np.random.randn(n_samples, dim).astype(np.float32)
    emb1 = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
    
    emb2 = np.random.randn(n_samples, dim).astype(np.float32)
    emb2 = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
    
    # Add correlation for some pairs
    mask = np.random.rand(n_samples) > 0.5
    emb2[mask] = emb1[mask] + 0.2 * np.random.randn(mask.sum(), dim)
    emb2 = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
    
    # Concatenate embeddings
    X = np.hstack([emb1, emb2])
    
    # True cosine similarities
    y_true = np.sum(emb1 * emb2, axis=1)
    
    print(f"\n1. Data shape: X={X.shape}, y={y_true.shape}")
    print(f"   y_true range: [{y_true.min():.3f}, {y_true.max():.3f}]")
    print(f"   y_true mean: {y_true.mean():.3f}, std: {y_true.std():.3f}")
    
    # Train LinearRegression
    lr = LinearRegression()
    lr.fit(X, y_true)
    
    # Analyze coefficients
    print(f"\n2. Linear Regression Analysis:")
    print(f"   Intercept: {lr.intercept_:.6f}")
    print(f"   Number of coefficients: {len(lr.coef_)}")
    
    # Split coefficients for first and second embedding
    coef1 = lr.coef_[:dim]
    coef2 = lr.coef_[dim:]
    
    print(f"\n3. Coefficient Analysis:")
    print(f"   First embedding coeffs - mean: {coef1.mean():.6f}, std: {coef1.std():.6f}")
    print(f"   Second embedding coeffs - mean: {coef2.mean():.6f}, std: {coef2.std():.6f}")
    
    # Check if coefficients approximate element-wise multiplication
    # For cosine similarity, we expect coef1[i] ≈ coef2[i] ≈ 1/dim for all i
    expected_coef = 1.0
    print(f"\n4. Coefficient Pattern Analysis:")
    print(f"   Expected coefficient value for element-wise multiplication: {expected_coef:.6f}")
    print(f"   Actual coefficient correlation: {np.corrcoef(coef1, coef2)[0,1]:.6f}")
    
    # Test on known examples
    print(f"\n5. Testing on Known Examples:")
    
    # Test 1: Identical embeddings
    test_emb = np.random.randn(dim).astype(np.float32)
    test_emb = test_emb / np.linalg.norm(test_emb)
    X_identical = np.hstack([test_emb, test_emb]).reshape(1, -1)
    pred_identical = lr.predict(X_identical)[0]
    print(f"   Identical embeddings: predicted={pred_identical:.6f}, true=1.000000")
    
    # Test 2: Orthogonal embeddings
    test_emb2 = np.random.randn(dim).astype(np.float32)
    test_emb2 = test_emb2 - np.dot(test_emb2, test_emb) * test_emb  # Make orthogonal
    test_emb2 = test_emb2 / np.linalg.norm(test_emb2)
    X_orthogonal = np.hstack([test_emb, test_emb2]).reshape(1, -1)
    pred_orthogonal = lr.predict(X_orthogonal)[0]
    actual_orthogonal = np.dot(test_emb, test_emb2)
    print(f"   Orthogonal embeddings: predicted={pred_orthogonal:.6f}, true={actual_orthogonal:.6f}")
    
    # Test 3: Opposite embeddings
    X_opposite = np.hstack([test_emb, -test_emb]).reshape(1, -1)
    pred_opposite = lr.predict(X_opposite)[0]
    print(f"   Opposite embeddings: predicted={pred_opposite:.6f}, true=-1.000000")
    
    # Analyze what the model actually learned
    print(f"\n6. What did the model learn?")
    
    # Check if it learned element-wise multiplication
    # The ideal would be: sum(emb1[i] * emb2[i] * 1.0) for all i
    # So coefficients should pair up
    
    # Create a simple test
    a = np.array([1, 0, 0])
    b = np.array([1, 0, 0])
    # Cosine similarity = 1
    
    # What would LinearRegression need to learn?
    # Input: [1, 0, 0, 1, 0, 0]
    # Output: 1
    # This means: coef[0] * 1 + coef[3] * 1 = 1
    # But coef[0] and coef[3] are independent!
    
    print("\n   KEY INSIGHT:")
    print("   LinearRegression CAN learn cosine similarity because:")
    print("   - For normalized vectors a and b, cos(a,b) = dot(a,b) = sum(a[i]*b[i])")
    print("   - Input is [a1,a2,...,an,b1,b2,...,bn]")
    print("   - The model learns: f(x) = sum(x[i]*x[i+n]) for i in 0..n-1")
    print("   - This is NOT a linear function of the concatenated input!")
    print("   - LinearRegression CANNOT learn element-wise multiplication")
    
    print("\n7. Why is the model failing?")
    print("   LinearRegression assumes: y = w0*x0 + w1*x1 + ... + wn*xn + b")
    print("   But cosine similarity needs: y = x0*xn + x1*xn+1 + ... + xn-1*x2n-1")
    print("   This requires MULTIPLICATION between features, not linear combination!")
    
    return lr, X, y_true

if __name__ == "__main__":
    analyze_linear_relationship()