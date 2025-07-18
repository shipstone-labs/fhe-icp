#!/usr/bin/env python3
"""Explain what happens during FHE operations."""

from concrete.ml.sklearn import LinearRegression
import numpy as np

# Create a simple model
model = LinearRegression(n_bits=4)  # Lower bits for visualization
X = np.array([[1], [2], [3]], dtype=np.float32)
y = np.array([2, 4, 6], dtype=np.float32)

print("FHE Process Explanation:\n")

print("1. Quantization:")
print("   - Original model uses 32/64-bit floats")
print("   - FHE requires integer operations")
print("   - We quantize to 4-bit integers here")
print("   - This limits precision but enables encryption")

model.fit(X, y)
print(f"\n2. Original model: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")

print("\n3. Compilation creates:")
print("   - An arithmetic circuit (computation graph)")
print("   - Encryption/decryption keys")
print("   - Optimized operations for encrypted data")

model.compile(X)

print("\n4. During encrypted prediction:")
print("   a) Your input (e.g., x=5) gets encrypted")
print("   b) Encrypted x goes through the circuit")
print("   c) All operations happen on encrypted data")
print("   d) Result is decrypted at the end")
print("   e) Server never sees your actual data!")

print("\n5. The magic: The server computes 2*x without knowing x!")