#!/usr/bin/env python3
"""Test basic FHE operations with Concrete-ML."""

from concrete.ml.sklearn import LinearRegression
import numpy as np
import time

print("Testing Fully Homomorphic Encryption with Concrete-ML\n")

# Step 1: Create simple training data
# We're creating a simple y = 2x relationship
print("1. Creating training data...")
X_train = np.array([[1], [2], [3], [4], [5], [6]], dtype=np.float32)
y_train = np.array([2, 4, 6, 8, 10, 12], dtype=np.float32)
print(f"   X_train shape: {X_train.shape}")
print(f"   y_train shape: {y_train.shape}")

# Step 2: Create and train the model
print("\n2. Training quantized model...")
model = LinearRegression(n_bits=8)  # 8-bit quantization for FHE
model.fit(X_train, y_train)
print(f"   Model coefficients: {model.coef_}")
print(f"   Model intercept: {model.intercept_}")

# Step 3: Compile the model for FHE
print("\n3. Compiling model for FHE (this may take 10-30 seconds)...")
start_time = time.time()
model.compile(X_train)
compile_time = time.time() - start_time
print(f"   Compilation completed in {compile_time:.2f} seconds")

# Step 4: Test predictions
test_value = np.array([[7]], dtype=np.float32)
print(f"\n4. Testing predictions for X = {test_value[0][0]}")

# Clear (non-encrypted) prediction
clear_start = time.time()
clear_pred = model.predict(test_value)
clear_time = time.time() - clear_start

# FHE (encrypted) prediction
print("   Running FHE prediction (this may take 5-10 seconds)...")
fhe_start = time.time()
fhe_pred = model.predict(test_value, fhe="execute")
fhe_time = time.time() - fhe_start

# Step 5: Compare results
print("\n5. Results:")
print(f"   Expected (2 * 7):      14.0")
print(f"   Clear prediction:      {clear_pred[0]:.6f} (took {clear_time*1000:.2f}ms)")
print(f"   FHE prediction:        {fhe_pred[0]:.6f} (took {fhe_time:.2f}s)")
print(f"   Difference:            {abs(clear_pred[0] - fhe_pred[0]):.6f}")
print(f"   FHE overhead:          {fhe_time/clear_time:.0f}x slower")

# Verify accuracy
tolerance = 0.01
if abs(fhe_pred[0] - clear_pred[0]) < tolerance:
    print("\n✅ FHE test PASSED! Encrypted computation matches clear computation.")
else:
    print("\n❌ FHE test FAILED! Results don't match within tolerance.")

# Additional info
print(f"\n6. FHE Circuit Information:")
print(f"   Max bit width: {model.fhe_circuit.graph.maximum_integer_bit_width()}")