# Session 1: Environment Setup & Basic FHE Test - Detailed Instructions

## Overview

In this session, we'll set up the development environment and verify that Concrete-ML (the FHE library) works correctly. This is crucial as FHE libraries can have complex dependencies.

## Pre-requisites

- Python 3.8 or higher (check with `python --version`)
- pip package manager
- 8GB+ RAM recommended (FHE operations are memory intensive)
- macOS, Linux, or Windows (WSL2 recommended for Windows)

## Step 1: Create Project Directory (5 min)

Open your terminal and run:

```bash
# Create and navigate to project directory
mkdir fhe-bert-similarity
cd fhe-bert-similarity

# Create initial project structure
mkdir tests docs
touch README.md requirements.txt .gitignore
```

Create `.gitignore` with:

```
venv/
__pycache__/
*.pyc
*.enc
*.pth
.DS_Store
```

## Step 2: Set Up Virtual Environment (5 min)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate

# You should see (venv) in your prompt
```

**Troubleshooting:**

- If `python` doesn't work, try `python3`
- On Windows, you may need to enable script execution: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

## Step 3: Install Dependencies (10 min)

```bash
# Upgrade pip first
pip install --upgrade pip

# Install packages (this may take several minutes)
pip install concrete-ml==1.7.0 transformers==4.36.2 scikit-learn==1.3.2 torch

# Save dependencies
pip freeze > requirements.txt
```

**Expected output:**

- Concrete-ML will download several components
- You'll see progress bars for package installation
- Total download size: ~2-3GB (including PyTorch)

**Common issues:**

- **Memory error**: Close other applications
- **Permission error**: Use `pip install --user`
- **Build errors on M1 Mac**: Install Rosetta 2 first

## Step 4: Verify Installation (5 min)

Create `verify_install.py`:

```python
#!/usr/bin/env python3
"""Verify all packages are correctly installed."""

import sys

def check_import(module_name, package_name=None):
    """Try to import a module and report status."""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"✓ {package_name} imported successfully")
        return True
    except ImportError as e:
        print(f"✗ {package_name} import failed: {e}")
        return False

# Check all required packages
packages = [
    ("concrete.ml", "Concrete-ML"),
    ("transformers", "Transformers"),
    ("sklearn", "Scikit-learn"),
    ("torch", "PyTorch"),
    ("numpy", "NumPy"),
]

all_good = True
for module, name in packages:
    if not check_import(module, name):
        all_good = False

if all_good:
    print("\n✅ All packages installed correctly!")
else:
    print("\n❌ Some packages failed to import. Check the errors above.")
    sys.exit(1)

# Check versions
import concrete.ml
import transformers
print(f"\nVersions:")
print(f"Concrete-ML: {concrete.ml.__version__}")
print(f"Transformers: {transformers.__version__}")
```

Run it:

```bash
python verify_install.py
```

## Step 5: Basic FHE Test (10 min)

Create `test_fhe.py`:

```python
#!/usr/bin/env python3
"""Test basic FHE operations with Concrete-ML."""

from concrete.ml.sklearn import LinearRegression
import numpy as np
import time

print("Testing Fully Homomorphic Encryption with Concrete-ML\n")

# Step 1: Create simple training data
# We're creating a simple y = 2x relationship
print("1. Creating training data...")
X_train = np.array([[1], [2], [3], [4], [5], [6]])
y_train = np.array([2, 4, 6, 8, 10, 12])
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
test_value = np.array([[7]])
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
```

Run the test:

```bash
python test_fhe.py
```

**Expected output:**

- Compilation takes 10-30 seconds
- FHE prediction takes 5-10 seconds
- FHE is typically 1000-10000x slower than clear
- Results should match within 0.01 tolerance

## Step 6: Understanding What Just Happened (5 min)

Create `fhe_explained.py` to understand the process:

```python
#!/usr/bin/env python3
"""Explain what happens during FHE operations."""

from concrete.ml.sklearn import LinearRegression
import numpy as np

# Create a simple model
model = LinearRegression(n_bits=4)  # Lower bits for visualization
X = np.array([[1], [2], [3]])
y = np.array([2, 4, 6])

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
```

## Checkpoint Questions

Before moving to Session 2, ensure you can answer:

1. **What is quantization and why is it needed for FHE?**
   - FHE works on integers, not floats
   - We convert model weights to low-bit integers
2. **Why is FHE so slow?**
   - Each operation on encrypted data is ~1000x more complex
   - Homomorphic operations require large polynomial computations
3. **What does `compile()` do?**
   - Converts the ML model into an arithmetic circuit
   - Generates encryption keys
   - Optimizes the circuit for FHE operations

## Common Issues & Solutions

| Issue                | Solution                                     |
| -------------------- | -------------------------------------------- |
| "Module not found"   | Ensure virtual environment is activated      |
| Compilation hangs    | Normal - can take up to 1 minute             |
| Memory error         | Close other applications, use smaller n_bits |
| Wrong Python version | Use Python 3.8+                              |

## Next Session Preview

In Session 2, we'll:

- Load a real BERT model
- Extract embeddings from text
- Understand the 768-dimensional output
- Prepare for FHE integration

## Additional Resources

- [Concrete-ML Concepts](https://docs.zama.ai/concrete-ml/getting-started/concepts)
- [FHE Introduction Video](https://www.youtube.com/watch?v=LhnGarHz_oM)
- [Why Quantization Matters](https://docs.zama.ai/concrete-ml/explanations/quantization)

------

**Time check**: This session should take 30-35 minutes. If you finished early, try:

- Changing `n_bits` to see precision vs speed tradeoffs
- Training a polynomial regression instead of linear
- Reading about the TFHE scheme Concrete-ML usescd