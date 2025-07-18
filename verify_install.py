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