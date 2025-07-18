#!/usr/bin/env python3
"""Test model initialization with the fix."""

import os
import logging
from key_management import FHEKeyManager
from batch_operations import BatchProcessor

logging.basicConfig(level=logging.INFO)

def test_model_init():
    """Test the fixed model initialization."""
    print("\n" + "="*60)
    print("TESTING FIXED MODEL INITIALIZATION")
    print("="*60)
    
    # First ensure we have keys
    km = FHEKeyManager()
    key = km.get_current_key()
    
    if not key:
        print("\nNo keys found. For testing, we'll set a test password via environment variable.")
        os.environ['FHE_MASTER_PASSWORD'] = 'test_password_123'
        
        # Modify key manager to accept password from env in test mode
        import key_management
        original_get_master_key = key_management.FHEKeyManager._get_master_key
        
        def test_get_master_key(self):
            if 'FHE_MASTER_PASSWORD' in os.environ:
                password = os.environ['FHE_MASTER_PASSWORD']
                import secrets
                salt = secrets.token_bytes(16)
                self._master_salt = salt
                return self._derive_master_key(password, salt)
            return original_get_master_key(self)
        
        key_management.FHEKeyManager._get_master_key = test_get_master_key
        
        print("Generating keys with test password...")
        km.generate_keys()
        print("Keys generated successfully!")
    
    # Now test model initialization
    print("\nInitializing batch processor...")
    bp = BatchProcessor()
    
    print("\nInitializing FHE model with proper training data...")
    bp._init_model()
    
    if bp.fhe_model:
        print("\n✅ Model initialized successfully!")
        
        # Test with some predictions
        import numpy as np
        print("\nTesting predictions:")
        
        # Create test embeddings
        emb1 = np.random.randn(128).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)
        
        # Test identical embeddings
        X_identical = np.hstack([emb1, emb1]).reshape(1, -1)
        sim_identical = bp.fhe_model.predict_clear(X_identical)[0]
        print(f"  Identical embeddings similarity: {sim_identical:.3f} (expected ~1.0)")
        
        # Test different embeddings
        emb2 = np.random.randn(128).astype(np.float32)
        emb2 = emb2 / np.linalg.norm(emb2)
        X_different = np.hstack([emb1, emb2]).reshape(1, -1)
        sim_different = bp.fhe_model.predict_clear(X_different)[0]
        print(f"  Different embeddings similarity: {sim_different:.3f} (expected ~0.0)")
        
        if abs(sim_identical - 1.0) < 0.3:
            print("\n✅ Model validation PASSED! Similarity scores look correct.")
        else:
            print("\n❌ Model validation FAILED! Similarity scores are still incorrect.")
    else:
        print("\n❌ Model initialization failed!")

if __name__ == "__main__":
    test_model_init()