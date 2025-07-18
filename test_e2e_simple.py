#!/usr/bin/env python3
"""Simple end-to-end test of the fixed similarity computation."""

import numpy as np
import os
os.environ['FHE_MASTER_PASSWORD'] = 'test_password_123'

def test_e2e():
    """Test the complete flow with the fixes."""
    
    print("="*60)
    print("END-TO-END TEST: Fixed FHE Similarity")
    print("="*60)
    
    # 1. Generate keys
    print("\n1. Generating FHE keys...")
    from key_management import FHEKeyManager
    km = FHEKeyManager()
    
    # Check if keys exist
    if not km.get_current_key():
        print("   Creating new keys...")
        km.generate_keys()
    else:
        print("   Using existing keys")
    
    # 2. Initialize batch processor
    print("\n2. Initializing batch processor...")
    from batch_operations import BatchProcessor
    bp = BatchProcessor()
    
    # 3. Create test documents
    print("\n3. Creating test documents...")
    texts = [
        "Machine learning and artificial intelligence are transforming technology",
        "AI and ML revolutionize how computers learn from data", 
        "I love cooking Italian pasta with fresh tomatoes"
    ]
    
    # Encrypt documents
    print("\n4. Encrypting documents...")
    doc_ids = bp.encrypt_documents(texts, ['doc1', 'doc2', 'doc3'])
    print(f"   Encrypted {len(doc_ids)} documents")
    
    # 5. Compare documents
    print("\n5. Computing similarities...")
    
    # Similar documents (ML/AI topics)
    sim_12 = bp.compare_encrypted('doc1', 'doc2')
    print(f"   Doc1 vs Doc2 (similar topics): {sim_12:.3f}")
    
    # Different documents
    sim_13 = bp.compare_encrypted('doc1', 'doc3')
    print(f"   Doc1 vs Doc3 (different topics): {sim_13:.3f}")
    
    # 6. Verify results
    print("\n6. RESULTS:")
    if sim_12 > 0.5 and sim_13 < 0.3:
        print("   ✅ SUCCESS: Similar documents have high similarity!")
        print("   ✅ SUCCESS: Different documents have low similarity!")
        print("\n   The fixed pipeline correctly computes document similarity!")
    else:
        print("   ❌ FAILED: Similarity scores don't match expectations")
        print(f"   Expected: similar > 0.5, different < 0.3")
        print(f"   Got: similar = {sim_12:.3f}, different = {sim_13:.3f}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    test_e2e()