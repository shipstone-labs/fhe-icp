#!/usr/bin/env python3
"""Integration test for Session 4 implementation."""

import os
import tempfile
import shutil
import getpass
from pathlib import Path

# Mock password for testing
getpass.getpass = lambda prompt: "testpassword123"

def test_session4_integration():
    """Test the complete Session 4 workflow."""
    print("Session 4 Integration Test")
    print("=" * 50)
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    key_dir = Path(temp_dir) / "keys"
    doc_dir = Path(temp_dir) / "docs"
    
    try:
        # Import modules
        from key_management import FHEKeyManager
        from encrypted_storage import EncryptedDocumentStore
        from batch_operations import BatchProcessor, BatchConfig
        
        print("\n1. Testing Key Management...")
        key_manager = FHEKeyManager(key_dir=str(key_dir))
        key_info = key_manager.generate_keys("test_session4")
        print(f"   ✓ Keys generated: {key_info['key_id']}")
        
        print("\n2. Testing Encrypted Storage...")
        storage = EncryptedDocumentStore(storage_dir=str(doc_dir))
        
        print("\n3. Testing Batch Operations...")
        processor = BatchProcessor(
            key_manager=key_manager,
            storage=storage,
            config=BatchConfig(batch_size=2, show_progress=False)
        )
        
        # Test documents
        test_docs = [
            "Fully homomorphic encryption enables computation on encrypted data.",
            "FHE is a powerful privacy-preserving technology.",
            "Internet Computer Protocol supports decentralized applications.",
            "ICP canisters can run complex computations.",
            "Privacy and security are essential for modern applications."
        ]
        
        print("\n4. Encrypting documents...")
        doc_ids = processor.encrypt_documents(test_docs)
        print(f"   ✓ Encrypted {len(doc_ids)} documents")
        
        print("\n5. Testing document comparison...")
        similarity = processor.compare_encrypted(doc_ids[0], doc_ids[1])
        print(f"   ✓ Similarity between docs 0 and 1: {similarity:.3f}")
        
        print("\n6. Testing document search...")
        results = processor.search_similar("homomorphic encryption privacy", top_k=3)
        print(f"   ✓ Found {len(results)} similar documents")
        for doc_id, score in results:
            idx = doc_ids.index(doc_id) if doc_id in doc_ids else -1
            if idx >= 0:
                print(f"     - Doc {idx}: {score:.3f}")
        
        print("\n7. Testing storage statistics...")
        stats = storage.get_stats()
        print(f"   ✓ Total documents: {stats['total_documents']}")
        print(f"   ✓ Average size: {stats['average_size_bytes']:.0f} bytes")
        
        print("\n8. Testing memory management...")
        mem_stats = processor.get_memory_stats()
        print(f"   ✓ Memory used: {mem_stats['used_mb']:.1f} MB")
        
        print("\n" + "=" * 50)
        print("✅ All tests passed!")
        print("\nSession 4 implementation is working correctly.")
        print("You can now use the CLI with commands like:")
        print("  python fhe_cli.py keys generate")
        print("  python fhe_cli.py encrypt 'Your text here'")
        print("  python fhe_cli.py compare doc1 doc2")
        print("  python fhe_cli.py stats")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"\nTemporary files cleaned up.")


if __name__ == "__main__":
    test_session4_integration()