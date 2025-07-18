#!/usr/bin/env python3
"""Comprehensive test suite for FHE document system."""

import unittest
import tempfile
import shutil
import json
import hashlib
from pathlib import Path
import numpy as np
import time
import os
import pickle

from key_management import FHEKeyManager
from encrypted_storage import EncryptedDocument, EncryptedDocumentStore
from batch_operations import BatchProcessor, BatchConfig
from bert_embeddings import BertEmbedder
from dimension_reduction import DimensionReducer


class TestKeyManagement(unittest.TestCase):
    """Test key management functionality."""
    
    def setUp(self):
        """Create temporary directory for keys."""
        self.temp_dir = tempfile.mkdtemp()
        self.key_manager = FHEKeyManager(key_dir=self.temp_dir)
        
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
        
    def test_generate_keys(self):
        """Test key generation."""
        # Mock password input
        import key_management
        original_getpass = key_management.getpass.getpass
        key_management.getpass.getpass = lambda prompt: "testpassword"
        
        try:
            key_info = self.key_manager.generate_keys("test_key_001")
            
            self.assertEqual(key_info['key_id'], "test_key_001")
            self.assertIn('model_file', key_info)
            self.assertTrue(Path(key_info['model_file']).exists())
            
            # Check file permissions
            file_stat = os.stat(key_info['model_file'])
            self.assertEqual(file_stat.st_mode & 0o777, 0o600)
            
        finally:
            key_management.getpass.getpass = original_getpass
            
    def test_list_keys(self):
        """Test listing keys."""
        # Initially empty
        keys = self.key_manager.list_keys()
        self.assertEqual(len(keys), 0)
        
        # Generate a key
        import key_management
        key_management.getpass.getpass = lambda prompt: "testpassword"
        
        self.key_manager.generate_keys("test_key_001")
        
        # Should have one key
        keys = self.key_manager.list_keys()
        self.assertEqual(len(keys), 1)
        self.assertIn("test_key_001", keys)
        
    def test_load_model(self):
        """Test model loading."""
        import key_management
        key_management.getpass.getpass = lambda prompt: "testpassword"
        
        # Generate keys first
        self.key_manager.generate_keys("test_key_001")
        
        # Load model
        model = self.key_manager.load_model("test_key_001")
        self.assertIsNotNone(model)
        
        # Test with invalid key
        with self.assertRaises(ValueError):
            self.key_manager.load_model("invalid_key")


class TestEncryptedStorage(unittest.TestCase):
    """Test encrypted document storage."""
    
    def setUp(self):
        """Create temporary storage directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = EncryptedDocumentStore(storage_dir=self.temp_dir)
        
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
        
    def test_document_creation(self):
        """Test creating encrypted documents."""
        doc = EncryptedDocument(
            doc_id="test_001",
            content_hash=hashlib.sha256(b"test content").hexdigest(),
            timestamp="2024-01-01T00:00:00",
            encrypted_embedding=np.random.randn(128).astype(np.float32),
            metadata={"test": True}
        )
        
        self.assertEqual(doc.doc_id, "test_001")
        self.assertEqual(doc.encrypted_embedding.shape, (128,))
        
        # Test validation
        with self.assertRaises(ValueError):
            bad_doc = EncryptedDocument(
                doc_id="bad",
                content_hash="hash",
                timestamp="now",
                encrypted_embedding=np.random.randn(128)  # Wrong size
            )
            
    def test_save_load(self):
        """Test saving and loading documents."""
        # Create document
        embedding = np.random.randn(128).astype(np.float32)
        doc = EncryptedDocument(
            doc_id="test_001",
            content_hash="abcdef123456",
            timestamp="2024-01-01T00:00:00",
            encrypted_embedding=embedding
        )
        
        # Save
        path = self.storage.save(doc)
        self.assertTrue(Path(path).exists())
        
        # Load
        loaded = self.storage.load("test_001")
        self.assertEqual(loaded.doc_id, doc.doc_id)
        self.assertEqual(loaded.content_hash, doc.content_hash)
        np.testing.assert_array_almost_equal(
            loaded.encrypted_embedding, 
            doc.encrypted_embedding
        )
        
    def test_search_metadata(self):
        """Test metadata search."""
        # Create documents with metadata
        for i in range(5):
            doc = EncryptedDocument(
                doc_id=f"doc_{i}",
                content_hash=f"hash_{i}",
                timestamp="2024-01-01T00:00:00",
                encrypted_embedding=np.random.randn(128).astype(np.float32),
                metadata={"category": "test" if i < 3 else "other"}
            )
            self.storage.save(doc)
            
        # Search
        matches = self.storage.search_by_metadata("category", "test")
        self.assertEqual(len(matches), 3)
        
    def test_compression(self):
        """Test compression effectiveness."""
        # Create large embedding
        large_embedding = np.ones(256, dtype=np.float32) * 0.5
        doc = EncryptedDocument(
            doc_id="compression_test",
            content_hash="hash",
            timestamp="2024-01-01T00:00:00",
            encrypted_embedding=large_embedding
        )
        
        # Compare sizes
        uncompressed = pickle.dumps(doc)
        compressed = doc.to_bytes()
        
        compression_ratio = len(compressed) / len(uncompressed)
        self.assertLess(compression_ratio, 0.9)  # At least 10% compression


class TestBatchOperations(unittest.TestCase):
    """Test batch processing operations."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.key_manager = FHEKeyManager(key_dir=f"{self.temp_dir}/keys")
        self.storage = EncryptedDocumentStore(storage_dir=f"{self.temp_dir}/docs")
        
        # Mock password
        import key_management
        key_management.getpass.getpass = lambda prompt: "testpassword"
        
        # Generate keys
        self.key_manager.generate_keys("test_batch_key")
        
        # Create processor
        self.processor = BatchProcessor(
            key_manager=self.key_manager,
            storage=self.storage,
            config=BatchConfig(batch_size=2, show_progress=False)
        )
        
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)
        
    def test_batch_encryption(self):
        """Test batch document encryption."""
        texts = [
            "Document one about machine learning",
            "Document two about deep learning",
            "Document three about neural networks"
        ]
        
        doc_ids = self.processor.encrypt_documents(texts)
        
        self.assertEqual(len(doc_ids), 3)
        
        # Verify documents were saved
        for doc_id in doc_ids:
            doc = self.storage.load(doc_id)
            self.assertIsNotNone(doc)
            self.assertEqual(doc.encrypted_embedding.shape, (128,))
            
    def test_memory_management(self):
        """Test memory tracking."""
        initial_stats = self.processor.get_memory_stats()
        
        # Process some documents
        texts = ["Test document"] * 10
        self.processor.encrypt_documents(texts)
        
        final_stats = self.processor.get_memory_stats()
        
        # Memory should have increased
        self.assertGreater(final_stats['current_mb'], initial_stats['current_mb'])
        self.assertGreater(final_stats['used_mb'], 0)
        
    def test_document_comparison(self):
        """Test comparing encrypted documents."""
        # Encrypt two similar documents
        texts = [
            "Machine learning is a subset of artificial intelligence",
            "Machine learning is part of AI technology"
        ]
        doc_ids = self.processor.encrypt_documents(texts)
        
        # Compare
        similarity = self.processor.compare_encrypted(doc_ids[0], doc_ids[1])
        
        # Should be somewhat similar
        self.assertGreater(similarity, 0.5)
        self.assertLess(similarity, 1.0)


class TestCLI(unittest.TestCase):
    """Test CLI functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Monkey patch the default directories
        import key_management
        import encrypted_storage
        
        self.original_key_dir = key_management.FHEKeyManager.__init__.__defaults__
        self.original_storage_dir = encrypted_storage.EncryptedDocumentStore.__init__.__defaults__
        
        key_management.FHEKeyManager.__init__.__defaults__ = (f"{self.temp_dir}/keys",)
        encrypted_storage.EncryptedDocumentStore.__init__.__defaults__ = (f"{self.temp_dir}/docs",)
        
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)
        
        # Restore defaults
        import key_management
        import encrypted_storage
        key_management.FHEKeyManager.__init__.__defaults__ = self.original_key_dir
        encrypted_storage.EncryptedDocumentStore.__init__.__defaults__ = self.original_storage_dir
        
    def test_cli_import(self):
        """Test CLI can be imported."""
        try:
            from fhe_cli import FHEDocumentCLI
            cli = FHEDocumentCLI()
            self.assertIsNotNone(cli)
        except ImportError:
            self.fail("Failed to import CLI")


class TestSecurity(unittest.TestCase):
    """Test security features."""
    
    def test_password_protection(self):
        """Test master password protection."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create key manager with password
            import key_management
            key_management.getpass.getpass = lambda prompt: "secure123"
            
            km1 = FHEKeyManager(key_dir=temp_dir)
            km1.generate_keys("secure_key")
            
            # Try to access with wrong password
            key_management.getpass.getpass = lambda prompt: "wrongpassword"
            km2 = FHEKeyManager(key_dir=temp_dir)
            
            with self.assertRaises(ValueError):
                km2.load_model("secure_key")
                
        finally:
            shutil.rmtree(temp_dir)
            
    def test_file_permissions(self):
        """Test secure file permissions."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            import key_management
            key_management.getpass.getpass = lambda prompt: "testpass"
            
            km = FHEKeyManager(key_dir=temp_dir)
            km.generate_keys("perm_test")
            
            # Check permissions on key files
            for file in Path(temp_dir).rglob("*.enc"):
                stat = os.stat(file)
                self.assertEqual(stat.st_mode & 0o777, 0o600)
                
        finally:
            shutil.rmtree(temp_dir)


class TestPerformance(unittest.TestCase):
    """Test performance characteristics."""
    
    def test_encryption_speed(self):
        """Test encryption performance."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Setup
            import key_management
            key_management.getpass.getpass = lambda prompt: "testpass"
            
            processor = BatchProcessor(
                key_manager=FHEKeyManager(key_dir=f"{temp_dir}/keys"),
                storage=EncryptedDocumentStore(storage_dir=f"{temp_dir}/docs"),
                config=BatchConfig(show_progress=False)
            )
            
            # Generate keys
            processor.key_manager.generate_keys()
            processor._load_model()
            
            # Time encryption
            text = "Test document for performance measurement"
            start = time.time()
            doc_ids = processor.encrypt_documents([text])
            elapsed = time.time() - start
            
            # Should complete in reasonable time
            self.assertLess(elapsed, 5.0)  # 5 seconds max
            
        finally:
            shutil.rmtree(temp_dir)
            
    def test_storage_efficiency(self):
        """Test storage space efficiency."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            storage = EncryptedDocumentStore(storage_dir=temp_dir)
            
            # Create and save multiple documents
            for i in range(10):
                doc = EncryptedDocument(
                    doc_id=f"efficiency_test_{i}",
                    content_hash=f"hash_{i}",
                    timestamp="2024-01-01T00:00:00",
                    encrypted_embedding=np.random.randn(128).astype(np.float32)
                )
                storage.save(doc)
                
            # Check average size
            stats = storage.get_stats()
            avg_size = stats['average_size_bytes']
            
            # Should be reasonably small (compressed)
            self.assertLess(avg_size, 5000)  # Less than 5KB per doc
            
        finally:
            shutil.rmtree(temp_dir)


def run_all_tests():
    """Run all tests with summary."""
    print("Running FHE Document System Test Suite")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestKeyManagement,
        TestEncryptedStorage,
        TestBatchOperations,
        TestCLI,
        TestSecurity,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
        
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)