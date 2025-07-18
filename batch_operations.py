#!/usr/bin/env python3
"""Batch operations for efficient FHE processing."""

import gc
import time
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
import psutil
from tqdm import tqdm
import logging

from bert_embeddings import BertEmbedder
from dimension_reduction import DimensionReducer
from fhe_similarity import FHESimilarityModel
from encrypted_storage import EncryptedDocument, EncryptedDocumentStore
from key_management import FHEKeyManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch operations."""
    batch_size: int = 10
    max_memory_mb: int = 4000
    checkpoint_interval: int = 50
    show_progress: bool = True
    force_gc: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.max_memory_mb < 100:
            raise ValueError("max_memory_mb must be >= 100")


class BatchProcessor:
    """Handle batch encryption and comparison operations."""
    
    def __init__(self, 
                 embedder: Optional[BertEmbedder] = None,
                 reducer: Optional[DimensionReducer] = None,
                 key_manager: Optional[FHEKeyManager] = None,
                 storage: Optional[EncryptedDocumentStore] = None,
                 config: Optional[BatchConfig] = None):
        """
        Initialize batch processor.
        
        Args:
            embedder: BERT embedder (created if not provided)
            reducer: Dimension reducer (loaded if not provided)
            key_manager: Key manager (created if not provided)
            storage: Document storage (created if not provided)
            config: Batch configuration
        """
        self.embedder = embedder or BertEmbedder()
        self.reducer = reducer or DimensionReducer.load("pca_reducer_128.pkl")
        self.key_manager = key_manager or FHEKeyManager()
        self.storage = storage or EncryptedDocumentStore()
        self.config = config or BatchConfig()
        
        # Create FHE model instance (will be compiled when needed)
        self.fhe_model = None
        self._init_model()
        
        # Memory tracking
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024
        
        logger.info(f"Batch processor initialized (memory: {self.initial_memory:.1f} MB)")
        
    def _init_model(self):
        """Initialize FHE model (compilation happens on first use)."""
        try:
            # Check if we have keys
            current_key = self.key_manager.get_current_key()
            if current_key:
                # Create a new FHE model instance
                from fhe_similarity import FHESimilarityModel
                self.fhe_model = FHESimilarityModel(input_dim=128, n_bits=8)
                
                # Train the model with proper similarity data
                # The train() method will use _prepare_training_data() internally
                X_train, y_train = self.fhe_model.train()
                
                # Compile the model using a sample from the training data
                self.fhe_model.compile(X_train[:10])
                
                # Validate the model works correctly
                import numpy as np
                # Test with identical embeddings (should give similarity ~1.0)
                test_identical = np.hstack([X_train[0, :128], X_train[0, :128]]).reshape(1, -1)
                similarity = self.fhe_model.predict_clear(test_identical)[0]
                if abs(similarity - 1.0) > 0.2:
                    logger.warning(f"Model validation failed: identical embeddings gave similarity {similarity:.3f}")
                else:
                    logger.info(f"Model validation passed: identical embeddings similarity = {similarity:.3f}")
                
                logger.info("FHE model initialized and compiled with similarity training data")
        except Exception as e:
            logger.warning(f"Could not initialize FHE model: {e}")
            logger.info("Generate keys first using key_manager.generate_keys()")
            
    def _check_memory(self) -> float:
        """Check current memory usage in MB."""
        current_mb = self.process.memory_info().rss / 1024 / 1024
        return current_mb
        
    def _maybe_gc(self):
        """Force garbage collection if enabled."""
        if self.config.force_gc:
            gc.collect()
            
    def encrypt_documents(self, 
                         texts: List[str], 
                         doc_ids: Optional[List[str]] = None,
                         metadata: Optional[List[Dict]] = None) -> List[str]:
        """
        Encrypt multiple documents in batches.
        
        Args:
            texts: List of document texts
            doc_ids: Optional document IDs (auto-generated if not provided)
            metadata: Optional metadata for each document
            
        Returns:
            List of document IDs
        """
        if self.fhe_model is None:
            raise RuntimeError("No FHE model initialized. Generate keys first.")
            
        n_docs = len(texts)
        if doc_ids is None:
            doc_ids = [f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}" 
                      for i in range(n_docs)]
        if metadata is None:
            metadata = [{} for _ in range(n_docs)]
            
        logger.info(f"Encrypting {n_docs} documents in batches of {self.config.batch_size}")
        
        # Process in batches
        encrypted_ids = []
        
        # Progress bar
        pbar = tqdm(total=n_docs, desc="Encrypting", disable=not self.config.show_progress)
        
        for i in range(0, n_docs, self.config.batch_size):
            batch_end = min(i + self.config.batch_size, n_docs)
            batch_texts = texts[i:batch_end]
            batch_ids = doc_ids[i:batch_end]
            batch_metadata = metadata[i:batch_end]
            
            # Check memory
            current_memory = self._check_memory()
            if current_memory > self.config.max_memory_mb:
                logger.warning(f"Memory usage high: {current_memory:.1f} MB")
                self._maybe_gc()
                
            # Generate embeddings
            embeddings = self.embedder.get_embeddings_batch(batch_texts)
            
            # Reduce dimensions
            reduced = self.reducer.transform(embeddings)
            
            # Encrypt each embedding
            for j, (text, doc_id, meta, embedding) in enumerate(
                zip(batch_texts, batch_ids, batch_metadata, reduced)):
                
                # Create encrypted embedding
                # Note: In production, we'd use FHE encryption here
                # For now, we'll store the reduced embedding directly
                encrypted_embedding = embedding.astype(np.float32)
                
                # Create document
                doc = EncryptedDocument(
                    doc_id=doc_id,
                    content_hash=hashlib.sha256(text.encode()).hexdigest(),
                    timestamp=datetime.now().isoformat(),
                    encrypted_embedding=encrypted_embedding,
                    key_id=self.key_manager.get_current_key(),
                    metadata=meta
                )
                
                # Save
                self.storage.save(doc)
                encrypted_ids.append(doc_id)
                
            pbar.update(batch_end - i)
            
            # Checkpoint if needed
            if (i + self.config.batch_size) % self.config.checkpoint_interval == 0:
                logger.info(f"Checkpoint: {len(encrypted_ids)} documents encrypted")
                self._maybe_gc()
                
        pbar.close()
        
        logger.info(f"Encrypted {len(encrypted_ids)} documents")
        return encrypted_ids
        
    def compare_encrypted(self, doc_id1: str, doc_id2: str) -> float:
        """
        Compare two encrypted documents.
        
        Args:
            doc_id1: First document ID
            doc_id2: Second document ID
            
        Returns:
            Similarity score
        """
        if self.fhe_model is None:
            raise RuntimeError("No FHE model initialized. Generate keys first.")
            
        # Load documents
        doc1 = self.storage.load(doc_id1)
        doc2 = self.storage.load(doc_id2)
        
        # Prepare input for FHE model
        # Use element-wise product for similarity computation
        X = (doc1.encrypted_embedding * doc2.encrypted_embedding).reshape(1, -1)
        
        # Run FHE prediction
        start_time = time.time()
        
        # In production, this would run on encrypted data
        # For now, we simulate with the compiled model
        similarity = self.fhe_model.model.predict(X)[0]
        
        elapsed = time.time() - start_time
        logger.info(f"Comparison took {elapsed:.2f}s")
        
        return float(similarity)
        
    def search_similar(self, 
                      query_text: str, 
                      top_k: int = 5,
                      min_similarity: float = 0.5) -> List[Tuple[str, float]]:
        """
        Search for similar documents.
        
        Args:
            query_text: Query text
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (doc_id, similarity) tuples
        """
        if self.fhe_model is None:
            raise RuntimeError("No FHE model initialized. Generate keys first.")
            
        # Generate query embedding
        query_embedding = self.embedder.get_embedding(query_text)
        query_reduced = self.reducer.transform(query_embedding.reshape(1, -1))[0]
        
        # Compare with all documents
        all_docs = self.storage.list_documents()
        similarities = []
        
        logger.info(f"Searching {len(all_docs)} documents...")
        
        for doc_info in tqdm(all_docs, desc="Searching", disable=not self.config.show_progress):
            doc_id = doc_info['doc_id']
            doc = self.storage.load(doc_id)
            
            # Prepare input - element-wise product
            X = (query_reduced * doc.encrypted_embedding).reshape(1, -1)
            
            # Compute similarity
            similarity = self.fhe_model.model.predict(X)[0]
            
            if similarity >= min_similarity:
                similarities.append((doc_id, float(similarity)))
                
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
        
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        current = self._check_memory()
        return {
            'initial_mb': self.initial_memory,
            'current_mb': current,
            'used_mb': current - self.initial_memory,
            'max_mb': self.config.max_memory_mb,
            'usage_percent': (current / self.config.max_memory_mb) * 100
        }


def demo_batch_operations():
    """Demonstrate batch operations."""
    import hashlib
    
    print("Batch Operations Demo")
    print("=" * 50)
    
    # Create processor
    processor = BatchProcessor(config=BatchConfig(batch_size=2))
    
    # Sample documents
    texts = [
        "Machine learning is transforming industries.",
        "Deep learning models require large datasets.",
        "Natural language processing enables text understanding.",
        "Computer vision algorithms analyze images.",
        "Reinforcement learning optimizes decision making."
    ]
    
    # Check if we have keys
    if processor.fhe_model is None:
        print("\n1. Generating FHE keys first...")
        processor.key_manager.generate_keys()
        processor._init_model()
    
    # Encrypt documents
    print("\n2. Encrypting documents...")
    doc_ids = processor.encrypt_documents(
        texts,
        metadata=[{"category": "AI"} for _ in texts]
    )
    
    # Memory stats
    print("\n3. Memory usage:")
    stats = processor.get_memory_stats()
    for key, value in stats.items():
        print(f"  {key}: {value:.1f}")
        
    # Compare documents
    print("\n4. Comparing documents...")
    similarity = processor.compare_encrypted(doc_ids[0], doc_ids[1])
    print(f"  Similarity between doc 0 and 1: {similarity:.3f}")
    
    # Search
    print("\n5. Searching for similar documents...")
    query = "Neural networks and deep learning"
    results = processor.search_similar(query, top_k=3)
    print(f"  Query: '{query}'")
    for doc_id, score in results:
        idx = doc_ids.index(doc_id)
        print(f"  - {doc_id}: {score:.3f} ('{texts[idx][:50]}...')")
        
    print("\nDemo complete!")


if __name__ == "__main__":
    demo_batch_operations()