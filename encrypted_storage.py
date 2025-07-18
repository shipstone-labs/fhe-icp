#!/usr/bin/env python3
"""Standardized encrypted document storage format."""

import gzip
import json
import pickle
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EncryptedDocument:
    """Encrypted document with metadata."""
    doc_id: str
    content_hash: str  # SHA-256 of original content
    timestamp: str  # ISO format timestamp
    encrypted_embedding: np.ndarray  # Encrypted embedding (128-dim or 256-dim)
    model_version: str = "1.0"
    key_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate document after creation."""
        if self.encrypted_embedding is not None:
            if not isinstance(self.encrypted_embedding, np.ndarray):
                raise TypeError("encrypted_embedding must be numpy array")
            if len(self.encrypted_embedding.shape) != 1:
                raise ValueError(f"Expected 1D embedding, got shape {self.encrypted_embedding.shape}")
            if self.encrypted_embedding.shape[0] not in (128, 256):
                raise ValueError(f"Expected 128-dim or 256-dim embedding, got {self.encrypted_embedding.shape}")
                
    def to_bytes(self) -> bytes:
        """Serialize to bytes using pickle+gzip."""
        return gzip.compress(pickle.dumps(self))
        
    @classmethod
    def from_bytes(cls, data: bytes) -> 'EncryptedDocument':
        """Deserialize from bytes."""
        return pickle.loads(gzip.decompress(data))
        
    def size_bytes(self) -> int:
        """Get serialized size in bytes."""
        return len(self.to_bytes())


class EncryptedDocumentStore:
    """Manage encrypted document storage."""
    
    def __init__(self, storage_dir: str = "./encrypted_docs"):
        """
        Initialize document store.
        
        Args:
            storage_dir: Directory for storing documents
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Index file for fast lookups
        self.index_file = self.storage_dir / "index.json"
        self.index = self._load_index()
        
        logger.info(f"Document store initialized at: {self.storage_dir}")
        
    def save(self, doc: EncryptedDocument) -> str:
        """
        Save encrypted document.
        
        Args:
            doc: Document to save
            
        Returns:
            Path to saved document
        """
        # Validate document
        doc.__post_init__()
        
        # Create filename
        filename = f"{doc.doc_id}.enc"
        filepath = self.storage_dir / filename
        
        # Serialize and save
        data = doc.to_bytes()
        with open(filepath, 'wb') as f:
            f.write(data)
            
        # Update index
        self.index[doc.doc_id] = {
            'filename': filename,
            'timestamp': doc.timestamp,
            'content_hash': doc.content_hash,
            'size_bytes': len(data),
            'model_version': doc.model_version,
            'key_id': doc.key_id,
            'metadata': doc.metadata
        }
        self._save_index()
        
        logger.info(f"Saved document {doc.doc_id} ({len(data)} bytes)")
        return str(filepath)
        
    def load(self, doc_id: str) -> EncryptedDocument:
        """
        Load encrypted document.
        
        Args:
            doc_id: Document ID to load
            
        Returns:
            Encrypted document
        """
        if doc_id not in self.index:
            raise KeyError(f"Document {doc_id} not found")
            
        filename = self.index[doc_id]['filename']
        filepath = self.storage_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Document file missing: {filepath}")
            
        with open(filepath, 'rb') as f:
            data = f.read()
            
        doc = EncryptedDocument.from_bytes(data)
        logger.info(f"Loaded document {doc_id}")
        return doc
        
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents with metadata."""
        return [
            {'doc_id': doc_id, **info}
            for doc_id, info in self.index.items()
        ]
        
    def delete(self, doc_id: str) -> bool:
        """
        Delete document.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        if doc_id not in self.index:
            return False
            
        # Delete file
        filename = self.index[doc_id]['filename']
        filepath = self.storage_dir / filename
        if filepath.exists():
            filepath.unlink()
            
        # Update index
        del self.index[doc_id]
        self._save_index()
        
        logger.info(f"Deleted document {doc_id}")
        return True
        
    def search_by_metadata(self, key: str, value: Any) -> List[str]:
        """
        Search documents by metadata.
        
        Args:
            key: Metadata key to search
            value: Value to match
            
        Returns:
            List of matching document IDs
        """
        matches = []
        for doc_id, info in self.index.items():
            if key in info.get('metadata', {}) and info['metadata'][key] == value:
                matches.append(doc_id)
        return matches
        
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total_size = sum(info['size_bytes'] for info in self.index.values())
        
        return {
            'total_documents': len(self.index),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / 1024 / 1024,
            'average_size_bytes': total_size / len(self.index) if self.index else 0,
            'storage_dir': str(self.storage_dir)
        }
        
    def validate_all(self) -> Dict[str, List[str]]:
        """
        Validate all stored documents.
        
        Returns:
            Dictionary with 'valid' and 'invalid' document lists
        """
        valid = []
        invalid = []
        
        for doc_id in self.index:
            try:
                doc = self.load(doc_id)
                doc.__post_init__()  # Validate structure
                valid.append(doc_id)
            except Exception as e:
                logger.error(f"Validation failed for {doc_id}: {e}")
                invalid.append(doc_id)
                
        return {'valid': valid, 'invalid': invalid}
        
    def _load_index(self) -> Dict[str, Dict]:
        """Load document index."""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                return json.load(f)
        return {}
        
    def _save_index(self):
        """Save document index."""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)


def demo_storage():
    """Demonstrate storage operations."""
    print("Encrypted Document Storage Demo")
    print("=" * 50)
    
    # Create store
    store = EncryptedDocumentStore()
    
    # Create sample document
    print("\n1. Creating sample document...")
    doc = EncryptedDocument(
        doc_id="sample_001",
        content_hash=hashlib.sha256(b"Sample document content").hexdigest(),
        timestamp=datetime.now().isoformat(),
        encrypted_embedding=np.random.randn(128).astype(np.float32),
        metadata={"category": "demo", "importance": "high"}
    )
    
    # Save document
    print("\n2. Saving document...")
    path = store.save(doc)
    print(f"  Saved to: {path}")
    print(f"  Size: {doc.size_bytes()} bytes")
    
    # List documents
    print("\n3. Listing documents:")
    for doc_info in store.list_documents():
        print(f"  - {doc_info['doc_id']}: {doc_info['size_bytes']} bytes")
        
    # Load document
    print("\n4. Loading document...")
    loaded_doc = store.load("sample_001")
    print(f"  Loaded: {loaded_doc.doc_id}")
    print(f"  Verified: {np.allclose(loaded_doc.encrypted_embedding, doc.encrypted_embedding)}")
    
    # Search by metadata
    print("\n5. Searching by metadata...")
    matches = store.search_by_metadata("category", "demo")
    print(f"  Found {len(matches)} documents with category='demo'")
    
    # Get stats
    print("\n6. Storage statistics:")
    stats = store.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
        
    # Validate
    print("\n7. Validating all documents...")
    validation = store.validate_all()
    print(f"  Valid: {len(validation['valid'])}")
    print(f"  Invalid: {len(validation['invalid'])}")
    
    print("\nDemo complete!")


if __name__ == "__main__":
    demo_storage()