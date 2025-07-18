#!/usr/bin/env python3
"""
Session 4 Quick Start - Encrypted Storage Example

This is a simplified example to get started with Session 4.
Use the full implementations in the docs for production.
"""

from dataclasses import dataclass, asdict
import json
import gzip
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import hashlib


@dataclass
class SimpleEncryptedDoc:
    """Simplified encrypted document format."""
    doc_id: str
    content_hash: str
    timestamp: str
    encrypted_data: bytes
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert bytes to hex for JSON serialization
        data['encrypted_data'] = data['encrypted_data'].hex()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SimpleEncryptedDoc':
        """Create from dictionary."""
        # Convert hex back to bytes
        data['encrypted_data'] = bytes.fromhex(data['encrypted_data'])
        return cls(**data)


def quick_encrypt_test():
    """Quick test of document encryption concept."""
    print("Session 4 Quick Start - Encrypted Storage")
    print("="*50)
    
    # 1. Create a test document
    doc = SimpleEncryptedDoc(
        doc_id="test_001",
        content_hash=hashlib.sha256(b"Hello, FHE World!").hexdigest(),
        timestamp=datetime.now().isoformat(),
        encrypted_data=b"This would be encrypted embedding data"
    )
    
    print(f"Created document: {doc.doc_id}")
    print(f"Content hash: {doc.content_hash[:16]}...")
    
    # 2. Save to file (JSON + gzip)
    storage_dir = Path("./test_storage")
    storage_dir.mkdir(exist_ok=True)
    
    filepath = storage_dir / f"{doc.doc_id}.json.gz"
    
    # Serialize and compress
    json_data = json.dumps(doc.to_dict(), indent=2).encode()
    compressed = gzip.compress(json_data)
    
    with open(filepath, 'wb') as f:
        f.write(compressed)
    
    original_size = len(json_data)
    compressed_size = len(compressed)
    print(f"\nSaved to: {filepath}")
    print(f"Original size: {original_size} bytes")
    print(f"Compressed size: {compressed_size} bytes")
    print(f"Compression ratio: {compressed_size/original_size:.1%}")
    
    # 3. Load back
    with open(filepath, 'rb') as f:
        loaded_compressed = f.read()
    
    loaded_json = gzip.decompress(loaded_compressed)
    loaded_dict = json.loads(loaded_json)
    loaded_doc = SimpleEncryptedDoc.from_dict(loaded_dict)
    
    print(f"\nLoaded document: {loaded_doc.doc_id}")
    print(f"Verified: {loaded_doc.doc_id == doc.doc_id}")
    
    # 4. Next steps
    print("\n" + "="*50)
    print("Next Steps for Full Implementation:")
    print("1. Add encryption using FHE circuit")
    print("2. Implement batch processing")
    print("3. Add document indexing")
    print("4. Create CLI interface")
    print("\nRefer to docs/session4-*.md for complete implementation")


if __name__ == "__main__":
    quick_encrypt_test()
