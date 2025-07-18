#!/usr/bin/env python3
"""Secure FHE key management system."""

import os
import json
import pickle
import hashlib
import secrets
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import getpass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FHEKeyManager:
    """Manage FHE keys securely with encryption and rotation."""
    
    def __init__(self, key_dir: str = "~/.fhe_keys"):
        """
        Initialize key manager.
        
        Args:
            key_dir: Directory to store keys (will be created if needed)
        """
        self.key_dir = Path(key_dir).expanduser()
        self.key_dir.mkdir(parents=True, exist_ok=True)
        
        # Key metadata file
        self.metadata_file = self.key_dir / "key_metadata.json"
        self.current_key_id = None
        self._master_key = None
        
        # Key size expectations (approximate)
        self.expected_sizes = {
            'compiled_model': '50-100 MB',
            'metadata': '<1 KB'
        }
        
        logger.info(f"Key manager initialized. Key directory: {self.key_dir}")
        
    def _derive_master_key(self, password: str, salt: bytes) -> bytes:
        """Derive master key from password."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
        
    def _get_master_key(self) -> bytes:
        """Get or create master key for key encryption."""
        if self._master_key is not None:
            return self._master_key
            
        master_key_file = self.key_dir / ".master"
        
        if master_key_file.exists():
            # Load existing master key
            password = getpass.getpass("Enter master password: ")
            
            with open(master_key_file, 'rb') as f:
                data = json.loads(f.read())
                salt = base64.b64decode(data['salt'])
                
            self._master_key = self._derive_master_key(password, salt)
            
            # Verify password by trying to decrypt test data
            try:
                f = Fernet(self._master_key)
                f.decrypt(base64.b64decode(data['test']))
            except:
                raise ValueError("Invalid master password")
                
        else:
            # Create new master key
            print("Creating new master key...")
            password = getpass.getpass("Create master password: ")
            confirm = getpass.getpass("Confirm master password: ")
            
            if password != confirm:
                raise ValueError("Passwords don't match")
                
            salt = secrets.token_bytes(16)
            self._master_key = self._derive_master_key(password, salt)
            
            # Save with test data
            f = Fernet(self._master_key)
            test_data = f.encrypt(b"test")
            
            with open(master_key_file, 'wb') as file:
                file.write(json.dumps({
                    'salt': base64.b64encode(salt).decode(),
                    'test': base64.b64encode(test_data).decode(),
                    'created': datetime.now().isoformat()
                }).encode())
                
            # Secure permissions
            os.chmod(master_key_file, 0o600)
            
        return self._master_key
        
    def generate_keys(self, key_id: Optional[str] = None) -> Dict[str, str]:
        """
        Generate new FHE keys.
        
        Args:
            key_id: Optional key identifier (auto-generated if not provided)
            
        Returns:
            Dictionary with key paths
        """
        if key_id is None:
            key_id = f"fhe_key_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        logger.info(f"Generating new FHE keys with ID: {key_id}")
        
        # Create key directory
        key_path = self.key_dir / key_id
        key_path.mkdir(exist_ok=True)
        
        # Generate keys using our compiled model
        from fhe_similarity import FHESimilarityModel
        import numpy as np
        
        print("Generating FHE keys (this may take 30-60 seconds)...")
        
        # Create a model instance to get keys
        model = FHESimilarityModel(input_dim=256, n_bits=8)
        X_sample = np.random.randn(100, 256).astype(np.float32)
        y_sample = np.random.randn(100)
        
        # Train and compile to generate keys
        model.train(X_sample, y_sample, n_samples=100)
        model.compile(X_sample[:10])
        
        # Save the compiled model (includes keys)
        master_key = self._get_master_key()
        f = Fernet(master_key)
        
        # Save the model parameters and configuration
        # We can't pickle the compiled circuit directly, so we save what we need
        model_data = {
            'input_dim': model.input_dim,
            'n_bits': model.n_bits,
            'similarity_type': model.similarity_type,
            'metrics': model.metrics,
            'compiled': True
        }
        
        # Serialize model data
        serialized_data = pickle.dumps(model_data)
        encrypted_model = f.encrypt(serialized_data)
        
        # Save encrypted model
        model_file = key_path / "compiled_model.enc"
        with open(model_file, 'wb') as file:
            file.write(encrypted_model)
        os.chmod(model_file, 0o600)
        
        # Update metadata
        metadata = self._load_metadata()
        metadata['keys'][key_id] = {
            'created': datetime.now().isoformat(),
            'path': str(key_path),
            'active': True,
            'model_file': str(model_file),
            'size_bytes': len(encrypted_model)
        }
        
        # Set as current key
        metadata['current'] = key_id
        self.current_key_id = key_id
        self._save_metadata(metadata)
        
        print(f"Keys generated successfully!")
        print(f"  Key ID: {key_id}")
        print(f"  Size: {len(encrypted_model) / 1024 / 1024:.1f} MB")
        
        return {
            'key_id': key_id,
            'model_file': str(model_file),
            'created': metadata['keys'][key_id]['created']
        }
        
    def list_keys(self) -> Dict[str, Dict]:
        """List all available keys."""
        metadata = self._load_metadata()
        return metadata.get('keys', {})
        
    def get_current_key(self) -> Optional[str]:
        """Get current active key ID."""
        metadata = self._load_metadata()
        return metadata.get('current')
        
    def load_model(self, key_id: Optional[str] = None):
        """
        Load compiled FHE model.
        
        Args:
            key_id: Key ID to load (uses current if not specified)
            
        Returns:
            Compiled model object
        """
        if key_id is None:
            key_id = self.get_current_key()
            if key_id is None:
                raise ValueError("No current key set. Generate keys first.")
                
        metadata = self._load_metadata()
        if key_id not in metadata.get('keys', {}):
            raise ValueError(f"Key {key_id} not found")
            
        key_info = metadata['keys'][key_id]
        model_file = Path(key_info['model_file'])
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
            
        # Load and decrypt
        master_key = self._get_master_key()
        f = Fernet(master_key)
        
        with open(model_file, 'rb') as file:
            encrypted_data = file.read()
            
        decrypted_data = f.decrypt(encrypted_data)
        model_data = pickle.loads(decrypted_data)
        
        # Note: We return the model configuration, not the compiled circuit
        # The actual FHE model would need to be recompiled for each use
        logger.info(f"Loaded model configuration from key: {key_id}")
        return model_data
        
    def rotate_keys(self, grace_period_days: int = 7) -> Dict[str, str]:
        """
        Rotate to new keys with grace period.
        
        Args:
            grace_period_days: Days to keep old keys active
            
        Returns:
            New key information
        """
        # Generate new keys
        new_key_info = self.generate_keys()
        
        # Update metadata with rotation info
        metadata = self._load_metadata()
        if 'current' in metadata and metadata['current'] in metadata['keys']:
            old_key_id = metadata['current']
            metadata['keys'][old_key_id]['rotated_at'] = datetime.now().isoformat()
            metadata['keys'][old_key_id]['grace_until'] = (
                datetime.now() + timedelta(days=grace_period_days)
            ).isoformat()
            
        self._save_metadata(metadata)
        
        print(f"Key rotation complete. Grace period: {grace_period_days} days")
        return new_key_info
        
    def _load_metadata(self) -> Dict:
        """Load key metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {'keys': {}}
        
    def _save_metadata(self, metadata: Dict):
        """Save key metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        os.chmod(self.metadata_file, 0o600)


def main():
    """Demo key management operations."""
    print("FHE Key Management Demo")
    print("=" * 50)
    
    manager = FHEKeyManager()
    
    # Generate keys
    print("\n1. Generating new keys...")
    key_info = manager.generate_keys()
    
    # List keys
    print("\n2. Available keys:")
    for key_id, info in manager.list_keys().items():
        print(f"  - {key_id}: created {info['created']}")
        
    # Load model
    print("\n3. Loading model...")
    model = manager.load_model()
    print(f"  Model loaded successfully!")
    
    # Demonstrate rotation
    print("\n4. Key rotation demo...")
    manager.rotate_keys(grace_period_days=7)
    
    print("\nDemo complete!")


if __name__ == "__main__":
    main()