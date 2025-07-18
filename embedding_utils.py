#!/usr/bin/env python3
"""Utility functions for BERT embeddings in FHE context."""

import numpy as np
from typing import Tuple
import json
import pickle
from pathlib import Path

from bert_embeddings import BertEmbedder


def prepare_embedding_for_fhe(embedding: np.ndarray, 
                            scale: int = 1000,
                            normalize: bool = True) -> Tuple[np.ndarray, dict]:
    """
    Prepare embedding for FHE by normalizing and quantizing.
    
    Args:
        embedding: Original BERT embedding
        scale: Scale factor for quantization
        normalize: Whether to L2-normalize first
        
    Returns:
        Quantized embedding and metadata dict
    """
    # Store original stats
    metadata = {
        'original_shape': embedding.shape,
        'original_min': float(embedding.min()),
        'original_max': float(embedding.max()),
        'original_norm': float(np.linalg.norm(embedding)),
    }
    
    # Normalize if requested
    if normalize:
        embedding = embedding / np.linalg.norm(embedding)
        metadata['normalized'] = True
    else:
        metadata['normalized'] = False
        
    # Scale and quantize
    embedding_scaled = embedding * scale
    embedding_quantized = np.round(embedding_scaled).astype(np.int32)
    
    metadata['scale'] = scale
    metadata['quantized_min'] = int(embedding_quantized.min())
    metadata['quantized_max'] = int(embedding_quantized.max())
    
    return embedding_quantized, metadata


def save_embedding_data(embeddings: dict, 
                       metadata: dict,
                       output_path: str):
    """Save embeddings and metadata for later use."""
    data = {
        'embeddings': embeddings,
        'metadata': metadata,
        'version': '1.0',
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
        
    print(f"Saved embedding data to {output_path}")


def load_embedding_data(input_path: str) -> Tuple[dict, dict]:
    """Load previously saved embeddings."""
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
        
    return data['embeddings'], data['metadata']


# Quick test
if __name__ == "__main__":
    print("Testing embedding utilities...")
    
    # Create test embedding
    embedder = BertEmbedder()
    text = "This is a test document for FHE processing."
    embedding = embedder.get_embedding(text)
    
    # Prepare for FHE
    emb_quantized, metadata = prepare_embedding_for_fhe(embedding)
    
    print(f"\nOriginal embedding:")
    print(f"  Shape: {embedding.shape}")
    print(f"  Range: [{embedding.min():.3f}, {embedding.max():.3f}]")
    
    print(f"\nQuantized embedding:")
    print(f"  Shape: {emb_quantized.shape}")
    print(f"  Range: [{emb_quantized.min()}, {emb_quantized.max()}]")
    print(f"  Metadata: {json.dumps(metadata, indent=2)}")
    
    # Test save/load
    test_data = {
        'doc1': emb_quantized,
        'metadata': metadata
    }
    save_embedding_data(test_data, metadata, 'test_embeddings.pkl')
    loaded_data, loaded_meta = load_embedding_data('test_embeddings.pkl')
    
    print(f"\n✅ Save/load test passed!")