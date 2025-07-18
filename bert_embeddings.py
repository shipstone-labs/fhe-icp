#!/usr/bin/env python3
"""BERT embeddings extraction for FHE similarity."""

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from typing import List, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BertEmbedder:
    """Extract BERT embeddings for text documents."""
    
    def __init__(self, model_name: str = 'bert-base-uncased', 
                 max_length: int = 100,
                 device: str = None):
        """
        Initialize BERT model for embedding extraction.
        
        Args:
            model_name: Hugging Face model name
            max_length: Maximum sequence length (must be ≤ 512 for BERT)
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.model_name = model_name
        self.max_length = min(max_length, 512)  # BERT's limit
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        logger.info(f"Loading model: {model_name}")
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Model info
        self.hidden_size = self.model.config.hidden_size
        logger.info(f"Model loaded. Hidden size: {self.hidden_size}")
        
    def get_embedding(self, text: str, pooling: str = 'mean') -> np.ndarray:
        """
        Extract embedding for a single text.
        
        Args:
            text: Input text
            pooling: How to pool token embeddings ('mean', 'cls', 'max')
            
        Returns:
            Numpy array of shape (hidden_size,)
        """
        # Tokenize
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**encoded)
            hidden_states = outputs.last_hidden_state
            
        # Apply pooling
        if pooling == 'mean':
            # Mean pooling (excluding padding tokens)
            attention_mask = encoded['attention_mask'].unsqueeze(-1)
            masked_hidden = hidden_states * attention_mask
            summed = masked_hidden.sum(dim=1)
            count = attention_mask.sum(dim=1)
            embedding = summed / count
        elif pooling == 'cls':
            # Use [CLS] token
            embedding = hidden_states[:, 0, :]
        elif pooling == 'max':
            # Max pooling
            embedding = hidden_states.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
            
        # Convert to numpy
        embedding = embedding.cpu().numpy()[0]  # Remove batch dimension
        
        return embedding
    
    def get_embeddings_batch(self, texts: List[str], 
                           batch_size: int = 8,
                           pooling: str = 'mean') -> np.ndarray:
        """
        Extract embeddings for multiple texts efficiently.
        
        Args:
            texts: List of input texts
            batch_size: Process this many texts at once
            pooling: Pooling method
            
        Returns:
            Numpy array of shape (n_texts, hidden_size)
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**encoded)
                hidden_states = outputs.last_hidden_state
                
            # Apply pooling
            if pooling == 'mean':
                attention_mask = encoded['attention_mask'].unsqueeze(-1)
                masked_hidden = hidden_states * attention_mask
                summed = masked_hidden.sum(dim=1)
                count = attention_mask.sum(dim=1)
                batch_embeddings = summed / count
            elif pooling == 'cls':
                batch_embeddings = hidden_states[:, 0, :]
            elif pooling == 'max':
                batch_embeddings = hidden_states.max(dim=1)[0]
                
            # Convert to numpy and append
            batch_embeddings = batch_embeddings.cpu().numpy()
            embeddings.append(batch_embeddings)
            
        # Concatenate all batches
        embeddings = np.vstack(embeddings)
        
        return embeddings
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            
        Returns:
            Similarity score between -1 and 1
        """
        # Normalize
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)
        
        # Compute cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        
        return similarity


def test_embedder():
    """Test the BertEmbedder class."""
    print("Testing BertEmbedder...\n")
    
    # Initialize embedder
    embedder = BertEmbedder(max_length=100)
    
    # Test texts
    texts = [
        "The cat sat on the mat.",
        "A feline rested on the rug.",
        "Dogs are great pets.",
        "Machine learning is fascinating.",
        "Artificial intelligence transforms technology."
    ]
    
    print("Extracting embeddings for test texts...")
    embeddings = embedder.get_embeddings_batch(texts)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Compute similarity matrix
    print("\nSimilarity matrix:")
    print("Text pairs:", end="")
    for i in range(len(texts)):
        print(f"\t{i+1}", end="")
    print()
    
    for i in range(len(texts)):
        print(f"Text {i+1}:", end="")
        for j in range(len(texts)):
            sim = embedder.compute_similarity(embeddings[i], embeddings[j])
            print(f"\t{sim:.3f}", end="")
        print()
    
    print("\nText legend:")
    for i, text in enumerate(texts):
        print(f"  {i+1}: '{text[:30]}...'")
    
    # Test pooling methods
    print("\n\nTesting different pooling methods...")
    test_text = "Understanding natural language processing."
    
    for pooling in ['mean', 'cls', 'max']:
        emb = embedder.get_embedding(test_text, pooling=pooling)
        print(f"  {pooling} pooling: shape={emb.shape}, mean={emb.mean():.4f}, std={emb.std():.4f}")


if __name__ == "__main__":
    test_embedder()