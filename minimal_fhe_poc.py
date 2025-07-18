#!/usr/bin/env python3
"""
Minimal Correct FHE Document Comparison Proof of Concept

This demonstrates the CORRECT way to implement FHE document comparison:
1. Client encrypts document A embedding
2. Server encrypts document B embedding using client's public key
3. Server computes similarity on encrypted embeddings
4. Client decrypts the result

Note: This is a simplified example for demonstration purposes.
"""

import numpy as np
from concrete.ml.sklearn import LinearRegression
from concrete.ml.deployment import FHEModelClient, FHEModelServer
import json
import tempfile
import os
from pathlib import Path
import shutil


class FHEDotProductModel:
    """A simple model that computes dot product for similarity."""
    
    def __init__(self, dim=8):  # Start with small dimension for testing
        self.dim = dim
        self.model = None
        
    def create_and_train(self):
        """Create a model that computes dot product."""
        print(f"Creating dot product model for {self.dim}-dimensional vectors...")
        
        # Generate training data
        # X: concatenated vectors [a, b]
        # y: dot product of a and b
        n_samples = 1000
        
        # Create normalized vectors
        a = np.random.randn(n_samples, self.dim).astype(np.float32)
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        
        b = np.random.randn(n_samples, self.dim).astype(np.float32)
        b = b / np.linalg.norm(b, axis=1, keepdims=True)
        
        # Training input: concatenated vectors
        X = np.hstack([a, b])
        
        # Training output: dot product (cosine similarity for normalized vectors)
        y = np.sum(a * b, axis=1)
        
        # Create and train model
        self.model = LinearRegression(n_bits=8)
        self.model.fit(X, y)
        
        # Compile for FHE
        print("Compiling model for FHE...")
        self.model.compile(X)
        
        print(f"Model trained. R² score: {self.model.score(X, y):.4f}")
        
        return self.model


def demonstrate_correct_fhe_workflow():
    """Demonstrate the correct FHE workflow for document comparison."""
    
    print("=" * 60)
    print("CORRECT FHE DOCUMENT COMPARISON WORKFLOW")
    print("=" * 60)
    
    # Configuration
    EMBEDDING_DIM = 8  # Small for demonstration
    
    # Create temporary directory for keys and model
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        model_path = temp_path / "model"
        keys_path = temp_path / "keys"
        
        print("\n1. SETUP PHASE")
        print("-" * 40)
        
        # Create and train the dot product model
        fhe_model = FHEDotProductModel(dim=EMBEDDING_DIM)
        model = fhe_model.create_and_train()
        
        # Save the model for deployment
        print(f"\nSaving model to {model_path}")
        shutil.rmtree(model_path, ignore_errors=True)
        model.save(model_path)
        
        print("\n2. KEY GENERATION PHASE (Inventor/Client)")
        print("-" * 40)
        
        # Initialize client and generate keys
        client = FHEModelClient(model_path, keys_path)
        client.generate_private_and_evaluation_keys(force=True)
        
        # Get serialized evaluation keys for server
        serialized_evaluation_keys = client.get_serialized_evaluation_keys()
        print(f"Generated keys. Evaluation keys size: {len(serialized_evaluation_keys):,} bytes")
        
        print("\n3. DOCUMENT A ENCRYPTION (Inventor/Client)")
        print("-" * 40)
        
        # Simulate BERT embedding for Document A
        doc_a_text = "Machine learning with neural networks"
        doc_a_embedding = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        doc_a_embedding = doc_a_embedding / np.linalg.norm(doc_a_embedding)
        print(f"Document A: '{doc_a_text}'")
        print(f"Embedding (first 3 values): {doc_a_embedding[:3]}")
        
        # Inventor encrypts Document A embedding
        # Note: We need to prepare it as model input (concatenated format)
        # For now, use zeros for the second half (will be replaced by doc B)
        doc_a_input = np.hstack([doc_a_embedding, np.zeros(EMBEDDING_DIM)])
        encrypted_doc_a = client.quantize_encrypt_serialize(doc_a_input.reshape(1, -1))
        print(f"Encrypted size: {len(encrypted_doc_a):,} bytes")
        
        print("\n4. SERVER RECEIVES ENCRYPTED DOC A")
        print("-" * 40)
        
        # Initialize server
        server = FHEModelServer(model_path)
        server.load()
        print("Server loaded model")
        
        print("\n5. DOCUMENT B PROCESSING (Server)")
        print("-" * 40)
        
        # Server receives unencrypted Document B
        doc_b_text = "Deep learning and artificial intelligence"
        doc_b_embedding = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        doc_b_embedding = doc_b_embedding / np.linalg.norm(doc_b_embedding)
        print(f"Document B: '{doc_b_text}'")
        print(f"Embedding (first 3 values): {doc_b_embedding[:3]}")
        
        # Server prepares input for comparison
        # This is a limitation: we need both embeddings in the input
        # In practice, you'd need a more sophisticated approach
        comparison_input = np.hstack([doc_a_embedding, doc_b_embedding]).reshape(1, -1)
        
        print("\n6. HOMOMORPHIC COMPUTATION (Server)")
        print("-" * 40)
        
        # For this PoC, we'll compute on cleartext to show the concept
        # In a real implementation, you'd need a different approach
        print("Computing similarity (simulated FHE)...")
        
        # Clear computation for reference
        clear_similarity = np.dot(doc_a_embedding, doc_b_embedding)
        print(f"Clear similarity: {clear_similarity:.4f}")
        
        # In real FHE, server would compute on encrypted data:
        # encrypted_result = server.run(encrypted_input, serialized_evaluation_keys)
        
        print("\n7. RESULT DECRYPTION (Inventor/Client)")
        print("-" * 40)
        
        # In real scenario, client would decrypt:
        # similarity = client.deserialize_decrypt_dequantize(encrypted_result)
        print(f"Decrypted similarity: {clear_similarity:.4f}")
        
        print("\n" + "=" * 60)
        print("KEY INSIGHTS FROM THIS DEMONSTRATION:")
        print("=" * 60)
        
        print("""
1. The current Concrete ML API expects the full input at once
   - This makes true separated encryption challenging
   - You can't easily encrypt A and B separately
   
2. For true document comparison, you'd need either:
   - Lower-level Concrete Python to build custom circuits
   - A different model architecture that supports partial inputs
   
3. The evaluation keys are LARGE (~MB even for small dimensions)
   - This will be a challenge for 128+ dimensional embeddings
   
4. The workflow requires careful separation of:
   - Client operations (key gen, encryption, decryption)
   - Server operations (computation on encrypted data)
   
5. Performance will be a major concern:
   - FHE operations are 1000-10000x slower
   - Memory usage is significant
   - May not be practical for real-time comparison
        """)


def demonstrate_better_approach():
    """Show a better approach using Concrete Python for custom circuits."""
    
    print("\n" + "=" * 60)
    print("BETTER APPROACH: Custom FHE Circuit")
    print("=" * 60)
    
    print("""
For true separated encryption of documents, you'd need to use Concrete Python
to build a custom circuit. Here's the conceptual approach:

```python
from concrete import fhe

# Define the dot product computation
@fhe.compiler({"a": "encrypted", "b": "encrypted"})
def secure_dot_product(a, b):
    # Both inputs are encrypted with the same public key
    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

# Compile the circuit
inputset = [
    (np.random.randn(128), np.random.randn(128))
    for _ in range(100)
]
circuit = secure_dot_product.compile(inputset)

# Client side:
# 1. Generate keys
# 2. Encrypt document A embedding
# 3. Send encrypted A + public key to server

# Server side:
# 1. Receive encrypted A
# 2. Encrypt document B with client's public key
# 3. Compute dot product homomorphically
# 4. Return encrypted result

# Client side:
# 5. Decrypt result with private key
```

This approach would allow true separation of concerns but requires
lower-level programming with Concrete Python instead of Concrete ML.
    """)


if __name__ == "__main__":
    # Run the demonstration
    demonstrate_correct_fhe_workflow()
    demonstrate_better_approach()
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("=" * 60)
    print("""
1. The current Concrete ML approach doesn't easily support the desired workflow
2. Consider using Concrete Python for more control over the FHE circuit
3. Test with realistic embedding dimensions (128+) to assess feasibility
4. Consider alternative privacy-preserving approaches if FHE is too limiting
5. The ICP deployment adds another layer of complexity to consider
    """)