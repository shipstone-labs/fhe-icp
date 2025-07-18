# FHE-BERT Document Similarity: Claude Code Development Plan

## Project Overview

Build a CLI tool that encrypts documents using FHE and compares them with unencrypted documents using BERT embeddings.

## Session 1: Environment Setup & Basic FHE Test (30 min)

**Goal:** Install dependencies and verify FHE works with a simple example

**Tasks:**

1. Create project directory and virtual environment
2. Install Concrete-ML and dependencies
3. Run basic FHE multiplication test

```bash
# Commands to paste
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install concrete-ml transformers scikit-learn
```

**Test:**

```python
# test_fhe.py
from concrete.ml.sklearn import LinearRegression
import numpy as np

# Simple FHE test
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])

model = LinearRegression(n_bits=8)
model.fit(X, y)
model.compile(X)

# Test prediction
clear_pred = model.predict(np.array([[5]]))
fhe_pred = model.predict(np.array([[5]]), fhe="execute")
print(f"Clear: {clear_pred}, FHE: {fhe_pred}")
```

**Documentation:**

- [Concrete-ML Installation](https://docs.zama.ai/concrete-ml/getting-started/installing)
- [Concrete-ML Quick Start](https://docs.zama.ai/concrete-ml/getting-started/quick_start)

------

## Session 2: BERT Embeddings Pipeline (45 min)

**Goal:** Extract BERT embeddings from text documents

**Tasks:**

1. Load pre-trained BERT model
2. Create function to convert text → embeddings
3. Test with sample documents

```python
# bert_embeddings.py
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class BertEmbedder:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
    
    def get_embedding(self, text, max_length=100):
        # Tokenize
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            max_length=max_length, 
            truncation=True,
            padding=True
        )
        
        # Get BERT output
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Average pooling over tokens
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.numpy()[0]

# Test
embedder = BertEmbedder()
emb1 = embedder.get_embedding("The cat sat on the mat")
emb2 = embedder.get_embedding("A feline rested on the rug")
print(f"Embedding shape: {emb1.shape}")
print(f"Cosine similarity: {np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))}")
```

**Documentation:**

- [Hugging Face BERT Tutorial](https://huggingface.co/docs/transformers/model_doc/bert)
- [Concrete-ML Sentiment Analysis Example](https://huggingface.co/blog/sentiment-analysis-fhe)

------

## Session 3: FHE-Compatible Similarity Model (45 min)

**Goal:** Build and compile an FHE-compatible model for similarity computation

**Tasks:**

1. Create similarity computation model
2. Compile for FHE execution
3. Test encrypted vs unencrypted predictions

```python
# fhe_similarity.py
from concrete.ml.sklearn import SGDRegressor
import numpy as np
from sklearn.preprocessing import normalize

class FHESimilarity:
    def __init__(self, n_bits=12):
        # Using SGDRegressor for dot product approximation
        self.model = SGDRegressor(n_bits=n_bits, random_state=42)
        self.compiled = False
    
    def prepare_training_data(self, n_samples=1000, dim=768):
        # Generate synthetic training data
        X1 = np.random.randn(n_samples, dim)
        X2 = np.random.randn(n_samples, dim)
        
        # Normalize
        X1 = normalize(X1)
        X2 = normalize(X2)
        
        # Concatenate for model input
        X = np.hstack([X1, X2])
        
        # Compute cosine similarities as targets
        y = np.sum(X1 * X2, axis=1)
        
        return X, y
    
    def train_and_compile(self):
        X, y = self.prepare_training_data()
        
        # Train
        self.model.fit(X, y)
        
        # Compile for FHE
        self.model.compile(X)
        self.compiled = True
        
        return self.model.score(X, y)

# Test
fhe_sim = FHESimilarity()
score = fhe_sim.train_and_compile()
print(f"Model R² score: {score}")
```

**Documentation:**

- [Concrete-ML Deep Learning](https://docs.zama.ai/concrete-ml/deep-learning/fhe_friendly_models)
- [FHE-Compatible Models Guide](https://docs.zama.ai/concrete-ml/getting-started/concepts)

------

## Session 4: Encryption and Storage Functions (30 min)

**Goal:** Implement document encryption and storage

**Tasks:**

1. Create encryption function
2. Implement file storage for encrypted data
3. Test encryption/decryption cycle

```python
# encryption.py
import pickle
import json
from pathlib import Path

class DocumentEncryptor:
    def __init__(self, embedder, fhe_model):
        self.embedder = embedder
        self.fhe_model = fhe_model
    
    def encrypt_document(self, text, output_path):
        # Get embedding
        embedding = self.embedder.get_embedding(text)
        
        # Create encryption (using FHE client)
        encrypted_embedding = self.fhe_model.model.fhe_circuit.encrypt(embedding)
        
        # Save encrypted data and metadata
        data = {
            'encrypted': encrypted_embedding,
            'length': len(text.split()),
            'model': 'bert-base-uncased'
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        return output_path
    
    def load_encrypted(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

# Test
encryptor = DocumentEncryptor(embedder, fhe_sim)
enc_path = encryptor.encrypt_document("This is a test document", "test.enc")
print(f"Encrypted document saved to: {enc_path}")
```

**Documentation:**

- [Concrete-ML Encryption Guide](https://docs.zama.ai/concrete-ml/getting-started/concepts#key-generation)
- [FHE Circuit Documentation](https://docs.zama.ai/concrete-ml/guides/client_server)

------

## Session 5: Comparison Function (45 min)

**Goal:** Implement the comparison function between encrypted and plain documents

**Tasks:**

1. Create comparison function
2. Handle encrypted-plaintext operations
3. Return similarity percentage

```python
# comparison.py
import numpy as np

class DocumentComparator:
    def __init__(self, embedder, fhe_model, encryptor):
        self.embedder = embedder
        self.fhe_model = fhe_model
        self.encryptor = encryptor
    
    def compare(self, encrypted_path, plain_text):
        # Load encrypted document
        enc_data = self.encryptor.load_encrypted(encrypted_path)
        
        # Get plaintext embedding
        plain_embedding = self.embedder.get_embedding(plain_text)
        
        # Prepare input for FHE model
        # Concatenate encrypted and plain embeddings
        model_input = np.concatenate([
            enc_data['encrypted'],
            plain_embedding
        ])
        
        # Run FHE comparison
        if self.fhe_model.compiled:
            similarity = self.fhe_model.model.predict(
                model_input.reshape(1, -1), 
                fhe="execute"
            )[0]
        else:
            raise RuntimeError("Model not compiled for FHE")
        
        # Convert to percentage
        percentage = min(100, max(0, similarity * 100))
        
        return {
            'similarity': percentage,
            'encrypted_tokens': enc_data['length'],
            'plain_tokens': len(plain_text.split())
        }

# Test
comparator = DocumentComparator(embedder, fhe_sim, encryptor)
result = comparator.compare("test.enc", "This is another test document")
print(f"Similarity: {result['similarity']:.1f}%")
```

**Documentation:**

- [Concrete-ML Inference Guide](https://docs.zama.ai/concrete-ml/guides/prediction)
- [Client-Server Deployment](https://docs.zama.ai/concrete-ml/guides/client_server)

------

## Session 6: CLI Interface (30 min)

**Goal:** Create the command-line interface

**Tasks:**

1. Implement argument parsing
2. Create encrypt and compare commands
3. Add error handling

```python
# cli.py
import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='FHE Document Similarity')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Encrypt command
    encrypt_parser = subparsers.add_parser('encrypt', help='Encrypt a document')
    encrypt_parser.add_argument('text', help='Text to encrypt (or @filename)')
    encrypt_parser.add_argument('-o', '--output', default='document.enc', 
                              help='Output file (default: document.enc)')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare documents')
    compare_parser.add_argument('encrypted', help='Encrypted document path')
    compare_parser.add_argument('plaintext', help='Plain text (or @filename)')
    
    args = parser.parse_args()
    
    if args.command == 'encrypt':
        # Read text from file if @ prefix
        text = args.text
        if text.startswith('@'):
            with open(text[1:], 'r') as f:
                text = f.read()
        
        # Initialize models (cached after first run)
        embedder = BertEmbedder()
        fhe_sim = FHESimilarity()
        fhe_sim.train_and_compile()
        encryptor = DocumentEncryptor(embedder, fhe_sim)
        
        # Encrypt
        output = encryptor.encrypt_document(text, args.output)
        print(f"Encrypted document saved to: {output}")
    
    elif args.command == 'compare':
        # Read plaintext
        plaintext = args.plaintext
        if plaintext.startswith('@'):
            with open(plaintext[1:], 'r') as f:
                plaintext = f.read()
        
        # Initialize and compare
        embedder = BertEmbedder()
        fhe_sim = FHESimilarity()
        fhe_sim.train_and_compile()
        encryptor = DocumentEncryptor(embedder, fhe_sim)
        comparator = DocumentComparator(embedder, fhe_sim, encryptor)
        
        result = comparator.compare(args.encrypted, plaintext)
        print(f"Similarity: {result['similarity']:.1f}%")
        print(f"Encrypted tokens: {result['encrypted_tokens']}")
        print(f"Plaintext tokens: {result['plain_tokens']}")

if __name__ == '__main__':
    main()
```

**Test commands:**

```bash
python cli.py encrypt "The quick brown fox jumps over the lazy dog" -o fox.enc
python cli.py compare fox.enc "A fast brown fox leaps over a sleepy dog"
```

------

## Session 7: Optimization and Testing (45 min)

**Goal:** Optimize performance and add comprehensive tests

**Tasks:**

1. Add model caching
2. Implement proper error handling
3. Create test suite

```python
# test_suite.py
import unittest
import tempfile
import os

class TestFHESimilarity(unittest.TestCase):
    def setUp(self):
        self.embedder = BertEmbedder()
        self.fhe_sim = FHESimilarity()
        self.fhe_sim.train_and_compile()
    
    def test_similar_documents(self):
        # Test highly similar documents
        doc1 = "The weather is nice today"
        doc2 = "The weather is pleasant today"
        
        with tempfile.NamedTemporaryFile(suffix='.enc', delete=False) as tmp:
            encryptor = DocumentEncryptor(self.embedder, self.fhe_sim)
            encryptor.encrypt_document(doc1, tmp.name)
            
            comparator = DocumentComparator(self.embedder, self.fhe_sim, encryptor)
            result = comparator.compare(tmp.name, doc2)
            
            self.assertGreater(result['similarity'], 70)
            os.unlink(tmp.name)
    
    def test_different_documents(self):
        # Test different documents
        doc1 = "Machine learning is fascinating"
        doc2 = "I love cooking pasta"
        
        with tempfile.NamedTemporaryFile(suffix='.enc', delete=False) as tmp:
            encryptor = DocumentEncryptor(self.embedder, self.fhe_sim)
            encryptor.encrypt_document(doc1, tmp.name)
            
            comparator = DocumentComparator(self.embedder, self.fhe_sim, encryptor)
            result = comparator.compare(tmp.name, doc2)
            
            self.assertLess(result['similarity'], 30)
            os.unlink(tmp.name)

if __name__ == '__main__':
    unittest.main()
```

**Documentation:**

- [Concrete-ML Optimization](https://docs.zama.ai/concrete-ml/guides/optimization)
- [Python unittest Documentation](https://docs.python.org/3/library/unittest.html)

------

## Final Session: ICP Canister Prep (Optional, 45 min)

**Goal:** Prepare code for ICP deployment

**Documentation:**

- [ICP Quick Start](https://internetcomputer.org/docs/current/developer-docs/getting-started/hello-world)
- [Azle TypeScript CDK](https://demergent-labs.github.io/azle/)

## Complete Project Structure

```
fhe-bert-similarity/
├── bert_embeddings.py
├── fhe_similarity.py
├── encryption.py
├── comparison.py
├── cli.py
├── test_suite.py
├── requirements.txt
└── README.md
```

## Next Steps

1. Performance optimization (GPU support)
2. Better similarity metrics
3. Multi-document batch processing
4. Web interface
5. ICP deployment