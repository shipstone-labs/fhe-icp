# FHE-BERT Document Similarity

A CLI tool that encrypts documents using Fully Homomorphic Encryption (FHE) and compares them with unencrypted documents using BERT embeddings.

## Features

- Encrypt documents using FHE to protect privacy
- Compare encrypted documents with plaintext documents
- Get similarity scores without decrypting the original document
- Built with Concrete-ML and Transformers

## Installation

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
# Encrypt a document
python cli.py encrypt "Your text here" -o document.enc

# Compare encrypted document with plaintext
python cli.py compare document.enc "Text to compare"
```

## Requirements

- Python 3.8+
- 8GB+ RAM
- See requirements.txt for dependencies