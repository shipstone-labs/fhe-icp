# FHE-BERT Document Similarity on ICP

A production-ready CLI tool for encrypting and comparing documents using Fully Homomorphic Encryption (FHE) with BERT embeddings, designed for deployment on Internet Computer Protocol (ICP) canisters.

## Features

- 🔐 Secure document encryption using FHE
- 🔍 Compare encrypted documents without decryption
- 📊 Similarity search across encrypted document collections
- 🚀 Optimized for ICP canister deployment
- 💾 Efficient storage with compression
- 🔑 Secure key management with master password

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fhe-icp.git
cd fhe-icp

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Clean Version (Recommended)

Simple commands with minimal output:

```bash
cd ~/projects/fhe-icp
source venv/bin/activate

# First time only - initialize
python fhe_clean.py init
# Output: ✓ Ready

# Add documents
python fhe_clean.py add "Machine learning transforms data into insights"
# Output: ✓ doc_1

python fhe_clean.py add "Deep learning uses neural networks for AI"
# Output: ✓ doc_2

python fhe_clean.py add "Italian pasta recipes for dinner"
# Output: ✓ doc_3

# Compare documents
python fhe_clean.py compare doc_1 doc_2
# Output: 75% - similar

python fhe_clean.py compare doc_1 doc_3
# Output: 23% - different

# Search documents
python fhe_clean.py search "artificial intelligence"
# Output: doc_2: 82%
#         doc_1: 71%
```

### Technical Version

For detailed logging and full control:

```bash
# Generate FHE keys (first time only)
python fhe_cli.py keys generate

# Encrypt documents
python fhe_cli.py encrypt "Your text here" --id doc1

# Compare documents
python fhe_cli.py compare doc1 doc2

# View statistics
python fhe_cli.py stats
```

## Project Structure

```
fhe-icp/
├── fhe_clean.py           # Clean CLI (minimal output)
├── fhe_cli.py             # Technical CLI (detailed output)
├── key_management.py      # Secure key handling
├── encrypted_storage.py   # Document storage
├── batch_operations.py    # Batch processing
├── bert_embeddings.py     # BERT integration
├── dimension_reduction.py # PCA for embeddings
├── fhe_similarity.py      # FHE model
└── test_suite.py         # Comprehensive tests
```

## Requirements

- Python 3.8+
- macOS, Linux, or Windows
- 8GB+ RAM (16GB recommended)
- 5GB+ free disk space

## Security Notes

- Keys are encrypted with a master password
- Key files have restricted permissions (0600)
- Never share your master password or key files
- Documents remain encrypted at rest

## License

MIT License - see LICENSE file for details