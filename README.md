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

## Quick Start Options

### Option 1: Interactive Chat Interface (Easiest!)

The simplest way to use the system - just have a conversation:

```bash
cd ~/projects/fhe-icp
source venv/bin/activate
python fhe_interactive.py
```

This gives you a friendly menu-driven interface:
- No need to remember commands
- Visual document selection
- Clear explanations of similarity scores
- Perfect for non-technical users

Example interaction:
```
🔐 Welcome to FHE Document Comparison!
Your documents are encrypted for privacy using advanced cryptography.

==================================================
What would you like to do?

1. 📝 Add a document
2. 🔍 Compare two documents
3. 🔎 Search for similar documents
4. ⚡ Quick compare (without saving)
5. 📊 View statistics
6. 👋 Exit

Enter your choice (1-6): 1

📝 Let's add a new document!

You can type or paste your text below.
(For multiple lines, end with a line containing just 'END')

Artificial intelligence is transforming how we work
END

💾 Saving as: artificial_intelligence_is_115423
📄 Preview: Artificial intelligence is transforming how we work

✅ Document encrypted and saved!
ID: artificial_intelligence_is_115423
```

### Option 2: Web Browser Interface

Use a simple web interface in your browser:

```bash
cd ~/projects/fhe-icp
source venv/bin/activate
python fhe_simple_web.py
```

Then open http://localhost:8080 in your browser!

### Option 3: Command Line Interface (Mac Terminal)

Here's how to perform a complete document comparison test on macOS:

```bash
# 1. Open Terminal (Cmd+Space, type "Terminal")
cd ~/projects/fhe-icp  # Navigate to project directory

# 2. Activate the virtual environment
source venv/bin/activate

# 3. Generate FHE keys (first time only)
python fhe_cli.py keys generate
# You'll be prompted to create a master password. Remember it!
# This takes 30-60 seconds and generates ~50-100MB of key files

# 4. Create test documents by encrypting some sample texts
python fhe_cli.py encrypt "Machine learning transforms data into insights" --id doc1
python fhe_cli.py encrypt "Deep learning uses neural networks for AI tasks" --id doc2  
python fhe_cli.py encrypt "Cooking recipes for delicious pasta dishes" --id doc3

# 5. Compare two AI-related documents (should show high similarity)
python fhe_cli.py compare doc1 doc2
# Expected output: Similarity score: 0.7-0.9 (Similar)

# 6. Compare AI document with cooking document (should show low similarity)
python fhe_cli.py compare doc1 doc3
# Expected output: Similarity score: 0.1-0.4 (Not very similar)

# 7. Search for documents similar to a query
python fhe_cli.py search "artificial intelligence and neural networks" --top-k 3
# Should rank doc2 and doc1 higher than doc3

# 8. View statistics about your encrypted documents
python fhe_cli.py stats
```

### Complete Example Session

```bash
# Terminal session example
$ cd ~/projects/fhe-icp
$ source venv/bin/activate
(venv) $ python fhe_cli.py keys generate
Generating new FHE keys...
Create master password: ********
Confirm master password: ********
Generating FHE keys (this may take 30-60 seconds)...
✅ Keys generated successfully!
Key ID: fhe_key_20250718_120000

(venv) $ python fhe_cli.py encrypt "Quantum computing will revolutionize cryptography" --id quantum_doc
Encrypting document...
✅ Document encrypted successfully!
Document ID: quantum_doc
Size: 1024 bytes

(venv) $ python fhe_cli.py encrypt "Blockchain ensures data integrity and security" --id blockchain_doc
Encrypting document...
✅ Document encrypted successfully!
Document ID: blockchain_doc
Size: 1019 bytes

(venv) $ python fhe_cli.py compare quantum_doc blockchain_doc
Comparing documents...
  Document 1: quantum_doc
  Document 2: blockchain_doc

Similarity score: 0.6234
Interpretation: Somewhat similar

(venv) $ python fhe_cli.py stats
FHE Document System Statistics
==================================================

Keys:
  Total keys: 1
  Current key: fhe_key_20250718_120000

Storage:
  Total documents: 2
  Total size: 0.002 MB
  Average size: 1021 bytes
```

## Advanced Usage

### Batch Encryption

```bash
# Create a JSON file with multiple documents
echo '[
  {"text": "First document about AI", "id": "ai_1"},
  {"text": "Second document about ML", "id": "ml_1"},
  {"text": "Third document about data", "id": "data_1"}
]' > documents.json

# Encrypt all documents at once
python fhe_cli.py encrypt-batch documents.json
```

### Key Management

```bash
# List all keys
python fhe_cli.py keys list

# Rotate keys (keeps old keys for grace period)
python fhe_cli.py keys rotate --grace-days 7
```

### ICP Resource Estimation

```bash
# Estimate resources needed for ICP deployment
python fhe_cli.py estimate
```

## Project Structure

```
fhe-icp/
├── fhe_cli.py              # Main CLI interface
├── key_management.py       # Secure key handling
├── encrypted_storage.py    # Document storage
├── batch_operations.py     # Batch processing
├── bert_embeddings.py      # BERT integration
├── dimension_reduction.py  # PCA for embeddings
├── fhe_similarity.py       # FHE model
└── test_suite.py          # Comprehensive tests
```

## Requirements

- Python 3.8+
- macOS, Linux, or Windows
- 8GB+ RAM (16GB recommended)
- 5GB+ free disk space
- See requirements.txt for Python dependencies

## Security Notes

- Keys are encrypted with a master password
- Key files have restricted permissions (0600)
- Never share your master password or key files
- Documents remain encrypted at rest

## Troubleshooting

### "No module named 'cryptography'"
```bash
pip install cryptography msgpack
```

### "No compiled model found"
```bash
python fhe_cli.py keys generate
```

### Memory errors during key generation
Close other applications and ensure 8GB+ RAM is available.

## Development

Run tests:
```bash
python test_suite.py
```

Integration test:
```bash
python session4_integration_test.py
```

## License

MIT License - see LICENSE file for details

## Next Steps

- Session 5: Deploy to ICP canisters
- Add web interface
- Implement real-time document streams
- Multi-language support