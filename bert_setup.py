#!/usr/bin/env python3
"""Initial BERT model setup and testing."""

import torch
from transformers import AutoTokenizer, AutoModel
import time

print("Setting up BERT model...\n")

# Step 1: Load tokenizer
print("1. Loading tokenizer...")
start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
print(f"   Tokenizer loaded in {time.time() - start_time:.2f}s")
print(f"   Vocabulary size: {tokenizer.vocab_size}")

# Step 2: Load model
print("\n2. Loading BERT model (first time will download ~440MB)...")
print("   This may take several minutes on first run...")
start_time = time.time()
try:
    model = AutoModel.from_pretrained('bert-base-uncased')
    model.eval()  # Set to evaluation mode
    print(f"   Model loaded in {time.time() - start_time:.2f}s")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"   Error loading model: {e}")
    print("   Try running again or check your internet connection")
    exit(1)

# Step 3: Test tokenization
print("\n3. Testing tokenization...")
test_text = "The quick brown fox jumps over the lazy dog."
tokens = tokenizer.tokenize(test_text)
print(f"   Original text: '{test_text}'")
print(f"   Tokens: {tokens}")
print(f"   Number of tokens: {len(tokens)}")

# Step 4: Convert to IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"\n4. Token IDs: {token_ids}")

# Step 5: Test full encoding
print("\n5. Full encoding with special tokens...")
encoded = tokenizer(test_text, return_tensors="pt")
print(f"   Input IDs shape: {encoded['input_ids'].shape}")
print(f"   Attention mask shape: {encoded['attention_mask'].shape}")
print(f"   Actual tokens: {tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])}")

# Step 6: Quick inference test
print("\n6. Testing model inference...")
with torch.no_grad():
    outputs = model(**encoded)
    
print(f"   Last hidden state shape: {outputs.last_hidden_state.shape}")
print(f"   Output dimensions: [batch_size=1, sequence_length={outputs.last_hidden_state.shape[1]}, hidden_size=768]")

print("\n✅ BERT setup complete!")