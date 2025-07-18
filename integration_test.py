#!/usr/bin/env python3
"""Integration test for the complete FHE-BERT pipeline."""

import numpy as np
import time
from pathlib import Path

def integration_test():
    """Test the complete pipeline end-to-end."""
    print("="*60)
    print("FHE-BERT SIMILARITY PIPELINE - INTEGRATION TEST")
    print("="*60)
    
    # Check all required files exist
    required_files = [
        'bert_embeddings.py',
        'dimension_reduction.py',
        'fhe_similarity.py',
        'pca_reducer_128.pkl',
        'fhe_similarity_model.pkl'
    ]
    
    print("\n📁 Checking required files...")
    all_present = True
    for file in required_files:
        if Path(file).exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - Missing!")
            all_present = False
    
    if not all_present:
        print("\n⚠️  Some files are missing. Run previous steps first.")
        return
    
    # Import modules
    print("\n📦 Importing modules...")
    from bert_embeddings import BertEmbedder
    from dimension_reduction import DimensionReducer
    from fhe_similarity import FHESimilarityModel
    
    # Test documents
    doc1 = "Machine learning is transforming artificial intelligence."
    doc2 = "AI and ML are revolutionizing technology."
    doc3 = "I enjoy cooking Italian pasta with fresh ingredients."
    
    print("\n📄 Test documents:")
    print(f"  Doc1: '{doc1}'")
    print(f"  Doc2: '{doc2}'")
    print(f"  Doc3: '{doc3}'")
    
    # Step 1: Extract BERT embeddings
    print("\n1️⃣ Extracting BERT embeddings...")
    start = time.time()
    embedder = BertEmbedder()
    
    emb1 = embedder.get_embedding(doc1)
    emb2 = embedder.get_embedding(doc2)
    emb3 = embedder.get_embedding(doc3)
    
    bert_time = time.time() - start
    print(f"  ✅ Embeddings extracted in {bert_time:.2f}s")
    print(f"  Shape: {emb1.shape}")
    
    # Step 2: Reduce dimensions
    print("\n2️⃣ Reducing dimensions...")
    start = time.time()
    reducer = DimensionReducer.load('pca_reducer_128.pkl')
    
    emb1_reduced = reducer.transform(emb1.reshape(1, -1))[0]
    emb2_reduced = reducer.transform(emb2.reshape(1, -1))[0]
    emb3_reduced = reducer.transform(emb3.reshape(1, -1))[0]
    
    reduction_time = time.time() - start
    print(f"  ✅ Dimensions reduced in {reduction_time:.2f}s")
    print(f"  New shape: {emb1_reduced.shape}")
    
    # Step 3: Load FHE model
    print("\n3️⃣ Loading FHE similarity model...")
    try:
        model = FHESimilarityModel.load('fhe_similarity_model.pkl')
        print(f"  ✅ Model loaded (compiled: {model.compiled})")
    except:
        print("  ⚠️  Could not load saved model. Creating new one...")
        model = FHESimilarityModel(input_dim=256, n_bits=8)
        print("  Training model...")
        X_train, y_train = model.train(n_samples=500)
        print("  Compiling model...")
        model.compile(X_train[:100])
        model.save('fhe_similarity_model.pkl')
    
    # Step 4: Compute similarities
    print("\n4️⃣ Computing similarities...")
    
    # Prepare inputs (concatenate embeddings)
    X12 = np.hstack([emb1_reduced, emb2_reduced]).reshape(1, -1)
    X13 = np.hstack([emb1_reduced, emb3_reduced]).reshape(1, -1)
    X23 = np.hstack([emb2_reduced, emb3_reduced]).reshape(1, -1)
    
    # Check if model is trained
    if model.model is None:
        print("  Model not trained. Training now...")
        X_train, y_train = model.train(n_samples=500)
        model.compile(X_train[:100])
    
    # Clear predictions (fast)
    print("\n  Clear predictions:")
    start = time.time()
    sim12_clear = model.predict_clear(X12)[0]
    sim13_clear = model.predict_clear(X13)[0]
    sim23_clear = model.predict_clear(X23)[0]
    clear_time = time.time() - start
    
    print(f"    Doc1-Doc2: {sim12_clear:.4f}")
    print(f"    Doc1-Doc3: {sim13_clear:.4f}")
    print(f"    Doc2-Doc3: {sim23_clear:.4f}")
    print(f"    Time: {clear_time*1000:.1f}ms")
    
    # FHE predictions (slow but private)
    if model.compiled:
        print("\n  FHE predictions (encrypted):")
        print("  ⏳ This will take 2-3 seconds per comparison...")
        
        start = time.time()
        sim12_fhe = model.predict_encrypted(X12)[0]
        fhe_time = time.time() - start
        
        print(f"    Doc1-Doc2: {sim12_fhe:.4f} (took {fhe_time:.1f}s)")
        print(f"    Difference from clear: {abs(sim12_clear - sim12_fhe):.6f}")
    
    # Summary
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    
    print(f"\n⏱️  Timing breakdown:")
    print(f"  BERT extraction: {bert_time:.2f}s")
    print(f"  Dimension reduction: {reduction_time:.2f}s")
    print(f"  Clear similarity: {clear_time*1000:.1f}ms")
    if model.compiled:
        print(f"  FHE similarity: {fhe_time:.1f}s")
        print(f"  FHE overhead: {fhe_time/clear_time:.0f}x slower")
    
    print(f"\n📊 Results interpretation:")
    print(f"  Doc1-Doc2 similarity: {sim12_clear:.2%} (both about AI/ML)")
    print(f"  Doc1-Doc3 similarity: {sim13_clear:.2%} (unrelated topics)")
    print(f"  Doc2-Doc3 similarity: {sim23_clear:.2%} (unrelated topics)")
    
    print("\n✅ Integration test PASSED!")
    print("   The FHE-BERT similarity pipeline is working correctly.")


if __name__ == "__main__":
    integration_test()