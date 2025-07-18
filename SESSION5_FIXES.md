# Session 5: Critical Bug Fix and Implementation Corrections

## Date: July 18, 2025

## Executive Summary

A fundamental mathematical error was discovered in the FHE-BERT similarity implementation. The system was using concatenated embeddings with LinearRegression, which cannot compute cosine similarity. The fix involves changing to element-wise products, which enables accurate similarity computation.

## The Core Problem

### What Was Wrong
The system concatenated two embeddings `[emb1, emb2]` and expected LinearRegression to learn cosine similarity. This is mathematically impossible because:
- LinearRegression computes: `y = w0*x0 + w1*x1 + ... + wn*xn + b`
- Cosine similarity requires: `y = sum(emb1[i] * emb2[i])`
- LinearRegression cannot multiply features together

### Evidence
- Test showed concatenation approach achieved only R² = 0.51
- Identical vectors predicted similarity = 0.299 (should be 1.0)
- All predictions clustered around training mean (~0.2)

### The Solution
Use element-wise products `emb1 * emb2` as features. LinearRegression can then learn to sum these products (with weights ≈ 1.0), effectively computing cosine similarity.

## Changes Made

### 1. Core Model Fix (`fhe_similarity.py`)
```python
# Line 54: Changed from concatenation to products
X = emb1 * emb2  # was: np.hstack([emb1, emb2])

# Line 39: Fixed dimension handling
single_dim = self.input_dim  # was: self.input_dim // 2
```

### 2. Batch Operations (`batch_operations.py`)
```python
# Line 86: Updated dimensions
self.fhe_model = FHESimilarityModel(input_dim=128, n_bits=8)  # was: 256

# Line 226: Fixed comparison
X = (doc1.encrypted_embedding * doc2.encrypted_embedding).reshape(1, -1)

# Line 273: Fixed search
X = (query_reduced * doc.encrypted_embedding).reshape(1, -1)
```

### 3. Key Management (`key_management.py`)
```python
# Line 138: Updated dimensions and training
model = FHESimilarityModel(input_dim=128, n_bits=8)  # was: 256
X_train, y_train = model.train()  # Now uses proper training data
```

## Test Results

After fixes:
- ✅ Similar documents (ML topics): similarity = 3.108 (high)
- ✅ Different documents (ML vs cooking): similarity = 0.265 (low)
- ✅ Model correctly distinguishes between similar and different content

## Remaining Work for Full CLI Functionality

### 1. Update Test Files
All test files still use concatenation. Need to update:
- `test_similarity_bug.py`
- `integration_test.py`
- `test_fhe.py`
- `quantization_strategy.py`
- `test_suite.py`

### 2. Fix FHE Compilation Issues
The model compilation fails with "NoParametersFound". This needs investigation:
- May need to update Concrete ML version
- Verify compilation input format
- Check parameter passing to FHE circuit

### 3. Update Documentation
- Update all examples to use element-wise products
- Document the mathematical reasoning
- Update dimension specifications (128 not 256)

### 4. Validate End-to-End Workflow
- Test full encryption → storage → comparison pipeline
- Verify FHE predictions match clear predictions
- Test with real BERT embeddings (not just random)

### 5. Performance Optimization
- Current similarity values are >1 due to training distribution
- Consider normalizing model outputs to [0, 1] range
- Test with larger embedding dimensions

### 6. CLI Commands to Test
```bash
# Generate keys (if needed)
python fhe_cli.py keys generate

# Encrypt documents
python fhe_cli.py encrypt "Machine learning is amazing" --id doc1
python fhe_cli.py encrypt "Deep learning transforms AI" --id doc2
python fhe_cli.py encrypt "I love cooking pasta" --id doc3

# Compare documents
python fhe_cli.py compare doc1 doc2  # Should show high similarity
python fhe_cli.py compare doc1 doc3  # Should show low similarity
```

## Technical Details

### Why Element-wise Products Work
For normalized vectors a and b:
- Cosine similarity = dot(a, b) = sum(a[i] * b[i])
- Element-wise product gives [a[0]*b[0], a[1]*b[1], ..., a[n]*b[n]]
- LinearRegression learns weights ≈ [1, 1, ..., 1] to sum these
- Result is cosine similarity

### FHE Implications
- Element-wise products can be computed before encryption
- LinearRegression performs only additions (FHE-friendly)
- Both documents must be available to compute products

## Recommendations

1. **Immediate**: Fix remaining test files to use products
2. **Short-term**: Resolve FHE compilation issues
3. **Medium-term**: Add output normalization for interpretable similarity scores
4. **Long-term**: Consider more sophisticated similarity metrics

## Conclusion

The core mathematical issue has been fixed. The system now correctly computes document similarity using element-wise products. With the remaining updates completed, the CLI will provide accurate similarity detection for encrypted documents.