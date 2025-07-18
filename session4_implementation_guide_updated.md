# Session 4 Implementation Guide (UPDATED)

## Updates Based on Claude Code Review

✅ **Key Serialization**: Use `pickle.dump(model.model, f)` for Concrete ML models  
✅ **Storage Format**: pickle+gzip for FHE objects, MessagePack optional for metadata  
✅ **Compilation**: Add `show_progress=True` parameter  
✅ **Memory Management**: psutil approach confirmed correct  

## Current Status
Sessions 1-3 are complete. The following components are ready:
- ✅ BERT embeddings (`bert_embeddings.py`)
- ✅ Dimension reduction (`dimension_reduction.py`)
- ✅ FHE similarity model (`fhe_similarity.py`)
- ✅ Various utilities and tests

## Files to Create for Session 4

### 1. `key_management.py`
**Purpose**: Secure FHE key generation and management
**Key Classes**:
- `FHEKeyManager`: Handle key generation, storage, rotation

**Implementation Notes (UPDATED)**:
- Use the existing `fhe_similarity.py` model to generate keys
- **Serialize with**: `pickle.dump(model.model, file)`
- Encrypt the pickled model with master password
- File permissions must be 0600
- Models will be 50-100MB each

### 2. `encrypted_storage.py`
**Purpose**: Standardized encrypted document storage
**Key Classes**:
- `EncryptedDocument`: Dataclass for document format
- `EncryptedDocumentStore`: Manage document storage

**Implementation Notes (UPDATED)**:
- **FHE objects**: Use pickle + gzip
- **Metadata**: Can use MessagePack or JSON
- **Document index**: Use JSON for readability
- Include validation methods

### 3. `batch_operations.py`
**Purpose**: Efficient batch processing
**Key Classes**:
- `BatchConfig`: Configuration dataclass
- `BatchProcessor`: Handle batch encryption/comparison

**Implementation Notes**:
- Use existing embedder and reducer
- Process in configurable batches
- Add memory monitoring with psutil (confirmed correct)
- Include progress bars with tqdm
- **Add** `show_progress=True` when compiling models

### 4. `fhe_cli.py`
**Purpose**: Production CLI interface
**Key Classes**:
- `FHEDocumentCLI`: Main CLI class

**Commands to implement**:
- `keys generate/list/rotate`
- `encrypt` (single document)
- `encrypt-batch` (multiple documents)
- `compare` (document vs query)
- `search` (search all documents)
- `stats` (system statistics)
- `validate` (check document integrity)

### 5. `test_suite.py`
**Purpose**: Comprehensive testing
**Test Classes**:
- `TestKeyManagement`
- `TestEncryptedStorage`
- `TestBatchOperations`
- `TestCLI`
- `TestSecurity`
- `TestPerformance`

**Additional Tests (UPDATED)**:
- Test pickle serialization/deserialization
- Verify progress bars work
- Test large model files (50-100MB)

## Updated Implementation Order

1. **Update FHE Model First**
   ```python
   # In fhe_similarity.py, update compile method:
   def compile(self, X_sample, show_progress=True):
       self.model.compile(X_sample, show_progress=show_progress)
   ```

2. **Start with Key Management**
   - Use pickle for model serialization
   - Test with real FHE model
   - Expect 50-100MB files

3. **Implement Storage Format**
   - Use pickle+gzip for FHE objects
   - Optional MessagePack for metadata
   - Test file sizes and compression

4. **Add Batch Operations**
   - No changes to memory monitoring
   - Add progress feedback during compilation
   - Test with multiple models

5. **Build CLI Interface**
   - Inform users about large file sizes
   - Show progress during long operations
   - Include security warnings about pickle

6. **Complete Test Suite**
   - Add pickle-specific tests
   - Test progress bar functionality
   - Verify file formats

## Integration Points

The new components need to work with existing code:
- `bert_embeddings.BertEmbedder` - For text embeddings
- `dimension_reduction.DimensionReducer` - For PCA reduction
- `fhe_similarity.FHESimilarityModel` - For FHE operations
- **Update**: Ensure FHE model supports `show_progress` parameter

## Testing Commands

After implementation, test with:
```bash
# Test individual components
python key_management.py
python encrypted_storage.py
python batch_operations.py

# Run full test suite
python test_suite.py

# Test CLI
python fhe_cli.py keys generate
python fhe_cli.py encrypt "Test document" --id test1
python fhe_cli.py stats
```

## Common Integration Issues (UPDATED)

1. **Pickle Security**: Only load trusted files
2. **Large Files**: 50-100MB models are normal
3. **Progress Bars**: Require updated Concrete ML
4. **Serialization**: Must use pickle for FHE models

## Key Code Snippets

### Saving FHE Model:
```python
# Correct way (Concrete ML standard)
with open('model.pkl', 'wb') as f:
    pickle.dump(model.model, f)

# With compression
with open('model.pkl.gz', 'wb') as f:
    f.write(gzip.compress(pickle.dumps(model.model)))
```

### Loading FHE Model:
```python
# Load pickled model
with open('model.pkl.gz', 'rb') as f:
    model_data = gzip.decompress(f.read())
    model = pickle.loads(model_data)
```

### Compilation with Progress:
```python
# Show progress during compilation
model.compile(X_sample, show_progress=True)
```

## Next Steps After Implementation

1. Run integration tests with updated serialization
2. Verify file sizes match expectations (50-100MB)
3. Test progress bars during compilation
4. Document any additional changes needed
5. Prepare for Session 5 (ICP deployment)

---

Remember: The core Session 4 architecture remains sound. These updates align the implementation with Concrete ML's conventions while maintaining all planned features.
