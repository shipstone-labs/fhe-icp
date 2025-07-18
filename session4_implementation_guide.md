# Session 4 Implementation Guide

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

**Implementation Notes**:
- Use the existing `fhe_similarity.py` model to generate keys
- Store keys encrypted with master password
- File permissions must be 0600
- Keys will be 50-100MB each

### 2. `encrypted_storage.py`
**Purpose**: Standardized encrypted document storage
**Key Classes**:
- `EncryptedDocument`: Dataclass for document format
- `EncryptedDocumentStore`: Manage document storage

**Implementation Notes**:
- Use MessagePack + gzip for best compression
- Create JSON index for fast lookups
- Include validation methods

### 3. `batch_operations.py`
**Purpose**: Efficient batch processing
**Key Classes**:
- `BatchConfig`: Configuration dataclass
- `BatchProcessor`: Handle batch encryption/comparison

**Implementation Notes**:
- Use existing embedder and reducer
- Process in configurable batches
- Add memory monitoring with psutil
- Include progress bars with tqdm

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

## Implementation Order

1. **Start with Key Management** (most complex)
   - Test with mock FHE model first
   - Then integrate with real model

2. **Implement Storage Format**
   - Start with simple pickle format
   - Add compression later
   - Test thoroughly

3. **Add Batch Operations**
   - Begin with sequential processing
   - Add parallelization
   - Monitor memory usage

4. **Build CLI Interface**
   - Start with basic commands
   - Add progress indicators
   - Include error handling

5. **Complete Test Suite**
   - Write tests as you go
   - Aim for 90% coverage
   - Include performance tests

## Integration Points

The new components need to work with existing code:
- `bert_embeddings.BertEmbedder` - For text embeddings
- `dimension_reduction.DimensionReducer` - For PCA reduction
- `fhe_similarity.FHESimilarityModel` - For FHE operations

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

## Common Integration Issues

1. **Import Errors**: Ensure all modules are in the same directory
2. **Model Loading**: The FHE model needs to be compiled before use
3. **Memory Issues**: Monitor usage, especially during key generation
4. **File Permissions**: Keys must have restricted permissions

## Next Steps After Implementation

1. Run integration tests
2. Benchmark performance
3. Document any deviations from spec
4. Prepare for Session 5 (ICP deployment)

---

Remember: The session 4 documents contain complete implementations. Use them as reference but adapt as needed for your specific environment.
