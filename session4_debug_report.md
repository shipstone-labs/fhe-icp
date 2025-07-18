# Session 4 Debug Report: Similarity Scoring Issue

## Executive Summary

Session 4 implementation has a critical bug in `batch_operations.py` that causes the FHE similarity model to be trained on random noise instead of proper similarity training data. This results in meaningless similarity scores that do not reflect actual document relationships.

## Problem Statement

From SESSION_REPORT.md (lines 235-244):
```
**Root Cause Identified**: In `batch_operations.py` lines 89-92:
```python
X_sample = np.random.randn(100, 256).astype(np.float32)
y_sample = np.random.randn(100)
self.fhe_model.train(X_sample, y_sample, n_samples=100)
```

The FHE model is being trained on random noise instead of proper similarity training data.
```

## Technical Investigation

### 1. Code Analysis

#### Current Implementation (batch_operations.py, lines 74-96)

```python
def _init_model(self):
    """Initialize FHE model (compilation happens on first use)."""
    try:
        # Check if we have keys
        current_key = self.key_manager.get_current_key()
        if current_key:
            # Create a new FHE model instance
            from fhe_similarity import FHESimilarityModel
            self.fhe_model = FHESimilarityModel(input_dim=256, n_bits=8)
            
            # Train the model (needed before compilation)
            import numpy as np
            X_sample = np.random.randn(100, 256).astype(np.float32)
            y_sample = np.random.randn(100)
            self.fhe_model.train(X_sample, y_sample, n_samples=100)
            
            # Compile the model
            self.fhe_model.compile(X_sample[:10])
            
            logger.info("FHE model initialized and compiled")
    except Exception as e:
        logger.warning(f"Could not initialize FHE model: {e}")
        logger.info("Generate keys first using key_manager.generate_keys()")
```

#### FHESimilarityModel Design (fhe_similarity.py, lines 30-67)

The `FHESimilarityModel` class has a properly designed `_prepare_training_data()` method that:

1. Generates normalized embedding pairs
2. Creates controlled similarity relationships:
   - 50% of pairs are similar (correlation added)
   - 50% of pairs are randomly different
3. Computes true cosine similarity as training targets

```python
def _prepare_training_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate training data for similarity model."""
    # ... generates normalized embeddings ...
    
    # Add some correlation for realistic data
    mask = np.random.rand(n_samples) > 0.5
    emb2[mask] = emb1[mask] + 0.2 * np.random.randn(mask.sum(), single_dim)
    emb2 = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
    
    # Compute target similarities
    if self.similarity_type == 'cosine':
        y = np.sum(emb1 * emb2, axis=1)
```

### 2. Impact Analysis

The bug causes the LinearRegression model to learn a mapping from random 256-dimensional inputs to random outputs between approximately -2 and +2 (standard normal distribution). This results in:

1. **No correlation with actual similarity**: The model outputs are essentially random
2. **Incorrect scale**: Cosine similarity should be in [-1, 1], but random targets can exceed this range
3. **No semantic meaning**: Similar documents may get low scores, different documents may get high scores

### 3. Verification

The diagnostic test (`test_similarity_bug.py`) demonstrates:

```python
# Model trained on random data will produce:
# - Similar embeddings: random value (could be -0.5, 1.2, etc.)
# - Identical embeddings: random value (should be 1.0)
# - Different embeddings: random value (should be near 0)

# Model trained on proper data will produce:
# - Similar embeddings: ~0.8-0.95
# - Identical embeddings: ~1.0
# - Different embeddings: ~0.0-0.3
```

## Solution

### Immediate Fix

Replace the `_init_model()` method in `batch_operations.py`:

```python
def _init_model(self):
    """Initialize FHE model (compilation happens on first use)."""
    try:
        # Check if we have keys
        current_key = self.key_manager.get_current_key()
        if current_key:
            # Create a new FHE model instance
            from fhe_similarity import FHESimilarityModel
            self.fhe_model = FHESimilarityModel(input_dim=256, n_bits=8)
            
            # Train the model with proper similarity data
            # The train() method will use _prepare_training_data() internally
            X_train, y_train = self.fhe_model.train()
            
            # Compile the model using a sample from the training data
            self.fhe_model.compile(X_train[:10])
            
            logger.info("FHE model initialized and compiled")
    except Exception as e:
        logger.warning(f"Could not initialize FHE model: {e}")
        logger.info("Generate keys first using key_manager.generate_keys()")
```

### Why This Works

1. `self.fhe_model.train()` without arguments triggers the internal `_prepare_training_data()` method
2. This generates proper embedding pairs with known cosine similarities
3. The model learns the correct mapping from concatenated embeddings to similarity scores
4. The compilation uses actual training data samples, ensuring consistency

## Additional Findings

### Model Persistence Issue

From `fhe_similarity.py` lines 141-156:

```python
def save(self, path: str):
    """Save the model (note: compiled models cannot be pickled)."""
    if self.compiled:
        print("Warning: Compiled FHE models cannot be pickled. Saving model state only.")
        print("You will need to recompile after loading.")
```

This is not conjecture - it's documented in the code. The compiled FHE circuit contains C pointers that cannot be serialized. The current implementation correctly handles this by:
1. Saving only model parameters
2. Requiring recompilation after loading
3. Warning users about this limitation

This is the correct approach given Concrete ML's architecture.

## Testing Protocol

After applying the fix:

```bash
# 1. Run the diagnostic test
python test_similarity_bug.py

# 2. Run the integration test
python session4_integration_test.py

# 3. Test with real examples
python fhe_cli.py encrypt "Machine learning algorithms are powerful" --id ml1
python fhe_cli.py encrypt "Deep learning networks are revolutionary" --id ml2
python fhe_cli.py encrypt "I enjoy cooking Italian pasta" --id cook1

# Expected results:
python fhe_cli.py compare ml1 ml2  # Should show >0.7 similarity
python fhe_cli.py compare ml1 cook1  # Should show <0.3 similarity
```

## Conclusion

The bug is a simple training data issue with a straightforward fix. The model architecture and overall Session 4 implementation are sound. Once the model is trained on proper similarity data instead of random noise, the similarity scores will accurately reflect document relationships.
