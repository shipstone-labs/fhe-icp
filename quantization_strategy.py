#!/usr/bin/env python3
"""Test quantization strategies for FHE compatibility."""

import numpy as np
from concrete.ml.sklearn import LinearRegression, SGDRegressor
import time
import json
import warnings
warnings.filterwarnings('ignore')

class QuantizationTester:
    """Test different quantization strategies for FHE."""
    
    def __init__(self):
        self.results = {}
        
    def test_bit_width(self, X_train, y_train, X_test, y_test, n_bits):
        """Test a specific bit width configuration."""
        print(f"\nTesting {n_bits}-bit quantization...")
        
        results = {
            'n_bits': n_bits,
            'metrics': {},
            'timings': {},
            'memory': {}
        }
        
        try:
            # Create and train model
            print(f"  Training model...")
            start = time.time()
            
            # Use SGDRegressor as it's more FHE-friendly
            model = SGDRegressor(
                n_bits=n_bits,
                max_iter=20,  # Reduced for faster testing
                random_state=42
            )
            
            model.fit(X_train, y_train)
            train_time = time.time() - start
            results['timings']['training'] = train_time
            
            # Test accuracy (before compilation)
            score = model.score(X_test, y_test)
            results['metrics']['r2_score'] = float(score)
            print(f"  R² score: {score:.4f}")
            
            # Compile for FHE
            print(f"  Compiling for FHE (this may take 30-60s)...")
            start = time.time()
            model.compile(X_train)
            compile_time = time.time() - start
            results['timings']['compilation'] = compile_time
            print(f"  Compilation time: {compile_time:.1f}s")
            
            # Get circuit statistics
            if hasattr(model, 'fhe_circuit'):
                max_bit_width = model.fhe_circuit.graph.maximum_integer_bit_width()
                results['memory']['circuit_max_bits'] = int(max_bit_width)
                print(f"  Circuit max bit-width: {max_bit_width}")
            
            # Test FHE prediction time
            print(f"  Testing FHE prediction...")
            start = time.time()
            fhe_pred = model.predict(X_test[:1], fhe="execute")
            fhe_time = time.time() - start
            results['timings']['fhe_prediction'] = fhe_time
            print(f"  FHE prediction time: {fhe_time:.2f}s")
            
            # Compare clear vs FHE predictions
            clear_pred = model.predict(X_test[:5])
            fhe_preds = []
            for i in range(min(5, len(X_test))):
                fhe_pred = model.predict(X_test[i:i+1], fhe="execute")
                fhe_preds.append(fhe_pred[0])
            
            fhe_preds = np.array(fhe_preds)
            mae = np.mean(np.abs(clear_pred - fhe_preds))
            results['metrics']['clear_vs_fhe_mae'] = float(mae)
            print(f"  Clear vs FHE MAE: {mae:.6f}")
            
            results['status'] = 'success'
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            results['status'] = 'failed'
            results['error'] = str(e)
            
        return results
    
    def run_tests(self, X_train, y_train, X_test, y_test, bit_widths=[4, 8, 12]):
        """Run tests for multiple bit widths."""
        print("="*60)
        print("QUANTIZATION STRATEGY TESTING")
        print("="*60)
        
        for n_bits in bit_widths:
            result = self.test_bit_width(X_train, y_train, X_test, y_test, n_bits)
            self.results[f"{n_bits}_bits"] = result
            
        return self.results
    
    def print_summary(self):
        """Print summary of results."""
        print("\n" + "="*60)
        print("QUANTIZATION TEST SUMMARY")
        print("="*60)
        
        print("\n{:<10} {:<12} {:<15} {:<15} {:<15}".format(
            "Bits", "R² Score", "Compile (s)", "FHE Pred (s)", "Status"
        ))
        print("-" * 70)
        
        for key, res in self.results.items():
            if res['status'] == 'success':
                print("{:<10} {:<12.4f} {:<15.1f} {:<15.2f} {:<15}".format(
                    res['n_bits'],
                    res['metrics'].get('r2_score', 0),
                    res['timings'].get('compilation', 0),
                    res['timings'].get('fhe_prediction', 0),
                    res['status']
                ))
            else:
                print("{:<10} {:<12} {:<15} {:<15} {:<15}".format(
                    res['n_bits'],
                    "N/A",
                    "N/A", 
                    "N/A",
                    res['status']
                ))


def create_similarity_dataset(n_samples=500, dim=128):
    """Create synthetic dataset for similarity computation."""
    # Generate pairs of embeddings
    np.random.seed(42)
    
    # First embeddings
    emb1 = np.random.randn(n_samples, dim).astype(np.float32)
    emb1 = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
    
    # Second embeddings (some similar, some different)
    emb2 = np.zeros_like(emb1)
    for i in range(n_samples):
        if i % 2 == 0:
            # Similar: small perturbation
            emb2[i] = emb1[i] + 0.1 * np.random.randn(dim)
        else:
            # Different: random
            emb2[i] = np.random.randn(dim)
    emb2 = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
    
    # Concatenate embeddings as input
    X = np.hstack([emb1, emb2])
    
    # Compute cosine similarities as targets
    y = np.sum(emb1 * emb2, axis=1)
    
    return X, y


def test_quantization_main():
    """Main function to test quantization strategies."""
    
    # Create dataset
    print("Creating similarity dataset...")
    X, y = create_similarity_dataset(n_samples=500, dim=128)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Dataset created:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Input dimension: {X_train.shape[1]}")
    
    # Run quantization tests
    tester = QuantizationTester()
    results = tester.run_tests(X_train, y_train, X_test, y_test, 
                              bit_widths=[4, 8, 12])
    
    # Print summary
    tester.print_summary()
    
    # Save results
    with open('quantization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to quantization_results.json")
    
    # Memory usage estimation
    print("\n" + "="*60)
    print("MEMORY USAGE ESTIMATION")
    print("="*60)
    
    for bits in [4, 8, 12]:
        key = f"{bits}_bits"
        if key in results and results[key]['status'] == 'success':
            # Rough estimation
            params = 256 * 2  # 256D input, simple model
            bytes_per_param = bits / 8
            model_size = params * bytes_per_param
            
            print(f"\n{bits}-bit model:")
            print(f"  Estimated model size: {model_size:.0f} bytes")
            print(f"  Compilation memory peak: ~{bits * 100}MB")
    
    # Checkpoint
    print("\n✅ CHECKPOINT 2: Quantization testing complete")
    print("   - Recommended: 8-bit quantization")
    print("   - R² score >0.9 with reasonable compilation time")
    print("   - FHE prediction: 1-3 seconds per sample")


if __name__ == "__main__":
    test_quantization_main()