#!/usr/bin/env python3
"""Debugging utilities for FHE development."""

import sys
import platform
import importlib
import subprocess
import psutil
import os

def check_environment():
    """Check system environment for FHE compatibility."""
    print("="*60)
    print("FHE ENVIRONMENT CHECK")
    print("="*60)
    
    # System info
    print("\n📊 SYSTEM INFORMATION:")
    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Architecture: {platform.machine()}")
    
    # Memory
    memory = psutil.virtual_memory()
    print(f"\n💾 MEMORY:")
    print(f"  Total: {memory.total / (1024**3):.1f} GB")
    print(f"  Available: {memory.available / (1024**3):.1f} GB")
    print(f"  Used: {memory.percent}%")
    
    if memory.available < 4 * (1024**3):
        print("  ⚠️  WARNING: Less than 4GB available. FHE compilation may fail.")
    
    # CPU
    print(f"\n🖥️  CPU:")
    print(f"  Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
    print(f"  Usage: {psutil.cpu_percent(interval=1)}%")
    
    # Check imports
    print("\n📦 PACKAGE VERSIONS:")
    packages = {
        'concrete.ml': 'concrete-ml',
        'transformers': 'transformers',
        'torch': 'torch',
        'sklearn': 'scikit-learn',
        'numpy': 'numpy'
    }
    
    for module, name in packages.items():
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✅ {name}: {version}")
        except ImportError:
            print(f"  ❌ {name}: NOT INSTALLED")
    
    # OpenSSL check (common issue)
    print("\n🔐 OPENSSL CHECK:")
    try:
        result = subprocess.run(['openssl', 'version'], 
                              capture_output=True, text=True)
        print(f"  {result.stdout.strip()}")
    except:
        print("  ⚠️  OpenSSL not found in PATH")
    
    # Common issues and solutions
    print("\n🔧 COMMON ISSUES AND SOLUTIONS:")
    print("-" * 60)
    
    issues = [
        {
            'error': 'MemoryError during compilation',
            'cause': 'Insufficient RAM for FHE compilation',
            'solution': '1. Reduce n_bits (try 4 or 6)\n         2. Use smaller input dimensions\n         3. Close other applications'
        },
        {
            'error': 'SSL certificate verify failed',
            'cause': 'Corporate firewall or outdated certificates',
            'solution': '1. Update certifi: pip install --upgrade certifi\n         2. Set REQUESTS_CA_BUNDLE environment variable'
        },
        {
            'error': 'Compilation takes forever',
            'cause': 'Normal for first compilation',
            'solution': '1. Be patient (60-120s is normal)\n         2. Check CPU usage (should be 100%)\n         3. Reduce model complexity'
        },
        {
            'error': 'ImportError: concrete.ml',
            'cause': 'Installation issues',
            'solution': '1. pip uninstall concrete-ml\n         2. pip install concrete-ml==1.7.0'
        },
        {
            'error': 'Killed (signal 9)',
            'cause': 'Out of memory (OOM killer)',
            'solution': '1. Monitor with: watch -n 1 free -h\n         2. Increase swap space\n         3. Use cloud instance with more RAM'
        }
    ]
    
    for issue in issues:
        print(f"\n❗ Error: {issue['error']}")
        print(f"   Cause: {issue['cause']}")
        print(f"   Fix:   {issue['solution']}")
    
    # Performance tips
    print("\n⚡ PERFORMANCE OPTIMIZATION TIPS:")
    print("-" * 60)
    
    tips = [
        "Use batch processing for multiple predictions",
        "Pre-compute and cache embeddings when possible",
        "Consider using GPU for BERT embeddings extraction",
        "Use smaller models (DistilBERT) for faster processing",
        "Implement progressive computation (start with low precision)",
        "Monitor memory with: htop or Activity Monitor",
        "Set CONCRETE_ML_VERBOSE=1 for detailed logs"
    ]
    
    for i, tip in enumerate(tips, 1):
        print(f"  {i}. {tip}")
    
    # Environment variables
    print("\n🌍 USEFUL ENVIRONMENT VARIABLES:")
    env_vars = {
        'CONCRETE_ML_VERBOSE': 'Enable detailed logging',
        'OMP_NUM_THREADS': 'Control parallelism',
        'REQUESTS_CA_BUNDLE': 'Custom SSL certificates',
        'TRANSFORMERS_CACHE': 'Model cache directory'
    }
    
    for var, desc in env_vars.items():
        current = os.environ.get(var, 'Not set')
        print(f"  {var}: {current}")
        print(f"    Purpose: {desc}")
    
    print("\n✅ Environment check complete!")


def memory_monitor(func):
    """Decorator to monitor memory usage of a function."""
    def wrapper(*args, **kwargs):
        import tracemalloc
        
        # Start tracing
        tracemalloc.start()
        
        # Get starting memory
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024
        
        # Run function
        result = func(*args, **kwargs)
        
        # Get memory stats
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        end_memory = process.memory_info().rss / 1024 / 1024
        
        print(f"\n📊 Memory usage for {func.__name__}:")
        print(f"  Start: {start_memory:.1f} MB")
        print(f"  End: {end_memory:.1f} MB")
        print(f"  Peak: {peak / 1024 / 1024:.1f} MB")
        print(f"  Allocated: {(end_memory - start_memory):.1f} MB")
        
        return result
    
    return wrapper


def test_memory_usage():
    """Test memory usage of key operations."""
    print("\n" + "="*60)
    print("MEMORY USAGE TESTING")
    print("="*60)
    
    @memory_monitor
    def test_embedding_extraction():
        from bert_embeddings import BertEmbedder
        embedder = BertEmbedder()
        texts = ["Test document"] * 10
        embeddings = embedder.get_embeddings_batch(texts)
        return embeddings
    
    @memory_monitor
    def test_fhe_compilation():
        from concrete.ml.sklearn import SGDRegressor
        import numpy as np
        
        X = np.random.randn(100, 256).astype(np.float32)
        y = np.random.randn(100)
        
        model = SGDRegressor(n_bits=8)
        model.fit(X, y)
        model.compile(X)
        
        return model
    
    # Run tests
    print("\n1. Testing BERT embedding extraction...")
    test_embedding_extraction()
    
    print("\n2. Testing FHE compilation...")
    test_fhe_compilation()


if __name__ == "__main__":
    # Run environment check
    check_environment()
    
    # Optionally run memory tests
    if len(sys.argv) > 1 and sys.argv[1] == "--memory":
        test_memory_usage()