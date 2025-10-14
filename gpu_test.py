#!/usr/bin/env python3
"""
GPU-Accelerated Toolchain Test Script
Test your RTX 5070 Ti GPU setup with various ML frameworks
"""

import sys
import time
import numpy as np

def test_pytorch_gpu():
    """Test PyTorch GPU functionality"""
    print("=== PyTorch GPU Test ===")
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ cuDNN version: {torch.backends.cudnn.version()}")
            print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"✓ GPU {i}: {props.name}")
                print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
                print(f"  Compute capability: {props.major}.{props.minor}")
            
            # Test GPU operations
            device = torch.device('cuda:0')
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            
            start_time = time.time()
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time
            
            print(f"✓ GPU matrix multiplication (1000x1000): {gpu_time:.4f} seconds")
            print(f"✓ GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
            
        else:
            print("❌ CUDA not available")
            return False
            
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False
    
    return True

def test_cupy_gpu():
    """Test CuPy GPU functionality"""
    print("\n=== CuPy GPU Test ===")
    try:
        import cupy as cp
        print(f"✓ CuPy version: {cp.__version__}")
        
        # Test basic operations
        x_gpu = cp.array([1, 2, 3, 4, 5])
        y_gpu = cp.array([6, 7, 8, 9, 10])
        
        start_time = time.time()
        z_gpu = cp.dot(x_gpu, y_gpu)
        cp.cuda.Stream.null.synchronize()
        gpu_time = time.time() - start_time
        
        print(f"✓ GPU dot product: {z_gpu} (time: {gpu_time:.6f}s)")
        
        # Test larger computation
        size = 5000
        a_gpu = cp.random.rand(size, size)
        b_gpu = cp.random.rand(size, size)
        
        start_time = time.time()
        c_gpu = cp.matmul(a_gpu, b_gpu)
        cp.cuda.Stream.null.synchronize()
        gpu_time = time.time() - start_time
        
        print(f"✓ GPU matrix multiplication ({size}x{size}): {gpu_time:.4f} seconds")
        
        # Memory info
        mempool = cp.get_default_memory_pool()
        print(f"✓ GPU memory used: {mempool.used_bytes() / 1024**2:.1f} MB")
        
    except Exception as e:
        print(f"❌ CuPy test failed: {e}")
        return False
    
    return True

def test_ml_libraries():
    """Test ML libraries for GPU support"""
    print("\n=== ML Libraries Test ===")
    
    # XGBoost GPU test
    try:
        import xgboost as xgb
        print(f"✓ XGBoost version: {xgb.__version__}")
        
        # Create sample data
        X = np.random.rand(1000, 10)
        y = np.random.rand(1000)
        
        # Test GPU training
        dtrain = xgb.DMatrix(X, label=y)
        params = {
            'tree_method': 'gpu_hist',
            'gpu_id': 0,
            'objective': 'reg:squarederror'
        }
        
        start_time = time.time()
        model = xgb.train(params, dtrain, num_boost_round=10, verbose_eval=False)
        gpu_time = time.time() - start_time
        
        print(f"✓ XGBoost GPU training: {gpu_time:.4f} seconds")
        
    except Exception as e:
        print(f"❌ XGBoost GPU test failed: {e}")
    
    # LightGBM GPU test
    try:
        import lightgbm as lgb
        print(f"✓ LightGBM version: {lgb.__version__}")
        
        params = {
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'objective': 'regression'
        }
        
        train_data = lgb.Dataset(X, label=y)
        start_time = time.time()
        model = lgb.train(params, train_data, num_boost_round=10, verbose_eval=False)
        gpu_time = time.time() - start_time
        
        print(f"✓ LightGBM GPU training: {gpu_time:.4f} seconds")
        
    except Exception as e:
        print(f"❌ LightGBM GPU test failed: {e}")
    
    return True

def test_transformers_gpu():
    """Test Transformers library with GPU"""
    print("\n=== Transformers GPU Test ===")
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        
        if not torch.cuda.is_available():
            print("❌ CUDA not available for Transformers test")
            return False
        
        print("✓ Testing Transformers with GPU...")
        
        # Load a small model
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Move to GPU
        device = torch.device('cuda:0')
        model = model.to(device)
        
        # Test inference
        text = "This is a test sentence for GPU acceleration."
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        print(f"✓ Transformers GPU inference: {gpu_time:.4f} seconds")
        
    except Exception as e:
        print(f"❌ Transformers GPU test failed: {e}")
        return False
    
    return True

def main():
    """Run all GPU tests"""
    print("🚀 GPU-Accelerated Toolchain Test for RTX 5070 Ti")
    print("=" * 60)
    
    tests = [
        test_pytorch_gpu,
        test_cupy_gpu,
        test_ml_libraries,
        test_transformers_gpu
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All GPU tests passed! Your RTX 5070 Ti is ready for ML workloads!")
    else:
        print("⚠️  Some tests failed. Check GPU drivers and CUDA installation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)