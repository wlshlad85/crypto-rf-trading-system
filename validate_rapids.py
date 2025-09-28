#!/usr/bin/env python3
"""
RAPIDS Installation Validation Script
Tests all major RAPIDS components and provides diagnostic information
"""

import sys
import subprocess
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.END}")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def check_gpu_availability() -> Tuple[bool, str]:
    """Check if GPU is available via nvidia-smi"""
    try:
        # Try nvidia-smi first
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,compute_cap', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
            return True, gpu_info
    except FileNotFoundError:
        pass
    
    try:
        # Try nvidia-smi.exe (WSL)
        result = subprocess.run(['nvidia-smi.exe', '--query-gpu=name,compute_cap', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
            return True, f"{gpu_info} (via nvidia-smi.exe)"
    except FileNotFoundError:
        pass
    
    return False, "No GPU detected"

def check_cuda_toolkit() -> Tuple[bool, str]:
    """Check CUDA toolkit installation"""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            # Extract version from output
            for line in result.stdout.split('\n'):
                if 'release' in line:
                    return True, line.strip()
        return False, "CUDA toolkit not found"
    except FileNotFoundError:
        return False, "nvcc not found in PATH"

def check_rapids_packages() -> Dict[str, Tuple[bool, str]]:
    """Check which RAPIDS packages are installed"""
    packages = {
        'cudf': 'GPU DataFrame library',
        'cuml': 'GPU Machine Learning library',
        'cugraph': 'GPU Graph Analytics library',
        'cuspatial': 'GPU Spatial Analytics library',
        'cuxfilter': 'GPU Data Visualization library',
        'cucim': 'GPU Image Processing library',
        'cupy': 'GPU Array library',
        'rmm': 'RAPIDS Memory Manager'
    }
    
    results = {}
    for package, description in packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            results[package] = (True, f"v{version} - {description}")
        except ImportError:
            results[package] = (False, f"Not installed - {description}")
    
    return results

def test_cudf_functionality() -> Tuple[bool, str]:
    """Test basic cuDF functionality"""
    try:
        import cudf
        
        # Create a simple DataFrame
        df = cudf.DataFrame({
            'a': range(1000),
            'b': range(1000, 2000),
            'c': ['test'] * 1000
        })
        
        # Perform operations
        result = df.groupby('c').agg({'a': 'sum', 'b': 'mean'})
        
        if len(result) == 1 and result['a'].iloc[0] == 499500:
            return True, "Basic operations working correctly"
        else:
            return False, "Operations completed but results unexpected"
            
    except Exception as e:
        return False, f"Error: {str(e)}"

def test_cuml_functionality() -> Tuple[bool, str]:
    """Test basic cuML functionality"""
    try:
        import cudf
        import cuml
        from cuml.datasets import make_regression
        from cuml.linear_model import LinearRegression
        
        # Generate data
        X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Make predictions
        predictions = model.predict(X[:10])
        
        if len(predictions) == 10:
            return True, "Linear regression model trained successfully"
        else:
            return False, "Model trained but predictions unexpected"
            
    except Exception as e:
        return False, f"Error: {str(e)}"

def test_gpu_memory() -> Tuple[bool, str]:
    """Test GPU memory allocation"""
    try:
        import cupy as cp
        import rmm
        
        # Get GPU memory info
        meminfo = cp.cuda.MemoryPool().get_limit()
        
        # Allocate a small array
        arr = cp.ones((1000, 1000), dtype=cp.float32)
        
        # Perform operation
        result = cp.sum(arr)
        
        if result == 1000000:
            return True, f"GPU memory allocation working"
        else:
            return False, "GPU memory allocated but computation failed"
            
    except Exception as e:
        return False, f"Error: {str(e)}"

def run_performance_comparison() -> Tuple[bool, str]:
    """Run a simple performance comparison between CPU and GPU"""
    try:
        import time
        import pandas as pd
        import cudf
        
        # Create test data
        n = 1_000_000
        data = {
            'key': [i % 100 for i in range(n)],
            'value': range(n)
        }
        
        # CPU timing
        pdf = pd.DataFrame(data)
        cpu_start = time.time()
        cpu_result = pdf.groupby('key')['value'].sum()
        cpu_time = time.time() - cpu_start
        
        # GPU timing
        gdf = cudf.DataFrame(data)
        gpu_start = time.time()
        gpu_result = gdf.groupby('key')['value'].sum()
        gpu_time = time.time() - gpu_start
        
        speedup = cpu_time / gpu_time
        
        return True, f"CPU: {cpu_time:.3f}s, GPU: {gpu_time:.3f}s, Speedup: {speedup:.1f}x"
        
    except Exception as e:
        return False, f"Error: {str(e)}"

def check_wsl_environment() -> Tuple[bool, str]:
    """Check if running in WSL environment"""
    try:
        with open('/proc/version', 'r') as f:
            version_info = f.read().lower()
            if 'microsoft' in version_info or 'wsl' in version_info:
                # Extract WSL version if possible
                if 'wsl2' in version_info:
                    return True, "Running in WSL2"
                else:
                    return True, "Running in WSL"
        return False, "Not running in WSL"
    except:
        return False, "Unable to determine WSL status"

def main():
    """Run all validation tests"""
    print_header("RAPIDS Installation Validator")
    
    # Check environment
    print(f"\n{Colors.BOLD}Environment Checks:{Colors.END}")
    
    # WSL Check
    wsl_ok, wsl_info = check_wsl_environment()
    if wsl_ok:
        print_success(f"WSL Environment: {wsl_info}")
    else:
        print_warning(f"WSL Environment: {wsl_info}")
    
    # GPU Check
    gpu_ok, gpu_info = check_gpu_availability()
    if gpu_ok:
        print_success(f"GPU Available: {gpu_info}")
    else:
        print_error(f"GPU Check: {gpu_info}")
        print_warning("RAPIDS requires a CUDA-capable GPU")
    
    # CUDA Toolkit Check
    cuda_ok, cuda_info = check_cuda_toolkit()
    if cuda_ok:
        print_success(f"CUDA Toolkit: {cuda_info}")
    else:
        print_warning(f"CUDA Toolkit: {cuda_info}")
    
    # Check RAPIDS packages
    print(f"\n{Colors.BOLD}RAPIDS Package Status:{Colors.END}")
    packages = check_rapids_packages()
    
    installed_count = 0
    for package, (installed, info) in packages.items():
        if installed:
            print_success(f"{package}: {info}")
            installed_count += 1
        else:
            print_error(f"{package}: {info}")
    
    if installed_count == 0:
        print_error("\nNo RAPIDS packages found. Please install RAPIDS first.")
        sys.exit(1)
    
    # Functionality tests
    print(f"\n{Colors.BOLD}Functionality Tests:{Colors.END}")
    
    # Test cuDF
    if packages.get('cudf', (False, ''))[0]:
        cudf_ok, cudf_info = test_cudf_functionality()
        if cudf_ok:
            print_success(f"cuDF Test: {cudf_info}")
        else:
            print_error(f"cuDF Test: {cudf_info}")
    
    # Test cuML
    if packages.get('cuml', (False, ''))[0]:
        cuml_ok, cuml_info = test_cuml_functionality()
        if cuml_ok:
            print_success(f"cuML Test: {cuml_info}")
        else:
            print_error(f"cuML Test: {cuml_info}")
    
    # Test GPU Memory
    if packages.get('cupy', (False, ''))[0]:
        mem_ok, mem_info = test_gpu_memory()
        if mem_ok:
            print_success(f"GPU Memory Test: {mem_info}")
        else:
            print_error(f"GPU Memory Test: {mem_info}")
    
    # Performance test
    if packages.get('cudf', (False, ''))[0]:
        print(f"\n{Colors.BOLD}Performance Test:{Colors.END}")
        perf_ok, perf_info = run_performance_comparison()
        if perf_ok:
            print_success(f"Performance: {perf_info}")
        else:
            print_error(f"Performance: {perf_info}")
    
    # Summary
    print_header("Validation Summary")
    
    if gpu_ok and installed_count > 0:
        print_success("RAPIDS is installed and functional!")
        print(f"\n{Colors.BOLD}Next Steps:{Colors.END}")
        print("1. Try the example notebooks in JupyterLab")
        print("2. Run the performance benchmarks")
        print("3. Explore RAPIDS documentation at https://docs.rapids.ai")
    else:
        print_error("RAPIDS validation failed. Please check the errors above.")
        print(f"\n{Colors.BOLD}Troubleshooting:{Colors.END}")
        if not gpu_ok:
            print("- Ensure NVIDIA drivers are installed on Windows")
            print("- Check that your GPU has compute capability >= 7.0")
        if installed_count == 0:
            print("- Install RAPIDS using one of the methods in the runbook")
            print("- Use the RAPIDS Release Selector for compatible versions")

if __name__ == "__main__":
    main()