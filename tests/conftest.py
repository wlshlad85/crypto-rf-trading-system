"""
Pytest configuration and fixtures.
"""
import pytest
import subprocess
from pathlib import Path

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "gpu: marks tests as requiring GPU")
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "benchmark: marks tests as benchmarks")

def pytest_collection_modifyitems(config, items):
    """Add markers based on test names and availability."""
    # Check if GPU is available
    gpu_available = check_gpu_available()
    
    for item in items:
        # Auto-mark GPU tests
        if "gpu" in item.nodeid.lower():
            item.add_marker(pytest.mark.gpu)
        
        # Skip GPU tests if no GPU
        if item.get_closest_marker("gpu") and not gpu_available:
            item.add_marker(pytest.mark.skip(reason="GPU not available"))
        
        # Mark slow tests
        if "latency" in item.nodeid or "scaling" in item.nodeid:
            item.add_marker(pytest.mark.slow)

def check_gpu_available():
    """Check if GPU is available for testing."""
    try:
        # Check CUDA
        result = subprocess.run(
            ["python", "-c", "import torch; print(torch.cuda.is_available())"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and "True" in result.stdout:
            return True
    except:
        pass
    
    try:
        # Check nvidia-smi
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except:
        return False

# Shared fixtures
@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "data"

@pytest.fixture(scope="session")
def repo_root():
    """Path to repository root."""
    return Path(__file__).parent.parent