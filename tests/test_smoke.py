"""
Smoke tests to verify the testing framework is set up correctly.
"""
import subprocess
import sys
from pathlib import Path
import pytest

def test_python_version():
    """Test Python version is supported."""
    assert sys.version_info >= (3, 9), "Python 3.9+ required"

def test_imports():
    """Test that core dependencies are importable."""
    import numpy
    import pandas
    import sklearn
    import xgboost
    import yaml
    
    # Print versions for debugging
    print(f"\nDependency versions:")
    print(f"  numpy: {numpy.__version__}")
    print(f"  pandas: {pandas.__version__}")
    print(f"  sklearn: {sklearn.__version__}")
    print(f"  xgboost: {xgboost.__version__}")

def test_project_structure(repo_root):
    """Test that project structure is correct."""
    expected_files = [
        "scripts/run_pipeline.py",
        "scripts/generate_golden_data.py",
        "scripts/update_baseline.py",
        "tests/expectations.yaml",
        "tests/test_regression.py",
        "tests/test_latency.py",
        ".github/workflows/ci.yml",
        "requirements.txt",
        "Makefile",
        "README.md"
    ]
    
    for file_path in expected_files:
        full_path = repo_root / file_path
        assert full_path.exists(), f"Missing file: {file_path}"

def test_scripts_executable(repo_root):
    """Test that scripts can be executed."""
    scripts = [
        "scripts/run_pipeline.py",
        "scripts/generate_golden_data.py",
        "scripts/update_baseline.py"
    ]
    
    for script in scripts:
        result = subprocess.run(
            [sys.executable, str(repo_root / script), "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"Script failed: {script}\n{result.stderr}"

@pytest.mark.gpu
def test_gpu_availability():
    """Test GPU availability (optional)."""
    try:
        import torch
        assert torch.cuda.is_available(), "CUDA not available"
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    except ImportError:
        pytest.skip("PyTorch not installed")

def test_make_targets(repo_root):
    """Test that Makefile targets work."""
    # Test help target
    result = subprocess.run(
        ["make", "help"],
        cwd=repo_root,
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, "make help failed"
    assert "GPU/CPU Regression Testing" in result.stdout