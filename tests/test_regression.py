#!/usr/bin/env python3
"""
Regression tests for ML pipeline.
Tests metrics against expectations and CPU/GPU parity.
"""
import json
import os
import numpy as np
import pandas as pd
import yaml
import subprocess
import tempfile
import pytest
from pathlib import Path
from typing import Dict, Any

# Test configuration
REPO_ROOT = Path(__file__).parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_pipeline.py"
GOLDEN_DATA = REPO_ROOT / "tests" / "data" / "golden.parquet"
EXPECTATIONS = REPO_ROOT / "tests" / "expectations.yaml"

def run_pipeline(mode: str, model_type: str = 'xgboost') -> Dict[str, Any]:
    """Run the pipeline and return results."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        output_path = f.name
    
    cmd = [
        "python", str(SCRIPT_PATH),
        "--mode", mode,
        "--input", str(GOLDEN_DATA),
        "--out", output_path,
        "--model", model_type
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"\n{mode.upper()} pipeline output:")
        print(result.stdout)
        
        with open(output_path) as fh:
            return json.load(fh)
    except subprocess.CalledProcessError as e:
        print(f"Pipeline failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)

@pytest.fixture(scope="module")
def expectations():
    """Load expectations file."""
    with open(EXPECTATIONS) as f:
        return yaml.safe_load(f)

@pytest.fixture(scope="module")
def results(expectations):
    """Run both CPU and GPU pipelines once and cache results."""
    if not GOLDEN_DATA.exists():
        # Generate golden data if it doesn't exist
        subprocess.run([
            "python", str(REPO_ROOT / "scripts" / "generate_golden_data.py")
        ], check=True)
    
    cpu_result = run_pipeline("cpu")
    
    # Try GPU, but don't fail if not available
    try:
        gpu_result = run_pipeline("gpu")
    except Exception as e:
        print(f"GPU mode failed (might not be available): {e}")
        gpu_result = None
    
    return {"cpu": cpu_result, "gpu": gpu_result}

class TestMetricsRegression:
    """Test that metrics match expectations."""
    
    def test_cpu_metrics(self, results, expectations):
        """Test CPU metrics against expectations."""
        cpu_metrics = results["cpu"]["metrics"]
        expected_metrics = expectations["metrics"]
        
        for metric_name, config in expected_metrics.items():
            actual = cpu_metrics[metric_name]
            expected = config["expected"]
            tolerance = config["atol"]
            
            assert abs(actual - expected) <= tolerance, \
                f"CPU {metric_name} drift: {actual:.6f} vs {expected:.6f} (tol={tolerance})"
    
    def test_gpu_metrics(self, results, expectations):
        """Test GPU metrics against expectations."""
        if results["gpu"] is None:
            pytest.skip("GPU not available")
        
        gpu_metrics = results["gpu"]["metrics"]
        expected_metrics = expectations["metrics"]
        
        for metric_name, config in expected_metrics.items():
            actual = gpu_metrics[metric_name]
            expected = config["expected"]
            tolerance = config["atol"]
            
            assert abs(actual - expected) <= tolerance, \
                f"GPU {metric_name} drift: {actual:.6f} vs {expected:.6f} (tol={tolerance})"

class TestCPUGPUParity:
    """Test that CPU and GPU produce consistent results."""
    
    def test_prediction_parity(self, results, expectations):
        """Test that predictions are consistent between CPU and GPU."""
        if results["gpu"] is None:
            pytest.skip("GPU not available")
        
        yhat_cpu = np.array(results["cpu"]["predictions"])
        yhat_gpu = np.array(results["gpu"]["predictions"])
        
        # Check same length
        assert len(yhat_cpu) == len(yhat_gpu), \
            f"Prediction lengths differ: CPU={len(yhat_cpu)}, GPU={len(yhat_gpu)}"
        
        # RMSE check
        rmse = np.sqrt(np.mean((yhat_cpu - yhat_gpu)**2))
        rmse_tol = expectations["parity"]["yhat_rmse_atol"]
        assert rmse <= rmse_tol, \
            f"CPU/GPU prediction RMSE too high: {rmse:.6f} > {rmse_tol}"
        
        # Max difference check
        max_diff = np.max(np.abs(yhat_cpu - yhat_gpu))
        max_diff_tol = expectations["parity"]["yhat_max_diff"]
        assert max_diff <= max_diff_tol, \
            f"CPU/GPU max prediction diff too high: {max_diff:.6f} > {max_diff_tol}"
        
        print(f"\nPrediction parity: RMSE={rmse:.6f}, max_diff={max_diff:.6f}")
    
    def test_metric_parity(self, results, expectations):
        """Test that metrics are consistent between CPU and GPU."""
        if results["gpu"] is None:
            pytest.skip("GPU not available")
        
        cpu_metrics = results["cpu"]["metrics"]
        gpu_metrics = results["gpu"]["metrics"]
        rtol = expectations["parity"]["metric_rtol"]
        
        for metric_name in cpu_metrics:
            cpu_val = cpu_metrics[metric_name]
            gpu_val = gpu_metrics[metric_name]
            
            if abs(cpu_val) > 1e-9:  # Avoid division by zero
                rel_diff = abs(cpu_val - gpu_val) / abs(cpu_val)
                assert rel_diff <= rtol, \
                    f"CPU/GPU {metric_name} mismatch: {cpu_val:.6f} vs {gpu_val:.6f} (rel_diff={rel_diff:.6f})"

class TestDataQuality:
    """Test data quality and feature drift."""
    
    def test_feature_statistics(self, expectations):
        """Test that feature statistics haven't drifted."""
        if not GOLDEN_DATA.exists():
            pytest.skip("Golden data not found")
        
        df = pd.read_parquet(GOLDEN_DATA)
        expected_stats = expectations["data_quality"]["features"]
        
        for feature, stats in expected_stats.items():
            if feature not in df.columns:
                continue
            
            actual_mean = df[feature].mean()
            actual_std = df[feature].std()
            
            if "atol" in stats:
                # Absolute tolerance
                assert abs(actual_mean - stats["mean"]) <= stats["atol"], \
                    f"{feature} mean drift: {actual_mean:.6f} vs {stats['mean']:.6f}"
                assert abs(actual_std - stats["std"]) <= stats["atol"], \
                    f"{feature} std drift: {actual_std:.6f} vs {stats['std']:.6f}"
            else:
                # Relative tolerance
                rtol = stats.get("rtol", 0.1)
                assert abs(actual_mean - stats["mean"]) / abs(stats["mean"]) <= rtol, \
                    f"{feature} mean drift: {actual_mean:.6f} vs {stats['mean']:.6f}"
    
    def test_missing_data(self, expectations):
        """Test that missing data is within acceptable bounds."""
        if not GOLDEN_DATA.exists():
            pytest.skip("Golden data not found")
        
        df = pd.read_parquet(GOLDEN_DATA)
        missing_pct = df.isna().sum().sum() / (df.shape[0] * df.shape[1])
        max_missing = expectations["data_quality"]["max_missing_pct"]
        
        assert missing_pct <= max_missing, \
            f"Too much missing data: {missing_pct:.4f} > {max_missing}"

class TestNumericalStability:
    """Test numerical stability across runs."""
    
    def test_deterministic_cpu(self, results):
        """Test that CPU results are deterministic."""
        # Run again
        cpu_result2 = run_pipeline("cpu")
        
        # Check predictions are identical
        yhat1 = np.array(results["cpu"]["predictions"])
        yhat2 = np.array(cpu_result2["predictions"])
        
        assert np.allclose(yhat1, yhat2, atol=1e-9), \
            "CPU results not deterministic"
    
    def test_no_nans(self, results):
        """Test that predictions don't contain NaNs."""
        for mode in ["cpu", "gpu"]:
            if results[mode] is None:
                continue
            
            yhat = np.array(results[mode]["predictions"])
            assert not np.any(np.isnan(yhat)), f"{mode} predictions contain NaN"
            assert not np.any(np.isinf(yhat)), f"{mode} predictions contain Inf"

def test_golden_data_exists():
    """Test that golden data exists or can be generated."""
    if not GOLDEN_DATA.exists():
        # Generate it
        subprocess.run([
            "python", str(REPO_ROOT / "scripts" / "generate_golden_data.py")
        ], check=True)
    
    assert GOLDEN_DATA.exists(), "Golden data file not found"
    
    # Check it's readable and has expected structure
    df = pd.read_parquet(GOLDEN_DATA)
    assert len(df) > 0, "Golden data is empty"
    assert 'y' in df.columns, "Target column 'y' not found"
    print(f"\nGolden data: {df.shape} shape, {df.memory_usage(deep=True).sum()/1024**2:.2f} MB")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])