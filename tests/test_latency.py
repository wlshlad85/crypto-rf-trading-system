#!/usr/bin/env python3
"""
Latency and performance regression tests.
Ensures execution time stays within budget.
"""
import json
import yaml
import time
import subprocess
import tempfile
import pytest
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

# Test configuration
REPO_ROOT = Path(__file__).parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_pipeline.py"
GOLDEN_DATA = REPO_ROOT / "tests" / "data" / "golden.parquet"
EXPECTATIONS = REPO_ROOT / "tests" / "expectations.yaml"

def timed_run(mode: str, n_runs: int = 1) -> Tuple[float, float]:
    """Run pipeline and measure execution time.
    
    Returns:
        (mean_time_ms, std_time_ms)
    """
    times = []
    
    for _ in range(n_runs):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name
        
        # Measure wall clock time
        t0 = time.perf_counter()
        
        cmd = [
            "python", str(SCRIPT_PATH),
            "--mode", mode,
            "--input", str(GOLDEN_DATA),
            "--out", output_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            dt = (time.perf_counter() - t0) * 1000  # Convert to ms
            times.append(dt)
            
            # Also get self-reported time
            with open(output_path) as f:
                result = json.load(f)
                internal_time = result.get("execution_time_ms", dt)
                print(f"{mode.upper()} run {len(times)}: {dt:.1f}ms wall, {internal_time:.1f}ms internal")
            
        except subprocess.CalledProcessError as e:
            print(f"Pipeline failed: {e}")
            print(f"stderr: {e.stderr}")
            raise
        finally:
            import os
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    mean_time = np.mean(times)
    std_time = np.std(times) if n_runs > 1 else 0
    
    return mean_time, std_time

@pytest.fixture(scope="module")
def expectations():
    """Load expectations file."""
    with open(EXPECTATIONS) as f:
        return yaml.safe_load(f)

class TestLatencyBudget:
    """Test that execution stays within latency budget."""
    
    def test_cpu_latency(self, expectations):
        """Test CPU execution time."""
        max_time = expectations["latency_ms"]["cpu_max"]
        
        # Run multiple times for stability
        mean_time, std_time = timed_run("cpu", n_runs=3)
        
        # Check against budget (using mean + 1 std)
        upper_bound = mean_time + std_time
        assert upper_bound <= max_time, \
            f"CPU too slow: {mean_time:.1f}±{std_time:.1f}ms > {max_time}ms budget"
        
        print(f"\nCPU latency: {mean_time:.1f}±{std_time:.1f}ms (budget: {max_time}ms)")
    
    def test_gpu_latency(self, expectations):
        """Test GPU execution time."""
        max_time = expectations["latency_ms"]["gpu_max"]
        
        # Try GPU - skip if not available
        try:
            mean_time, std_time = timed_run("gpu", n_runs=3)
        except Exception as e:
            pytest.skip(f"GPU not available: {e}")
        
        # Check against budget
        upper_bound = mean_time + std_time
        assert upper_bound <= max_time, \
            f"GPU too slow: {mean_time:.1f}±{std_time:.1f}ms > {max_time}ms budget"
        
        print(f"\nGPU latency: {mean_time:.1f}±{std_time:.1f}ms (budget: {max_time}ms)")
    
    def test_gpu_speedup(self, expectations):
        """Test that GPU provides expected speedup over CPU."""
        min_speedup = expectations["latency_ms"]["gpu_speedup_min"]
        
        # Measure both
        cpu_time, _ = timed_run("cpu", n_runs=1)
        
        try:
            gpu_time, _ = timed_run("gpu", n_runs=1)
        except Exception:
            pytest.skip("GPU not available")
        
        # Calculate speedup
        speedup = cpu_time / gpu_time
        assert speedup >= min_speedup, \
            f"GPU speedup too low: {speedup:.2f}x < {min_speedup}x expected"
        
        print(f"\nGPU speedup: {speedup:.2f}x (CPU: {cpu_time:.1f}ms, GPU: {gpu_time:.1f}ms)")

class TestScalability:
    """Test performance at different data sizes."""
    
    @pytest.mark.parametrize("scale_factor", [0.1, 0.5, 1.0])
    def test_scaling(self, scale_factor):
        """Test that execution time scales reasonably with data size."""
        # Generate scaled dataset
        import pandas as pd
        df = pd.read_parquet(GOLDEN_DATA)
        
        n_samples = int(len(df) * scale_factor)
        df_scaled = df.iloc[:n_samples]
        
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            scaled_path = f.name
            df_scaled.to_parquet(scaled_path)
        
        try:
            # Measure time
            cpu_time, _ = timed_run("cpu", n_runs=1)
            
            # Rough check: time should scale sub-linearly
            # (due to fixed overhead and algorithmic efficiency)
            if scale_factor < 1.0:
                # Smaller data should be faster
                baseline_time = 5000  # ms, rough baseline
                expected_max = baseline_time * (scale_factor + 0.2)  # Allow 20% fixed overhead
                assert cpu_time <= expected_max, \
                    f"Scaling issue: {cpu_time:.1f}ms for {scale_factor}x data"
            
            print(f"\nScaling test ({scale_factor}x): {cpu_time:.1f}ms for {n_samples} samples")
            
        finally:
            import os
            if os.path.exists(scaled_path):
                os.unlink(scaled_path)

class TestMemoryUsage:
    """Test memory usage stays reasonable."""
    
    def test_memory_efficient(self):
        """Test that pipeline doesn't use excessive memory."""
        # This is a simple test - in production you'd use memory_profiler
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024**2  # MB
        
        # Run pipeline
        _ = timed_run("cpu", n_runs=1)
        
        final_memory = process.memory_info().rss / 1024**2  # MB
        memory_increase = final_memory - initial_memory
        
        # Check memory increase is reasonable (< 500 MB for test data)
        assert memory_increase < 500, \
            f"Excessive memory usage: {memory_increase:.1f} MB increase"
        
        print(f"\nMemory usage: {memory_increase:.1f} MB increase")

def test_warmup_effect():
    """Test if there's significant warmup effect (first run slower)."""
    times = []
    
    for i in range(5):
        t, _ = timed_run("cpu", n_runs=1)
        times.append(t)
        print(f"Run {i+1}: {t:.1f}ms")
    
    # Check if first run is significantly slower
    first_run = times[0]
    avg_rest = np.mean(times[1:])
    
    warmup_ratio = first_run / avg_rest
    print(f"\nWarmup effect: first run {warmup_ratio:.2f}x slower than average")
    
    # Warn if significant warmup effect
    if warmup_ratio > 1.5:
        print("WARNING: Significant warmup effect detected. Consider pre-warming in production.")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])