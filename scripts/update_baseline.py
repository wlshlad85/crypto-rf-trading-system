#!/usr/bin/env python3
"""
Update baseline expectations based on current pipeline results.
Run this when making intentional model changes.
"""
import argparse
import json
import yaml
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_pipeline.py"
GOLDEN_DATA = REPO_ROOT / "tests" / "data" / "golden.parquet"
EXPECTATIONS = REPO_ROOT / "tests" / "expectations.yaml"

def run_pipeline_multiple(mode: str, n_runs: int = 5):
    """Run pipeline multiple times and collect results."""
    results = []
    
    for i in range(n_runs):
        print(f"Running {mode} pipeline {i+1}/{n_runs}...")
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name
        
        cmd = [
            "python", str(SCRIPT_PATH),
            "--mode", mode,
            "--input", str(GOLDEN_DATA),
            "--out", output_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            with open(output_path) as fh:
                results.append(json.load(fh))
        finally:
            import os
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    return results

def compute_stable_metrics(results):
    """Compute stable metrics from multiple runs."""
    # Extract metrics from all runs
    all_metrics = {}
    for result in results:
        for metric, value in result["metrics"].items():
            if metric not in all_metrics:
                all_metrics[metric] = []
            all_metrics[metric].append(value)
    
    # Compute median and reasonable tolerance
    stable_metrics = {}
    for metric, values in all_metrics.items():
        values = np.array(values)
        median = np.median(values)
        std = np.std(values)
        
        # Set tolerance as max of:
        # - 3 standard deviations
        # - 1% of absolute value
        # - Minimum threshold based on metric type
        min_tol = {
            "auc": 1e-3,
            "accuracy": 5e-3,
            "logloss": 1e-2,
            "sharpe_1d": 0.05,
            "pnl_sum": 10.0,
            "max_drawdown": 0.002,
            "win_rate": 0.01
        }.get(metric, 1e-3)
        
        tolerance = max(3 * std, abs(median) * 0.01, min_tol)
        
        stable_metrics[metric] = {
            "expected": float(median),
            "atol": float(tolerance),
            "_std": float(std),
            "_values": [float(v) for v in values]
        }
    
    return stable_metrics

def compute_latency_budgets(cpu_results, gpu_results):
    """Compute reasonable latency budgets."""
    cpu_times = [r["execution_time_ms"] for r in cpu_results]
    gpu_times = [r["execution_time_ms"] for r in gpu_results] if gpu_results else []
    
    # Set budget as 95th percentile + 20% margin
    cpu_p95 = np.percentile(cpu_times, 95)
    cpu_budget = int(cpu_p95 * 1.2)
    
    if gpu_times:
        gpu_p95 = np.percentile(gpu_times, 95)
        gpu_budget = int(gpu_p95 * 1.2)
        speedup = np.mean(cpu_times) / np.mean(gpu_times)
    else:
        gpu_budget = 1000  # Default
        speedup = 3.0  # Default expectation
    
    return {
        "cpu_max": cpu_budget,
        "gpu_max": gpu_budget,
        "gpu_speedup_min": float(max(speedup * 0.8, 2.0))  # 80% of current speedup
    }

def compute_parity_tolerances(cpu_results, gpu_results):
    """Compute CPU/GPU parity tolerances."""
    if not gpu_results:
        return {
            "yhat_rmse_atol": 1e-3,
            "yhat_max_diff": 1e-2,
            "metric_rtol": 1e-3
        }
    
    # Compare predictions
    rmses = []
    max_diffs = []
    
    for cpu_r, gpu_r in zip(cpu_results[:3], gpu_results[:3]):
        cpu_pred = np.array(cpu_r["predictions"])
        gpu_pred = np.array(gpu_r["predictions"])
        
        rmse = np.sqrt(np.mean((cpu_pred - gpu_pred)**2))
        max_diff = np.max(np.abs(cpu_pred - gpu_pred))
        
        rmses.append(rmse)
        max_diffs.append(max_diff)
    
    return {
        "yhat_rmse_atol": float(np.max(rmses) * 2),  # 2x current max
        "yhat_max_diff": float(np.max(max_diffs) * 2),
        "metric_rtol": 1e-3
    }

def main():
    parser = argparse.ArgumentParser(description='Update baseline expectations')
    parser.add_argument('--n-runs', type=int, default=5,
                        help='Number of runs for stability')
    parser.add_argument('--skip-gpu', action='store_true',
                        help='Skip GPU tests')
    parser.add_argument('--output', default=None,
                        help='Output path (default: overwrite expectations.yaml)')
    
    args = parser.parse_args()
    
    # Ensure golden data exists
    if not GOLDEN_DATA.exists():
        print("Generating golden dataset...")
        subprocess.run([
            "python", str(REPO_ROOT / "scripts" / "generate_golden_data.py")
        ], check=True)
    
    # Run CPU pipeline
    print(f"\nRunning CPU pipeline {args.n_runs} times...")
    cpu_results = run_pipeline_multiple("cpu", args.n_runs)
    
    # Run GPU pipeline
    gpu_results = []
    if not args.skip_gpu:
        try:
            print(f"\nRunning GPU pipeline {args.n_runs} times...")
            gpu_results = run_pipeline_multiple("gpu", args.n_runs)
        except Exception as e:
            print(f"GPU tests failed (might not be available): {e}")
            print("Continuing with CPU-only baseline...")
    
    # Compute stable metrics
    print("\nComputing stable metrics...")
    metrics = compute_stable_metrics(cpu_results)
    
    # Remove internal fields
    for metric in metrics.values():
        metric.pop("_std", None)
        metric.pop("_values", None)
    
    # Compute latency budgets
    latency = compute_latency_budgets(cpu_results, gpu_results)
    
    # Compute parity tolerances
    parity = compute_parity_tolerances(cpu_results, gpu_results)
    
    # Load existing expectations for data quality
    if EXPECTATIONS.exists():
        with open(EXPECTATIONS) as f:
            old_exp = yaml.safe_load(f)
        data_quality = old_exp.get("data_quality", {})
    else:
        # Compute from golden data
        import pandas as pd
        df = pd.read_parquet(GOLDEN_DATA)
        
        data_quality = {
            "features": {
                "return_0": {
                    "mean": float(df["return_0"].mean()),
                    "std": float(df["return_0"].std()),
                    "atol": 0.005
                },
                "volume_0": {
                    "mean": float(df["volume_0"].mean()),
                    "std": float(df["volume_0"].std()),
                    "rtol": 0.1
                }
            },
            "max_missing_pct": 0.03
        }
    
    # Create new expectations
    new_expectations = {
        "metrics": metrics,
        "latency_ms": latency,
        "parity": parity,
        "data_quality": data_quality,
        "feature_importance": {
            "top_10_overlap_min": 0.7
        },
        "version": {
            "model_version": "1.0.0",
            "data_version": datetime.now().strftime("%Y.%m"),
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "updated_by": "update_baseline.py"
        }
    }
    
    # Save expectations
    output_path = args.output or str(EXPECTATIONS)
    with open(output_path, 'w') as f:
        yaml.dump(new_expectations, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nBaseline updated: {output_path}")
    
    # Print summary
    print("\nNew baseline summary:")
    print(f"- Metrics: {len(metrics)} tracked")
    print(f"- CPU latency budget: {latency['cpu_max']}ms")
    if gpu_results:
        print(f"- GPU latency budget: {latency['gpu_max']}ms")
        print(f"- Expected GPU speedup: {latency['gpu_speedup_min']:.1f}x")
    
    # Verify new baseline works
    print("\nVerifying new baseline...")
    test_cmd = ["pytest", str(REPO_ROOT / "tests" / "test_regression.py"), "-v"]
    result = subprocess.run(test_cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ All tests pass with new baseline!")
    else:
        print("❌ Tests failed with new baseline:")
        print(result.stdout)
        print(result.stderr)

if __name__ == '__main__':
    main()