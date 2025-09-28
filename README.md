# GPU/CPU Regression Testing Framework

A comprehensive testing framework for maintaining GPU speedups and catching performance regressions in ML pipelines. This framework ensures consistency between CPU and GPU execution paths while monitoring latency and accuracy metrics.

## Features

- **Dual-path testing**: Runs identical pipelines on CPU and GPU
- **Golden dataset**: Small, representative dataset with realistic characteristics
- **Metric tracking**: AUC, Sharpe ratio, PnL, latency benchmarks
- **Parity checks**: Ensures CPU/GPU produce consistent results
- **CI integration**: GitHub Actions with self-hosted GPU runners
- **Baseline management**: Track and update expected metrics

## Quick Start

### 1. Install Dependencies

```bash
# CPU-only setup
make install

# GPU setup (CUDA 11.8)
make install-gpu

# GPU setup (CUDA 12.1)
make install-gpu-cu12
```

### 2. Generate Golden Dataset

```bash
make golden
# or
python scripts/generate_golden_data.py
```

### 3. Run Tests

```bash
# Run all tests
make test

# CPU tests only
make test-cpu

# GPU tests only
make test-gpu

# Quick smoke test
make test-quick
```

### 4. Update Baseline

When making intentional model changes:

```bash
# Update baseline with CPU and GPU
make baseline

# Update baseline (CPU only)
make baseline-no-gpu
```

## Project Structure

```
.
├── scripts/
│   ├── run_pipeline.py         # Main pipeline with CPU/GPU modes
│   ├── generate_golden_data.py # Generate test dataset
│   └── update_baseline.py      # Update expected metrics
├── tests/
│   ├── data/
│   │   └── golden.parquet      # Golden dataset (generated)
│   ├── expectations.yaml       # Expected metrics & tolerances
│   ├── test_regression.py      # Metric & parity tests
│   └── test_latency.py         # Performance tests
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions workflow
├── requirements*.txt           # Dependencies
├── pytest.ini                  # Test configuration
└── Makefile                    # Convenience commands
```

## Test Types

### 1. Metric Regression Tests
- Compares model metrics (AUC, accuracy, Sharpe) against baseline
- Fails if metrics drift beyond tolerance

### 2. CPU/GPU Parity Tests
- Ensures predictions are consistent between CPU and GPU
- Checks RMSE and max difference tolerances

### 3. Latency Tests
- Monitors execution time against budgets
- Verifies GPU speedup (typically 3-5x)

### 4. Data Quality Tests
- Detects feature drift
- Monitors missing data rates

## CI/CD Integration

The framework includes GitHub Actions workflows for:

1. **CPU Testing**: Runs on GitHub-hosted runners
2. **GPU Testing**: Runs on self-hosted GPU runners
3. **Matrix Testing**: Multiple Python/CUDA versions

### Setting Up Self-Hosted GPU Runner

1. On your GPU machine:
```bash
# Follow GitHub's self-hosted runner setup
# https://docs.github.com/en/actions/hosting-your-own-runners

# Add labels: self-hosted, gpu
```

2. Ensure GPU dependencies are available:
```bash
nvidia-smi  # Verify GPU
conda create -n ci python=3.10
conda activate ci
pip install -r requirements-gpu-cuda11.8.txt
```

## Customization

### Adding New Metrics

1. Update `compute_metrics()` in `scripts/run_pipeline.py`
2. Add expectations to `tests/expectations.yaml`
3. Run baseline update: `make baseline`

### Adjusting Tolerances

Edit `tests/expectations.yaml`:
```yaml
metrics:
  your_metric:
    expected: 0.75
    atol: 0.01  # Absolute tolerance
```

### Custom Models

Modify `model_fit_predict()` in `scripts/run_pipeline.py` to use your model:
- XGBoost (default)
- cuML Random Forest
- Custom implementations

## Performance Tips

1. **Fix random seeds** everywhere for reproducible metrics
2. **Use smaller golden datasets** (1-5 MB) for fast CI
3. **Profile before optimizing**:
   ```bash
   make profile-cpu
   make profile-memory
   ```
4. **Monitor warmup effects** - first run may be slower

## Troubleshooting

### GPU Tests Failing
- Check CUDA version compatibility
- Verify GPU memory availability
- Ensure cuDF/cuML installed correctly

### Metric Drift
- Review recent model changes
- Check data preprocessing consistency
- Update baseline if changes are intentional

### Flaky Tests
- Increase tolerances in `expectations.yaml`
- Add more runs in `update_baseline.py`
- Fix random seeds more aggressively

## Best Practices

1. **Run tests before committing**: `make test-quick`
2. **Update baseline for model changes**: `make baseline`
3. **Keep golden data small** but representative
4. **Version control expectations.yaml**
5. **Monitor CI times** - keep under 5 minutes

## License

MIT