# GPU-Accelerated Parameter Sweep Infrastructure

This directory contains the infrastructure for running reproducible, GPU-accelerated hyperparameter sweeps for the crypto RF trading system.

## 🚀 Quick Start

1. **Install GPU dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run a manual sweep:**
   ```bash
   python gpu_sweep_runner.py --config sweep_config_template.json --gpu-ids 0,1,2,3
   ```

3. **Setup nightly automated sweeps:**
   ```bash
   # Add to crontab (runs at 2 AM daily)
   crontab -e
   0 2 * * * /workspace/meta_optim/nightly_gpu_sweep.sh >> /var/log/crypto_sweep.log 2>&1
   ```

## 📁 Key Components

### 1. **crypto_gpu_reproducibility.py**
Comprehensive reproducibility module ensuring deterministic GPU operations:
- Sets all random seeds (Python, NumPy, PyTorch, CUDA)
- Configures deterministic algorithms
- Manages GPU-specific seeds for multi-GPU setups
- Provides framework-specific parameters (CatBoost, LightGBM, XGBoost)

```python
from crypto_gpu_reproducibility import setup_crypto_gpu_reproducibility

# Setup reproducibility
repro = setup_crypto_gpu_reproducibility(seed=123456)

# Get framework parameters
catboost_params = repro.get_catboost_params()
lightgbm_params = repro.get_lightgbm_params()
```

### 2. **hyperband_runner.py**
Enhanced Hyperband optimization with GPU reproducibility:
- Integrated reproducibility management
- Support for distributed GPU execution
- Checkpoint/resume capability
- Deterministic configuration sampling

```python
from hyperband_runner import HyperbandRunner

runner = HyperbandRunner(seed=123456, enable_gpu=True)
results = runner.run_hyperband_optimization(n_iterations=10)
```

### 3. **gpu_sweep_runner.py**
Main orchestrator for GPU parameter sweeps:
- Multiple sweep strategies (Hyperband, Grid, Random, Bayesian)
- Multi-GPU distribution and load balancing
- Resource monitoring and optimization
- Comprehensive logging and reporting

```bash
python gpu_sweep_runner.py \
    --config sweep_config.json \
    --gpu-ids 0,1,2,3 \
    --seed 123456 \
    --log-dir sweep_logs
```

### 4. **nightly_gpu_sweep.sh**
Production deployment script for automated sweeps:
- Environment setup and validation
- GPU availability checking
- Automatic checkpoint/resume
- Error handling and notifications
- Log rotation and cleanup

## 🔧 Configuration

### Sweep Configuration (sweep_config_template.json)

```json
{
    "sweep_type": "hyperband",  // Options: hyperband, grid, random, bayesian
    "hyperband": {
        "iterations": 10,        // Number of Hyperband iterations
        "max_iter": 81,         // Maximum resource allocation
        "eta": 3                // Reduction factor
    },
    "parameter_ranges": {
        // Define parameter search spaces for each model
    },
    "gpu_settings": {
        "memory_fraction": 0.8,  // GPU memory allocation
        "deterministic": true    // Force deterministic ops
    }
}
```

### Environment Variables

```bash
# GPU configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Select specific GPUs
export CUDA_LAUNCH_BLOCKING=1        # Force synchronous execution
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # cuBLAS determinism

# Reproducibility
export PYTHONHASHSEED=123456         # Python hash seed
export TF_DETERMINISTIC_OPS=1        # TensorFlow determinism

# Resource limits
export OMP_NUM_THREADS=1             # CPU thread limiting
```

## 📊 Sweep Strategies

### 1. **Hyperband** (Recommended)
- Efficient multi-armed bandit approach
- Progressive resource allocation
- Early stopping of poor configurations
- Best for large parameter spaces

### 2. **Grid Search**
- Exhaustive parameter combinations
- Suitable for small parameter spaces
- Guaranteed coverage

### 3. **Random Search**
- Random sampling from parameter distributions
- Good for initial exploration
- Can find unexpected optima

### 4. **Bayesian Optimization**
- Model-based optimization using Optuna
- Efficient for expensive evaluations
- Learns from previous trials

## 🖥️ Multi-GPU Execution

The infrastructure supports multiple GPU configurations:

### Single GPU
```bash
python gpu_sweep_runner.py --gpu-ids 0
```

### Multiple GPUs (Data Parallel)
```bash
python gpu_sweep_runner.py --gpu-ids 0,1,2,3
```

### Distributed Execution
Each GPU evaluates different configurations in parallel, with deterministic seed derivation ensuring reproducibility.

## 📈 Monitoring and Logging

### Real-time Monitoring
- GPU utilization and memory usage
- CPU and system memory usage
- Progress tracking and ETA
- Live performance metrics

### Logging Structure
```
sweep_logs/
├── nightly_20250929_020000/
│   ├── sweep_output.log           # Main execution log
│   ├── gpu_repro_*.json          # Reproducibility info
│   ├── sweep_results_*.json      # Detailed results
│   ├── sweep_report_*.txt        # Human-readable report
│   └── checkpoints/              # Resume checkpoints
```

### Reports Generated
1. **JSON Results**: Complete sweep data for analysis
2. **Text Report**: Summary statistics and best configurations
3. **Best Configs**: Top performing configurations for deployment
4. **Resource Usage**: GPU/CPU utilization statistics

## 🔄 Checkpoint and Resume

Sweeps can be interrupted and resumed:

```bash
# Resume from latest checkpoint
python gpu_sweep_runner.py --resume sweep_logs/checkpoints/checkpoint_latest.pkl

# Resume from specific checkpoint
python gpu_sweep_runner.py --resume path/to/checkpoint.pkl
```

## 🚨 Troubleshooting

### Common Issues

1. **Non-deterministic results**
   - Ensure CUDA_LAUNCH_BLOCKING=1 is set
   - Check cuDNN deterministic mode
   - Verify all seeds are properly set

2. **GPU memory errors**
   - Reduce batch size or model complexity
   - Adjust gpu_memory_fraction in config
   - Use gradient checkpointing

3. **Slow execution**
   - Deterministic ops are slower than default
   - Consider reducing parameter search space
   - Use early stopping more aggressively

### Debug Mode
```bash
# Enable verbose logging
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
python gpu_sweep_runner.py --config config.json --gpu-ids 0 --log-level DEBUG
```

## 📚 Best Practices

1. **Reproducibility First**
   - Always use the same seed for comparable results
   - Document exact GPU models and driver versions
   - Save complete system info with each sweep

2. **Resource Management**
   - Monitor GPU temperature and throttling
   - Use appropriate batch sizes for GPU memory
   - Implement checkpointing for long runs

3. **Parameter Selection**
   - Start with wide ranges, then narrow
   - Use log-scale for learning rates
   - Consider parameter interactions

4. **Production Deployment**
   - Test configurations on holdout data
   - Validate on different time periods
   - Monitor live performance metrics

## 🔗 Integration

### With Training Pipeline
```python
# Load best configuration
with open('models/best_configs_latest.json', 'r') as f:
    best_config = json.load(f)

# Use in training
model = train_model(best_config['configurations'][0])
```

### With Production System
```python
# Apply optimized parameters
trading_params = best_config['trading_params']
entry_model_params = best_config['entry_model']
```

## 📞 Support

For issues or questions:
1. Check logs in `sweep_logs/` directory
2. Review system info in reproducibility logs
3. Verify GPU drivers and CUDA installation
4. Ensure all dependencies are installed

---

Remember: **Reproducibility is key!** Always use consistent seeds and document your environment for comparable results across sweeps.