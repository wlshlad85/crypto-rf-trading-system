# RAPIDS on Windows WSL2 - Complete Setup Kit

This repository contains everything you need to get RAPIDS (GPU-accelerated data science) running on Windows 11 via WSL2.

## 📁 Repository Contents

- **`RAPIDS_WSL2_Runbook.md`** - Comprehensive setup guide with three installation methods
- **`rapids-quick-setup.sh`** - Interactive setup script for quick installation
- **`validate_rapids.py`** - Validation script to test your RAPIDS installation
- **`rapids_examples.py`** - Interactive examples demonstrating RAPIDS features
- **`docker-compose.yml`** - Docker Compose configuration for easy deployment

## 🚀 Quick Start

### Option 1: Docker (Fastest)
```bash
# Using docker-compose
docker-compose up rapids-notebook

# Or manually
docker run --rm -it --gpus all -p 8888:8888 \
  rapidsai/notebooks:25.08-cuda12.5-py3.11
```

### Option 2: Quick Setup Script
```bash
# Run the interactive setup
./rapids-quick-setup.sh

# Or use specific commands
./rapids-quick-setup.sh --check-gpu
./rapids-quick-setup.sh --install-miniforge
./rapids-quick-setup.sh --create-env
```

### Option 3: Manual Setup
Follow the detailed instructions in `RAPIDS_WSL2_Runbook.md`

## 🧪 Testing Your Installation

### Basic Validation
```bash
# Check if everything is working
python validate_rapids.py
```

### Interactive Examples
```bash
# Run example demonstrations
python rapids_examples.py
```

## 📊 What's Included

### RAPIDS Libraries
- **cuDF** - GPU DataFrames (like pandas, but faster)
- **cuML** - GPU Machine Learning (like scikit-learn, but faster)
- **cuGraph** - GPU Graph Analytics
- **cuSpatial** - GPU Spatial Analytics
- **cuXFilter** - GPU Data Visualization

### Scripts and Tools
1. **Quick Setup Script** (`rapids-quick-setup.sh`)
   - Automated installation options
   - GPU checking and validation
   - Environment creation

2. **Validation Script** (`validate_rapids.py`)
   - Comprehensive system checks
   - Package verification
   - Performance benchmarks

3. **Examples Script** (`rapids_examples.py`)
   - Interactive demonstrations
   - Performance comparisons
   - Real-world use cases

## 🔧 System Requirements

- **GPU**: NVIDIA GPU with compute capability ≥ 7.0 (RTX 2000 series or newer)
- **OS**: Windows 11 with WSL2
- **RAM**: 16GB minimum, 32GB recommended
- **Driver**: Latest NVIDIA Windows driver

## 📈 Performance Example

```python
import cudf
import pandas as pd

# 10 million rows
n = 10_000_000
pdf = pd.DataFrame({'x': range(n), 'y': range(n)})
gdf = cudf.from_pandas(pdf)

# Pandas: ~500ms
# cuDF: ~50ms
# Speedup: 10x!
```

## 🆘 Troubleshooting

### Common Issues

1. **GPU not detected**
   ```bash
   ./rapids-quick-setup.sh --check-gpu
   ```

2. **Docker can't access GPU**
   - Ensure Docker Desktop uses WSL2 backend
   - Restart Docker Desktop

3. **Import errors**
   ```bash
   python validate_rapids.py
   ```

## 📚 Resources

- [RAPIDS Documentation](https://docs.rapids.ai/)
- [RAPIDS Release Selector](https://rapids.ai/start.html)
- [NVIDIA WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/)
- [RAPIDS Community](https://rapids.ai/community.html)

## 🎯 Next Steps

1. Run `./rapids-quick-setup.sh` to set up your environment
2. Validate with `python validate_rapids.py`
3. Explore examples with `python rapids_examples.py`
4. Start building GPU-accelerated data science applications!

---

**Note**: This setup is specifically designed for Windows 11 with WSL2. For native Linux or other platforms, refer to the official RAPIDS documentation.