# RAPIDS on Windows via WSL2 - Implementation Runbook
*Last Updated: September 28, 2025*

This runbook provides three proven paths to get RAPIDS (GPU-accelerated data science libraries) running on Windows 11 via WSL2. Choose based on your needs:
- **Docker Desktop**: Fastest setup, best isolation
- **Conda/Miniforge**: Most flexible, best for custom environments
- **pip**: Minimal approach, good for CI/CD

## Prerequisites & System Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with compute capability ≥ 7.0 (Volta or newer)
  - Supported: RTX 2000/3000/4000 series, Tesla V100, A100, etc.
  - NOT supported: GTX 10xx series (Pascal) - support removed in RAPIDS 24.02
- **RAM**: 16GB minimum, 32GB+ recommended
- **Storage**: 50GB+ free space

### Software Requirements
- **Windows 11** (WSL2 requires Windows 10 version 2004+ but Windows 11 recommended)
- **WSL2** with Ubuntu 22.04 or 24.04
- **Latest NVIDIA Windows Driver** (GeForce/Quadro/Data Center)

### Critical WSL2 Constraints
- ⚠️ **Single GPU only** - Multi-GPU setups not supported in WSL2
- ⚠️ **No GPU Direct Storage** - Feature not available in WSL2
- ⚠️ **Windows driver only** - Never install Linux NVIDIA drivers inside WSL

---

## Step 0: Initial Setup (Required for All Options)

### 1. Enable WSL2 and Install Ubuntu

Open PowerShell as Administrator:
```powershell
# Install WSL2 with Ubuntu 24.04
wsl --install -d Ubuntu-24.04

# Update WSL to latest version
wsl --update

# Set WSL2 as default
wsl --set-default-version 2
```

### 2. Install NVIDIA Windows Driver

1. Download latest driver from [NVIDIA Downloads](https://www.nvidia.com/Download/index.aspx)
2. Install the Windows driver (NOT the WSL-specific one)
3. **Reboot your system**

### 3. Verify GPU Access

Windows PowerShell:
```powershell
nvidia-smi
```
Should show your GPU details and CUDA version.

Ubuntu WSL shell:
```bash
nvidia-smi
# If not found, try:
nvidia-smi.exe
```

---

## Option A: Docker Desktop (Recommended)

### Installation Steps

1. **Install Docker Desktop**
   - Download from [docker.com](https://www.docker.com/products/docker-desktop/)
   - During setup, ensure "Use WSL 2 instead of Hyper-V" is checked
   - After installation: Settings → General → Enable "Use the WSL 2 based engine"

2. **Verify GPU in Docker**
   ```bash
   docker run --rm -it --gpus=all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
   ```
   If this shows GPU performance metrics, you're ready.

3. **Run RAPIDS Container**
   
   Visit [RAPIDS Release Selector](https://rapids.ai/start.html) and select:
   - Method: Docker
   - CUDA Version: Match your nvidia-smi output
   - Container: notebooks (includes JupyterLab)
   
   Example command (adjust tag from selector):
   ```bash
   docker run --rm -it --gpus all \
     -p 8888:8888 \
     -v "$PWD":/workspace \
     rapidsai/notebooks:25.08-cuda12.5-py3.11
   ```

4. **Access JupyterLab**
   - Open browser to `http://localhost:8888`
   - Token is shown in terminal output

### Quick Test
In Jupyter or container shell:
```python
import cudf
df = cudf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
print(df)
print(f"GPU Memory: {df.memory_usage(deep=True).sum()} bytes")
```

---

## Option B: Conda/Miniforge (Most Flexible)

### Installation Steps

1. **Install Miniforge in WSL**
   ```bash
   # Download Miniforge installer
   curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
   
   # Install
   bash Miniforge3-$(uname)-$(uname -m).sh -b
   
   # Initialize shell
   ~/miniforge3/bin/conda init bash
   exec $SHELL
   
   # Configure for RAPIDS
   conda config --set channel_priority flexible
   ```

2. **Create RAPIDS Environment**
   
   Use [RAPIDS Release Selector](https://rapids.ai/start.html) with:
   - Method: Conda
   - CUDA Version: Match your system
   - Python Version: 3.10 or 3.11
   
   Example (get exact command from selector):
   ```bash
   mamba create -n rapids-25.08 -c rapidsai -c conda-forge -c nvidia \
     rapids=25.08 python=3.11 cuda-version=12.5
   
   conda activate rapids-25.08
   ```

3. **Verify Installation**
   ```python
   import cudf
   import cuml
   import cugraph
   
   # Test cuDF
   df = cudf.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
   print("cuDF DataFrame:")
   print(df)
   
   # Test cuML
   from cuml.linear_model import LinearRegression
   model = LinearRegression()
   print("\ncuML LinearRegression initialized")
   ```

### Environment Management
```bash
# Export environment
conda env export > rapids-env.yml

# Recreate elsewhere
mamba env create -f rapids-env.yml
```

---

## Option C: pip Installation (Advanced)

### Installation Steps

1. **Install WSL-Ubuntu CUDA Toolkit**
   
   Inside WSL Ubuntu:
   ```bash
   # Add NVIDIA package repositories
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt-get update
   
   # Install CUDA toolkit (NOT the driver)
   sudo apt-get install -y cuda-toolkit-12-5
   
   # Add to PATH
   echo 'export PATH=/usr/local/cuda-12.5/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

2. **Create Python Environment**
   ```bash
   python3 -m venv rapids-env
   source rapids-env/bin/activate
   python -m pip install --upgrade pip
   ```

3. **Install RAPIDS Packages**
   ```bash
   # For CUDA 12.x
   python -m pip install --extra-index-url=https://pypi.nvidia.com \
     cudf-cu12 cuml-cu12 cugraph-cu12 cuspatial-cu12 cuxfilter-cu12 \
     cucim-cu12 pylibraft-cu12 raft-dask-cu12
   ```

4. **Verify Installation**
   ```python
   import cudf
   print(cudf.__version__)
   
   # Create sample DataFrame
   df = cudf.DataFrame({
       'id': range(1000),
       'value': range(1000, 2000)
   })
   print(df.describe())
   ```

---

## Troubleshooting

### Common Issues & Solutions

#### 1. Conda `__cuda` Constraint Conflicts
```bash
# Solution: Use exact versions from Release Selector
mamba create -n rapids-clean -c rapidsai -c conda-forge -c nvidia \
  rapids=25.08 python=3.11 cuda-version=12.5 --solver=libmamba
```

#### 2. WSL Connection Errors
```bash
# Restart WSL
wsl --shutdown
# Wait 10 seconds, then restart
wsl
```

#### 3. libcuda.so Not Found
```bash
# Check if cuda libraries are in path
ldconfig -p | grep cuda

# If missing, add to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

#### 4. Docker Can't Access GPU
1. Ensure Docker Desktop uses WSL2 backend
2. Restart Docker Desktop
3. Run test: `docker run --rm --gpus all nvidia/cuda:12.5.0-base-ubuntu22.04 nvidia-smi`

#### 5. Memory Issues
```bash
# Increase WSL2 memory limit
# Create/edit %USERPROFILE%\.wslconfig
[wsl2]
memory=16GB
processors=8
```

### Performance Optimization

1. **WSL2 Configuration** (.wslconfig):
   ```ini
   [wsl2]
   memory=32GB
   processors=16
   swap=8GB
   localhostForwarding=true
   ```

2. **RAPIDS Memory Management**:
   ```python
   import rmm
   # Use managed memory pool
   rmm.mr.set_current_device_resource(
       rmm.mr.PoolMemoryResource(
           rmm.mr.CudaMemoryResource(),
           initial_pool_size=2**30  # 1GB
       )
   )
   ```

---

## Validation Tests

### Basic GPU Check
```python
import cupy as cp
import cudf

# Check GPU
print(f"GPU Available: {cp.cuda.runtime.getDeviceCount() > 0}")
print(f"GPU Name: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")

# Test computation
x = cp.arange(1000000)
print(f"Sum of 1M numbers on GPU: {x.sum()}")
```

### RAPIDS Feature Test
```python
import cudf
import cuml
from cuml.datasets import make_regression

# Generate data
X, y = make_regression(n_samples=10000, n_features=20)

# Convert to GPU DataFrame
X_df = cudf.DataFrame(X)
y_series = cudf.Series(y)

# Train model
from cuml.linear_model import Ridge
model = Ridge()
model.fit(X_df, y_series)

print(f"Model trained on {len(X_df)} samples")
print(f"First 5 predictions: {model.predict(X_df[:5])}")
```

### Performance Benchmark
```python
import time
import pandas as pd
import cudf

# Create large dataset
n = 10_000_000
pdf = pd.DataFrame({
    'a': range(n),
    'b': range(n, 2*n),
    'c': range(2*n, 3*n)
})

# CPU timing
start = time.time()
pdf_result = pdf.groupby('a').agg({'b': 'sum', 'c': 'mean'})
cpu_time = time.time() - start

# GPU timing
gdf = cudf.from_pandas(pdf)
start = time.time()
gdf_result = gdf.groupby('a').agg({'b': 'sum', 'c': 'mean'})
gpu_time = time.time() - start

print(f"CPU Time: {cpu_time:.2f}s")
print(f"GPU Time: {gpu_time:.2f}s")
print(f"Speedup: {cpu_time/gpu_time:.2f}x")
```

---

## Quick Reference

### Check Versions
```bash
# System info
nvidia-smi
python --version
conda --version

# RAPIDS versions
python -c "import cudf; print(f'cuDF: {cudf.__version__}')"
python -c "import cuml; print(f'cuML: {cuml.__version__}')"
python -c "import cugraph; print(f'cuGraph: {cugraph.__version__}')"
```

### Useful Aliases
Add to `~/.bashrc`:
```bash
alias rapids-docker='docker run --rm -it --gpus all -p 8888:8888 -v "$PWD":/workspace rapidsai/notebooks:25.08-cuda12.5-py3.11'
alias gpu-check='nvidia-smi'
alias rapids-test='python -c "import cudf; print(cudf.Series([1,2,3]))"'
```

### Resource Links
- [RAPIDS Documentation](https://docs.rapids.ai/)
- [RAPIDS Release Selector](https://rapids.ai/start.html)
- [NVIDIA WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/)
- [RAPIDS GitHub](https://github.com/rapidsai)
- [RAPIDS Community](https://rapids.ai/community.html)