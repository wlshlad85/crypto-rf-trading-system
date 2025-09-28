# RAPIDS on WSL2 - Quick Reference

## 🚀 Quick Commands

### Check System
```bash
# GPU check
nvidia-smi

# RAPIDS validation
python validate_rapids.py

# Interactive setup
./rapids-quick-setup.sh
```

### Docker Quick Start
```bash
# Using docker-compose
docker-compose up rapids-notebook

# Manual run
docker run --rm -it --gpus all -p 8888:8888 \
  rapidsai/notebooks:25.08-cuda12.5-py3.11

# Test GPU in Docker
docker run --rm --gpus all nvidia/cuda:12.5.0-base-ubuntu22.04 nvidia-smi
```

### Conda Quick Start
```bash
# Create environment from file
mamba env create -f environment.yml

# Or create manually
mamba create -n rapids -c rapidsai -c conda-forge -c nvidia \
  rapids=25.08 python=3.11 cuda-version=12.5

# Activate
conda activate rapids
```

### Pip Quick Start
```bash
# Create virtual environment
python3 -m venv rapids-env
source rapids-env/bin/activate

# Install RAPIDS
pip install -r requirements.txt --extra-index-url=https://pypi.nvidia.com
```

## 📊 Common RAPIDS Operations

### cuDF (GPU DataFrames)
```python
import cudf

# Create DataFrame
df = cudf.DataFrame({'a': [1,2,3], 'b': [4,5,6]})

# Read CSV
df = cudf.read_csv('file.csv')

# Operations
df['c'] = df['a'] + df['b']
grouped = df.groupby('a').mean()

# Convert to/from pandas
pdf = df.to_pandas()
gdf = cudf.from_pandas(pdf)
```

### cuML (GPU Machine Learning)
```python
from cuml.linear_model import LinearRegression
from cuml.cluster import KMeans
from cuml.datasets import make_regression

# Generate data
X, y = make_regression(n_samples=1000)

# Train model
model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)

# Clustering
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(X)
```

### Memory Management
```python
import rmm

# Set memory pool
rmm.mr.set_current_device_resource(
    rmm.mr.PoolMemoryResource(
        rmm.mr.CudaMemoryResource(),
        initial_pool_size=2**30  # 1GB
    )
)

# Check GPU memory
import cupy as cp
mempool = cp.get_default_memory_pool()
print(f"Used: {mempool.used_bytes()/1e9:.2f} GB")
```

## 🔧 Troubleshooting Commands

### WSL Issues
```bash
# Restart WSL
wsl --shutdown
wsl

# Update WSL
wsl --update

# Check WSL version
wsl -l -v
```

### GPU Issues
```bash
# Check driver version
nvidia-smi --query-gpu=driver_version --format=csv,noheader

# Check CUDA version
nvcc --version

# Test GPU in Python
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
```

### Docker Issues
```bash
# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.5.0-base-ubuntu22.04 nvidia-smi

# Clean up Docker
docker system prune -a

# View container logs
docker logs rapids-jupyter
```

### Conda Issues
```bash
# Update conda/mamba
conda update -n base conda
mamba update -n base mamba

# Clean conda cache
conda clean --all

# List environments
conda env list

# Remove environment
conda env remove -n rapids
```

## 📈 Performance Tips

### 1. Use Column Operations
```python
# Good - vectorized
df['new'] = df['a'] + df['b']

# Avoid - iterative
for i in range(len(df)):
    df.loc[i, 'new'] = df.loc[i, 'a'] + df.loc[i, 'b']
```

### 2. Minimize Host-Device Transfers
```python
# Keep data on GPU
gdf1 = cudf.read_csv('file1.csv')
gdf2 = cudf.read_csv('file2.csv')
result = gdf1.merge(gdf2)  # All on GPU

# Avoid unnecessary conversions
# Bad: GPU -> CPU -> GPU
pdf = gdf.to_pandas()
# ... some pandas operation
gdf = cudf.from_pandas(pdf)
```

### 3. Use Appropriate Data Types
```python
# Use smaller dtypes when possible
df['id'] = df['id'].astype('int32')  # Instead of int64
df['flag'] = df['flag'].astype('int8')  # For boolean/small values
```

## 🔍 Useful Validation Tests

```python
# Quick RAPIDS test
import cudf
print(f"cuDF version: {cudf.__version__}")
df = cudf.DataFrame({'a': [1,2,3]})
print(f"Sum: {df['a'].sum()}")

# Check all RAPIDS packages
for pkg in ['cudf', 'cuml', 'cugraph', 'cuspatial']:
    try:
        mod = __import__(pkg)
        print(f"✓ {pkg}: {mod.__version__}")
    except ImportError:
        print(f"✗ {pkg}: not installed")

# Memory info
import cupy as cp
free, total = cp.cuda.MemoryPool().free_bytes(), cp.cuda.MemoryPool().total_bytes()
print(f"GPU Memory: {free/1e9:.1f}/{total/1e9:.1f} GB free")
```

## 📚 Key Resources

- **RAPIDS Docs**: https://docs.rapids.ai/
- **Release Selector**: https://rapids.ai/start.html
- **API Reference**: https://docs.rapids.ai/api/
- **Examples**: https://github.com/rapidsai/notebooks
- **Community**: https://rapids.ai/community.html

---

**Remember**: 
- Windows driver only (no Linux driver in WSL)
- Single GPU only in WSL2
- Use Release Selector for compatible versions
- Docker is fastest to get started