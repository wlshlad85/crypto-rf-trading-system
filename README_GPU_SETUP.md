# GPU-Accelerated ML/AI Environment for RTX 5070 Ti

## 🎯 Overview

This environment provides a complete GPU-accelerated setup for machine learning and AI development using your **NVIDIA GeForce RTX 5070 Ti**. The setup includes:

- **CUDA 12.9** toolkit
- **PyTorch 2.8.0** with CUDA 12.x support
- **CuPy** for GPU-accelerated NumPy operations
- **XGBoost** and **LightGBM** with GPU support
- **Transformers**, **Diffusers**, and **Accelerate** for AI workloads
- **Jupyter Notebook** for interactive development

## 🚀 Quick Start

### 1. Activate the Environment
```bash
source activate_gpu_env.sh
```

### 2. Test Your GPU Setup
```bash
python gpu_test.py
```

### 3. Start Jupyter Notebook
```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

## 📦 Installed Packages

### Core ML/AI Frameworks
- **PyTorch**: 2.8.0+cu128 (with CUDA support)
- **CuPy**: 13.6.0 (GPU-accelerated NumPy)
- **NumPy**: 2.3.3
- **Pandas**: 2.3.2
- **Scikit-learn**: 1.7.2

### GPU-Accelerated Libraries
- **XGBoost**: 3.0.5 (with GPU support)
- **LightGBM**: 4.6.0 (with GPU support)

### Deep Learning & AI
- **Transformers**: 4.56.2 (Hugging Face)
- **Diffusers**: 0.35.1 (Stable Diffusion)
- **Accelerate**: 1.10.1 (Optimized training)

### Development Tools
- **Jupyter**: 1.1.1 (Notebook & Lab)
- **Matplotlib**: 3.10.6
- **Seaborn**: 0.13.2

### CUDA Components
- **CUDA Toolkit**: 12.9
- **cuDNN**: 9.10.2.21
- **NCCL**: 2.27.3 (Multi-GPU communication)

## 🔧 System Requirements

- **GPU**: NVIDIA GeForce RTX 5070 Ti (or compatible)
- **Driver**: NVIDIA Driver 560+ (recommended)
- **CUDA**: 12.x compatible
- **Python**: 3.13.3
- **OS**: Linux (Ubuntu 22.04+)

## 🧪 Testing GPU Acceleration

The `gpu_test.py` script will verify:

1. **PyTorch GPU Detection**
   - CUDA availability
   - GPU memory and compute capability
   - Matrix multiplication performance

2. **CuPy GPU Operations**
   - Basic array operations
   - Large matrix computations
   - Memory usage monitoring

3. **ML Libraries GPU Support**
   - XGBoost GPU training
   - LightGBM GPU training

4. **Transformers GPU Inference**
   - Model loading and inference
   - Performance benchmarking

## 💡 Usage Examples

### PyTorch GPU Operations
```python
import torch

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

# Create tensors on GPU
device = torch.device('cuda:0')
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)

# GPU matrix multiplication
z = torch.matmul(x, y)
```

### CuPy GPU Operations
```python
import cupy as cp
import numpy as np

# Create GPU arrays
x_gpu = cp.array([1, 2, 3, 4, 5])
y_gpu = cp.array([6, 7, 8, 9, 10])

# GPU computation
result = cp.dot(x_gpu, y_gpu)
print(result)
```

### XGBoost GPU Training
```python
import xgboost as xgb
import numpy as np

# Create sample data
X = np.random.rand(1000, 10)
y = np.random.rand(1000)

# GPU training parameters
params = {
    'tree_method': 'gpu_hist',
    'gpu_id': 0,
    'objective': 'reg:squarederror'
}

dtrain = xgb.DMatrix(X, label=y)
model = xgb.train(params, dtrain, num_boost_round=100)
```

### Transformers with GPU
```python
import torch
from transformers import AutoTokenizer, AutoModel

# Load model and move to GPU
device = torch.device('cuda:0')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModel.from_pretrained('distilbert-base-uncased')
model = model.to(device)

# GPU inference
text = "Your text here"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
```

## 🐛 Troubleshooting

### CUDA Not Available
- Verify NVIDIA drivers are installed: `nvidia-smi`
- Check CUDA installation: `nvcc --version`
- Ensure environment variables are set correctly

### Out of Memory Errors
- Reduce batch size
- Use gradient accumulation
- Clear GPU cache: `torch.cuda.empty_cache()`

### Performance Issues
- Ensure GPU is not being used by other processes
- Check GPU utilization: `nvidia-smi`
- Verify CUDA compute capability compatibility

## 📚 Additional Resources

- [PyTorch CUDA Documentation](https://pytorch.org/docs/stable/cuda.html)
- [CuPy User Guide](https://docs.cupy.dev/en/stable/user_guide/)
- [XGBoost GPU Support](https://xgboost.readthedocs.io/en/stable/gpu/index.html)
- [Transformers GPU Documentation](https://huggingface.co/docs/transformers/accelerate)

## 🎉 Ready to Go!

Your RTX 5070 Ti is now configured for high-performance ML/AI workloads. The environment supports:

- ✅ Deep learning model training and inference
- ✅ GPU-accelerated data processing
- ✅ Large-scale machine learning experiments
- ✅ AI model development and deployment

Happy coding! 🚀