#!/bin/bash
# GPU-Accelerated Environment Activation Script

echo "🚀 Activating GPU-Accelerated Environment for RTX 5070 Ti"
echo "=================================================="

# Set CUDA environment variables
export PATH=/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.9

# Activate virtual environment
source gpu_env/bin/activate

echo "✅ Environment activated!"
echo "📋 Available tools:"
echo "  - Python: $(python --version)"
echo "  - PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not available')"
echo "  - CUDA: $(nvcc --version 2>/dev/null | grep release || echo 'Not available')"
echo "  - CuPy: $(python -c 'import cupy; print(cupy.__version__)' 2>/dev/null || echo 'Not available')"

echo ""
echo "🧪 To test your GPU setup, run:"
echo "  python gpu_test.py"
echo ""
echo "📝 To start Jupyter notebook:"
echo "  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
echo ""
echo "💡 Environment variables set:"
echo "  - PATH: /usr/local/cuda-12.9/bin added"
echo "  - LD_LIBRARY_PATH: /usr/local/cuda-12.9/lib64 added"
echo "  - CUDA_HOME: /usr/local/cuda-12.9"