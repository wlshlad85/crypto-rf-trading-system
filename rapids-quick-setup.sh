#!/bin/bash
# RAPIDS WSL2 Quick Setup Script
# This script provides quick commands for RAPIDS setup and testing

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to check GPU availability
check_gpu() {
    print_status "Checking GPU availability..."
    
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
        print_status "GPU detected successfully"
    elif command -v nvidia-smi.exe &> /dev/null; then
        nvidia-smi.exe --query-gpu=name,driver_version,memory.total --format=csv,noheader
        print_status "GPU detected via nvidia-smi.exe"
    else
        print_error "nvidia-smi not found. Please ensure NVIDIA drivers are installed."
        return 1
    fi
}

# Function to test Docker GPU access
test_docker_gpu() {
    print_status "Testing Docker GPU access..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker not found. Please install Docker Desktop first."
        return 1
    fi
    
    docker run --rm --gpus all nvidia/cuda:12.5.0-base-ubuntu22.04 nvidia-smi
    
    if [ $? -eq 0 ]; then
        print_status "Docker can access GPU successfully"
    else
        print_error "Docker cannot access GPU. Check Docker Desktop WSL2 settings."
        return 1
    fi
}

# Function to install Miniforge
install_miniforge() {
    print_status "Installing Miniforge..."
    
    if [ -d "$HOME/miniforge3" ]; then
        print_warning "Miniforge already installed at $HOME/miniforge3"
        return 0
    fi
    
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh -b
    rm Miniforge3-$(uname)-$(uname -m).sh
    
    # Initialize conda
    $HOME/miniforge3/bin/conda init bash
    
    print_status "Miniforge installed. Please run 'exec $SHELL' to reload shell."
}

# Function to create RAPIDS conda environment
create_rapids_env() {
    local env_name="${1:-rapids-25.08}"
    local cuda_version="${2:-12.5}"
    local python_version="${3:-3.11}"
    
    print_status "Creating RAPIDS environment: $env_name"
    
    if ! command -v mamba &> /dev/null; then
        print_error "Mamba not found. Please install Miniforge first."
        return 1
    fi
    
    mamba create -n "$env_name" -c rapidsai -c conda-forge -c nvidia \
        rapids=25.08 python="$python_version" cuda-version="$cuda_version" -y
    
    print_status "RAPIDS environment created: $env_name"
    print_status "Activate with: conda activate $env_name"
}

# Function to run RAPIDS Docker container
run_rapids_docker() {
    local port="${1:-8888}"
    local tag="${2:-25.08-cuda12.5-py3.11}"
    
    print_status "Starting RAPIDS Docker container..."
    print_status "JupyterLab will be available at http://localhost:$port"
    
    docker run --rm -it --gpus all \
        -p "$port:8888" \
        -v "$PWD":/workspace \
        rapidsai/notebooks:"$tag"
}

# Function to test RAPIDS installation
test_rapids() {
    print_status "Testing RAPIDS installation..."
    
    python3 - <<'EOF'
try:
    import cudf
    import cuml
    import cugraph
    
    # Test cuDF
    df = cudf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    print("✓ cuDF working:", cudf.__version__)
    
    # Test cuML
    from cuml.linear_model import LinearRegression
    model = LinearRegression()
    print("✓ cuML working:", cuml.__version__)
    
    # Test cuGraph
    print("✓ cuGraph working:", cugraph.__version__)
    
    # Show GPU memory
    print(f"\nGPU Memory Used: {df.memory_usage(deep=True).sum()} bytes")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Please ensure RAPIDS is installed in your current environment")
except Exception as e:
    print(f"✗ Error: {e}")
EOF
}

# Function to benchmark CPU vs GPU
benchmark_rapids() {
    print_status "Running RAPIDS benchmark (CPU vs GPU)..."
    
    python3 - <<'EOF'
import time
try:
    import pandas as pd
    import cudf
    
    # Create test data
    n = 5_000_000
    print(f"Creating dataset with {n:,} rows...")
    
    pdf = pd.DataFrame({
        'key': [i % 1000 for i in range(n)],
        'value1': range(n),
        'value2': range(n, 2*n)
    })
    
    # CPU benchmark
    start = time.time()
    cpu_result = pdf.groupby('key').agg({'value1': 'sum', 'value2': 'mean'})
    cpu_time = time.time() - start
    
    # GPU benchmark
    gdf = cudf.from_pandas(pdf)
    start = time.time()
    gpu_result = gdf.groupby('key').agg({'value1': 'sum', 'value2': 'mean'})
    gpu_time = time.time() - start
    
    print(f"\nResults:")
    print(f"CPU Time: {cpu_time:.3f}s")
    print(f"GPU Time: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time/gpu_time:.1f}x")
    
except ImportError:
    print("Please install RAPIDS first (cudf required)")
except Exception as e:
    print(f"Error: {e}")
EOF
}

# Main menu
show_menu() {
    echo ""
    echo "RAPIDS WSL2 Quick Setup Menu"
    echo "============================"
    echo "1) Check GPU availability"
    echo "2) Test Docker GPU access"
    echo "3) Install Miniforge"
    echo "4) Create RAPIDS Conda environment"
    echo "5) Run RAPIDS Docker container"
    echo "6) Test RAPIDS installation"
    echo "7) Benchmark RAPIDS (CPU vs GPU)"
    echo "8) Show all system info"
    echo "9) Exit"
    echo ""
}

# Function to show all system info
show_system_info() {
    print_status "System Information"
    echo "=================="
    
    echo -e "\n${GREEN}OS Info:${NC}"
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        echo "Distribution: $NAME $VERSION"
    fi
    uname -r
    
    echo -e "\n${GREEN}WSL Info:${NC}"
    if [ -f /proc/version ]; then
        grep -i microsoft /proc/version || echo "Not running in WSL"
    fi
    
    echo -e "\n${GREEN}GPU Info:${NC}"
    check_gpu || true
    
    echo -e "\n${GREEN}CUDA Info:${NC}"
    if command -v nvcc &> /dev/null; then
        nvcc --version | grep release
    else
        echo "CUDA toolkit not found in PATH"
    fi
    
    echo -e "\n${GREEN}Python Info:${NC}"
    python3 --version 2>/dev/null || echo "Python3 not found"
    
    echo -e "\n${GREEN}Conda Info:${NC}"
    if command -v conda &> /dev/null; then
        conda --version
        echo "Active environment: ${CONDA_DEFAULT_ENV:-base}"
    else
        echo "Conda not found"
    fi
    
    echo -e "\n${GREEN}Docker Info:${NC}"
    if command -v docker &> /dev/null; then
        docker --version
    else
        echo "Docker not found"
    fi
}

# Parse command line arguments
case "$1" in
    --check-gpu)
        check_gpu
        ;;
    --test-docker)
        test_docker_gpu
        ;;
    --install-miniforge)
        install_miniforge
        ;;
    --create-env)
        create_rapids_env "$2" "$3" "$4"
        ;;
    --run-docker)
        run_rapids_docker "$2" "$3"
        ;;
    --test-rapids)
        test_rapids
        ;;
    --benchmark)
        benchmark_rapids
        ;;
    --info)
        show_system_info
        ;;
    *)
        # Interactive menu
        while true; do
            show_menu
            read -p "Select option: " choice
            
            case $choice in
                1) check_gpu ;;
                2) test_docker_gpu ;;
                3) install_miniforge ;;
                4) create_rapids_env ;;
                5) run_rapids_docker ;;
                6) test_rapids ;;
                7) benchmark_rapids ;;
                8) show_system_info ;;
                9) exit 0 ;;
                *) print_error "Invalid option" ;;
            esac
            
            echo ""
            read -p "Press Enter to continue..."
        done
        ;;
esac