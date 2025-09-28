#!/bin/bash
# RAPIDS Environment Setup Helper
# This script helps set up RAPIDS using different methods

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Default values
CUDA_VERSION="12.5"
PYTHON_VERSION="3.11"
RAPIDS_VERSION="25.08"
ENV_NAME="rapids-${RAPIDS_VERSION}"

# Function to print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --method METHOD     Installation method: conda, pip, or docker (default: conda)"
    echo "  --cuda VERSION      CUDA version (default: $CUDA_VERSION)"
    echo "  --python VERSION    Python version (default: $PYTHON_VERSION)"
    echo "  --env-name NAME     Environment name (default: $ENV_NAME)"
    echo "  --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --method conda"
    echo "  $0 --method pip --cuda 12.0"
    echo "  $0 --method docker"
}

# Parse arguments
METHOD="conda"
while [[ $# -gt 0 ]]; do
    case $1 in
        --method)
            METHOD="$2"
            shift 2
            ;;
        --cuda)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Function to check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    # Check WSL
    if ! grep -qi microsoft /proc/version; then
        echo -e "${YELLOW}Warning: Not running in WSL${NC}"
    fi
    
    # Check GPU
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✓ GPU detected${NC}"
        nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
    else
        echo -e "${RED}✗ nvidia-smi not found${NC}"
    fi
}

# Conda installation
install_conda() {
    echo -e "${GREEN}Setting up RAPIDS with Conda/Mamba...${NC}"
    
    # Check if conda/mamba exists
    if ! command -v conda &> /dev/null && ! command -v mamba &> /dev/null; then
        echo -e "${YELLOW}Conda/Mamba not found. Installing Miniforge...${NC}"
        curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
        bash Miniforge3-$(uname)-$(uname -m).sh -b -p "$HOME/miniforge3"
        rm Miniforge3-$(uname)-$(uname -m).sh
        
        # Initialize
        "$HOME/miniforge3/bin/conda" init bash
        source "$HOME/.bashrc"
    fi
    
    # Use mamba if available
    if command -v mamba &> /dev/null; then
        CONDA_CMD="mamba"
    else
        CONDA_CMD="conda"
    fi
    
    # Create environment
    echo -e "${YELLOW}Creating environment: $ENV_NAME${NC}"
    
    if [ -f "environment.yml" ]; then
        echo "Using environment.yml file..."
        $CONDA_CMD env create -f environment.yml -n "$ENV_NAME" || true
    else
        echo "Creating environment from scratch..."
        $CONDA_CMD create -n "$ENV_NAME" -c rapidsai -c conda-forge -c nvidia \
            rapids="$RAPIDS_VERSION" python="$PYTHON_VERSION" cuda-version="$CUDA_VERSION" -y
    fi
    
    echo -e "${GREEN}✓ Conda environment created${NC}"
    echo ""
    echo "To activate the environment, run:"
    echo -e "${YELLOW}conda activate $ENV_NAME${NC}"
}

# Pip installation
install_pip() {
    echo -e "${GREEN}Setting up RAPIDS with pip...${NC}"
    
    # Create virtual environment
    echo -e "${YELLOW}Creating virtual environment: $ENV_NAME${NC}"
    python3 -m venv "$ENV_NAME"
    source "$ENV_NAME/bin/activate"
    
    # Upgrade pip
    python -m pip install --upgrade pip
    
    # Install CUDA toolkit if needed
    if ! command -v nvcc &> /dev/null; then
        echo -e "${YELLOW}CUDA toolkit not found. Please install it manually:${NC}"
        echo "sudo apt-get update"
        echo "sudo apt-get install -y cuda-toolkit-${CUDA_VERSION/./-}"
    fi
    
    # Install RAPIDS
    if [ -f "requirements.txt" ]; then
        echo "Using requirements.txt file..."
        pip install -r requirements.txt --extra-index-url=https://pypi.nvidia.com
    else
        echo "Installing RAPIDS packages..."
        CUDA_SUFFIX="cu${CUDA_VERSION//./}"
        pip install --extra-index-url=https://pypi.nvidia.com \
            "cudf-${CUDA_SUFFIX}>=25.08" \
            "cuml-${CUDA_SUFFIX}>=25.08" \
            "cugraph-${CUDA_SUFFIX}>=25.08"
    fi
    
    echo -e "${GREEN}✓ Pip environment created${NC}"
    echo ""
    echo "To activate the environment, run:"
    echo -e "${YELLOW}source $ENV_NAME/bin/activate${NC}"
}

# Docker installation
install_docker() {
    echo -e "${GREEN}Setting up RAPIDS with Docker...${NC}"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker not found. Please install Docker Desktop first.${NC}"
        exit 1
    fi
    
    # Test GPU access
    echo -e "${YELLOW}Testing Docker GPU access...${NC}"
    docker run --rm --gpus all nvidia/cuda:12.5.0-base-ubuntu22.04 nvidia-smi
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Docker cannot access GPU. Check Docker Desktop settings.${NC}"
        exit 1
    fi
    
    # Use docker-compose if available
    if [ -f "docker-compose.yml" ]; then
        echo "Using docker-compose.yml..."
        docker-compose up -d rapids-notebook
        echo -e "${GREEN}✓ RAPIDS container started${NC}"
        echo ""
        echo "Access JupyterLab at: http://localhost:8888"
        echo "To view logs: docker-compose logs rapids-notebook"
    else
        # Run standalone container
        echo "Starting RAPIDS container..."
        docker run -d --name rapids-jupyter \
            --gpus all \
            -p 8888:8888 \
            -v "$PWD:/workspace" \
            "rapidsai/notebooks:${RAPIDS_VERSION}-cuda${CUDA_VERSION}-py${PYTHON_VERSION}"
        
        echo -e "${GREEN}✓ RAPIDS container started${NC}"
        echo ""
        echo "Access JupyterLab at: http://localhost:8888"
        echo "To view logs: docker logs rapids-jupyter"
    fi
}

# Main execution
echo -e "${GREEN}RAPIDS Environment Setup${NC}"
echo "========================"

# Check prerequisites
check_prerequisites

# Install based on method
case $METHOD in
    conda)
        install_conda
        ;;
    pip)
        install_pip
        ;;
    docker)
        install_docker
        ;;
    *)
        echo -e "${RED}Invalid method: $METHOD${NC}"
        echo "Valid methods: conda, pip, docker"
        exit 1
        ;;
esac

# Run validation
echo ""
echo -e "${YELLOW}Running validation...${NC}"

if [ "$METHOD" != "docker" ]; then
    # For conda/pip, run validation in the environment
    if [ "$METHOD" = "conda" ]; then
        conda run -n "$ENV_NAME" python validate_rapids.py 2>/dev/null || true
    else
        "$ENV_NAME/bin/python" validate_rapids.py 2>/dev/null || true
    fi
else
    # For docker, suggest validation command
    echo "To validate Docker installation, run:"
    echo "docker exec rapids-jupyter python -c 'import cudf; print(cudf.__version__)'"
fi

echo ""
echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Activate your environment (see instructions above)"
echo "2. Run validation: python validate_rapids.py"
echo "3. Try examples: python rapids_examples.py"
echo "4. Check the runbook for detailed documentation"