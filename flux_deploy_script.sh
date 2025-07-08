#!/bin/bash

# Simple FLUX.1-Kontext Deployment Script
# Installs conda, creates environment, clones repo, installs dependencies

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Install conda if not present
install_conda() {
    if command -v conda &> /dev/null; then
        print_success "Conda already installed"
        return 0
    fi
    
    print_status "Installing Miniconda..."
    cd /tmp
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda
    rm Miniconda3-latest-Linux-x86_64.sh
    
    export PATH="/root/miniconda/bin:$PATH"
    echo 'export PATH="/root/miniconda/bin:$PATH"' >> /root/.bashrc
    
    /root/miniconda/bin/conda init bash
    source /root/.bashrc
    
    print_success "Miniconda installed"
}

# Setup conda environment
setup_environment() {
    print_status "Setting up conda environment..."
    
    # Initialize conda
    source /root/miniconda/etc/profile.d/conda.sh
    
    # Remove existing environment if it exists
    if conda env list | grep -q "flux-kontext"; then
        print_warning "Removing existing flux-kontext environment..."
        conda env remove -n flux-kontext -y
    fi
    
    # Create new environment
    conda create -n flux-kontext python=3.10 -y
    conda activate flux-kontext
    
    print_success "Environment created and activated"
}

# Clone repository
clone_repo() {
    print_status "Cloning repository..."
    
    if [[ -d "/root/FLUX.1-Kontext-multi-image" ]]; then
        print_warning "Repository exists, updating..."
        cd /root/FLUX.1-Kontext-multi-image
        git pull
    else
        cd /root
        git clone https://github.com/nexusjuan12/FLUX.1-Kontext-multi-image.git
    fi
    
    print_success "Repository ready"
}

# Install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    
    # Activate environment
    source /root/miniconda/etc/profile.d/conda.sh
    conda activate flux-kontext
    
    # Install PyTorch with CUDA
    if command -v nvidia-smi &> /dev/null; then
        print_status "Installing PyTorch with CUDA support..."
        conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y
    else
        print_warning "No NVIDIA GPU detected, installing CPU-only PyTorch..."
        conda install pytorch cpuonly -c pytorch -y
    fi
    
    # Install core packages via conda
    conda install -c conda-forge numpy pillow requests opencv -y
    
    # Install Python packages via pip
    pip install \
        diffusers>=0.30.0 \
        transformers>=4.44.0 \
        accelerate>=0.33.0 \
        sentencepiece>=0.2.0 \
        gradio>=4.0.0 \
        huggingface_hub>=0.24.0 \
        safetensors>=0.4.0 \
        protobuf>=3.19.0 \
        spaces \
        scipy \
        matplotlib \
        tqdm
    
    # Install latest diffusers with GGUF support
    pip install git+https://github.com/huggingface/diffusers.git
    
    print_success "Dependencies installed"
}

# Main execution
main() {
    echo "=========================================="
    echo "FLUX.1-Kontext Simple Deployment"
    echo "=========================================="
    
    install_conda
    setup_environment
    clone_repo
    install_dependencies
    
    print_success "Deployment completed!"
    echo ""
    echo "To use:"
    echo "1. conda activate flux-kontext"
    echo "2. cd /root/FLUX.1-Kontext-multi-image"
    echo "3. python gguf_flux_app.py"
}

main "$@"