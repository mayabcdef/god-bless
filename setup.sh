#!/bin/bash
set -e  # Exit on error
apt update && apt install sudo -y

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting local environment setup...${NC}"

# Check if running on Ubuntu/Debian
if ! command -v apt-get &> /dev/null; then
    echo -e "${RED}Error: This script requires apt-get (Ubuntu/Debian).${NC}"
    exit 1
fi

# Set environment variables
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1

# CUDA arch list for torch (8.0 A100 - 8.6 A40, A6000 - 8.9 6000 ADA - 12.0 6000 PRO BW)
export TORCH_CUDA_ARCH_LIST="8.6;8.0;8.9;12.0"

# Update package lists
echo -e "${YELLOW}Updating package lists...${NC}"
sudo apt update -y

# Install python3-apt first (required for add-apt-repository)
echo -e "${YELLOW}Installing python3-apt...${NC}"
sudo apt-get install -y python3-apt

# Install software-properties-common for add-apt-repository
echo -e "${YELLOW}Installing software-properties-common...${NC}"
sudo apt-get install -y software-properties-common

# Check if Python 3.11 is already available
if command -v python3.11 &> /dev/null; then
    echo -e "${GREEN}Python 3.11 is already installed. Skipping PPA addition.${NC}"
else
    # Add deadsnakes PPA for Python 3.11
    echo -e "${YELLOW}Adding deadsnakes PPA for Python 3.11...${NC}"
    sudo add-apt-repository ppa:deadsnakes/ppa -y || {
        echo -e "${YELLOW}Warning: Could not add PPA. Trying alternative method...${NC}"
        # Alternative: manually add the PPA
        sudo apt-get install -y lsb-release
        sudo sh -c 'echo "deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/deadsnakes-ppa.list'
        sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F23C5A6CF475977595C89F51BA6932366A755776 || true
    }
    sudo apt update -y
fi

# Install wget and gnupg
echo -e "${YELLOW}Installing wget and gnupg...${NC}"
sudo apt install -y wget gnupg

# Install CUDA keyring
echo -e "${YELLOW}Setting up CUDA repository...${NC}"
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb

# Install basic system dependencies
echo -e "${YELLOW}Installing system dependencies...${NC}"
sudo apt-get update && sudo apt-get install -y \
    wget \
    unzip \
    git \
    build-essential \
    cmake \
    ninja-build \
    python3.11 \
    python3-pip \
    python3-wheel \
    python3.11-dev \
    python3.11-distutils \
    libjpeg-dev \
    libpng-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    bash \
    cuda-toolkit-12-8 \
    cuda-nvcc-12-8 \
    cuda-libraries-dev-12-8 \
    cuda-nvtx-12-8

# Set up Python 3.11 as default (optional, creates alternatives)
echo -e "${YELLOW}Setting up Python 3.11...${NC}"
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 120 || true
sudo update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3 120 || true

# Create symlinks for python and pip if they don't exist
if [ ! -f /usr/bin/python ]; then
    sudo ln -s /usr/bin/python3.11 /usr/bin/python || true
fi
if [ ! -f /usr/bin/pip ]; then
    sudo ln -s /usr/bin/pip3 /usr/bin/pip || true
fi

# Set up pip for Python 3.11
python3.11 -m pip install --upgrade pip setuptools wheel

# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

# Add to bashrc for persistence (optional)
if ! grep -q "CUDA_HOME=/usr/local/cuda-12.8" ~/.bashrc 2>/dev/null; then
    echo "" >> ~/.bashrc
    echo "# CUDA Environment Variables" >> ~/.bashrc
    echo "export CUDA_HOME=/usr/local/cuda-12.8" >> ~/.bashrc
    echo "export PATH=\$CUDA_HOME/bin:\$PATH" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
    echo "export TORCH_CUDA_ARCH_LIST=\"8.6;8.0;8.9;12.0\"" >> ~/.bashrc
fi

# Install basic dependencies from requirements.txt
if [ -f "docker/requirements.txt" ]; then
    echo -e "${YELLOW}Installing Python packages from requirements.txt...${NC}"
    python3.11 -m pip install -r docker/requirements.txt
else
    echo -e "${RED}Warning: docker/requirements.txt not found. Skipping requirements installation.${NC}"
fi

# Install flash-attn 2.8.3 prebuilt wheel for PyTorch 2.7 + Python 3.11 + CUDA 12.8
echo -e "${YELLOW}Installing flash-attn...${NC}"
python3.11 -m pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# Install SPZ
echo -e "${YELLOW}Installing SPZ...${NC}"
wget https://github.com/404-Repo/spz/releases/download/v.0.1.2/pyspz-wheels-ubuntu-latest.zip
unzip pyspz-wheels-ubuntu-latest.zip
python3.11 -m pip install pyspz-1.0.0-cp311-cp311-manylinux_2_34_x86_64.whl
rm -rf pyspz-wheels-ubuntu-latest.zip *.whl

# Install ben2
echo -e "${YELLOW}Installing ben2...${NC}"
python3.11 -m pip install -e "git+https://github.com/PramaLLC/BEN2.git#egg=ben2"

echo -e "${GREEN}Setup complete!${NC}"
echo -e "${YELLOW}Note: You may need to run 'source ~/.bashrc' or restart your terminal for environment variables to take effect.${NC}"
echo -e "${YELLOW}Note: Make sure CUDA 12.8 is properly installed and accessible.${NC}"