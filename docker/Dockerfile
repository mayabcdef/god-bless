FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.6;8.0;8.9;9.0;12.0"

# --- ENV CHO VLLM ---
ENV VLLM_GPU_MEMORY_UTILIZATION=0.275
ENV VLLM_PORT=8095
ENV VLLM_MODEL="THUDM/GLM-4.1V-9B-Thinking"
ENV VLLM_API_KEY="local"

# --- 1. SYSTEM DEPENDENCIES ---
# Gộp các lệnh apt để giảm layer và cache tốt hơn
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    curl \
    unzip \
    git \
    gnupg \
    build-essential \
    cmake \
    ninja-build \
    libjpeg-dev \
    libpng-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    bash \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt CUDA Toolkit (Nếu base image chưa đủ tool để build)
# LƯU Ý: CUDA 12.8 yêu cầu GPU driver >= 550.54.15
# cuda-compat-12-8 giúp hỗ trợ PTX compatibility với driver cũ hơn
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && apt-get install -y \
    cuda-toolkit-12-8 \
    cuda-nvcc-12-8 \
    cuda-libraries-dev-12-8 \
    cuda-nvtx-12-8 \
    cuda-compat-12-8 \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập Python 3.11 làm mặc định
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Thiết lập CUDA Env
# Thêm cuda-compat vào LD_LIBRARY_PATH để hỗ trợ PTX compatibility
ENV LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda-12.8
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/compat:$LD_LIBRARY_PATH

# --- 2. BASE ENVIRONMENT (Cho Trellis, SPZ, BEN2) ---
# Cài đặt pip mới nhất cho system python
RUN python -m ensurepip --upgrade && python -m pip install --upgrade pip

# Copy requirements và cài đặt vào SYSTEM python
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Cài đặt gsplat với TORCH_CUDA_ARCH_LIST để support H200 (compute capability 9.0)
# QUAN TRỌNG: gsplat compile từ source, cần set architecture list để build kernels cho 9.0
RUN TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0" pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3
RUN pip install bitsandbytes


# Install specific flash-attn wheel (Cho Base Env nếu Trellis cần)
# Nếu vLLM cần flash-attn, nó sẽ được cài riêng trong venv ở dưới
# RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# Install SPZ (System Python)
RUN wget https://github.com/404-Repo/spz/releases/download/v.0.1.2/pyspz-wheels-ubuntu-latest.zip && \
    unzip pyspz-wheels-ubuntu-latest.zip && \
    pip install pyspz-1.0.0-cp311-cp311-manylinux_2_34_x86_64.whl && \
    rm -rf pyspz-wheels-ubuntu-latest.zip *.whl

# Install ben2 (System Python)
RUN pip install -e "git+https://github.com/PramaLLC/BEN2.git#egg=ben2"

# --- 3. VLLM ENVIRONMENT (ISOLATED) ---
# Tạo môi trường ảo tại /opt/vllm-env
RUN python3.11 -m venv /opt/vllm-env

# Cài đặt PyTorch trước (cùng version với base image)
# Điều này đảm bảo vLLM sử dụng PyTorch version tương thích
RUN /opt/vllm-env/bin/pip install --upgrade pip
RUN /opt/vllm-env/bin/pip install hf_transfer

# Cài đặt vLLM và OpenAI vào trong VENV
# QUAN TRỌNG: Set TORCH_CUDA_ARCH_LIST để đảm bảo nếu vLLM compile gì đó, nó sẽ include 9.0 (H200)
# Sử dụng latest vLLM version để có support tốt hơn cho H200
RUN TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0" /opt/vllm-env/bin/pip install vllm==0.10.2 --extra-index-url https://download.pytorch.org/whl/cu128  openai==2.9.0

# --- 4. SETUP WORKSPACE ---
WORKDIR /workspace

# Copy toàn bộ project
COPY pipeline_service /workspace

# Copy start.sh (nếu nó nằm ngoài folder pipeline_service trong context build)
# Giả sử file start.sh nằm ở root context
#COPY start.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh

EXPOSE 10006 8095

CMD ["./start.sh"]