# ==============================================================================
# Singapore Embodied AI Challenge (ECM 2027) - RoboMamba Policy Inference Stack
# Universal Container: Runs on CPU, Apple Silicon, RTX 30/40 Series, and RTX 5090
# ==============================================================================

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/root/.cache/torch

# 1. System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    wget \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Set python3.11 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# 2. Install PyTorch with Blackwell (sm_120) & legacy CUDA/CPU support
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel
RUN python -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. Install Policy, Vision, & Robotics Dependencies
RUN python -m pip install --no-cache-dir \
    pyyaml>=6.0 \
    numpy>=1.24.0 \
    timm>=1.0.7 \
    einops>=0.8.0 \
    matplotlib>=3.8.0 \
    tqdm>=4.66.0 \
    h5py>=3.11.0 \
    opencv-python-headless>=4.10.0 \
    huggingface_hub>=0.24.0 \
    datasets>=2.20.0 \
    lerobot \
    wandb

# 4. Pre-bake DINOv2 ViT-B14 backbone into the image cache for instant, offline loading
RUN python -c "import torch; print('Pre-caching DINOv2 backbone...'); torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')"

# 5. Set up working directory & copy repository
WORKDIR /workspace/RoboMamba_Policy_Inference
COPY . /workspace/RoboMamba_Policy_Inference

# Make entrypoint executable
RUN chmod +x /workspace/RoboMamba_Policy_Inference/entrypoint.sh || true

ENTRYPOINT ["/bin/bash", "/workspace/RoboMamba_Policy_Inference/entrypoint.sh"]
CMD ["test"]
