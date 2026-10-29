#!/usr/bin/env bash
set -e

# ==============================================================================
# 1-Click Docker Runner for RoboMamba Policy Inference
# Automatically detects NVIDIA GPU or runs in CPU mode
# ==============================================================================

IMAGE_NAME="robomamba-inference:latest"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Build image if it doesn't exist
if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
    echo "Image '$IMAGE_NAME' not found locally. Building Docker container..."
    docker build -t "$IMAGE_NAME" -f Dockerfile .
else
    echo "✓ Docker image '$IMAGE_NAME' ready."
fi

# 2. Check for NVIDIA GPU support
GPU_FLAG=""
if command -v nvidia-smi &> /dev/null && docker run --help | grep -q "\-\-gpus"; then
    if docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 echo "GPU OK" &> /dev/null; then
        echo "✓ NVIDIA GPU runtime detected! Enabling GPU acceleration (--gpus all)..."
        GPU_FLAG="--gpus all"
    else
        echo "ℹ NVIDIA driver found, but NVIDIA container toolkit not configured. Falling back to CPU..."
    fi
else
    echo "ℹ Running in CPU mode..."
fi

# 3. Run container
echo "Starting RoboMamba Inference Container..."
docker run --rm -it \
    $GPU_FLAG \
    -v "$SCRIPT_DIR/checkpoints":/workspace/RoboMamba_Policy_Inference/checkpoints \
    "$IMAGE_NAME" "$@"
