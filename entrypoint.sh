#!/bin/bash
set -e

echo "===================================================================="
echo "  Singapore Embodied AI Challenge (ECM 2027) - RoboMamba Policy Env "
echo "===================================================================="

# Check Device
python -c "
import torch
if torch.cuda.is_available():
    print(f'✓ Hardware: NVIDIA GPU Detected ({torch.cuda.get_device_name(0)})')
    print(f'✓ Device Capability: {torch.cuda.get_device_capability(0)}')
else:
    print('ℹ Hardware: Running on CPU (CPU Inference Mode)')
"

echo "--------------------------------------------------------------------"

# Handle Checkpoint
CKPT="checkpoints/ecm2027_rtx5090/best_model.pt"
if [ -f "$CKPT" ]; then
    echo "✓ Found trained checkpoint at: $CKPT"
elif [ -f "best_model.pt" ]; then
    CKPT="best_model.pt"
    echo "✓ Found trained checkpoint at: $CKPT"
else
    echo "ℹ Checkpoint '$CKPT' not found. Using initialized weights for sanity inference."
fi

echo "===================================================================="

if [ "$1" = "test" ] || [ -z "$1" ]; then
    echo "Running RoboMamba-DINO-ACT Sanity Inference Benchmark..."
    exec python test_inference.py --config config_rtx5090.yaml --checkpoint "$CKPT"
elif [ "$1" = "bash" ] || [ "$1" = "sh" ]; then
    exec /bin/bash
else
    exec "$@"
fi
