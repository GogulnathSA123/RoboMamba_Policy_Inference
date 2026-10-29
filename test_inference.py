"""
Quick Sanity Inference Test for DINOv2 + RoboMamba + ACT Policy
Tests model loading, forward inference speed, and action output shapes for the Singapore ECM 2027 setup.
"""

import os
import time
import yaml
import argparse
import torch
from models.policy import DINOv2RoboMambaACTPolicy

def test_inference(config_path="config_rtx5090.yaml", checkpoint_path="checkpoints/ecm2027_rtx5090/best_model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Testing RoboMamba-DINO-ACT Inference on {device} ===")

    # 1. Load config
    if not os.path.exists(config_path):
        config_path = "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    print(f"Dataset Target: {cfg['dataset']['repo_id']}")
    print(f"Cameras ({len(cfg['dataset']['cameras'])}): {cfg['dataset']['cameras']}")
    print(f"State Dim: {cfg['model']['state_dim']}, Action Dim: {cfg['model']['action_dim']}, Chunk Size: {cfg['model']['chunk_size']}")

    # 2. Build model
    model = DINOv2RoboMambaACTPolicy(
        dino_model=cfg["model"]["dino_backbone"],
        freeze_dino=cfg["model"]["freeze_dino"],
        d_model=cfg["model"]["d_model"],
        n_mamba_layers=cfg["model"]["n_mamba_layers"],
        state_dim=cfg["model"]["state_dim"],
        action_dim=cfg["model"]["action_dim"],
        chunk_size=cfg["model"]["chunk_size"],
        latent_dim=cfg["model"]["latent_dim"],
    ).to(device)

    # 3. Load checkpoint if available
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint weights from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print("✓ Checkpoint successfully loaded!")
    else:
        print(f"Notice: Checkpoint '{checkpoint_path}' not found yet. Running test with initialized weights.")

    model.eval()

    # 4. Create dummy input batch matching ECM 2027 GalBot specs
    num_cams = len(cfg["dataset"]["cameras"])
    dummy_images = torch.randn(1, num_cams, 3, 224, 224, device=device)
    dummy_state = torch.randn(1, cfg["model"]["state_dim"], device=device)

    # 5. Benchmark inference latency
    print("\nWarmup forward pass...")
    with torch.no_grad():
        for _ in range(2 if device.type == "cpu" else 5):
            _ = model(dummy_images, dummy_state)

    num_passes = 3 if device.type == "cpu" else 50
    print(f"Timing {num_passes} forward passes...")
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_passes):
            out = model(dummy_images, dummy_state)
    elapsed = (time.time() - start_time) / num_passes

    action_chunk = out[0] if isinstance(out, tuple) else out
    print(f"\n✓ Forward pass successful!")
    print(f"  Action chunk output shape: {action_chunk.shape} (Expected: [1, {cfg['model']['chunk_size']}, {cfg['model']['action_dim']}])")
    print(f"  Average inference latency: {elapsed * 1000.0:.2f} ms ({1.0 / elapsed:.1f} FPS)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RoboMamba-DINO-ACT Policy Inference")
    parser.add_argument("--config", type=str, default="config_rtx5090.yaml", help="Path to YAML configuration")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ecm2027_rtx5090/best_model.pt", help="Path to checkpoint weights (.pt)")
    args = parser.parse_args()

    test_inference(config_path=args.config, checkpoint_path=args.checkpoint)
