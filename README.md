# RoboMamba-DINO-ACT Policy Inference Stack

Standalone inference & evaluation container for the **RoboMamba + DINOv2 + ACT** policy, developed for the **Singapore Embodied AI Challenge for Manufacturing (ECM 2027 / SEA Challenge)**.

Designed to run seamlessly on **any machine**:
- **CPU Mode:** Standard laptops (Intel / AMD / Apple Silicon).
- **GPU Mode:** NVIDIA GPUs (RTX 30-series, RTX 40-series, and Blackwell RTX 50-series).

---

## 🚀 1-Click Run with Docker (Recommended)

### 1. Clone this Repository
```bash
git clone https://github.com/GogulnathSA123/RoboMamba_Policy_Inference.git
cd RoboMamba_Policy_Inference
```

### 2. Put Trained Checkpoints (Optional)
If you have trained weights (`best_model.pt`), place them into the `checkpoints/` directory:
```bash
mkdir -p checkpoints/ecm2027_rtx5090
cp /path/to/best_model.pt checkpoints/ecm2027_rtx5090/
```
*(If no checkpoint is provided, the script runs a sanity check with initialized weights).*

### 3. Launch Inference in One Go
```bash
bash run_docker.sh
```

**What happens automatically:**
1. Builds the Docker container once (pre-caching DINOv2 backbone for offline execution).
2. Auto-detects whether an NVIDIA GPU is available (`--gpus all`) or runs on CPU.
3. Mounts the `./checkpoints` directory directly.
4. Executes `test_inference.py` and reports forward pass speed (FPS / latency) and action chunk shapes.

---

## 🐳 Alternative Docker Commands

* **Using Docker Compose:**
  ```bash
  docker compose up
  ```

* **Interactive Bash Shell inside Container:**
  ```bash
  bash run_docker.sh bash
  ```

---

## 🐍 Native Python Execution (Without Docker)

If you prefer to run directly in your local Python environment:

```bash
# 1. Install lightweight dependencies
pip install torch torchvision pyyaml numpy tqdm

# 2. Run inference sanity benchmark
python test_inference.py \
  --config config_rtx5090.yaml \
  --checkpoint checkpoints/ecm2027_rtx5090/best_model.pt
```

---

## 🤖 Official Competition Simulator Track Integration (`ecm2027_sim_track`)

This repository includes [`robomamba_policy.py`](./robomamba_policy.py), which connects the policy to the official Isaac Sim competition evaluation track:

1. Copy `robomamba_policy.py` into your competition simulator directory:
   ```bash
   cp robomamba_policy.py /path/to/ecm2027_sim_track/policy/
   ```
2. Run the simulation benchmark in Isaac Sim:
   ```bash
   cd /path/to/ecm2027_sim_track
   bash scripts/run_policy_eval.sh configs/evaluation/robomamba_eval.yml
   ```

---

## 📊 Model Specifications (ECM 2027 Cleanroom Wafer Chamber)

| Component | Specification |
| :--- | :--- |
| **Vision Backbone** | DINOv2 ViT-B/14 (86M params, 768-dim patch tokens) |
| **Sequence Model** | 6-Layer Bidirectional Mamba SSM |
| **Action Decoder** | Conditional VAE (CVAE) + ACT Transformer (32-step chunking) |
| **Camera Inputs (3)** | `first_person_camera_rgb`, `left_hand_camera_rgb`, `right_hand_camera_rgb` (224x224) |
| **Proprioception** | 77 dimensions (GalBot chassis odometry + dual 7-DoF arms + grippers) |
| **Action Output** | 20 dimensions (bimanual delta commands + chassis velocity) |
