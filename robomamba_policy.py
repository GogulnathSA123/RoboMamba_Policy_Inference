"""
RoboMamba-DINO-ACT Competition Policy Adapter for Singapore ECM 2027
Integrates DINOv2RoboMambaACTPolicy into the ecm2027_sim_track simulation evaluation framework.
"""

import sys
import os
import yaml
import numpy as np
import torch
import torchvision.transforms as T
from pathlib import Path
from typing import Dict, Any

try:
    from policy.base import BasePolicy
except ImportError:
    class BasePolicy:
        def __init__(self, usr_args: Dict[str, Any]):
            self.usr_args = usr_args

# Ensure current directory is in sys.path for local models import
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from models.policy import DINOv2RoboMambaACTPolicy


class TemporalEnsembleBuffer:
    """Temporal ensembling buffer for smooth action trajectory execution."""

    def __init__(self, action_dim=20, chunk_size=32, k=0.01):
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.k = k
        self.predictions = {}

    def reset(self):
        self.predictions.clear()

    def update(self, current_t, pred_chunk):
        for offset in range(self.chunk_size):
            t = current_t + offset
            if t not in self.predictions:
                self.predictions[t] = []
            self.predictions[t].append((current_t, pred_chunk[offset]))

    def get_action(self, current_t):
        if current_t not in self.predictions or len(self.predictions[current_t]) == 0:
            return torch.zeros(self.action_dim)

        actions_at_t = self.predictions[current_t]
        weights = []
        actions = []

        for start_t, act in actions_at_t:
            age = current_t - start_t
            w = np.exp(-self.k * age)
            weights.append(w)
            actions.append(act)

        weights = np.array(weights)
        weights /= weights.sum()

        actions_tensor = torch.stack(actions)
        weighted_action = (actions_tensor * torch.tensor(weights, device=actions_tensor.device).unsqueeze(1)).sum(dim=0)
        return weighted_action.detach().cpu().numpy()


class RoboMambaPolicy(BasePolicy):
    """Competition policy adapter for RoboMamba + DINOv2 + ACT."""

    def __init__(self, usr_args: Dict[str, Any]):
        super().__init__(usr_args)

    def get_model(self, usr_args: Dict[str, Any]) -> Any:
        """Instantiate model and load checkpoint."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Resolve paths dynamically
        checkpoint_path = usr_args.get("checkpoint")
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            chk_candidates = [
                os.path.join(robomamba_dir, "checkpoints/ecm2027_rtx5090/best_model.pt") if robomamba_dir else "",
                os.path.join(robomamba_dir, "checkpoints/best_model.pt") if robomamba_dir else "",
                "checkpoints/ecm2027_rtx5090/best_model.pt",
                "best_model.pt",
            ]
            for c in chk_candidates:
                if c and os.path.exists(c):
                    checkpoint_path = c
                    break

        config_path = usr_args.get("config")
        if not config_path or not os.path.exists(config_path):
            cfg_candidates = [
                os.path.join(robomamba_dir, "config_rtx5090.yaml") if robomamba_dir else "",
                os.path.join(robomamba_dir, "config.yaml") if robomamba_dir else "",
                "config_rtx5090.yaml",
                "config.yaml",
            ]
            for c in cfg_candidates:
                if c and os.path.exists(c):
                    config_path = c
                    break

        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.action_dim = self.cfg["model"].get("action_dim", 20)
        self.state_dim = self.cfg["model"].get("state_dim", 77)
        self.chunk_size = self.cfg["model"].get("chunk_size", 32)

        # Build policy network
        self.model = DINOv2RoboMambaACTPolicy(
            dino_model=self.cfg["model"]["dino_backbone"],
            freeze_dino=self.cfg["model"]["freeze_dino"],
            d_model=self.cfg["model"]["d_model"],
            n_mamba_layers=self.cfg["model"]["n_mamba_layers"],
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            chunk_size=self.chunk_size,
            latent_dim=self.cfg["model"]["latent_dim"],
        ).to(self.device)

        # Load weights
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[RoboMambaPolicy] Loading weights from: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            self.model.load_state_dict(state_dict)
            print("[RoboMambaPolicy] Weights successfully loaded!")
        else:
            print(f"[RoboMambaPolicy] Warning: Checkpoint '{checkpoint_path}' not found. Using initialized weights.")

        self.model.eval()

        # Image transform (640x480 -> 224x224 with ImageNet z-scoring)
        self.img_transform = T.Compose([
            T.Resize((224, 224), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.temporal_buffer = TemporalEnsembleBuffer(action_dim=self.action_dim, chunk_size=self.chunk_size, k=0.01)
        self.step_idx = 0
        return self.model

    def reset_model(self) -> None:
        """Reset temporal buffer and timestep tracking at the start of each episode."""
        if hasattr(self, "temporal_buffer"):
            self.temporal_buffer.reset()
        self.step_idx = 0

    def get_action(self, obs: Dict[str, Any] = None) -> np.ndarray:
        """Predict the next action using temporal ensembling over action chunks."""
        if obs is None:
            return np.zeros(self.action_dim, dtype=np.float32)

        policy_obs = obs.get("policy", obs)

        # 1. Extract 3 camera RGB views: Head, Left Wrist, Right Wrist
        head_cam = None
        left_wrist = None
        right_wrist = None

        for k, v in policy_obs.items():
            if "first_person" in k and head_cam is None:
                head_cam = v
            elif "left_hand" in k and left_wrist is None:
                left_wrist = v
            elif "right_hand" in k and right_wrist is None:
                right_wrist = v

        # Convert images to [3, 224, 224] tensors
        cam_tensors = []
        for cam in [head_cam, left_wrist, right_wrist]:
            if cam is not None:
                t = cam if torch.is_tensor(cam) else torch.from_numpy(cam)
                if t.dim() == 4:
                    t = t.squeeze(0)
                if t.shape[-1] == 3:  # H, W, C -> C, H, W
                    t = t.permute(2, 0, 1)
                t = t.float()
                if t.max() > 1.0:
                    t = t / 255.0
                t = self.img_transform(t)
            else:
                t = torch.zeros(3, 224, 224)
            cam_tensors.append(t)

        images = torch.stack(cam_tensors).unsqueeze(0).to(self.device)  # [1, 3, 3, 224, 224]

        # 2. Extract proprioception state [77]
        raw_state = policy_obs.get("state", policy_obs.get("joint_pos", None))
        if raw_state is not None:
            st = raw_state if torch.is_tensor(raw_state) else torch.from_numpy(raw_state)
            if st.dim() == 2:
                st = st.squeeze(0)
            st = st.float().to(self.device)
            if st.shape[0] < self.state_dim:
                st = torch.nn.functional.pad(st, (0, self.state_dim - st.shape[0]))
            elif st.shape[0] > self.state_dim:
                st = st[:self.state_dim]
            state = st.unsqueeze(0)
        else:
            state = torch.zeros(1, self.state_dim, device=self.device)

        # 3. Predict action chunk
        with torch.no_grad():
            out = self.model(images, state)
            pred_chunk = (out[0] if isinstance(out, tuple) else out).squeeze(0)  # [chunk_size, action_dim]

        # 4. Temporal ensembling
        self.temporal_buffer.update(self.step_idx, pred_chunk)
        action = self.temporal_buffer.get_action(self.step_idx)
        self.step_idx += 1

        if torch.is_tensor(action):
            action = action.cpu().numpy()
        return action.astype(np.float32)

    def eval(self, task_env: Any, observation: Dict[str, Any],
             usr_args: Dict[str, Any], video_writer: Any) -> bool:
        """Execute full evaluation episode rollout."""
        time_out_limit = usr_args.get("time_out_limit", 300)
        camera_keys = ["first_person_camera_rgb", "left_hand_camera_rgb", "right_hand_camera_rgb"]
        obs = observation
        success = False

        for t in range(time_out_limit):
            action = self.get_action(obs)
            obs, terminated, extras = self.step_environment(task_env, action, usr_args)
            self.add_video_frame(video_writer, obs, camera_keys)

            if isinstance(extras, dict) and extras.get("success", False):
                success = True

            if terminated:
                break

        return success
