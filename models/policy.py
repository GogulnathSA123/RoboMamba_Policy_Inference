import torch
import torch.nn as nn
from .dino_backbone import DINOv2VisionBackbone
from .mamba_block import RoboMambaSequenceEncoder
from .act_decoder import CVAEEncoder, ACTActionDecoder

class DINOv2RoboMambaACTPolicy(nn.Module):
    """
    Unified Policy Architecture:
      - Vision Backbone: Pre-trained DINOv2 (Meta AI) with spatial pooling
      - Sequence Model: RoboMamba (Selective State Space Model)
      - Action Decoder: ACT (Action Chunking with Transformers + CVAE)
      - Integrated Normalization: Stores dataset statistics as buffers (LeRobot style)
    """

    def __init__(
        self,
        dino_model="dinov2_vits14",
        freeze_dino=True,
        d_model=512,
        n_mamba_layers=4,
        state_dim=6,
        action_dim=6,
        chunk_size=16,
        latent_dim=32,
        pool_grid=4,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.pool_grid = pool_grid

        # Normalization buffers (filled before training, restored automatically from checkpoint)
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))
        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_std", torch.ones(action_dim))

        # 1. Vision Backbone (DINOv2)
        self.vision = DINOv2VisionBackbone(
            model_name=dino_model,
            d_model=d_model,
            freeze=freeze_dino,
            pool_spatial=True,
            pool_grid=pool_grid,
        )

        # 2. Joint State Embedding (for Mamba sequence input)
        self.state_encoder = nn.Linear(state_dim, d_model)

        # 3. Spatiotemporal Sequence Encoder (Mamba SSM)
        self.mamba = RoboMambaSequenceEncoder(d_model=d_model, n_layers=n_mamba_layers)

        # 4. CVAE Encoder (training only — discarded at inference)
        self.cvae = CVAEEncoder(
            action_dim=action_dim,
            state_dim=state_dim,
            latent_dim=latent_dim,
            d_model=d_model,
            chunk_size=chunk_size,
        )

        # 5. ACT Action Decoder (with latent + proprio as additional memory tokens)
        self.decoder = ACTActionDecoder(
            action_dim=action_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            latent_dim=latent_dim,
            state_dim=state_dim,
        )

    def forward(self, images, state, actions=None):
        """
        Input:
            images: Multi-camera RGB tensor [B, Num_Cams, 3, 224, 224] (z-scored)
            state: Raw robot joint state [B, state_dim]
            actions: (Training only) Raw ground-truth action chunk [B, chunk_size, action_dim]
        Output:
            pred_actions: Action chunk [B, chunk_size, action_dim]
                          (Normalized during training, Unnormalized raw actions during inference)
            mu: CVAE mean (None during inference)
            logvar: CVAE log variance (None during inference)
        """
        B = images.shape[0]

        # Normalize state internally: [B, state_dim]
        state_norm = (state - self.state_mean) / self.state_std

        # --- Observation Encoding ---

        # Extract visual tokens using DINOv2: [B, Num_Visual_Tokens, d_model]
        vis_tokens = self.vision(images)

        # Encode joint state: [B, 1, d_model]
        state_token = self.state_encoder(state_norm).unsqueeze(1)

        # Concatenate visual tokens + state token along sequence dimension
        seq_tokens = torch.cat([vis_tokens, state_token], dim=1)  # [B, Seq_Len, d_model]

        # Pass through Mamba SSM for spatiotemporal encoding
        memory = self.mamba(seq_tokens)  # [B, Seq_Len, d_model]

        # --- Action Decoding ---

        if actions is not None:
            # Training: Normalize actions internally
            actions_norm = (actions - self.action_mean) / self.action_std
            
            # CVAE encoder produces latent z from (actions_norm, state_norm)
            z, mu, logvar = self.cvae(actions_norm, state_norm)
            pred_actions_norm = self.decoder(memory, z, state_norm)
            
            # Return normalized actions so loss is computed in normalized space
            return pred_actions_norm, mu, logvar
        else:
            # Inference: z = 0 (mean of standard Gaussian prior, matching original ACT)
            z = torch.zeros(B, self.latent_dim, device=images.device)
            pred_actions_norm = self.decoder(memory, z, state_norm)
            
            # Unnormalize actions back to raw values for the hardware/environment
            pred_actions = pred_actions_norm * self.action_std + self.action_mean
            return pred_actions, None, None
