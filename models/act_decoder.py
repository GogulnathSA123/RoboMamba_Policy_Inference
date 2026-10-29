import math
import torch
import torch.nn as nn
import numpy as np

def get_sinusoid_encoding_table(n_position, d_hid):
    """Sinusoidal position encoding table (matching original ACT implementation)."""
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class CVAEEncoder(nn.Module):
    """
    CVAE Encoder for ACT policy (training-only module).

    Matches the original ACT (tonyzhaozh/act) detr_vae.py:
    - cls_embed: nn.Embedding(1, hidden_dim) — learned CLS token
    - encoder_action_proj: nn.Linear(action_dim, hidden_dim) — per-step action projection
    - encoder_joint_proj: nn.Linear(state_dim, hidden_dim) — proprioception projection
    - pos_table: sinusoidal position encoding registered as buffer
    - latent_proj: nn.Linear(hidden_dim, latent_dim * 2) — outputs [mu, logvar] concatenated
    - Reparameterization: std = logvar.div(2).exp() (matching original)
    """

    def __init__(self, action_dim=6, state_dim=6, latent_dim=32, d_model=512, chunk_size=16, n_heads=8, n_encoder_layers=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model

        # Learned [CLS] token (matching original: nn.Embedding(1, hidden_dim))
        self.cls_embed = nn.Embedding(1, d_model)

        # Per-action step projection (matching original: encoder_action_proj)
        self.encoder_action_proj = nn.Linear(action_dim, d_model)

        # Joint state projection (matching original: encoder_joint_proj)
        self.encoder_joint_proj = nn.Linear(state_dim, d_model)

        # Sinusoidal position encoding registered as buffer (matching original: pos_table)
        # Sequence: [CLS] + [joint_state] + [action_0, ..., action_{T-1}]
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1 + 1 + chunk_size, d_model))

        # Latent projection: outputs mu and logvar concatenated (matching original)
        self.latent_proj = nn.Linear(d_model, latent_dim * 2)

        # Transformer Encoder (matching original ACT: post-norm, relu activation)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, activation="relu", batch_first=True, norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

    def forward(self, actions, state):
        """
        Input:
            actions: Ground-truth action chunk [B, chunk_size, action_dim]
            state: Current joint state [B, state_dim]
        Output:
            z: Sampled latent variable [B, latent_dim]
            mu: Gaussian mean [B, latent_dim]
            logvar: Gaussian log-variance [B, latent_dim]
        """
        B = actions.shape[0]

        # Project each action step: [B, chunk_size, d_model]
        action_embed = self.encoder_action_proj(actions)

        # Project joint state: [B, 1, d_model]
        joint_embed = self.encoder_joint_proj(state).unsqueeze(1)

        # CLS token: [B, 1, d_model]
        cls_embed = self.cls_embed.weight.unsqueeze(0).repeat(B, 1, 1)

        # Concatenate: [CLS, joint_state, action_0, ..., action_{T-1}]
        encoder_input = torch.cat([cls_embed, joint_embed, action_embed], dim=1)

        # Add sinusoidal positional encoding
        pos_embed = self.pos_table.clone().detach()[:, :encoder_input.shape[1], :]
        encoder_input = encoder_input + pos_embed

        # Encode full sequence
        encoder_output = self.encoder(encoder_input)

        # Extract CLS token output: [B, d_model]
        cls_output = encoder_output[:, 0, :]

        # Project to latent space: [B, latent_dim * 2] -> split to mu, logvar
        latent_info = self.latent_proj(cls_output)
        mu = latent_info[:, :self.latent_dim]
        logvar = latent_info[:, self.latent_dim:]

        # Reparameterization trick (matching original: std = logvar.div(2).exp())
        std = logvar.div(2).exp()
        eps = torch.randn_like(std)
        z = mu + std * eps

        return z, mu, logvar


class ACTActionDecoder(nn.Module):
    """
    ACT Transformer Query Decoder for predicting action chunks.

    Matches the original ACT (tonyzhaozh/act) detr_vae.py:
    - query_embed: nn.Embedding(num_queries, hidden_dim) — learned action queries
    - action_head: nn.Linear(hidden_dim, action_dim) — single linear projection (not MLP)
    - latent_out_proj: nn.Linear(latent_dim, hidden_dim) — projects z to decoder memory
    - additional_pos_embed: nn.Embedding(2, hidden_dim) — positional embed for [latent, proprio]
    - Latent z and proprioception are prepended to visual memory as additional tokens
    - Decoder uses post-norm, relu activation (DETR standard)
    """

    def __init__(self, action_dim=6, chunk_size=16, d_model=512, latent_dim=32, state_dim=6,
                 n_heads=8, n_decoder_layers=1):
        super().__init__()
        self.chunk_size = chunk_size
        self.d_model = d_model

        # Latent z projection (matching original: latent_out_proj)
        self.latent_out_proj = nn.Linear(latent_dim, d_model)

        # Proprioception projection for decoder memory
        self.input_proj_robot_state = nn.Linear(state_dim, d_model)

        # Additional positional embeddings for [latent, proprio] tokens in decoder memory
        # (matching original: additional_pos_embed)
        self.additional_pos_embed = nn.Embedding(2, d_model)

        # Learned action query embeddings (matching original: query_embed)
        self.query_embed = nn.Embedding(chunk_size, d_model)

        # Transformer Decoder (matching original ACT: post-norm, relu, 1 layer default)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, activation="relu", batch_first=True, norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)

        # Action prediction head (matching original: single nn.Linear)
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(self, memory, z, state):
        """
        Input:
            memory: Encoded observation tokens from Mamba [B, Seq_Len, d_model]
            z: Latent variable [B, latent_dim] (zeros during inference)
            state: Normalized robot joint state [B, state_dim]
        Output:
            actions: Predicted action chunk [B, chunk_size, action_dim]
        """
        B = memory.shape[0]

        # Project latent z: [B, 1, d_model]
        latent_input = self.latent_out_proj(z).unsqueeze(1)

        # Project proprioception: [B, 1, d_model]
        proprio_input = self.input_proj_robot_state(state).unsqueeze(1)

        # Additional positional embeddings for [latent, proprio]
        additional_pos = self.additional_pos_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, 2, d_model]

        # Prepend [latent, proprio] to visual memory with their positional embeddings
        additional_tokens = torch.cat([latent_input, proprio_input], dim=1) + additional_pos
        memory_with_context = torch.cat([additional_tokens, memory], dim=1)  # [B, 2 + Seq_Len, d_model]

        # Generate action queries
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, chunk_size, d_model]

        # Decode: action queries cross-attend to observation memory + latent + proprio
        out = self.decoder(tgt=queries, memory=memory_with_context)  # [B, chunk_size, d_model]

        # Predict actions (single linear head, matching original ACT)
        actions = self.action_head(out)  # [B, chunk_size, action_dim]

        return actions
