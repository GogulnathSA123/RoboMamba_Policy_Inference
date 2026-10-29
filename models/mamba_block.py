import torch
import torch.nn as nn
import math

class MambaEncoderLayer(nn.Module):
    """
    Mamba State Space Model (SSM) layer with Selective Scan mechanism.

    Follows the Mamba architecture (Gu & Dao, 2023):
    - Input-dependent gating via selective scan (Δ, B, C are input-dependent)
    - Depthwise causal conv1d for local context
    - SiLU activations with gated residual connection
    - Pre-norm (LayerNorm) with residual
    """

    def __init__(self, d_model=512, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        # Pre-norm
        self.norm = nn.LayerNorm(d_model)

        # Input projection: splits into x (for SSM) and z (for gating)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Causal depthwise conv1d for local context mixing
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )
        self.act = nn.SiLU()

        # Selective SSM projections (input-dependent Δ, B, C)
        # Projects x -> (dt, B, C) where dt is scalar, B and C are d_state vectors
        self.x_proj = nn.Linear(self.d_inner, d_state + d_state + self.d_inner, bias=False)

        # dt (discretization step) projection and initialization
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        # Initialize dt bias for stable training (from Mamba paper)
        dt_init_std = self.d_inner ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # Structured state matrix A (HiPPO initialization)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))  # [d_inner, d_state]

        # Skip connection parameter D
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def selective_scan(self, x, dt, B, C):
        """
        Selective Scan SSM (sequential implementation).

        Input:
            x:  [B, L, d_inner]  - input sequence after conv1d
            dt: [B, L, d_inner]  - input-dependent discretization step (after softplus)
            B:  [B, L, d_state]  - input-dependent B matrix
            C:  [B, L, d_state]  - input-dependent C matrix

        Output:
            y:  [B, L, d_inner]  - SSM output

        Computes the recurrence:
            h_t = exp(Δ_t * A) * h_{t-1} + Δ_t * B_t * x_t
            y_t = C_t · h_t + D * x_t
        """
        batch, seq_len, d_inner = x.shape
        d_state = B.shape[-1]

        # Recover A from log-space: [d_inner, d_state]
        A = -torch.exp(self.A_log.float())

        # Initialize hidden state: [batch, d_inner, d_state]
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]          # [B, d_inner]
            dt_t = dt[:, t, :]        # [B, d_inner]
            B_t = B[:, t, :]          # [B, d_state]
            C_t = C[:, t, :]          # [B, d_state]

            # Discretize A: dA = exp(dt_t * A), shape [B, d_inner, d_state]
            dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))

            # Discretize B: dB = dt_t * B_t, shape [B, d_inner, d_state]
            dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)

            # State update: h = dA * h + dB * x_t
            h = dA * h + dB * x_t.unsqueeze(-1)

            # Output: y_t = C_t · h + D * x_t
            y_t = (h * C_t.unsqueeze(1)).sum(dim=-1) + self.D * x_t  # [B, d_inner]
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # [B, L, d_inner]
        return y

    def forward(self, x):
        """
        Input:
            x: Tensor of shape [B, Seq_Len, d_model]
        Output:
            Tensor of shape [B, Seq_Len, d_model]
        """
        residual = x
        x = self.norm(x)

        B, L, _ = x.shape

        # Dual projection: x_branch (SSM path) and z (gate path)
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)  # Each [B, L, d_inner]

        # Causal conv1d: local context mixing
        x_conv = x_branch.transpose(1, 2)  # [B, d_inner, L]
        x_conv = self.conv1d(x_conv)[:, :, :L]  # Causal: truncate future
        x_conv = x_conv.transpose(1, 2)  # [B, L, d_inner]
        x_conv = self.act(x_conv)

        # Selective SSM projections: input-dependent (Δ, B, C)
        x_ssm = self.x_proj(x_conv)  # [B, L, d_state + d_state + d_inner]
        B_proj, C_proj, dt_input = x_ssm.split(
            [self.d_state, self.d_state, self.d_inner], dim=-1
        )

        # Discretization step: softplus ensures positivity
        dt = nn.functional.softplus(self.dt_proj(dt_input))  # [B, L, d_inner]

        # Run Selective Scan SSM
        y = self.selective_scan(x_conv, dt, B_proj, C_proj)

        # Gated residual: y * SiLU(z)
        y = y * self.act(z)

        # Output projection + residual
        out = self.out_proj(y) + residual

        return out


class RoboMambaSequenceEncoder(nn.Module):
    """Stack of Mamba SSM layers for spatiotemporal sequence encoding."""

    def __init__(self, d_model=512, d_state=16, n_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaEncoderLayer(d_model=d_model, d_state=d_state)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
