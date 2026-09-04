"""
models/flow_encoder.py
------------------------
Stage C (Sec. 15) + time encoding (Sec. 16): converts each preprocessed flow
x_i in R^80 into h_i in R^d (d=128 initial), then adds a learned embedding of
its RELATIVE time within the window (never a raw absolute timestamp -- Sec.
77 item 2).

Architecture matches Figure 2 in the design doc exactly:
    Linear(80,128) -> LayerNorm -> GELU -> Dropout(0.1)
    -> Linear(128,128) -> LayerNorm -> GELU
"""

import torch
import torch.nn as nn

from ..config import FlowEncoderConfig


class TimeEncoder(nn.Module):
    """Sec. 16: tau_i = t_i - window_start, delta_i = t_i - t_{i-1}.

    T1 ("relative time only"): uses tau only.
    T2 ("relative + global"): uses tau and delta, giving the model both a
    within-window locality signal (delta) and a within-window position
    signal (tau) -- analogous to the dual local/global temporal encoding
    used in 2026 temporal-graph attack-prediction work referenced in Sec. 16.
    """

    def __init__(self, embed_dim: int, variant: str = "T2"):
        super().__init__()
        assert variant in ("T1", "T2")
        self.variant = variant
        in_dim = 1 if variant == "T1" else 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, tau: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        # tau, delta: [..., N] (seconds). Log1p-compress before the MLP so
        # long idle gaps don't dominate (same "log-transform long-tailed,
        # non-negative quantity" philosophy as Stage A, Sec. 7).
        tau_c = torch.log1p(tau.clamp(min=0)).unsqueeze(-1)
        if self.variant == "T1":
            feat = tau_c
        else:
            delta_c = torch.log1p(delta.clamp(min=0)).unsqueeze(-1)
            feat = torch.cat([tau_c, delta_c], dim=-1)
        return self.mlp(feat)


class FlowEncoder(nn.Module):
    """h_i = E_phi(x_i); h~_i = h_i + e_time_i (Sec. 15/16, Figure 2)."""

    def __init__(self, cfg: FlowEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
        )
        self.time_encoder = TimeEncoder(cfg.time_embed_dim, cfg.time_variant)
        assert (
            cfg.time_embed_dim == cfg.hidden_dim
        ), "time_embed_dim must match hidden_dim for additive combination (Sec. 16)"

    def forward(
        self, x: torch.Tensor, tau: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor:
        """x: [..., N, 80], tau/delta: [..., N] -> returns h_tilde [..., N, d]"""
        h = self.net(x)
        e_time = self.time_encoder(tau, delta)
        return h + e_time
