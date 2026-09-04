"""
models/dynamics.py
--------------------
Stage H (Sec. 25-30): the Dynamics / World Model.

    z_hat_{t+1:t+K} = D_theta(z_{t-L+1:t})

This is the core "world-model" piece: its job is to predict *future latent
network states*, not to classify the current one directly (Sec. 77 item 8 --
never replace this with a direct classifier; direct classifiers only exist
as baselines, see training/evaluate.py::BASELINE_LADDER).

All three variants below produce K future latent states directly
(F2 "direct multi-horizon", Sec. 28/29 preferred training strategy).
Autoregressive rollout (F3) for evaluation-time stability checks is provided
via `.autoregressive_rollout()`, reusing the same trained module.
"""

import math
from typing import cast

import torch
import torch.nn as nn

from ..config import DynamicsConfig


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positional_encoding = cast(torch.Tensor, self.pe)
        return x + positional_encoding[:, : x.shape[1]]


class LSTMSeq2SeqDynamics(nn.Module):
    """D1 (Sec. 26): literature-backed recurrent encoder-decoder baseline,
    adapted from LSTM encoder-decoder attack-stage-sequence forecasting to
    predict future NETWORK STATES rather than stage labels directly."""

    def __init__(self, cfg: DynamicsConfig, state_dim: int):
        super().__init__()
        self.cfg = cfg
        self.state_dim = state_dim
        self.encoder = nn.LSTM(
            state_dim,
            cfg.hidden_dim,
            cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.decoder_cell = nn.LSTMCell(state_dim, cfg.hidden_dim)
        self.out_proj = nn.Linear(cfg.hidden_dim, state_dim)
        self.h0_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.c0_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)

    def forward(self, z_history: torch.Tensor) -> torch.Tensor:
        """z_history: [B, L, d] -> z_hat_future: [B, K, d] (direct, F2)."""
        B = z_history.shape[0]
        _, (h_n, c_n) = self.encoder(z_history)
        h = h_n[-1]
        c = c_n[-1]
        dec_input = z_history[:, -1, :]  # seed decoder with last observed state
        outputs = []
        for _ in range(self.cfg.horizon):
            h, c = self.decoder_cell(dec_input, (h, c))
            z_hat = self.out_proj(h)
            outputs.append(z_hat)
            dec_input = z_hat  # teacher-forcing-free direct rollout of the decoder chain
        return torch.stack(outputs, dim=1)


class TransformerEncoderDynamics(nn.Module):
    """D2 (Sec. 27): primary candidate. Zhistory -> TransformerEncoder ->
    K-step future head, applied to a single pooled context vector
    (Figure: Input [L,128] -> PosEnc -> TransformerEncoderLayer -> Future
    Head -> [K,128])."""

    def __init__(self, cfg: DynamicsConfig, state_dim: int):
        super().__init__()
        self.cfg = cfg
        self.pos_enc = SinusoidalPositionalEncoding(
            state_dim, max_len=cfg.history_len + 8
        )
        layer = nn.TransformerEncoderLayer(
            d_model=state_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.future_head = nn.Linear(state_dim, state_dim * cfg.horizon)
        self.state_dim = state_dim

    def forward(self, z_history: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(z_history)
        enc = self.encoder(x)  # [B, L, d]
        pooled = enc[
            :, -1, :
        ]  # use most-recent-token representation as context
        out = self.future_head(pooled)  # [B, K*d]
        return out.view(z_history.shape[0], self.cfg.horizon, self.state_dim)


class TransformerEncDecDynamics(nn.Module):
    """D3 (Sec. 28): explicit seq2seq formulation with learned future-position
    queries as decoder input (standard "encoder -> K learned query tokens"
    pattern, avoiding autoregressive decoding at train time per Sec. 29)."""

    def __init__(self, cfg: DynamicsConfig, state_dim: int):
        super().__init__()
        self.cfg = cfg
        self.pos_enc = SinusoidalPositionalEncoding(
            state_dim, max_len=cfg.history_len + 8
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=state_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=cfg.num_layers
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=state_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(
            dec_layer, num_layers=cfg.num_layers
        )
        self.future_queries = nn.Parameter(
            torch.randn(cfg.horizon, state_dim) * 0.02
        )

    def forward(self, z_history: torch.Tensor) -> torch.Tensor:
        B = z_history.shape[0]
        memory = self.encoder(self.pos_enc(z_history))
        tgt = self.future_queries.unsqueeze(0).expand(B, -1, -1)
        out = self.decoder(tgt, memory)
        return out  # [B, K, d]


def build_dynamics_model(cfg: DynamicsConfig, state_dim: int) -> nn.Module:
    if cfg.variant == "D1":
        return LSTMSeq2SeqDynamics(cfg, state_dim)
    if cfg.variant == "D2":
        return TransformerEncoderDynamics(cfg, state_dim)
    if cfg.variant == "D3":
        return TransformerEncDecDynamics(cfg, state_dim)
    raise ValueError(f"Unknown dynamics variant: {cfg.variant}")


@torch.no_grad()
def autoregressive_rollout(
    model: nn.Module, z_history: torch.Tensor, steps: int
) -> torch.Tensor:
    """F3 (Sec. 28/29): evaluation-time-only recursive rollout, used to test
    whether the direct-trained model stays stable when fed its own previous
    predictions, WITHOUT retraining it autoregressively."""
    history = z_history.clone()
    preds = []
    for _ in range(steps):
        z_hat_seq = model(history)
        next_z = z_hat_seq[:, 0:1, :]
        preds.append(next_z)
        history = torch.cat([history[:, 1:, :], next_z], dim=1)
    return torch.cat(preds, dim=1)
