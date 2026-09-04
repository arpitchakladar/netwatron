"""
models/heads.py
-----------------
Stage I (Sec. 31): future-state decoder z_hat -> S_hat (observable projection).
Stage J (Sec. 34-38): risk head, 4 variants.
Stage K (Sec. 39-41): ATT&CK-aligned stage head, 2 variants.

Reminder (Sec. 42): stage outputs are ATT&CK-ALIGNED INFERENCES, never
claimed as ground truth -- CSE-CIC-IDS2018 provides no per-flow stage labels.
"""

import torch
import torch.nn as nn
from typing import Optional

from ..config import RiskConfig, StageConfig


class FutureStateDecoder(nn.Module):
    """Sec. 31: z_hat_{t+i} -> S_hat_{t+i}. Here S is taken to be the same
    dimensionality as the observable network-state features (i.e. it
    reconstructs the pooled Stage-A/D feature space), making L_s in Sec. 32
    computable directly against the *actual* pooled state at that future
    step (used only as a training signal, not fed back into the model)."""

    def __init__(self, state_dim: int, observable_dim: Optional[int] = None):
        super().__init__()
        observable_dim = observable_dim or state_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, observable_dim),
        )

    def forward(self, z_hat: torch.Tensor) -> torch.Tensor:
        return self.net(z_hat)


class TrajectoryAttention(nn.Module):
    """Sec. 34: pools the K future latent states into a single trajectory
    summary vector via learned attention (used by RISK-2 / T3+)."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1)
        )

    def forward(self, z_future: torch.Tensor) -> torch.Tensor:
        # z_future: [B, K, d] -> [B, d]  (no padding mask needed; K is fixed)
        scores = self.score(z_future).squeeze(-1)  # [B, K]
        alpha = torch.softmax(scores, dim=-1)
        return torch.einsum("bk,bkd->bd", alpha, z_future)


class RiskHead(nn.Module):
    """Implements RISK-1..4 (Sec. 35-38) behind one interface.

    RISK-1 per-step:      z_future [B,K,d] -> R [B,K]        (time-dependent risk curve)
    RISK-2 horizon:       z_future [B,K,d] -> R [B]           (single horizon probability, via TrajectoryAttention)
    RISK-3 current-state: z_now    [B,d]   -> R [B]           (non-world-model baseline)
    RISK-4 history-only:  z_history[B,L,d] -> R [B]           (history without generated future -- tests
                                                                 "future-state modelling > history-only", Sec. 38)
    """

    def __init__(
        self, cfg: RiskConfig, state_dim: int, history_len: Optional[int] = None
    ):
        super().__init__()
        self.cfg = cfg
        if cfg.variant == "RISK-1":
            self.per_step = nn.Sequential(
                nn.Linear(state_dim, cfg.hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.hidden_dim, 1),
            )
        elif cfg.variant == "RISK-2":
            self.traj_attn = TrajectoryAttention(state_dim)
            self.classifier = nn.Sequential(
                nn.Linear(state_dim, cfg.hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.hidden_dim, 1),
            )
        elif cfg.variant == "RISK-3":
            self.classifier = nn.Sequential(
                nn.Linear(state_dim, cfg.hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.hidden_dim, 1),
            )
        elif cfg.variant == "RISK-4":
            assert history_len is not None
            self.history_pool = nn.Sequential(
                nn.Linear(state_dim, cfg.hidden_dim), nn.GELU()
            )
            self.classifier = nn.Linear(cfg.hidden_dim, 1)
        else:
            raise ValueError(f"Unknown risk variant: {cfg.variant}")

    def forward(
        self,
        *,
        z_future: Optional[torch.Tensor] = None,
        z_now: Optional[torch.Tensor] = None,
        z_history: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.variant == "RISK-1":
            assert z_future is not None
            logits = self.per_step(z_future).squeeze(-1)  # [B, K]
            return torch.sigmoid(logits), logits
        if self.cfg.variant == "RISK-2":
            assert z_future is not None
            traj = self.traj_attn(z_future)
            logit = self.classifier(traj).squeeze(-1)  # [B]
            return torch.sigmoid(logit), logit
        if self.cfg.variant == "RISK-3":
            assert z_now is not None
            logit = self.classifier(z_now).squeeze(-1)
            return torch.sigmoid(logit), logit
        if self.cfg.variant == "RISK-4":
            assert z_history is not None
            pooled = self.history_pool(z_history).mean(
                dim=1
            )  # mean over L (simple, replace with attn if needed)
            logit = self.classifier(pooled).squeeze(-1)
            return torch.sigmoid(logit), logit
        raise RuntimeError(f"Unknown risk variant: {self.cfg.variant}")


class StageHead(nn.Module):
    """S1 (Sec. 40) independent per-step classification, or S2 (Sec. 41)
    sequence decoder with explicit inter-step conditioning via a GRU (a
    lightweight stand-in for a full stage-transition decoder)."""

    def __init__(self, cfg: StageConfig, state_dim: int, horizon: int):
        super().__init__()
        self.cfg = cfg
        self.horizon = horizon
        if cfg.variant == "S1":
            self.classifier = nn.Sequential(
                nn.Linear(state_dim, cfg.hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.hidden_dim, cfg.num_stages),
            )
        elif cfg.variant == "S2":
            self.gru = nn.GRU(state_dim, cfg.hidden_dim, batch_first=True)
            self.classifier = nn.Linear(cfg.hidden_dim, cfg.num_stages)
        else:
            raise ValueError(f"Unknown stage variant: {cfg.variant}")

    def forward(self, z_future: torch.Tensor) -> torch.Tensor:
        """z_future: [B, K, d] -> stage_logits [B, K, num_stages]"""
        if self.cfg.variant == "S1":
            return self.classifier(z_future)
        out, _ = self.gru(z_future)
        return self.classifier(out)
