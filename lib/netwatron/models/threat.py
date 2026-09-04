"""
models/threat.py
------------------
Stage N (Sec. 46-47): final threat assessment.

    T = f(R_trajectory, S_surprise, C_stage)

A lightweight calibration layer combines the three signals. Ablation
variants (Sec. 47 and Sec. 66/74):
    T1 = f(R)
    T2 = f(R, S)
    T3 = f(R)                (same as T1 but using *trajectory* risk specifically,
                               as opposed to T1/T2 in the baseline ladder which use
                               current/history risk -- see experiments/run_baseline_ladder.py)
    T4 = f(R, S)              trajectory risk + surprise
    T5 = f(R, S, C)           trajectory risk + surprise + stage consistency (full model)

We implement this as a single small MLP whose *input feature set* changes
with the variant, which is the cleanest way to run the T1..T5 ablation
sweep without duplicating code.
"""

import torch
import torch.nn as nn
from typing import Optional

from ..config import ThreatConfig


class ThreatCalibration(nn.Module):
    def __init__(self, cfg: ThreatConfig, hidden_dim: int = 32):
        super().__init__()
        self.cfg = cfg
        n_in = 1  # R is always present
        if cfg.use_surprise:
            n_in += 1
        if cfg.use_stage_consistency:
            n_in += 1
        self.n_in = n_in
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        # normalize surprise/consistency (unbounded) before combining with R (in [0,1])
        self.surprise_norm = nn.LayerNorm(1) if cfg.use_surprise else None
        self.consistency_norm = (
            nn.LayerNorm(1) if cfg.use_stage_consistency else None
        )

    def forward(
        self,
        risk_prob: torch.Tensor,
        surprise: Optional[torch.Tensor] = None,
        consistency: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feats = [risk_prob.unsqueeze(-1)]
        if self.cfg.use_surprise:
            assert surprise is not None, "T4/T5 require a surprise signal"
            assert self.surprise_norm is not None
            feats.append(self.surprise_norm(surprise.unsqueeze(-1)))
        if self.cfg.use_stage_consistency:
            assert (
                consistency is not None
            ), "T5 requires a stage-consistency signal"
            assert self.consistency_norm is not None
            feats.append(self.consistency_norm(consistency.unsqueeze(-1)))
        x = torch.cat(feats, dim=-1)
        logit = self.net(x).squeeze(-1)
        return torch.sigmoid(logit), logit


def make_threat_config(variant: str) -> ThreatConfig:
    """Sec. 66 ablation shorthand -> ThreatConfig. Kept separate from
    RiskConfig.variant, which selects RISK-1..4 for the risk head itself."""
    table = {
        "T1": ThreatConfig(
            variant="T1", use_surprise=False, use_stage_consistency=False
        ),
        "T2": ThreatConfig(
            variant="T2", use_surprise=True, use_stage_consistency=False
        ),
        "T3": ThreatConfig(
            variant="T3", use_surprise=False, use_stage_consistency=False
        ),
        "T4": ThreatConfig(
            variant="T4", use_surprise=True, use_stage_consistency=False
        ),
        "T5": ThreatConfig(
            variant="T5", use_surprise=True, use_stage_consistency=True
        ),
    }
    if variant not in table:
        raise ValueError(f"Unknown threat variant: {variant}")
    return table[variant]
