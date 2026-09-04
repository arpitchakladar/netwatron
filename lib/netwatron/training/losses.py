"""
training/losses.py
---------------------
Sec. 32: dynamics loss  L_dyn = lambda_z * L_z + lambda_s * L_s
Sec. 68: joint objective
    L = lambda_dyn*L_dyn + lambda_state*L_state + lambda_risk*L_risk
        + lambda_stage*L_stage + lambda_ssl*L_ssl
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ..config import LossWeights


def dynamics_loss(
    z_hat: torch.Tensor,
    z_actual: torch.Tensor,
    s_hat: torch.Tensor,
    s_actual: torch.Tensor,
    distance: str = "huber",
) -> tuple:
    if distance == "huber":
        l_z = F.smooth_l1_loss(z_hat, z_actual)
        l_s = F.smooth_l1_loss(s_hat, s_actual)
    else:
        l_z = F.mse_loss(z_hat, z_actual)
        l_s = F.mse_loss(s_hat, s_actual)
    return l_z, l_s


def risk_loss(
    risk_logit: torch.Tensor,
    risk_label: Optional[torch.Tensor],
    per_step: bool = False,
    per_step_label: Optional[torch.Tensor] = None,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """RISK-2/3/4: risk_logit [B] vs risk_label [B].
    RISK-1: risk_logit [B,K] vs per_step_label [B,K] (per_step=True)."""
    if per_step:
        assert per_step_label is not None
        return F.binary_cross_entropy_with_logits(
            risk_logit, per_step_label, pos_weight=pos_weight
        )
    assert risk_label is not None
    return F.binary_cross_entropy_with_logits(
        risk_logit, risk_label, pos_weight=pos_weight
    )


def stage_loss(
    stage_logits: torch.Tensor, stage_labels: torch.Tensor
) -> torch.Tensor:
    """stage_logits: [B, K, C], stage_labels: [B, K] -> mean CE over K steps.
    Reminder: these labels are ATT&CK-aligned pseudo-labels/inferences, not
    verified ground truth (Sec. 42)."""
    B, K, C = stage_logits.shape
    return F.cross_entropy(
        stage_logits.reshape(B * K, C), stage_labels.reshape(B * K)
    )


def joint_loss(
    outputs: dict,
    batch: dict,
    weights: LossWeights,
    risk_per_step: bool = False,
    risk_pos_weight: Optional[torch.Tensor] = None,
) -> dict:
    l_z, l_s = dynamics_loss(
        outputs["z_future_hat"],
        outputs["z_future_actual"],
        outputs["s_hat_future"],
        outputs["z_future_actual"],
    )
    l_dyn = l_z  # L_z term of Sec. 32; L_s tracked separately below as l_state
    l_state = l_s

    if risk_per_step:
        l_risk = risk_loss(
            outputs["risk_logit"],
            None,
            per_step=True,
            per_step_label=batch["per_step_risk_label"],
            pos_weight=risk_pos_weight,
        )
    else:
        l_risk = risk_loss(
            outputs["risk_logit"],
            batch["risk_label"],
            pos_weight=risk_pos_weight,
        )

    l_stage = stage_loss(outputs["stage_logits"], batch["stage_labels"])

    total = (
        weights.lambda_dyn * l_dyn
        + weights.lambda_state * l_state
        + weights.lambda_risk * l_risk
        + weights.lambda_stage * l_stage
    )

    return {
        "total": total,
        "l_dyn": l_dyn.detach(),
        "l_state": l_state.detach(),
        "l_risk": l_risk.detach(),
        "l_stage": l_stage.detach(),
    }
