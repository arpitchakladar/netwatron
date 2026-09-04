"""
training/train.py
--------------------
Sec. 69: Phase 1 (Representation) -> Phase 2 (Dynamics) -> Phase 3
(Risk/Stage) -> Phase 4 (End-to-end joint fine-tuning).

Phase 1 here pretrains the Flow Encoder + State pooling using the
self-supervised "future prediction" objective (SSL-1, Sec. 49/51 recommended
order: future prediction -> masked -> contrastive) rather than only
attack labels -- this is what lets zt be optimized for "predictable future"
rather than classification alone (Sec. 67).

Phases 2-4 progressively unfreeze the Dynamics Model, then the Risk/Stage
heads, then everything jointly -- implemented via simple param-group
freezing rather than separate optimizers, to keep this readable.
"""

import copy
from typing import Optional

import torch
from torch.utils.data import DataLoader

from ..config import PipelineConfig, LossWeights
from ..models.world_model import WorldModel
from .losses import joint_loss, dynamics_loss


def _set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def phase1_representation(
    model: WorldModel, loader: DataLoader, cfg: PipelineConfig, device
):
    """SSL-1 (Sec. 49): predict z_{t+1} from z_{t-L:t} on the *encoder's own*
    windows, ignoring labels entirely, to shape a representation whose
    future is predictable (Sec. 67). We reuse the dynamics module itself as
    the one-step predictor during this warm-up (it gets re-trained properly
    in Phase 2 with the real horizon)."""
    _set_requires_grad(model.risk_head, False)
    _set_requires_grad(model.stage_head, False)
    _set_requires_grad(model.threat_calibration, False)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    model.train()
    for epoch in range(cfg.train.epochs_phase1):
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            z_history = model.encode_sequence(
                batch["hist_X"],
                batch["hist_mask"],
                batch["hist_tau"],
                batch["hist_delta"],
            )
            z_future_actual = model.encode_sequence(
                batch["fut_X"],
                batch["fut_mask"],
                batch["fut_tau"],
                batch["fut_delta"],
            )
            z_future_hat = model.dynamics(z_history)
            l_z, _ = dynamics_loss(
                z_future_hat, z_future_actual, z_future_hat, z_future_actual
            )
            opt.zero_grad()
            l_z.backward()
            opt.step()
            total_loss += l_z.item()
            n_batches += 1
        print(
            f"[Phase 1 | epoch {epoch+1}/{cfg.train.epochs_phase1}] "
            f"future-prediction SSL loss = {total_loss / max(n_batches,1):.4f}"
        )

    _set_requires_grad(model.risk_head, True)
    _set_requires_grad(model.stage_head, True)
    _set_requires_grad(model.threat_calibration, True)


def phase2_dynamics(
    model: WorldModel, loader: DataLoader, cfg: PipelineConfig, device
):
    """Supervised dynamics fine-tuning with the real horizon K, still
    ignoring risk/stage heads."""
    _set_requires_grad(model.risk_head, False)
    _set_requires_grad(model.stage_head, False)
    _set_requires_grad(model.threat_calibration, False)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    model.train()
    for epoch in range(cfg.train.epochs_phase2):
        total = 0.0
        n_batches = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            l_z, l_s = dynamics_loss(
                out["z_future_hat"],
                out["z_future_actual"],
                out["s_hat_future"],
                out["z_future_actual"],
            )
            loss = cfg.loss.lambda_dyn * l_z + cfg.loss.lambda_state * l_s
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n_batches += 1
        print(
            f"[Phase 2 | epoch {epoch+1}/{cfg.train.epochs_phase2}] dynamics loss = {total / max(n_batches,1):.4f}"
        )

    _set_requires_grad(model.risk_head, True)
    _set_requires_grad(model.stage_head, True)
    _set_requires_grad(model.threat_calibration, True)


def phase3_risk_stage(
    model: WorldModel,
    loader: DataLoader,
    cfg: PipelineConfig,
    device,
    risk_pos_weight: Optional[torch.Tensor] = None,
):
    """Train risk + stage heads with the (now largely fixed) dynamics
    backbone; small LR on the backbone, full LR on the new heads."""
    backbone_params = (
        list(model.flow_encoder.parameters())
        + list(model.state_pool.parameters())
        + list(model.dynamics.parameters())
        + list(model.state_decoder.parameters())
    )
    head_params = (
        list(model.risk_head.parameters())
        + list(model.stage_head.parameters())
        + list(model.stage_consistency.parameters())
        + list(model.threat_calibration.parameters())
    )
    opt = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg.train.lr * 0.1},
            {"params": head_params, "lr": cfg.train.lr},
        ],
        weight_decay=cfg.train.weight_decay,
    )

    model.train()
    risk_per_step = cfg.risk.variant == "RISK-1"
    for epoch in range(cfg.train.epochs_phase3):
        total = 0.0
        n_batches = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            losses = joint_loss(
                out,
                batch,
                LossWeights(
                    lambda_dyn=0.0,
                    lambda_state=0.0,
                    lambda_risk=cfg.loss.lambda_risk,
                    lambda_stage=cfg.loss.lambda_stage,
                ),
                risk_per_step=risk_per_step,
                risk_pos_weight=risk_pos_weight,
            )
            opt.zero_grad()
            losses["total"].backward()
            opt.step()
            total += losses["total"].item()
            n_batches += 1
        print(
            f"[Phase 3 | epoch {epoch+1}/{cfg.train.epochs_phase3}] risk+stage loss = {total / max(n_batches,1):.4f}"
        )


def phase4_joint_finetune(
    model: WorldModel,
    loader: DataLoader,
    cfg: PipelineConfig,
    device,
    risk_pos_weight: Optional[torch.Tensor] = None,
):
    """End-to-end joint fine-tuning of everything with the full weighted
    objective (Sec. 68/69, final phase)."""
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr * 0.3,
        weight_decay=cfg.train.weight_decay,
    )
    model.train()
    risk_per_step = cfg.risk.variant == "RISK-1"
    for epoch in range(cfg.train.epochs_phase4):
        total = 0.0
        n_batches = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            losses = joint_loss(
                out,
                batch,
                cfg.loss,
                risk_per_step=risk_per_step,
                risk_pos_weight=risk_pos_weight,
            )
            opt.zero_grad()
            losses["total"].backward()
            opt.step()
            total += losses["total"].item()
            n_batches += 1
        print(
            f"[Phase 4 | epoch {epoch+1}/{cfg.train.epochs_phase4}] joint loss = {total / max(n_batches,1):.4f}"
        )


def train_world_model(
    train_loader: DataLoader,
    cfg: PipelineConfig,
    risk_pos_weight: Optional[float] = None,
) -> WorldModel:
    device = torch.device(
        cfg.train.device if torch.cuda.is_available() else "cpu"
    )
    torch.manual_seed(cfg.train.seed)

    model = WorldModel(cfg).to(device)
    risk_weight_tensor: Optional[torch.Tensor] = None
    if risk_pos_weight is not None:
        risk_weight_tensor = torch.tensor(risk_pos_weight, device=device)

    phase1_representation(model, train_loader, cfg, device)
    phase2_dynamics(model, train_loader, cfg, device)
    phase3_risk_stage(model, train_loader, cfg, device, risk_weight_tensor)
    phase4_joint_finetune(model, train_loader, cfg, device, risk_weight_tensor)
    return model
