"""
models/world_model.py
------------------------
Assembles the "Final Recommended Prototype" (Sec. 74, Figure 5):

    Flow Encoder -> Attention Pooling -> z_t -> [z_{t-L}..z_t]
    -> Temporal Transformer -> K future latent states
    -> {State Decoder, Risk Trajectory, Stage Sequence}
    -> Dynamics Surprise, Stage Consistency -> Final Threat Score

Formal summary this module implements (Sec. 71):
    Observation model: z_t        = E_phi(X_t)
    Dynamics model:     z_hat_{t+1:t+K} = D_theta(z_{t-L+1:t})
    Decoder:            S_hat_{t+i}     = G(z_hat_{t+i})
    Threat model:       R              = R_psi(z_hat_{t+1:t+K})
    Stage model:        C              = C_omega(z_hat_{t+1:t+K})
    Surprise:           S_t            = D(z_{t+1}, z_hat_{t+1})
    Final threat:       T              = F(R, S, C)
"""

import torch
import torch.nn as nn

from ..config import PipelineConfig
from .flow_encoder import FlowEncoder
from .state_pooling import build_state_pooling
from .dynamics import build_dynamics_model
from .heads import FutureStateDecoder, RiskHead, StageHead
from .surprise_consistency import dynamics_surprise, StageTransitionConsistency
from .threat import ThreatCalibration


class WorldModel(nn.Module):
    def __init__(self, cfg: PipelineConfig):
        super().__init__()
        self.cfg = cfg
        self.flow_encoder = FlowEncoder(cfg.flow_encoder)
        self.state_pool = build_state_pooling(
            cfg.state, cfg.data.has_ip_columns
        )
        # R5 (multi-token) flattens M tokens of width state_dim into a single
        # M*state_dim vector (see MultiTokenPooling docstring), so every
        # downstream module needs the WIDENED dimensionality, not the base
        # per-token width used internally by the pooling attention itself.
        effective_state_dim = (
            cfg.state.state_dim * cfg.state.num_tokens
            if cfg.state.variant == "R5"
            else cfg.state.state_dim
        )
        self.state_dim = effective_state_dim
        self.dynamics = build_dynamics_model(cfg.dynamics, effective_state_dim)
        self.state_decoder = FutureStateDecoder(effective_state_dim)
        self.risk_head = RiskHead(
            cfg.risk, effective_state_dim, history_len=cfg.dynamics.history_len
        )
        self.stage_head = StageHead(
            cfg.stage, effective_state_dim, cfg.dynamics.horizon
        )
        self.stage_consistency = StageTransitionConsistency(
            cfg.stage.num_stages
        )
        self.threat_calibration = ThreatCalibration(cfg.threat)

    def encode_window(self, X, mask, tau, delta) -> torch.Tensor:
        """X: [..., N, 80] -> z: [..., d] (or [..., M, d] for R5)."""
        h = self.flow_encoder(X, tau, delta)
        return self.state_pool(h, mask)

    def encode_sequence(
        self, X_seq, mask_seq, tau_seq, delta_seq
    ) -> torch.Tensor:
        """X_seq: [B, T, N, 80] -> z_seq: [B, T, d]. Encodes every window in
        the sequence with the SAME (shared) flow encoder + pooling, as
        required for zt to be a consistent representation across time
        (Sec. 24)."""
        B, T, N, D = X_seq.shape
        flat_X = X_seq.reshape(B * T, N, D)
        flat_mask = mask_seq.reshape(B * T, N)
        flat_tau = tau_seq.reshape(B * T, N)
        flat_delta = delta_seq.reshape(B * T, N)
        z_flat = self.encode_window(flat_X, flat_mask, flat_tau, flat_delta)
        return z_flat.reshape(B, T, -1)

    def forward(self, batch: dict) -> dict:
        z_history = self.encode_sequence(
            batch["hist_X"],
            batch["hist_mask"],
            batch["hist_tau"],
            batch["hist_delta"],
        )  # [B, L, d]
        z_future_actual = self.encode_sequence(
            batch["fut_X"],
            batch["fut_mask"],
            batch["fut_tau"],
            batch["fut_delta"],
        )  # [B, K, d]  (supervision target)

        z_future_hat = self.dynamics(z_history)  # [B, K, d]
        s_hat_future = self.state_decoder(
            z_future_hat
        )  # [B, K, d] observable projection

        stage_logits = self.stage_head(z_future_hat)  # [B, K, C]
        stage_probs = torch.softmax(stage_logits, dim=-1)
        consistency = self.stage_consistency(stage_probs)  # [B]

        risk_kwargs = {}
        if self.cfg.risk.variant == "RISK-1":
            risk_kwargs["z_future"] = z_future_hat
        elif self.cfg.risk.variant == "RISK-2":
            risk_kwargs["z_future"] = z_future_hat
        elif self.cfg.risk.variant == "RISK-3":
            risk_kwargs["z_now"] = z_history[:, -1, :]
        elif self.cfg.risk.variant == "RISK-4":
            risk_kwargs["z_history"] = z_history
        risk_prob, risk_logit = self.risk_head(**risk_kwargs)

        # Dynamics surprise (Sec. 44): only the *first* future step has a
        # true one-step-ahead target available at train time; use step 0.
        surprise = dynamics_surprise(
            z_future_actual[:, 0, :], z_future_hat[:, 0, :]
        )

        if self.cfg.risk.variant == "RISK-1":
            traj_risk_prob = risk_prob.mean(
                dim=-1
            )  # collapse per-step curve to a scalar for threat calib
        else:
            traj_risk_prob = risk_prob

        threat_prob, threat_logit = self.threat_calibration(
            traj_risk_prob,
            surprise=surprise if self.cfg.threat.use_surprise else None,
            consistency=(
                consistency if self.cfg.threat.use_stage_consistency else None
            ),
        )

        return {
            "z_history": z_history,
            "z_future_actual": z_future_actual,
            "z_future_hat": z_future_hat,
            "s_hat_future": s_hat_future,
            "stage_logits": stage_logits,
            "stage_consistency": consistency,
            "risk_prob": risk_prob,
            "risk_logit": risk_logit,
            "surprise": surprise,
            "threat_prob": threat_prob,
            "threat_logit": threat_logit,
        }
