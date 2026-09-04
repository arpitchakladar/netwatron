"""
models/surprise_consistency.py
---------------------------------
Stage M (Sec. 44-45): dynamics surprise -- how unexpected is the actual next
state relative to what the Dynamics Model predicted. This is the model's
primary lever for unseen/zero-day-*like* pattern flags (Sec. 45), but per
Sec. 77 item 10 we never claim this constitutes proven zero-day detection.

Stage L (Sec. 43): stage transition consistency -- structural plausibility
of the predicted ATT&CK-stage sequence under a learned/estimated transition
prior P(C_{t+1} | C_t).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def dynamics_surprise(
    z_actual_next: torch.Tensor, z_hat_next: torch.Tensor, reduction: str = "l2"
) -> torch.Tensor:
    """Surprise_{t+1} = D(z_{t+1}, z_hat_{t+1})  (Sec. 44).

    z_actual_next, z_hat_next: [B, d] (one step) or [B, K, d] (per-step over
    the horizon, once ground-truth future states become available at
    evaluation/online time).
    Returns per-example (or per-example-per-step) surprise, NOT yet
    normalized/calibrated -- see ThreatCalibration for that.
    """
    if reduction == "l2":
        return torch.linalg.vector_norm(z_actual_next - z_hat_next, dim=-1)
    if reduction == "mse":
        return F.mse_loss(z_hat_next, z_actual_next, reduction="none").mean(
            dim=-1
        )
    raise ValueError(reduction)


class GaussianNLLSurprise(nn.Module):
    """Optional probabilistic variant: Surprise = -log p(z_{t+1} | z_t),
    modelling p as a diagonal Gaussian with a learned, state-conditioned
    log-variance (Sec. 44, second formulation). Falls back to plain L2 if
    you don't want to train this extra head."""

    def __init__(self, state_dim: int):
        super().__init__()
        self.log_var_head = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, state_dim),
        )

    def forward(
        self,
        z_context: torch.Tensor,
        z_actual_next: torch.Tensor,
        z_hat_next: torch.Tensor,
    ) -> torch.Tensor:
        log_var = self.log_var_head(z_context).clamp(min=-8, max=8)
        var = torch.exp(log_var)
        nll = 0.5 * (((z_actual_next - z_hat_next) ** 2) / var + log_var)
        return nll.mean(
            dim=-1
        )  # per-example negative log-likelihood ("surprise")


class StageTransitionConsistency(nn.Module):
    """Sec. 43: C_consistency = sum_i log P(C_{t+i+1} | C_{t+i}).

    The transition matrix can be (a) fixed from a hand-specified ATT&CK
    tactic ordering prior, or (b) learned empirically from training-set
    stage-bucket co-occurrence. We support both; default is learned with a
    Dirichlet-smoothed prior so an unseen transition isn't -inf."""

    def __init__(
        self, num_stages: int, learnable: bool = True, smoothing: float = 1.0
    ):
        super().__init__()
        self.num_stages = num_stages
        self.smoothing = smoothing
        init = torch.full((num_stages, num_stages), smoothing)
        if learnable:
            self.log_transition_logits = nn.Parameter(init.log())
        else:
            self.register_buffer("log_transition_logits", init.log())

    def transition_log_probs(self) -> torch.Tensor:
        return F.log_softmax(
            self.log_transition_logits, dim=-1
        )  # [C, C], rows sum to 1 in prob space

    def forward(self, stage_probs: torch.Tensor) -> torch.Tensor:
        """stage_probs: [B, K, C] (softmax over stage logits from StageHead).
        Returns per-example consistency score [B] = sum_i E[log P(C_{i+1}|C_i)]
        under the predicted stage distributions (soft, differentiable
        version of Sec. 43's sum-of-log-transition-probs)."""
        log_T = self.transition_log_probs()  # [C, C]
        B, K, C = stage_probs.shape
        total = torch.zeros(B, device=stage_probs.device)
        for i in range(K - 1):
            p_i = stage_probs[:, i, :]  # [B, C]
            p_ip1 = stage_probs[:, i + 1, :]  # [B, C]
            # E_{c~p_i, c'~p_ip1}[log P(c'|c)] = p_i @ log_T @ p_ip1^T (diag)
            expected = torch.einsum("bc,cd,bd->b", p_i, log_T, p_ip1)
            total = total + expected
        return total

    def fit_prior_from_labels(self, stage_label_sequences: torch.Tensor):
        """Optional: warm-start the transition matrix from empirical counts
        over observed (pseudo-)stage-bucket sequences, stage_label_sequences:
        [N, K] long tensor of stage bucket ids."""
        counts = torch.full((self.num_stages, self.num_stages), self.smoothing)
        seqs = stage_label_sequences
        for i in range(seqs.shape[1] - 1):
            a = seqs[:, i]
            b = seqs[:, i + 1]
            idx = a * self.num_stages + b
            flat_counts = torch.bincount(
                idx, minlength=self.num_stages**2
            ).float()
            counts += flat_counts.view(self.num_stages, self.num_stages)
        with torch.no_grad():
            self.log_transition_logits.copy_(counts.log())
