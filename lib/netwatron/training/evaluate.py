"""
training/evaluate.py
-----------------------
Sec. 58-63: evaluation.
  58.1 Dynamics metrics:  MAE/RMSE per forecast step t+1..t+K, error-vs-horizon curve
  59   Risk metrics:      Precision/Recall/F1/PR-AUC/ROC-AUC/FPR (PR-AUC primary, Sec. 59,
                           given severe class imbalance)
  60   Stage metrics:     Macro-F1, per-class recall, confusion matrix
  61   Lead time:         LeadTime = T_a - T_w  (attack escalation time minus first
                           reliable-warning time)
  62   Unseen-attack:     train on known families, evaluate on held-out ones

Sec. 63-66: the baseline ladder + representation/dynamics/threat ablations.
Per Sec. 77 item 8, baselines 1-4 are direct/history classifiers -- they are
NEVER substituted for the Dynamics Model itself; they exist purely as
points of comparison to justify Hypothesis 1 (Sec. 74): "threat should be
assessed over the world model's future trajectory."
"""

from typing import Any, cast

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix,
    f1_score,
)

# ---------------------------------------------------------------------------
# Dynamics metrics (Sec. 58.1)
# ---------------------------------------------------------------------------


def dynamics_error_by_horizon(z_hat: np.ndarray, z_actual: np.ndarray) -> dict:
    """z_hat, z_actual: [N, K, d]. Returns per-step MAE/RMSE, useful for
    plotting the "error growth vs horizon" curve the design doc asks for."""
    err = z_hat - z_actual
    mae = np.abs(err).mean(axis=(0, 2))  # [K]
    rmse = np.sqrt((err**2).mean(axis=(0, 2)))  # [K]
    return {
        "mae_per_step": mae,
        "rmse_per_step": rmse,
        "mae_overall": float(mae.mean()),
        "rmse_overall": float(rmse.mean()),
    }


# ---------------------------------------------------------------------------
# Risk metrics (Sec. 59)
# ---------------------------------------------------------------------------


def risk_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=cast(Any, 0)
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    pr_auc = (
        average_precision_score(y_true, y_prob)
        if len(np.unique(y_true)) > 1
        else float("nan")
    )
    roc_auc = (
        roc_auc_score(y_true, y_prob)
        if len(np.unique(y_true)) > 1
        else float("nan")
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "fpr": fpr,
    }


# ---------------------------------------------------------------------------
# Stage metrics (Sec. 60)
# ---------------------------------------------------------------------------


def stage_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, num_stages: int
) -> dict:
    macro_f1 = f1_score(
        y_true, y_pred, average="macro", zero_division=cast(Any, 0)
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_stages)))
    per_class_recall = np.diag(cm) / cm.sum(axis=1).clip(min=1)
    return {
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "per_class_recall": per_class_recall,
    }


# ---------------------------------------------------------------------------
# Early-warning lead time (Sec. 61)
# ---------------------------------------------------------------------------


def early_warning_lead_time(
    threat_scores: np.ndarray,
    timestamps: np.ndarray,
    attack_escalation_time,
    threshold: float,
) -> float:
    """threat_scores: [T] threat score time series for one incident.
    timestamps: [T] aligned wall-clock (or relative-seconds) timestamps.
    attack_escalation_time (T_a): when the attack is considered to have
    actually escalated (from ground truth / analyst annotation).

    LeadTime = T_a - T_w, where T_w is the first time the threat score
    crosses `threshold` and STAYS at/above it up to T_a (a simple
    "first sustained crossing" rule to avoid flagging on noise).
    Returns NaN if no qualifying warning occurred before T_a (no lead time
    to report), and a NEGATIVE number if the first crossing occurred after
    T_a (late warning).
    """
    above = threat_scores >= threshold
    first_warn_idx = None
    for i in range(len(above)):
        if above[i] and (
            i == len(above) - 1
            or above[i:].all()
            or above[i : i + max(1, len(above) // 20)].all()
        ):
            first_warn_idx = i
            break
    if first_warn_idx is None:
        return float("nan")
    t_w = timestamps[first_warn_idx]
    return float(attack_escalation_time - t_w)


# ---------------------------------------------------------------------------
# Sec. 63: baseline ladder
# ---------------------------------------------------------------------------


class Baseline1_LightGBM:
    """Current state -> LightGBM -> Risk."""

    def __init__(self, **lgbm_kwargs):
        import lightgbm as lgb  # pyright: ignore[reportMissingImports]

        self.model = lgb.LGBMClassifier(**lgbm_kwargs)

    def fit(self, z_now, y):
        self.model.fit(z_now, y)

    def predict_proba(self, z_now):
        return self.model.predict_proba(z_now)[:, 1]


class Baseline2_MLP(torch.nn.Module):
    """Current state -> MLP -> Risk."""

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, z_now):
        return self.net(z_now).squeeze(-1)


class Baseline3_HistoryRNN(torch.nn.Module):
    """History -> GRU/LSTM -> Risk (no future rollout)."""

    def __init__(self, dim: int, hidden: int = 128, cell: str = "gru"):
        super().__init__()
        rnn_cls = torch.nn.GRU if cell == "gru" else torch.nn.LSTM
        self.rnn = rnn_cls(dim, hidden, batch_first=True)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, z_history):
        out, _ = self.rnn(z_history)
        return self.head(out[:, -1, :]).squeeze(-1)


class Baseline4_HistoryTransformer(torch.nn.Module):
    """History -> Transformer -> Risk (no future rollout)."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        hidden: int = 256,
    ):
        super().__init__()
        layer = torch.nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=hidden,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = torch.nn.Linear(dim, 1)

    def forward(self, z_history):
        out = self.encoder(z_history)
        return self.head(out[:, -1, :]).squeeze(-1)


BASELINE_LADDER_DESCRIPTIONS = {
    "baseline_1": "Current state -> LightGBM -> Risk",
    "baseline_2": "Current state -> MLP -> Risk",
    "baseline_3": "History -> GRU/LSTM -> Risk",
    "baseline_4": "History -> Transformer -> Risk",
    "baseline_5": "World Model: History -> Future states -> Risk",
    "baseline_6": "World Model + Surprise: History -> Future trajectory -> Risk + Surprise",
    "baseline_7": "Final Candidate: World Model + Surprise + Stage Consistency",
}
