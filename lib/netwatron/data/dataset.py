"""
data/dataset.py
-----------------
Turns a WindowBatch (Stage B output, chronologically ordered, one split at a
time) into training examples for the World Model:

    history:  windows[t-L+1 : t+1]   -> fed through Flow Encoder + State
               pooling -> Zhistory in R^{L x 128}                     (Sec. 24)
    future:   windows[t+1 : t+1+K]   -> used as supervision targets for
               the Dynamics Model's z-space and decoded-state losses
               (Sec. 26-32), and to derive the window-level risk label.

We keep the raw (unpooled) flow tensors for every window in the sequence
because the Flow Encoder + State pooling module is trained jointly (Phase 1)
rather than pre-baked into a fixed feature -- see training/train.py.
"""

from __future__ import annotations

import dataclasses
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from .windowing import WindowBatch


@dataclasses.dataclass
class SequenceSample:
    hist_X: np.ndarray  # [L, N_max, D]
    hist_mask: np.ndarray  # [L, N_max]
    hist_tau: np.ndarray  # [L, N_max]
    hist_delta: np.ndarray  # [L, N_max]
    fut_X: np.ndarray  # [K, N_max, D]
    fut_mask: np.ndarray  # [K, N_max]
    fut_tau: np.ndarray  # [K, N_max]
    fut_delta: np.ndarray  # [K, N_max]
    risk_label: int  # 1 if any future window in horizon is malicious
    per_step_risk_label: np.ndarray  # [K] per-step malicious indicator (RISK-1)
    stage_labels: (
        np.ndarray
    )  # [K] ATT&CK-aligned stage bucket per future window (NOT ground truth, Sec. 42)


class WindowSequenceDataset(Dataset):
    """history_len=L, horizon=K. Produces one sample per valid anchor t such
    that both the L-window history and the K-window future fit inside the
    (already train/val/test-split) window sequence -- see Sec. 11/24."""

    def __init__(
        self,
        windows: WindowBatch,
        window_labels: np.ndarray,
        history_len: int,
        horizon: int,
        num_stage_buckets: int = 5,
    ):
        self.windows = windows
        self.window_labels = window_labels
        self.L = history_len
        self.K = horizon
        self.num_stage_buckets = num_stage_buckets
        n = windows.X.shape[0]
        self.valid_anchors = (
            list(range(self.L - 1, n - self.K))
            if n >= (self.L + self.K)
            else []
        )

    def __len__(self):
        return len(self.valid_anchors)

    def _pseudo_stage_bucket(self, t: int) -> int:
        """Placeholder ATT&CK-aligned stage bucket derived only from the
        (already available) window label + a coarse escalation counter.
        This is explicitly an inference target, never claimed as ground
        truth (Sec. 42): CSE-CIC-IDS2018 has no per-flow stage annotation.
        Replace with a real stage-mapping heuristic (e.g. MITRE ATT&CK
        technique -> tactic lookup) once available.
        """
        if self.window_labels[t] == 0:
            return 0  # "benign / no observed stage"
        # crude escalation proxy: how many of the last few windows were malicious
        lo = max(0, t - self.num_stage_buckets + 1)
        recent = self.window_labels[lo : t + 1]
        bucket = min(int(recent.sum()), self.num_stage_buckets - 1)
        return bucket

    def __getitem__(self, idx: int) -> SequenceSample:
        t = self.valid_anchors[idx]
        hist_slice = slice(t - self.L + 1, t + 1)
        fut_slice = slice(t + 1, t + 1 + self.K)

        fut_labels = self.window_labels[fut_slice]
        stage_labels = np.array(
            [self._pseudo_stage_bucket(t + 1 + k) for k in range(self.K)],
            dtype=np.int64,
        )

        return SequenceSample(
            hist_X=self.windows.X[hist_slice],
            hist_mask=self.windows.mask[hist_slice],
            hist_tau=self.windows.tau[hist_slice],
            hist_delta=self.windows.delta[hist_slice],
            fut_X=self.windows.X[fut_slice],
            fut_mask=self.windows.mask[fut_slice],
            fut_tau=self.windows.tau[fut_slice],
            fut_delta=self.windows.delta[fut_slice],
            risk_label=int(fut_labels.any()),
            per_step_risk_label=fut_labels.astype(np.int64),
            stage_labels=stage_labels,
        )


def collate_sequences(batch: List[SequenceSample]):
    def stack(field):
        return torch.from_numpy(np.stack([getattr(b, field) for b in batch]))

    return {
        "hist_X": stack("hist_X").float(),
        "hist_mask": stack("hist_mask").float(),
        "hist_tau": stack("hist_tau").float(),
        "hist_delta": stack("hist_delta").float(),
        "fut_X": stack("fut_X").float(),
        "fut_mask": stack("fut_mask").float(),
        "fut_tau": stack("fut_tau").float(),
        "fut_delta": stack("fut_delta").float(),
        "risk_label": torch.tensor([b.risk_label for b in batch]).float(),
        "per_step_risk_label": stack("per_step_risk_label").float(),
        "stage_labels": stack("stage_labels").long(),
    }
