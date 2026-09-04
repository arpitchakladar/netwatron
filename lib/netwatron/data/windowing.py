"""
data/windowing.py
------------------
Stage B of the design doc (Sec. 12-14): time-window construction.

Xt = {x_t,1, ..., x_t,Nt}   where Nt varies window-to-window (Sec. 13).

CRITICAL ORDERING RULE (Sec. 11): this module must only ever be called on
data that has ALREADY been split into train/val/test (see
data.preprocessing.temporal_split). Building windows first and splitting
afterwards leaks near-duplicate overlapping windows across splits.

Each window also carries:
  - a validity mask M[t, i] in {0, 1}  (1 = real flow, 0 = padding)     (Sec. 14)
  - per-flow relative time features tau_i = t_i - window_start and
    delta_i = t_i - t_{i-1}                                              (Sec. 15)
These relative times -- NOT raw absolute timestamps -- are what the Flow
Encoder's time-encoding branch consumes (Sec. 77, item 2).
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclasses.dataclass
class WindowBatch:
    """A batch of time windows, padded to a common flow count.

    X:      float32 [B, N_max, D]   flow features (already Stage-A preprocessed)
    tau:    float32 [B, N_max]      time-since-window-start, seconds
    delta:  float32 [B, N_max]      time-since-previous-flow, seconds
    mask:   float32 [B, N_max]      1 = real flow, 0 = padding
    n_real: int64   [B]             number of real flows per window (for reference)
    window_start_ts: List[pd.Timestamp]
    window_index: List[int]         index of this window in the global window sequence
    """

    X: np.ndarray
    tau: np.ndarray
    delta: np.ndarray
    mask: np.ndarray
    n_real: np.ndarray
    window_start_ts: list
    window_index: list


def build_windows(
    X: np.ndarray,
    timestamps: pd.Series,
    window_seconds: float,
    max_flows_per_window: int,
    stride_seconds: Optional[float] = None,
) -> WindowBatch:
    """Group already-preprocessed, already-split flows into fixed-duration
    time windows and pad to a common flow count within this call.

    NOTE: `timestamps` must be sorted ascending (guaranteed if this is
    called on the output of preprocessing.transform() for a single split).
    """
    ts = pd.to_datetime(timestamps).reset_index(drop=True)
    assert (
        ts.is_monotonic_increasing
    ), "timestamps must be sorted before windowing"

    t0 = ts.iloc[0]
    rel_seconds = (ts - t0).dt.total_seconds().values

    stride = stride_seconds if stride_seconds is not None else window_seconds
    n_windows = int(np.floor((rel_seconds[-1]) / stride)) + 1

    windows_X, windows_tau, windows_delta, windows_mask, windows_n = (
        [],
        [],
        [],
        [],
        [],
    )
    window_start_ts = []

    flow_ptr = 0
    n_flows_total = len(rel_seconds)
    for w in range(n_windows):
        w_start = w * stride
        w_end = w_start + window_seconds
        # advance pointer past flows before this window (only matters if stride < window_seconds
        # i.e. overlapping windows are requested)
        idx = np.where((rel_seconds >= w_start) & (rel_seconds < w_end))[0]
        if len(idx) == 0:
            continue
        n_real = min(len(idx), max_flows_per_window)
        sel = idx[:n_real]

        Xw = np.zeros((max_flows_per_window, X.shape[1]), dtype=np.float32)
        tauw = np.zeros((max_flows_per_window,), dtype=np.float32)
        deltaw = np.zeros((max_flows_per_window,), dtype=np.float32)
        maskw = np.zeros((max_flows_per_window,), dtype=np.float32)

        Xw[:n_real] = X[sel]
        tauw[:n_real] = rel_seconds[sel] - w_start
        prev = np.concatenate([[rel_seconds[sel][0]], rel_seconds[sel][:-1]])
        deltaw[:n_real] = rel_seconds[sel] - prev
        maskw[:n_real] = 1.0

        windows_X.append(Xw)
        windows_tau.append(tauw)
        windows_delta.append(deltaw)
        windows_mask.append(maskw)
        windows_n.append(n_real)
        window_start_ts.append(t0 + pd.Timedelta(seconds=w_start))

    return WindowBatch(
        X=(
            np.stack(windows_X)
            if windows_X
            else np.zeros((0, max_flows_per_window, X.shape[1]), np.float32)
        ),
        tau=(
            np.stack(windows_tau)
            if windows_tau
            else np.zeros((0, max_flows_per_window), np.float32)
        ),
        delta=(
            np.stack(windows_delta)
            if windows_delta
            else np.zeros((0, max_flows_per_window), np.float32)
        ),
        mask=(
            np.stack(windows_mask)
            if windows_mask
            else np.zeros((0, max_flows_per_window), np.float32)
        ),
        n_real=np.array(windows_n, dtype=np.int64),
        window_start_ts=window_start_ts,
        window_index=list(range(len(windows_n))),
    )


def attach_window_labels(
    windows: WindowBatch,
    y: np.ndarray,
    timestamps: pd.Series,
    window_seconds: float,
    stride_seconds: Optional[float] = None,
) -> np.ndarray:
    """Derive a per-window label for supervised heads (e.g. any-malicious-in-window).

    This is a coarse aggregation used for the *baseline ladder* classifiers
    and for sanity-checking the state representation; the world-model risk
    head is trained against *future* windows, not this same-window label
    (see training/train.py). Not attack-specific feature engineering --
    purely a label aggregation.
    """
    ts = pd.to_datetime(timestamps).reset_index(drop=True)
    t0 = ts.iloc[0]
    rel_seconds = (ts - t0).dt.total_seconds().values
    stride = stride_seconds if stride_seconds is not None else window_seconds

    labels = []
    for w in windows.window_index:
        w_start = w * stride
        w_end = w_start + window_seconds
        idx = np.where((rel_seconds >= w_start) & (rel_seconds < w_end))[0]
        labels.append(int((y[idx] != 0).any()) if len(idx) else 0)
    return np.array(labels, dtype=np.int64)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 2000
    X = rng.normal(size=(n, 80)).astype(np.float32)
    ts = pd.Series(pd.date_range("2018-02-14", periods=n, freq="200ms"))
    y = (rng.random(n) < 0.05).astype(np.int64)

    wb = build_windows(X, ts, window_seconds=10.0, max_flows_per_window=64)
    labels = attach_window_labels(wb, y, ts, window_seconds=10.0)
    print("num windows:", wb.X.shape[0], "flows/window shape:", wb.X.shape[1:])
    print(
        "mean flows per window:",
        wb.n_real.mean(),
        "window-label positive rate:",
        labels.mean(),
    )
