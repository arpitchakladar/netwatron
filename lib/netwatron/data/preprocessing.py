"""
data/preprocessing.py
----------------------
Stage A of the design doc (Sec. 4-11): basic, attack-agnostic preprocessing.

Pipeline (Sec. 4):
    sort/preserve time -> label encoding -> temporal split -> log transform
    -> NaN/Inf handling -> standard scaling

Hard rules enforced here (do not relax these, see config.py docstring / Sec. 77):
  - Timestamp is kept OUT of the feature matrix; it is only used for ordering
    and window assignment (Sec. 5/6).
  - Protocol is one-hot encoded, never treated as an ordinary continuous
    scalar (Sec. 7).
  - The log1p / median / mean / std statistics are fit on the TRAIN split
    only, then applied to val/test (Sec. 8/9/11) -- this is what prevents
    temporal leakage.
  - The split is chronological, not random (Sec. 10/11).
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Optional, cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from ..config import DataConfig


@dataclasses.dataclass
class PreprocessArtifacts:
    """Everything fit on the training split; needed to transform new data
    consistently (and to reproduce the pipeline for the eventual full,
    multi-attack-class dataset -- Sec. 5.1)."""

    label_encoder: LabelEncoder
    protocol_encoder: OneHotEncoder
    log1p_columns: list
    train_median: pd.Series
    scaler: StandardScaler
    numeric_columns: list
    feature_names_out: list


def _select_log1p_columns(
    train_df: pd.DataFrame, numeric_cols, skew_threshold: float
) -> list:
    """Sec. 7: x_min >= 0 and skew(x) > 1, computed on TRAIN ONLY."""
    cols = []
    for c in numeric_cols:
        col = train_df[c]
        if col.min(skipna=True) >= 0 and col.skew(skipna=True) > skew_threshold:
            cols.append(c)
    return cols


def temporal_split(
    df: pd.DataFrame,
    timestamp_col: str,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> tuple:
    """Sec. 10/11: chronological split performed BEFORE any windowing.

    Never shuffle here. If the dataframe is not already sorted by time,
    sort it first (Sec. 4 preserves time order).
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train : n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val :].copy()
    return train_df, val_df, test_df


def fit_preprocessing(
    train_df: pd.DataFrame, cfg: DataConfig
) -> PreprocessArtifacts:
    """Fit label encoder, protocol one-hot, log1p column list, median
    imputer, and scaler -- all using ONLY the training split."""
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df[cfg.label_col].astype(str))

    protocol_encoder = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False
    )
    protocol_encoder.fit(train_df[[cfg.protocol_col]])

    numeric_cols = [
        c
        for c in train_df.columns
        if c not in cfg.excluded_input_cols + [cfg.protocol_col]
        and pd.api.types.is_numeric_dtype(train_df[c])
    ]

    log1p_cols = _select_log1p_columns(
        train_df, numeric_cols, cfg.skew_threshold
    )

    working = train_df[numeric_cols].copy()
    for c in log1p_cols:
        values = np.asarray(working[c], dtype=float)
        working[c] = np.log1p(np.maximum(values, 0.0))
    working = working.replace([np.inf, -np.inf], np.nan)
    train_median = working.median(numeric_only=True)
    working = working.fillna(train_median)

    scaler = StandardScaler()
    scaler.fit(working.values)

    protocol_feature_names = list(
        protocol_encoder.get_feature_names_out([cfg.protocol_col])
    )
    feature_names_out = numeric_cols + protocol_feature_names

    return PreprocessArtifacts(
        label_encoder=label_encoder,
        protocol_encoder=protocol_encoder,
        log1p_columns=log1p_cols,
        train_median=train_median,
        scaler=scaler,
        numeric_columns=numeric_cols,
        feature_names_out=feature_names_out,
    )


def transform_features(
    df: pd.DataFrame, cfg: DataConfig, art: PreprocessArtifacts
) -> np.ndarray:
    """Apply train-fitted feature transforms without requiring labels/timestamps."""
    # ``reindex`` deliberately retains the full training schema for streaming
    # inference. Columns absent from a lower-fidelity live parser become NaN
    # and are filled with the training median below, rather than changing the
    # feature order or dimensionality expected by the learned encoder.
    numeric = df.reindex(columns=art.numeric_columns).copy()
    for c in art.log1p_columns:
        values = np.asarray(numeric[c], dtype=float)
        numeric[c] = np.log1p(np.maximum(values, 0.0))
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(art.train_median)
    numeric_scaled = art.scaler.transform(numeric.values)

    protocol_oh = cast(
        np.ndarray, art.protocol_encoder.transform(df[[cfg.protocol_col]])
    )

    return np.concatenate([numeric_scaled, protocol_oh], axis=1).astype(
        np.float32
    )


def transform(df: pd.DataFrame, cfg: DataConfig, art: PreprocessArtifacts):
    """Apply a fitted preprocessor to a labelled, timestamped dataset split."""
    X = transform_features(df, cfg, art)
    y = art.label_encoder.transform(df[cfg.label_col].astype(str)).astype(
        np.int64
    )
    timestamps = df[cfg.timestamp_col].reset_index(drop=True)
    return X, y, timestamps


def run_stage_a(df: pd.DataFrame, cfg: DataConfig) -> dict[str, Any]:
    """Convenience wrapper: full Stage A end to end.

    Returns dict with X/y/timestamps for train/val/test plus the fitted
    PreprocessArtifacts (needed later to encode streaming/live traffic the
    same way, and to invert label encoding for reporting).
    """
    train_df, val_df, test_df = temporal_split(
        df, cfg.timestamp_col, cfg.train_frac, cfg.val_frac, cfg.test_frac
    )
    art = fit_preprocessing(train_df, cfg)

    out: dict[str, Any] = {"artifacts": art}
    for name, split_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        X, y, ts = transform(split_df, cfg, art)
        out[name] = {"X": X, "y": y, "timestamps": ts}

    print(
        f"[Stage A] feature dim = {X.shape[1]} "
        f"({len(art.numeric_columns)} numeric incl. "
        f"{len(art.log1p_columns)} log1p'd, "
        f"{X.shape[1] - len(art.numeric_columns)} protocol one-hot dims)"
    )
    return out


if __name__ == "__main__":
    # Smoke test with synthetic data matching the shapes reported in Sec. 79
    # (78 input columns before preprocessing: 77 numeric + Protocol + Timestamp + Label).
    rng = np.random.default_rng(0)
    n = 5000
    synth = pd.DataFrame(
        {f"feat_{i}": rng.exponential(scale=2.0, size=n) for i in range(76)}
    )
    synth["Dst Port"] = rng.integers(0, 65535, size=n)
    synth["Protocol"] = rng.choice([0, 6, 17], size=n)
    synth["Timestamp"] = pd.date_range("2018-02-14", periods=n, freq="s")
    labels = np.where(rng.random(n) < 0.1, "Bot", "Benign")
    synth["Label"] = labels

    cfg = DataConfig()
    result = run_stage_a(synth, cfg)
    print("train X shape:", result["train"]["X"].shape)
    print("val   X shape:", result["val"]["X"].shape)
    print("test  X shape:", result["test"]["X"].shape)
