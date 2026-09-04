"""
experiments/run_end_to_end.py
--------------------------------
Runs the ENTIRE pipeline once, top to bottom, on either:
  (a) real CSE-CIC-IDS2018 processed CSV data (point --csv at it), or
  (b) synthetic data shaped like Sec. 79's reported state
      (78 raw cols -> 80 preprocessed dims), so the code is runnable and
      testable even before the full multi-attack-class dataset is ready.

This corresponds to the "Immediate implementation state" -> "Next coding
stages" path described in Sec. 79:
    Timestamp -> time windows -> variable batches -> Flow Encoder ->
    Mean/Attention state -> [z_{t-L}] -> Dynamics Model -> z_hat_{t+1+K}

Usage:
    python scripts/train_model.py --synthetic
    python scripts/train_model.py --csv path/to/data.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "lib"))

from netwatron.config import PipelineConfig
from netwatron.data.dataset import WindowSequenceDataset, collate_sequences
from netwatron.data.preprocessing import run_stage_a
from netwatron.data.windowing import attach_window_labels, build_windows
from netwatron.training.evaluate import (
    dynamics_error_by_horizon,
    risk_metrics,
    stage_metrics,
)
from netwatron.training.train import train_world_model


def make_synthetic_dataframe(n: int = 20000, seed: int = 0) -> pd.DataFrame:
    """Matches the raw-column shape reported in Sec. 3-4: 77 numeric cols +
    Protocol (categorical, {0,6,17}) + Timestamp + Label (Benign/Bot only,
    per Sec. 4's current dataframe state)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {f"feat_{i}": rng.exponential(scale=2.0, size=n) for i in range(76)}
    )
    df["Dst Port"] = rng.integers(0, 65535, size=n).astype(float)
    df["Protocol"] = rng.choice([0, 6, 17], size=n)
    df["Timestamp"] = pd.date_range("2018-02-14", periods=n, freq="100ms")

    # inject a few synthetic "attack bursts" so windows have realistic,
    # temporally-clustered positive labels rather than i.i.d. noise
    labels = np.array(["Benign"] * n, dtype=object)
    burst_starts = rng.choice(np.arange(0, n - 500), size=6, replace=False)
    for s in burst_starts:
        length = rng.integers(200, 500)
        labels[s : s + length] = "Bot"
        # make attack-window features visibly different (NOT a manually
        # engineered attack signature fed to the model -- this only shapes
        # the synthetic data generator, the model never sees the label)
        for i in range(76):
            df.loc[s : s + length, f"feat_{i}"] *= rng.uniform(1.5, 4.0)
    df["Label"] = labels
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--n-synthetic", type=int, default=20000)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "netwatron_world_model.pt",
        help="output checkpoint; includes model, config, and fitted preprocessing",
    )
    args = parser.parse_args()

    cfg = PipelineConfig()

    if args.csv:
        df = pd.read_csv(args.csv, parse_dates=[cfg.data.timestamp_col])
    else:
        df = make_synthetic_dataframe(args.n_synthetic)

    # ---- Stage A: preprocessing (chronological split happens inside) ----
    stage_a = run_stage_a(df, cfg.data)
    cfg.flow_encoder.input_dim = len(stage_a["artifacts"].feature_names_out)

    # ---- Stage B: windowing, done independently per split (Sec. 11) ----
    splits = {}
    for name in ("train", "val", "test"):
        X, y, ts = (
            stage_a[name]["X"],
            stage_a[name]["y"],
            stage_a[name]["timestamps"],
        )
        windows = build_windows(
            X, ts, cfg.window.window_seconds, cfg.window.max_flows_per_window
        )
        window_labels = attach_window_labels(
            windows, y, ts, cfg.window.window_seconds
        )
        splits[name] = (windows, window_labels)
        print(
            f"[{name}] {windows.X.shape[0]} windows, "
            f"{window_labels.mean():.4f} positive-window rate"
        )

    train_windows, train_labels = splits["train"]
    val_windows, val_labels = splits["val"]

    train_ds = WindowSequenceDataset(
        train_windows,
        train_labels,
        cfg.dynamics.history_len,
        cfg.dynamics.horizon,
        cfg.stage.num_stages,
    )
    val_ds = WindowSequenceDataset(
        val_windows,
        val_labels,
        cfg.dynamics.history_len,
        cfg.dynamics.horizon,
        cfg.stage.num_stages,
    )

    if len(train_ds) == 0:
        raise RuntimeError(
            "Not enough windows for one history+horizon sequence -- lower "
            "window_seconds, raise n_synthetic, or shorten history_len/horizon."
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=collate_sequences,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        collate_fn=collate_sequences,
    )

    pos_rate = train_labels.mean().clip(min=1e-3)
    risk_pos_weight = (
        1 - pos_rate
    ) / pos_rate  # simple class-imbalance correction (Sec. 59 notes severe imbalance)

    model = train_world_model(
        train_loader, cfg, risk_pos_weight=float(risk_pos_weight)
    )

    # ---- Evaluation on held-out val split ----
    device = next(model.parameters()).device
    model.eval()
    all_z_hat, all_z_actual = [], []
    all_risk_prob, all_risk_label = [], []
    all_stage_pred, all_stage_true = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            all_z_hat.append(out["z_future_hat"].cpu().numpy())
            all_z_actual.append(out["z_future_actual"].cpu().numpy())
            risk_prob = out["risk_prob"]
            if risk_prob.dim() > 1:
                risk_prob = risk_prob.mean(dim=-1)
            all_risk_prob.append(risk_prob.cpu().numpy())
            all_risk_label.append(batch["risk_label"].cpu().numpy())
            all_stage_pred.append(out["stage_logits"].argmax(-1).cpu().numpy())
            all_stage_true.append(batch["stage_labels"].cpu().numpy())

    if all_z_hat:
        dyn_metrics = dynamics_error_by_horizon(
            np.concatenate(all_z_hat), np.concatenate(all_z_actual)
        )
        print(
            "\n[Dynamics] MAE per step:",
            np.round(dyn_metrics["mae_per_step"], 4),
        )
        print(
            "[Dynamics] RMSE per step:",
            np.round(dyn_metrics["rmse_per_step"], 4),
        )

        r_metrics = risk_metrics(
            np.concatenate(all_risk_label), np.concatenate(all_risk_prob)
        )
        print(
            "\n[Risk]",
            {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in r_metrics.items()
            },
        )

        s_metrics = stage_metrics(
            np.concatenate(all_stage_true).reshape(-1),
            np.concatenate(all_stage_pred).reshape(-1),
            cfg.stage.num_stages,
        )
        print("\n[Stage] macro-F1:", round(s_metrics["macro_f1"], 4))
        print(
            "[Stage] per-class recall:",
            np.round(s_metrics["per_class_recall"], 4),
        )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "preprocess_artifacts": stage_a["artifacts"],
        },
        args.checkpoint,
    )
    print(f"Saved checkpoint to {args.checkpoint}")
    return model


if __name__ == "__main__":
    main()
