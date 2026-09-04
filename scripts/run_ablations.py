"""
experiments/run_ablations.py
--------------------------------
Sec. 76-77: recommended order of experiments.
    Group 1: Representation R1 -> R2 (-> R3)
    Group 2: Dynamics D1 -> D2 (-> D3)
    Group 3: Horizons K = 1, 3, 5
    Group 4: Threat T1 -> T2 -> T3 -> T4 -> T5
    Group 5: Innovations T3 -> T4 -> T5 (subset of Group 4, kept for
             traceability against the doc's own numbering)
    Group 6: Unseen attacks G1 (known) -> G2 (held-out family) -- requires
             a dataset with >1 attack family; see note in run_unseen_attack_eval()
    Group 7: SSL variants (Stage O, SSL-1 -> SSL-2 -> SSL-3)

This script wires config.PipelineConfig variant switches into
train_world_model + evaluate, looping over each group and printing a
comparison table. It reuses the synthetic-data generator from
run_end_to_end.py so it is runnable standalone; swap in real data via
--csv the same way.
"""

import argparse
import copy
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
from netwatron.models.threat import make_threat_config
from netwatron.training.evaluate import dynamics_error_by_horizon, risk_metrics
from netwatron.training.train import train_world_model
from train_model import make_synthetic_dataframe


def build_loaders(df: pd.DataFrame, cfg: PipelineConfig):
    stage_a = run_stage_a(df, cfg.data)
    loaders = {}
    labels_by_split = {}
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
        ds = WindowSequenceDataset(
            windows,
            window_labels,
            cfg.dynamics.history_len,
            cfg.dynamics.horizon,
            cfg.stage.num_stages,
        )
        loaders[name] = DataLoader(
            ds,
            batch_size=cfg.train.batch_size,
            shuffle=(name == "train"),
            collate_fn=collate_sequences,
            drop_last=(name == "train"),
        )
        labels_by_split[name] = window_labels
    return loaders, labels_by_split


def evaluate_loader(model, loader, device):
    model.eval()
    all_z_hat, all_z_actual, all_risk_prob, all_risk_label = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            all_z_hat.append(out["z_future_hat"].cpu().numpy())
            all_z_actual.append(out["z_future_actual"].cpu().numpy())
            risk_prob = out["risk_prob"]
            if risk_prob.dim() > 1:
                risk_prob = risk_prob.mean(dim=-1)
            all_risk_prob.append(risk_prob.cpu().numpy())
            all_risk_label.append(batch["risk_label"].cpu().numpy())
    dyn = dynamics_error_by_horizon(
        np.concatenate(all_z_hat), np.concatenate(all_z_actual)
    )
    risk = risk_metrics(
        np.concatenate(all_risk_label), np.concatenate(all_risk_prob)
    )
    return dyn, risk


def run_group(
    name: str, base_cfg: PipelineConfig, loaders, variant_field_path, variants
):
    """variant_field_path: tuple of attribute names, e.g. ("state", "variant").
    Special-cased for ("threat", "variant"): ThreatConfig has extra boolean
    flags (use_surprise/use_stage_consistency) that must move together with
    the T1..T5 label, so we build the whole ThreatConfig via
    models.threat.make_threat_config instead of just setting one string."""
    print(f"\n{'='*70}\nGROUP: {name}\n{'='*70}")
    results = {}
    for v in variants:
        cfg = copy.deepcopy(base_cfg)
        if variant_field_path == ("threat", "variant"):
            cfg.threat = make_threat_config(v)
        else:
            obj = cfg
            for attr in variant_field_path[:-1]:
                obj = getattr(obj, attr)
            setattr(obj, variant_field_path[-1], v)

        # keep epochs small for a sweep; override for a real run
        cfg.train.epochs_phase1 = 2
        cfg.train.epochs_phase2 = 3
        cfg.train.epochs_phase3 = 3
        cfg.train.epochs_phase4 = 2

        try:
            model = train_world_model(loaders["train"], cfg)
            device = next(model.parameters()).device
            dyn, risk = evaluate_loader(model, loaders["val"], device)
            results[v] = {"rmse_overall": dyn["rmse_overall"], **risk}
            print(
                f"  variant={v:6s} | RMSE={dyn['rmse_overall']:.4f} | "
                f"PR-AUC={risk['pr_auc']:.4f} | ROC-AUC={risk['roc_auc']:.4f} | F1={risk['f1']:.4f}"
            )
        except Exception as e:
            print(f"  variant={v:6s} | FAILED: {e}")
            results[v] = {"error": str(e)}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--n_synthetic", type=int, default=20000)
    args = parser.parse_args()

    base_cfg = PipelineConfig()
    df = (
        pd.read_csv(args.csv, parse_dates=[base_cfg.data.timestamp_col])
        if args.csv
        else make_synthetic_dataframe(args.n_synthetic)
    )
    loaders, _ = build_loaders(df, base_cfg)

    all_results = {}
    # Group 1: representation (Sec. 76). R3/R4/R5 need bigger data/compute or IP columns; kept optional.
    all_results["representation"] = run_group(
        "Representation (R1 -> R2)",
        base_cfg,
        loaders,
        ("state", "variant"),
        ["R1", "R2"],
    )
    # Group 2: dynamics
    all_results["dynamics"] = run_group(
        "Dynamics (D1 -> D2)",
        base_cfg,
        loaders,
        ("dynamics", "variant"),
        ["D1", "D2"],
    )
    # Group 3: horizons
    all_results["horizon"] = run_group(
        "Horizon (K = 1, 3, 5)",
        base_cfg,
        loaders,
        ("dynamics", "horizon"),
        [1, 3, 5],
    )
    # Group 4/5: threat ablation
    all_results["threat"] = run_group(
        "Threat (T1 -> T5)",
        base_cfg,
        loaders,
        ("threat", "variant"),
        ["T1", "T2", "T3", "T4", "T5"],
    )

    print("\n\nSUMMARY:")
    for group, res in all_results.items():
        print(f"\n{group}:")
        for variant, metrics in res.items():
            print(f"  {variant}: {metrics}")

    return all_results


if __name__ == "__main__":
    main()
