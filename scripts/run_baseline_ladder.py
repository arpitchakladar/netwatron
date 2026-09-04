"""
experiments/run_baseline_ladder.py
--------------------------------------
Sec. 63: Critical Experimental Baseline Ladder.

    Baseline 1: Current state -> LightGBM -> Risk
    Baseline 2: Current state -> MLP -> Risk
    Baseline 3: History -> GRU/LSTM -> Risk
    Baseline 4: History -> Transformer -> Risk
    Baseline 5: World Model: History -> Future states -> Risk
    Baseline 6: World Model + Surprise
    Baseline 7: Final Candidate: World Model + Surprise + Stage Consistency

Baselines 1-4 are simple classifiers trained directly here (they do NOT use
models.world_model.WorldModel at all -- per Sec. 77 item 8, the Dynamics
Model must never be replaced by a direct classifier; these exist solely as
comparison points to test Hypothesis 1, Sec. 74: "threat should be assessed
over the world model's future trajectory"). Baselines 5-7 reuse WorldModel
with threat.variant set to T3/T4/T5 respectively.
"""

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
from netwatron.models.threat import make_threat_config
from netwatron.models.world_model import WorldModel
from netwatron.training.evaluate import (
    BASELINE_LADDER_DESCRIPTIONS,
    Baseline2_MLP,
    Baseline3_HistoryRNN,
    Baseline4_HistoryTransformer,
    risk_metrics,
)
from netwatron.training.train import train_world_model
from run_ablations import build_loaders, evaluate_loader
from train_model import make_synthetic_dataframe


def _extract_zs(model: WorldModel, loader, device):
    """Run a trained WorldModel's encoder only, to get z_now / z_history for
    the simple baselines (so baselines 1-4 compare on the SAME learned
    representation as the world model, isolating the effect of the
    dynamics/threat formulation itself)."""
    model.eval()
    z_now_all, z_hist_all, labels_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            z_history = model.encode_sequence(
                batch["hist_X"],
                batch["hist_mask"],
                batch["hist_tau"],
                batch["hist_delta"],
            )
            z_now_all.append(z_history[:, -1, :].cpu().numpy())
            z_hist_all.append(z_history.cpu().numpy())
            labels_all.append(batch["risk_label"].cpu().numpy())
    return (
        np.concatenate(z_now_all),
        np.concatenate(z_hist_all),
        np.concatenate(labels_all),
    )


def train_simple_torch_baseline(
    module,
    z_train,
    y_train,
    epochs=10,
    lr=1e-3,
    device: torch.device | str = "cpu",
):
    module = module.to(device)
    opt = torch.optim.Adam(module.parameters(), lr=lr)
    z_t = torch.from_numpy(z_train).float().to(device)
    y_t = torch.from_numpy(y_train).float().to(device)
    for _ in range(epochs):
        opt.zero_grad()
        logit = module(z_t)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y_t)
        loss.backward()
        opt.step()
    return module


def main():
    base_cfg = PipelineConfig()
    df = make_synthetic_dataframe(20000)
    loaders, _ = build_loaders(df, base_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    # --- First, train a representation encoder + a real world model with T5 (baseline 7 / final candidate) ---
    cfg_t5 = copy.deepcopy(base_cfg)
    cfg_t5.threat = make_threat_config("T5")
    model_t5 = train_world_model(loaders["train"], cfg_t5)
    dyn, risk = evaluate_loader(model_t5, loaders["val"], device)
    results["baseline_7"] = risk

    # --- Baseline 6: same but T4 (no stage consistency) ---
    cfg_t4 = copy.deepcopy(base_cfg)
    cfg_t4.threat = make_threat_config("T4")
    model_t4 = train_world_model(loaders["train"], cfg_t4)
    _, risk = evaluate_loader(model_t4, loaders["val"], device)
    results["baseline_6"] = risk

    # --- Baseline 5: World model, T3 (trajectory risk only, no surprise/consistency) ---
    cfg_t3 = copy.deepcopy(base_cfg)
    cfg_t3.threat = make_threat_config("T3")
    model_t3 = train_world_model(loaders["train"], cfg_t3)
    _, risk = evaluate_loader(model_t3, loaders["val"], device)
    results["baseline_5"] = risk

    # --- Baselines 1-4: reuse model_t3's encoder to get z_now / z_history, then train simple heads ---
    z_now_tr, z_hist_tr, y_tr = _extract_zs(model_t3, loaders["train"], device)
    z_now_val, z_hist_val, y_val = _extract_zs(model_t3, loaders["val"], device)

    # Baseline 1: LightGBM
    try:
        from netwatron.training.evaluate import Baseline1_LightGBM

        b1 = Baseline1_LightGBM(n_estimators=200, max_depth=6)
        b1.fit(z_now_tr, y_tr)
        prob = b1.predict_proba(z_now_val)
        results["baseline_1"] = risk_metrics(y_val, prob)
    except ImportError:
        print(
            "lightgbm not installed; skipping Baseline 1 (pip install lightgbm)"
        )
        results["baseline_1"] = None

    # Baseline 2: MLP on current state
    b2 = train_simple_torch_baseline(
        Baseline2_MLP(z_now_tr.shape[-1]), z_now_tr, y_tr, device=device
    )
    with torch.no_grad():
        prob = (
            torch.sigmoid(b2(torch.from_numpy(z_now_val).float().to(device)))
            .cpu()
            .numpy()
        )
    results["baseline_2"] = risk_metrics(y_val, prob)

    # Baseline 3: History GRU
    b3 = train_simple_torch_baseline(
        Baseline3_HistoryRNN(z_hist_tr.shape[-1]),
        z_hist_tr,
        y_tr,
        device=device,
    )
    with torch.no_grad():
        prob = (
            torch.sigmoid(b3(torch.from_numpy(z_hist_val).float().to(device)))
            .cpu()
            .numpy()
        )
    results["baseline_3"] = risk_metrics(y_val, prob)

    # Baseline 4: History Transformer
    b4 = train_simple_torch_baseline(
        Baseline4_HistoryTransformer(z_hist_tr.shape[-1]),
        z_hist_tr,
        y_tr,
        device=device,
    )
    with torch.no_grad():
        prob = (
            torch.sigmoid(b4(torch.from_numpy(z_hist_val).float().to(device)))
            .cpu()
            .numpy()
        )
    results["baseline_4"] = risk_metrics(y_val, prob)

    print("\n\n=== BASELINE LADDER RESULTS (Sec. 63) ===")
    for key in [
        "baseline_1",
        "baseline_2",
        "baseline_3",
        "baseline_4",
        "baseline_5",
        "baseline_6",
        "baseline_7",
    ]:
        desc = BASELINE_LADDER_DESCRIPTIONS[key]
        r = results.get(key)
        if r is None:
            print(f"{key} ({desc}): SKIPPED")
        else:
            print(
                f"{key} ({desc}): PR-AUC={r['pr_auc']:.4f}  ROC-AUC={r['roc_auc']:.4f}  F1={r['f1']:.4f}"
            )

    return results


if __name__ == "__main__":
    main()
