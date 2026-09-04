"""Inference adapter between live CIC flows and a trained Netwatron model."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from .config import PipelineConfig
from .data.preprocessing import PreprocessArtifacts, transform_features
from .models.world_model import WorldModel


class LoadedWorldModel:
    """Load a trained checkpoint and evaluate a history of flow windows.

    The model is trained on preprocessed per-flow vectors.  The caller must
    provide exactly ``input_dim`` features per flow and a chronological list of
    windows.  This keeps online inference semantically aligned with training;
    aggregate state vectors must not be passed to a flow-level checkpoint.
    """

    def __init__(
        self,
        model: WorldModel,
        cfg: PipelineConfig,
        artifacts: PreprocessArtifacts,
        device: str,
    ) -> None:
        self.model = model.eval()
        self.cfg = cfg
        self.artifacts = artifacts
        self.device = torch.device(device)
        self.missing_live_feature_columns: tuple[str, ...] = ()

    @classmethod
    def load(
        cls, checkpoint_path: str | Path, device: str = "cpu"
    ) -> "LoadedWorldModel":
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        cfg = checkpoint["config"]
        if not isinstance(cfg, PipelineConfig):
            raise TypeError("checkpoint does not contain a PipelineConfig")
        model = WorldModel(cfg).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        artifacts = checkpoint.get("preprocess_artifacts")
        if not isinstance(artifacts, PreprocessArtifacts):
            raise TypeError(
                "checkpoint does not contain fitted preprocessing artifacts"
            )
        return cls(model, cfg, artifacts, device)

    def encode_live_flows(
        self, flows: Sequence[dict[str, object]]
    ) -> np.ndarray:
        """Transform parser output using the preprocessing saved at training.

        Parser fields absent from the training schema are ignored. Training
        fields the parser cannot currently calculate (for example CIC active
        idle statistics) are imputed with the fitted training median, then
        scaled normally. This preserves dimension/order and maps unavailable
        information to the model's neutral training-space value.
        """
        if not flows:
            return np.empty(
                (0, self.cfg.flow_encoder.input_dim), dtype=np.float32
            )
        frame = pd.DataFrame(flows)
        if self.cfg.data.protocol_col not in frame.columns:
            raise ValueError(
                "live flow schema is missing required protocol column: "
                f"{self.cfg.data.protocol_col}"
            )
        self.missing_live_feature_columns = tuple(
            column
            for column in self.artifacts.numeric_columns
            if column not in frame.columns
        )
        return transform_features(frame, self.cfg.data, self.artifacts)

    @torch.no_grad()
    def forecast(
        self, history: Sequence[np.ndarray]
    ) -> dict[str, np.ndarray | float]:
        """Return forecast latents and model risk from ``history_len`` windows.

        Each window has shape ``[flows, input_dim]``.  Empty windows are valid.
        Time channels use monotonically increasing per-flow offsets because live
        flow dictionaries do not expose the preprocessor's absolute timestamps.
        """
        required = self.cfg.dynamics.history_len
        if len(history) != required:
            raise ValueError(
                f"expected {required} history windows, got {len(history)}"
            )
        max_flows = self.cfg.window.max_flows_per_window
        feature_dim = self.cfg.flow_encoder.input_dim
        X = np.zeros((1, required, max_flows, feature_dim), dtype=np.float32)
        mask = np.zeros((1, required, max_flows), dtype=np.float32)
        tau = np.zeros((1, required, max_flows), dtype=np.float32)
        delta = np.zeros((1, required, max_flows), dtype=np.float32)
        for index, window in enumerate(history):
            flows = np.asarray(window, dtype=np.float32)
            if flows.ndim != 2 or flows.shape[1] != feature_dim:
                raise ValueError(
                    f"window {index} must have shape [flows, {feature_dim}]"
                )
            count = min(len(flows), max_flows)
            X[0, index, :count] = flows[:count]
            mask[0, index, :count] = 1.0
            tau[0, index, :count] = np.arange(count, dtype=np.float32)
            if count > 1:
                delta[0, index, 1:count] = 1.0

        tensors = {
            "hist_X": torch.from_numpy(X).to(self.device),
            "hist_mask": torch.from_numpy(mask).to(self.device),
            "hist_tau": torch.from_numpy(tau).to(self.device),
            "hist_delta": torch.from_numpy(delta).to(self.device),
        }
        z_history = self.model.encode_sequence(
            tensors["hist_X"],
            tensors["hist_mask"],
            tensors["hist_tau"],
            tensors["hist_delta"],
        )
        z_future = self.model.dynamics(z_history)
        if self.cfg.risk.variant in ("RISK-1", "RISK-2"):
            risk_prob, _ = self.model.risk_head(z_future=z_future)
        elif self.cfg.risk.variant == "RISK-3":
            risk_prob, _ = self.model.risk_head(z_now=z_history[:, -1, :])
        else:
            risk_prob, _ = self.model.risk_head(z_history=z_history)
        if risk_prob.ndim > 1:
            risk_prob = risk_prob.mean(dim=-1)
        return {
            "current_latent_state": z_history[0, -1].cpu().numpy(),
            "future_latent_states": z_future[0].cpu().numpy(),
            "trajectory_risk": float(risk_prob[0].cpu()),
        }


class LiveFlowWindowBuffer:
    """Build model-ready, contiguous live flow windows from parser output."""

    def __init__(self, model: LoadedWorldModel, window_seconds: float) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self.model = model
        self.window_seconds = window_seconds
        self._window_start: float | None = None
        self._flows: list[dict[str, object]] = []

    def add(
        self, flow: dict[str, object], observed_at: float
    ) -> list[np.ndarray]:
        completed = self.advance(observed_at)
        self._flows.append(flow)
        return completed

    def advance(self, now: float) -> list[np.ndarray]:
        if self._window_start is None:
            self._window_start = now - (now % self.window_seconds)
            return []
        completed: list[np.ndarray] = []
        while now >= self._window_start + self.window_seconds:
            completed.append(self.model.encode_live_flows(self._flows))
            self._flows = []
            self._window_start += self.window_seconds
        return completed
