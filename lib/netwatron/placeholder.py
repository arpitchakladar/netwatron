"""Temporary interface for a learned temporal network world model.

The deterministic baseline here is intentionally not an intrusion classifier. It
only preserves the contract and provides visible scores while a trained sequence
network (Transformer, SSM, or recurrent latent dynamics model) is integrated.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class TrajectoryEvaluation:
    future_latent_states: NDArray[np.float64]
    dynamics_surprise_score: float
    trajectory_risk: float


class TemporalWorldModelPlaceholder:
    """Contract-compatible stand-in for a temporal world model.

    Replace ``_forecast`` with a trained model inference call.  A production
    model should encode the input sequence, autoregress latent states for
    ``forecast_steps``, decode expected observables, and score the likelihood of
    the observed final state.  The public return type remains unchanged.
    """

    def __init__(
        self, forecast_steps: int = 3, risk_midpoint: float = 2.0
    ) -> None:
        if forecast_steps < 1:
            raise ValueError("forecast_steps must be positive")
        self.forecast_steps = forecast_steps
        self.risk_midpoint = risk_midpoint

    def evaluate_trajectory(
        self, state_sequence: ArrayLike
    ) -> TrajectoryEvaluation:
        """Forecast K states and score deviation of the newest observed state.

        The score is a normalized mean absolute prediction error using the
        sequence's own historical scale.  It is a deterministic placeholder, not
        a learned normality distribution, and must be calibrated/replaced before
        using it to enforce network policy.
        """
        rows = np.asarray(state_sequence, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[0] < 2:
            raise ValueError("state_sequence requires at least two time steps")
        if rows.shape[1] == 0:
            raise ValueError(
                "state_sequence must be a non-empty rectangular matrix"
            )

        expected_current = self._next_state(rows[:-1])
        actual_current = rows[-1]
        scales = self._scales(rows[:-1])
        surprise = float(
            np.mean(np.abs(actual_current - expected_current) / scales)
        )

        trajectory = rows.copy()
        future = np.empty(
            (self.forecast_steps, rows.shape[1]), dtype=np.float64
        )
        for step in range(self.forecast_steps):
            next_state = self._next_state(trajectory)
            future[step] = next_state
            trajectory = np.vstack((trajectory, next_state))

        # Smoothly maps unbounded surprise into an operator-friendly [0, 1] risk.
        risk = 1.0 / (1.0 + exp(-(surprise - self.risk_midpoint)))
        return TrajectoryEvaluation(future, surprise, risk)

    @staticmethod
    def _next_state(history: NDArray[np.float64]) -> NDArray[np.float64]:
        """Placeholder forecast: persistence plus the latest observed velocity."""
        latest = history[-1]
        if history.shape[0] == 1:
            return latest
        previous = history[-2]
        return np.maximum(0.0, latest + (latest - previous))

    @staticmethod
    def _scales(history: NDArray[np.float64]) -> NDArray[np.float64]:
        """Per-feature scale prevents byte counters dominating ratio features."""
        return np.maximum(1.0, np.mean(np.abs(history), axis=0))
