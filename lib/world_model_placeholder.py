"""Temporary interface for a learned temporal network world model.

The deterministic baseline here is intentionally not an intrusion classifier. It
only preserves the contract and provides visible scores while a trained sequence
network (Transformer, SSM, or recurrent latent dynamics model) is integrated.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Sequence


@dataclass(frozen=True)
class TrajectoryEvaluation:
    future_latent_states: list[list[float]]
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
        self, state_sequence: Sequence[Sequence[float]]
    ) -> TrajectoryEvaluation:
        """Forecast K states and score deviation of the newest observed state.

        The score is a normalized mean absolute prediction error using the
        sequence's own historical scale.  It is a deterministic placeholder, not
        a learned normality distribution, and must be calibrated/replaced before
        using it to enforce network policy.
        """
        if len(state_sequence) < 2:
            raise ValueError("state_sequence requires at least two time steps")
        width = len(state_sequence[0])
        if not width or any(len(row) != width for row in state_sequence):
            raise ValueError(
                "state_sequence must be a non-empty rectangular matrix"
            )

        rows = [[float(value) for value in row] for row in state_sequence]
        expected_current = self._next_state(rows[:-1])
        actual_current = rows[-1]
        scales = self._scales(rows[:-1])
        surprise = (
            sum(
                abs(actual - expected) / scale
                for actual, expected, scale in zip(
                    actual_current, expected_current, scales
                )
            )
            / width
        )

        trajectory = list(rows)
        future: list[list[float]] = []
        for _ in range(self.forecast_steps):
            next_state = self._next_state(trajectory)
            future.append(next_state)
            trajectory.append(next_state)

        # Smoothly maps unbounded surprise into an operator-friendly [0, 1] risk.
        risk = 1.0 / (1.0 + exp(-(surprise - self.risk_midpoint)))
        return TrajectoryEvaluation(future, surprise, risk)

    @staticmethod
    def _next_state(history: Sequence[Sequence[float]]) -> list[float]:
        """Placeholder forecast: persistence plus the latest observed velocity."""
        latest = list(history[-1])
        if len(history) == 1:
            return latest
        previous = history[-2]
        return [
            max(0.0, value + (value - prior))
            for value, prior in zip(latest, previous)
        ]

    @staticmethod
    def _scales(history: Sequence[Sequence[float]]) -> list[float]:
        """Per-feature scale prevents byte counters dominating ratio features."""
        width = len(history[0])
        return [
            max(1.0, sum(abs(row[index]) for row in history) / len(history))
            for index in range(width)
        ]
