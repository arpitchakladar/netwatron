"""Sliding state-history buffer for temporal model inputs."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from state_aggregator import FEATURE_NAMES, NetworkState


class TemporalSequenceBuffer:
    """Keep the latest ``sequence_length`` state vectors in chronological order."""

    def __init__(
        self,
        sequence_length: int = 12,
        feature_names: tuple[str, ...] = FEATURE_NAMES,
    ) -> None:
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        self.sequence_length = sequence_length
        self.feature_names = feature_names
        self._states: deque[NetworkState] = deque(maxlen=sequence_length)

    def append(self, state: NetworkState) -> None:
        self._states.append(state)

    def extend(self, states: Iterable[NetworkState]) -> None:
        self._states.extend(states)

    @property
    def is_ready(self) -> bool:
        return len(self._states) == self.sequence_length

    def matrix(self) -> NDArray[np.float64]:
        """Return a contiguous ``[time, feature]`` NumPy matrix."""
        if not self._states:
            return np.empty((0, len(self.feature_names)), dtype=np.float64)
        return np.vstack(
            [state.to_vector(self.feature_names) for state in self._states]
        )

    def states(self) -> tuple[NetworkState, ...]:
        return tuple(self._states)

    def clear(self) -> None:
        self._states.clear()
