"""Convert completed CIC-style flows into fixed-duration network snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy as np
from numpy.typing import NDArray

FEATURE_NAMES = (
    "flow_count",
    "active_source_count",
    "active_destination_count",
    "total_packets",
    "total_bytes",
    "avg_flow_duration_ms",
    "forward_packet_ratio",
    "forward_byte_ratio",
    "syn_packets",
    "rst_packets",
    "syn_packet_ratio",
    "rst_packet_ratio",
)


@dataclass(frozen=True)
class NetworkState:
    """An immutable summary of all completed flows in one time bucket."""

    window_start: float
    window_end: float
    features: dict[str, float]

    def to_vector(
        self, feature_names: tuple[str, ...] = FEATURE_NAMES
    ) -> NDArray[np.float64]:
        """Return a stable feature order suitable for a temporal model."""
        return np.asarray(
            [self.features.get(name, 0.0) for name in feature_names],
            dtype=np.float64,
        )


class NetworkStateAggregator:
    """Aggregate completed flows into contiguous, fixed-width state windows.

    Windows are *tumbling* (non-overlapping) rather than event-count based.  This
    gives every temporal-model input the same time semantics.  Calling
    :meth:`advance` also emits empty windows, which is important because a quiet
    network is itself a meaningful state.
    """

    def __init__(self, window_seconds: float = 5.0) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self.window_seconds = float(window_seconds)
        self._window_start: float | None = None
        self._lock = Lock()
        self._reset_accumulators()

    def _reset_accumulators(self) -> None:
        self._flow_count = 0
        self._sources: set[str] = set()
        self._destinations: set[str] = set()
        self._total_packets = 0
        self._total_bytes = 0
        self._total_duration_us = 0.0
        self._forward_packets = 0
        self._forward_bytes = 0
        self._syn_packets = 0
        self._rst_packets = 0

    def add_flow(
        self, flow: Mapping[str, Any], observed_at: float | None = None
    ) -> list[NetworkState]:
        """Add one CIC flow and return every window closed by this observation.

        ``observed_at`` should normally be wall-clock time at which the collector
        receives the completed flow.  If omitted, the flow's ``Timestamp`` is
        used, which is convenient for replaying PCAP-derived flow dictionaries.
        """
        timestamp = float(
            observed_at if observed_at is not None else flow["Timestamp"]
        )
        with self._lock:
            completed = self._advance_unlocked(timestamp)
            self._accumulate(flow)
            return completed

    def advance(self, now: float) -> list[NetworkState]:
        """Close elapsed windows, including zero-traffic windows."""
        with self._lock:
            return self._advance_unlocked(float(now))

    def current_state(self, now: float | None = None) -> NetworkState | None:
        """Return a non-closing snapshot of the current window for display."""
        with self._lock:
            if self._window_start is None:
                return None
            end = self._window_start + self.window_seconds
            if now is not None:
                end = min(end, float(now))
            return NetworkState(self._window_start, end, self._feature_dict())

    def _advance_unlocked(self, now: float) -> list[NetworkState]:
        if self._window_start is None:
            self._window_start = now - (now % self.window_seconds)
            return []

        # A late flow belongs in the current bucket.  We cannot reopen an emitted
        # state, so retain it rather than silently discarding its telemetry.
        if now < self._window_start:
            return []

        completed: list[NetworkState] = []
        while now >= self._window_start + self.window_seconds:
            window_end = self._window_start + self.window_seconds
            completed.append(
                NetworkState(
                    self._window_start, window_end, self._feature_dict()
                )
            )
            self._window_start = window_end
            self._reset_accumulators()
        return completed

    def _accumulate(self, flow: Mapping[str, Any]) -> None:
        fwd_packets = int(flow.get("Tot Fwd Pkts", 0))
        bwd_packets = int(flow.get("Tot Bwd Pkts", 0))
        fwd_bytes = int(flow.get("TotLen Fwd Pkts", 0))
        bwd_bytes = int(flow.get("TotLen Bwd Pkts", 0))
        self._flow_count += 1
        self._sources.add(str(flow.get("Src IP", "unknown")))
        self._destinations.add(str(flow.get("Dst IP", "unknown")))
        self._forward_packets += fwd_packets
        self._forward_bytes += fwd_bytes
        self._total_packets += fwd_packets + bwd_packets
        self._total_bytes += fwd_bytes + bwd_bytes
        self._total_duration_us += float(flow.get("Flow Duration", 0))
        self._syn_packets += int(flow.get("SYN Flag Cnt", 0))
        self._rst_packets += int(flow.get("RST Flag Cnt", 0))

    def _feature_dict(self) -> dict[str, float]:
        packet_denominator = max(self._total_packets, 1)
        byte_denominator = max(self._total_bytes, 1)
        flow_denominator = max(self._flow_count, 1)
        return {
            "flow_count": float(self._flow_count),
            "active_source_count": float(len(self._sources)),
            "active_destination_count": float(len(self._destinations)),
            "total_packets": float(self._total_packets),
            "total_bytes": float(self._total_bytes),
            "avg_flow_duration_ms": self._total_duration_us
            / flow_denominator
            / 1_000,
            "forward_packet_ratio": self._forward_packets / packet_denominator,
            "forward_byte_ratio": self._forward_bytes / byte_denominator,
            "syn_packets": float(self._syn_packets),
            "rst_packets": float(self._rst_packets),
            "syn_packet_ratio": self._syn_packets / packet_denominator,
            "rst_packet_ratio": self._rst_packets / packet_denominator,
        }
