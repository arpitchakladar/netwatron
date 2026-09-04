"""Ground-truth logger for attack simulation traffic.

Every generated packet/session is tagged with:
  timestamp, src_ip, dst_ip, dst_port, label

Timestamps use Unix epoch with millisecond precision, aligned to
NIDS 5-second aggregation windows for easy downstream joins.
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import IO, Any, Optional


class GroundTruthLogger:
    """Append-only CSV logger for traffic ground truth labels."""

    HEADER = [
        "timestamp_ms",
        "src_ip",
        "dst_ip",
        "dst_port",
        "protocol",
        "label",
        "scenario",
        "extra",
    ]

    def __init__(
        self,
        log_path: str | Path,
        scenario: str = "",
        buffer_size: int = 50,
    ) -> None:
        self.log_path = Path(log_path)
        self.scenario = scenario
        self.buffer_size = buffer_size
        self._buffer: list[list[str]] = []
        self._handle: Optional[IO[str]] = None
        self._writer: Any = None
        self._ensure_file()

    def _ensure_file(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = (
            not self.log_path.exists() or self.log_path.stat().st_size == 0
        )
        self._handle = open(self.log_path, "a", newline="")
        self._writer = csv.writer(self._handle)
        if write_header:
            self._writer.writerow(self.HEADER)

    def log(
        self,
        src_ip: str,
        dst_ip: str,
        dst_port: int,
        label: str,
        protocol: str = "TCP",
        extra: str = "",
    ) -> None:
        """Log a single traffic event with current timestamp."""
        ts_ms = int(time.time() * 1000)
        row = [
            str(ts_ms),
            src_ip,
            dst_ip,
            str(dst_port),
            protocol,
            label,
            self.scenario,
            extra,
        ]
        self._buffer.append(row)
        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        if self._writer and self._handle and self._buffer:
            for row in self._buffer:
                self._writer.writerow(row)
            self._handle.flush()
            self._buffer.clear()

    def close(self) -> None:
        self.flush()
        if self._handle:
            self._handle.close()
            self._handle = None
            self._writer = None

    def __enter__(self) -> "GroundTruthLogger":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def align_to_window(epoch_ms: int, window_ms: int = 5000) -> int:
    """Snap a millisecond timestamp down to the nearest window boundary."""
    return (epoch_ms // window_ms) * window_ms
