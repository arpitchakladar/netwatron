#!/usr/bin/env python3
"""Live Temporal World Model monitor for completed network flows.

Run with the privileges required by Scapy, for example:
    sudo python scripts/advanced_tui.py --interface eth0
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

from rich.text import Text
from scapy.all import sniff
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Footer, Header, Static

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "lib"))

from flow_generator import PacketToFlowParser
from netwatron.placeholder import TemporalWorldModelPlaceholder
from netwatron.runtime import LiveFlowWindowBuffer, LoadedWorldModel
from sequence_buffer import TemporalSequenceBuffer
from state_aggregator import NetworkState, NetworkStateAggregator


class AdvancedNetworkMonitorApp(App[None]):
    """Textual UI with a capture thread and main-thread temporal evaluation."""

    CSS = """
    Screen { layout: vertical; }
    #summary {
        height: 3;
        padding: 1 2;
        background: $surface;
    }
    DataTable { height: 1fr; }
    """

    def __init__(
        self,
        interface: str,
        window_seconds: float = 5.0,
        sequence_length: int = 12,
        surprise_threshold: float = 2.5,
        checkpoint_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.interface = interface
        self.surprise_threshold = surprise_threshold
        self._aggregator = NetworkStateAggregator(window_seconds)
        self._states: queue.SimpleQueue[NetworkState] = queue.SimpleQueue()
        self._events: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._sequence = TemporalSequenceBuffer(sequence_length)
        self._world_model = TemporalWorldModelPlaceholder()
        self._trained_model = (
            LoadedWorldModel.load(checkpoint_path) if checkpoint_path else None
        )
        self._live_windows = (
            LiveFlowWindowBuffer(self._trained_model, window_seconds)
            if self._trained_model
            else None
        )
        self._model_history: deque = deque(
            maxlen=(
                self._trained_model.cfg.dynamics.history_len
                if self._trained_model
                else 1
            )
        )
        self._model_results: queue.SimpleQueue[dict] = queue.SimpleQueue()
        self._latest_model_result: dict | None = None
        self._stop_capture = threading.Event()
        self._capture_thread: threading.Thread | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Starting packet capture…", id="summary")
        yield DataTable(id="state-table")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#state-table", DataTable)
        table.cursor_type = "row"
        table.add_columns(
            "Window (local)",
            "Flows",
            "Packets",
            "Bytes",
            "Avg duration",
            "Surprise",
            "Risk",
            "Verdict",
        )
        self._capture_thread = threading.Thread(
            target=self._capture_flows,
            name="packet-flow-capture",
            daemon=True,
        )
        self._capture_thread.start()
        self.set_interval(0.25, self._drain_capture_queues)
        self._set_summary("Warming up temporal history")

    def on_unmount(self) -> None:
        self._stop_capture.set()

    def _capture_flows(self) -> None:
        """Capture and aggregate off the event loop; only queues immutable states."""
        parser = PacketToFlowParser(flow_timeout_sec=5)

        def on_packet(packet: object) -> None:
            parser.process_packet(packet)
            for flow in parser.extract_finished_flows(time.time()):
                self._submit_flow(flow)

        try:
            # A finite timeout lets the thread emit empty states and observe exit.
            while not self._stop_capture.is_set():
                sniff(
                    iface=self.interface,
                    filter="tcp",
                    prn=on_packet,
                    store=False,
                    timeout=1,
                )
                for flow in parser.extract_finished_flows(time.time()):
                    self._submit_flow(flow)
                self._states_put(self._aggregator.advance(time.time()))
                self._advance_live_model(time.time())
        except Exception as error:
            # Permissions and bad interfaces are operator errors.
            self._events.put(f"Capture stopped: {error}")

    def _submit_flow(self, flow: dict[str, object]) -> None:
        now = time.time()
        self._states_put(self._aggregator.add_flow(flow, observed_at=now))
        if self._live_windows:
            self._submit_live_windows(self._live_windows.add(flow, now))

    def _advance_live_model(self, now: float) -> None:
        if self._live_windows:
            self._submit_live_windows(self._live_windows.advance(now))

    def _submit_live_windows(self, windows: list) -> None:
        model = self._trained_model
        if model is None:
            return
        for window in windows:
            self._model_history.append(window)
            if len(self._model_history) == self._model_history.maxlen:
                self._model_results.put(
                    model.forecast(list(self._model_history))
                )

    def _states_put(self, states: list[NetworkState]) -> None:
        for state in states:
            self._states.put(state)

    def _drain_capture_queues(self) -> None:
        while True:
            try:
                self._latest_model_result = self._model_results.get_nowait()
            except queue.Empty:
                break
        while True:
            try:
                self._handle_state(self._states.get_nowait())
            except queue.Empty:
                break
        while True:
            try:
                self._set_summary(self._events.get_nowait())
            except queue.Empty:
                break

    def _handle_state(self, state: NetworkState) -> None:
        self._sequence.append(state)
        if not self._sequence.is_ready:
            self._set_summary(
                f"{self.interface} · history {len(self._sequence.states())}/"
                f"{self._sequence.sequence_length} windows · latest flows: "
                f"{int(state.features['flow_count'])}"
            )
            return

        result = self._world_model.evaluate_trajectory(self._sequence.matrix())
        risk = result.trajectory_risk
        if self._latest_model_result is not None:
            risk = float(self._latest_model_result["trajectory_risk"])
        blocked = (
            result.dynamics_surprise_score >= self.surprise_threshold
            or risk >= 0.5
        )
        verdict = "BLOCKED THREAT" if blocked else "OBSERVE"
        style = "bold red" if blocked else "green"
        cells = [
            time.strftime("%H:%M:%S", time.localtime(state.window_start)),
            str(int(state.features["flow_count"])),
            str(int(state.features["total_packets"])),
            str(int(state.features["total_bytes"])),
            f"{state.features['avg_flow_duration_ms']:.1f} ms",
            f"{result.dynamics_surprise_score:.3f}",
            f"{risk:.1%}",
            verdict,
        ]
        table = self.query_one("#state-table", DataTable)
        table.add_row(*(Text(cell, style=style) for cell in cells))
        self._set_summary(
            f"{self.interface} · {self._sequence.sequence_length}×"
            f"{len(self._sequence.feature_names)} state sequence · "
            f"surprise {result.dynamics_surprise_score:.3f} · "
            f"risk {risk:.1%} · {verdict}"
        )

    def _set_summary(self, message: str) -> None:
        self.query_one("#summary", Static).update(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interface", "-i", required=True, help="network interface"
    )
    parser.add_argument(
        "--window", type=float, default=5.0, help="window seconds"
    )
    parser.add_argument(
        "--history", type=int, default=12, help="state windows per sequence"
    )
    parser.add_argument(
        "--surprise-threshold", type=float, default=2.5, help="block threshold"
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="checkpoint produced by scripts/train_model.py",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    AdvancedNetworkMonitorApp(
        interface=args.interface,
        window_seconds=args.window,
        sequence_length=args.history,
        surprise_threshold=args.surprise_threshold,
        checkpoint_path=args.model,
    ).run()
