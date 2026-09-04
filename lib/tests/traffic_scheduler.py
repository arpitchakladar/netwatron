"""Utility to interleave/schedule multiple traffic generators concurrently."""

from __future__ import annotations

import random
import threading
import time
from typing import Callable, List


class TrafficJob:
    """A single traffic generation job with a callable and timing config."""

    def __init__(
        self,
        name: str,
        func: Callable[..., None],
        args: tuple = (),
        kwargs: dict | None = None,
        interval: float = 1.0,
        weight: float = 1.0,
    ) -> None:
        self.name = name
        self.func = func
        self.args = args
        self.kwargs = kwargs or {}
        self.interval = interval
        self.weight = weight
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class TrafficScheduler:
    """Schedule and run multiple traffic generators concurrently with
    weighted random ordering."""

    def __init__(self, duration: float = 30.0) -> None:
        self.duration = duration
        self.jobs: List[TrafficJob] = []
        self._threads: List[threading.Thread] = []

    def add_job(self, job: TrafficJob) -> None:
        self.jobs.append(job)

    def _run_job(self, job: TrafficJob, end_time: float) -> None:
        while time.time() < end_time and not job.stopped:
            try:
                job.func(*job.args, **job.kwargs)
            except Exception:
                pass
            jitter = job.interval * random.uniform(0.5, 1.5)
            time.sleep(jitter)

    def run(self) -> None:
        """Run all jobs concurrently for the configured duration."""
        end_time = time.time() + self.duration
        for job in self.jobs:
            t = threading.Thread(
                target=self._run_job, args=(job, end_time), daemon=True
            )
            self._threads.append(t)
            t.start()
        remaining = end_time - time.time()
        if remaining > 0:
            time.sleep(remaining)
        for job in self.jobs:
            job.stop()
        for t in self._threads:
            t.join(timeout=5.0)

    def stop_all(self) -> None:
        for job in self.jobs:
            job.stop()
        for t in self._threads:
            t.join(timeout=2.0)
