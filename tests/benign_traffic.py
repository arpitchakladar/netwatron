"""Benign traffic generator: continuously produces normal HTTP requests
against the target for a realistic baseline during all scenarios.
"""

from __future__ import annotations

import os
import random
import signal
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.tests.logger import GroundTruthLogger
from lib.tests.scenario_config import load_from_env, wait_for_target

RUNNING = True
ENDPOINTS = ["/", "/index.html", "/api/status", "/api/users", "/api/data"]


def _shutdown(signum, frame):
    global RUNNING
    RUNNING = False


def main() -> None:
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    cfg = load_from_env()
    target_ip = cfg.target.ip
    attacker_ip = os.environ.get("ATTACKER_IP", "172.28.0.100")
    duration = cfg.duration
    intensity = cfg.intensity  # requests per second

    base_url = f"http://{target_ip}:8080"
    log_path = f"/app/logs/benign_{attacker_ip}.csv"
    print(f"[benign] Target: {base_url}, Source: {attacker_ip}")
    print(f"[benign] Duration: {duration}s, Rate: {intensity} req/s")

    if not wait_for_target(target_ip, 8080):
        print("[benign] ERROR: target not reachable, aborting.")
        return

    with GroundTruthLogger(log_path, scenario="benign") as logger:
        count = 0
        end_time = time.time() + duration
        delay = 1.0 / max(intensity, 1)

        while RUNNING and time.time() < end_time:
            endpoint = random.choice(ENDPOINTS)
            try:
                requests.get(f"{base_url}{endpoint}", timeout=3)
                count += 1
            except requests.RequestException:
                pass

            logger.log(
                src_ip=attacker_ip,
                dst_ip=target_ip,
                dst_port=8080,
                label="benign",
                extra=f"endpoint={endpoint}",
            )
            time.sleep(delay)

    print(f"[benign] Completed. Sent {count} requests.")


if __name__ == "__main__":
    main()
