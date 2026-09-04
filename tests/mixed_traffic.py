"""Mixed traffic: interleaves normal HTTP requests with attack traffic
(80% benign, 20% attack) on a randomized schedule.

Simulates a compromised host producing blended traffic.
"""

from __future__ import annotations

import os
import random
import signal
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.tests.packet_utils import build_syn_packet, send_raw_packet
from lib.tests.traffic_scheduler import TrafficJob, TrafficScheduler
from lib.tests.logger import GroundTruthLogger
from lib.tests.scenario_config import load_from_env, wait_for_target

RUNNING = True
ENDPOINTS = ["/", "/index.html", "/api/status", "/api/users", "/api/data"]
BENIGN_RATIO = 0.8


def _shutdown(signum, frame):
    global RUNNING
    RUNNING = False


def benign_request(
    base_url: str, src_ip: str, target_ip: str, logger: GroundTruthLogger
) -> None:
    endpoint = random.choice(ENDPOINTS)
    try:
        requests.get(f"{base_url}{endpoint}", timeout=2)
    except requests.RequestException:
        pass
    logger.log(
        src_ip=src_ip,
        dst_ip=target_ip,
        dst_port=8080,
        label="benign",
        extra=f"endpoint={endpoint}",
    )


def port_scan_burst(
    base_url: str, src_ip: str, target_ip: str, logger: GroundTruthLogger
) -> None:
    for _ in range(random.randint(5, 20)):
        port = random.randint(1, 1024)
        pkt = build_syn_packet(target_ip, port, src_ip=src_ip)
        send_raw_packet(pkt)
        logger.log(
            src_ip=src_ip,
            dst_ip=target_ip,
            dst_port=port,
            label="port_scan",
            extra="mode=mixed",
        )


def main() -> None:
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    cfg = load_from_env()
    target_ip = cfg.target.ip
    attacker_ip = os.environ.get("ATTACKER_IP", "172.28.0.11")
    duration = cfg.duration
    intensity = cfg.intensity

    base_url = f"http://{target_ip}:8080"
    log_path = f"/app/logs/mixed_{attacker_ip}.csv"
    print(f"[mixed] Target: {base_url}, Attacker: {attacker_ip}")
    print(f"[mixed] Duration: {duration}s, Ratio: {BENIGN_RATIO:.0%} benign")

    if not wait_for_target(target_ip, 8080):
        print("[mixed] ERROR: target not reachable, aborting.")
        return

    with GroundTruthLogger(log_path, scenario="mixed") as logger:
        scheduler = TrafficScheduler(duration=duration)

        benign_interval = 1.0 / max(intensity * BENIGN_RATIO, 0.1)
        attack_interval = 1.0 / max(intensity * (1 - BENIGN_RATIO), 0.1)

        scheduler.add_job(
            TrafficJob(
                name="benign",
                func=benign_request,
                args=(base_url, attacker_ip, target_ip, logger),
                interval=benign_interval,
                weight=BENIGN_RATIO,
            )
        )
        scheduler.add_job(
            TrafficJob(
                name="attack",
                func=port_scan_burst,
                args=(base_url, attacker_ip, target_ip, logger),
                interval=attack_interval,
                weight=1 - BENIGN_RATIO,
            )
        )

        scheduler.run()

    print("[mixed] Completed.")


if __name__ == "__main__":
    main()
