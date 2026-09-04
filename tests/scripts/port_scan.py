"""TCP SYN/connect scan across a port range against the target.

CIC-IDS2018 category: Port Scan.
"""

from __future__ import annotations

import os
import signal
import sys
import time

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)

from lib.tests.packet_utils import build_syn_packet, send_raw_packet
from lib.tests.logger import GroundTruthLogger
from lib.tests.scenario_config import load_from_env, wait_for_target

RUNNING = True


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
    intensity = cfg.intensity  # ports per second

    log_path = f"/app/logs/port_scan_{attacker_ip}.csv"
    print(f"[port_scan] Target: {target_ip}, Attacker: {attacker_ip}")
    print(f"[port_scan] Duration: {duration}s, Rate: {intensity} ports/s")

    if not wait_for_target(target_ip, 8080):
        print("[port_scan] ERROR: target not reachable, aborting.")
        return

    with GroundTruthLogger(log_path, scenario="port_scan") as logger:
        start = time.time()
        port = 1
        delay = 1.0 / max(intensity, 1)

        while RUNNING and (time.time() - start) < duration:
            pkt = build_syn_packet(target_ip, port, src_ip=attacker_ip)
            send_raw_packet(pkt)
            logger.log(
                src_ip=attacker_ip,
                dst_ip=target_ip,
                dst_port=port,
                label="port_scan",
                extra=f"flags=S",
            )
            port += 1
            if port > 65535:
                port = 1
            time.sleep(delay)

    print(f"[port_scan] Completed. Scanned {port - 1} ports.")


if __name__ == "__main__":
    main()
