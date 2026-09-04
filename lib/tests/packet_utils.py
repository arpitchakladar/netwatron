"""Scapy helper functions for packet crafting and rate limiting."""

from __future__ import annotations

import random
import time
from typing import Optional

from scapy.layers.inet import TCP, IP
from scapy.packet import Packet
from scapy.sendrecv import send, sr1


def build_syn_packet(
    dst_ip: str,
    dst_port: int,
    src_ip: Optional[str] = None,
    src_port: Optional[int] = None,
) -> Packet:
    """Build a single TCP SYN packet."""
    if src_port is None:
        src_port = random.randint(1024, 65535)
    pkt = IP(dst=dst_ip) / TCP(
        sport=src_port,
        dport=dst_port,
        flags="S",
        seq=random.randint(0, 2**32),
    )
    if src_ip:
        pkt[IP].src = src_ip
    return pkt


def send_syn_scan(
    dst_ip: str,
    port_range: tuple[int, int] = (1, 1024),
    src_ip: Optional[str] = None,
    rate_limit: float = 0.0,
) -> list[dict]:
    """Send SYN packets across a port range. Returns list of results."""
    results = []
    for port in range(port_range[0], port_range[1] + 1):
        pkt = build_syn_packet(dst_ip, port, src_ip=src_ip)
        ans = sr1(pkt, timeout=0.5, verbose=0)
        if ans is not None and ans.haslayer(TCP):
            flags = str(ans[TCP].flags)
            results.append({"port": port, "flags": flags, "responded": True})
        else:
            results.append({"port": port, "flags": "", "responded": False})
        if rate_limit > 0:
            time.sleep(rate_limit)
    return results


def send_syn_flood(
    dst_ip: str,
    dst_port: int = 8080,
    src_ip: Optional[str] = None,
    duration: float = 10.0,
    rate: int = 100,
) -> int:
    """Send SYN flood packets. Returns count sent."""
    count = 0
    end_time = time.time() + duration
    delay = 1.0 / max(rate, 1)
    while time.time() < end_time:
        pkt = build_syn_packet(dst_ip, dst_port, src_ip=src_ip)
        send(pkt, verbose=0)
        count += 1
        time.sleep(delay)
    return count


def send_raw_packet(pkt: Packet) -> None:
    """Send a single raw packet."""
    send(pkt, verbose=0)
