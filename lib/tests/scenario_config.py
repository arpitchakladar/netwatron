"""Scenario configuration schema for attack-simulation test harness."""

from __future__ import annotations

import os
import socket
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class AttackType(str, Enum):
    PORT_SCAN = "port_scan"
    SYN_FLOOD = "syn_flood"
    HTTP_FLOOD = "http_flood"
    SLOWLORIS = "slowloris"
    BRUTE_FORCE = "brute_force"
    DDOS_DISTRIBUTED = "ddos_distributed"
    MIXED_TRAFFIC = "mixed_traffic"


class TrafficLabel(str, Enum):
    BENIGN = "benign"
    PORT_SCAN = "port_scan"
    SYN_FLOOD = "syn_flood"
    HTTP_FLOOD = "http_flood"
    SLOWLORIS = "slowloris"
    BRUTE_FORCE = "brute_force"
    DDOS = "ddos"


@dataclass
class AttackerConfig:
    ip: str
    name: str
    attack_types: List[AttackType] = field(default_factory=list)


@dataclass
class TargetConfig:
    ip: str = "172.28.0.10"
    name: str = "target"
    http_port: int = 8080


@dataclass
class NetworkConfig:
    subnet: str = "172.28.0.0/24"
    gateway: str = "172.28.0.1"
    name: str = "nids-test-net"


@dataclass
class ScenarioConfig:
    attack_type: AttackType
    target: TargetConfig
    attackers: List[AttackerConfig]
    network: NetworkConfig
    duration: float = 30.0
    intensity: int = 100
    label: TrafficLabel = TrafficLabel.BENIGN
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "attack_type": self.attack_type.value,
            "target_ip": self.target.ip,
            "target_port": self.target.http_port,
            "attacker_ips": [a.ip for a in self.attackers],
            "duration": self.duration,
            "intensity": self.intensity,
            "label": self.label.value,
        }


# Default topology
DEFAULT_TARGET = TargetConfig(ip="172.28.0.10", name="target", http_port=8080)
DEFAULT_NETWORK = NetworkConfig()
DEFAULT_ATTACKERS = [
    AttackerConfig(ip="172.28.0.11", name="attacker1"),
    AttackerConfig(ip="172.28.0.12", name="attacker2"),
    AttackerConfig(ip="172.28.0.13", name="attacker3"),
    AttackerConfig(ip="172.28.0.14", name="attacker4"),
    AttackerConfig(ip="172.28.0.15", name="attacker5"),
]
DEFAULT_BENIGN = AttackerConfig(
    ip="172.28.0.100", name="benign", attack_types=[]
)


def env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


def env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def load_from_env() -> ScenarioConfig:
    """Build a ScenarioConfig from environment variables set in compose."""
    attack_str = env_str("SCENARIO", "port_scan")
    try:
        attack_type = AttackType(attack_str)
    except ValueError:
        attack_type = AttackType.PORT_SCAN

    target_ip = env_str("TARGET_IP", DEFAULT_TARGET.ip)
    duration = env_float("DURATION", 30.0)
    intensity = env_int("INTENSITY", 100)

    label_map: Dict[AttackType, TrafficLabel] = {
        AttackType.PORT_SCAN: TrafficLabel.PORT_SCAN,
        AttackType.SYN_FLOOD: TrafficLabel.SYN_FLOOD,
        AttackType.HTTP_FLOOD: TrafficLabel.HTTP_FLOOD,
        AttackType.SLOWLORIS: TrafficLabel.SLOWLORIS,
        AttackType.BRUTE_FORCE: TrafficLabel.BRUTE_FORCE,
        AttackType.DDOS_DISTRIBUTED: TrafficLabel.DDOS,
        AttackType.MIXED_TRAFFIC: TrafficLabel.BENIGN,
    }

    return ScenarioConfig(
        attack_type=attack_type,
        target=TargetConfig(ip=target_ip, http_port=8080),
        attackers=DEFAULT_ATTACKERS,
        network=DEFAULT_NETWORK,
        duration=duration,
        intensity=intensity,
        label=label_map.get(attack_type, TrafficLabel.BENIGN),
        description=f"Auto-configured from environment: {attack_str}",
    )


def wait_for_target(
    target_ip: str,
    port: int = 8080,
    timeout: float = 30.0,
    interval: float = 1.0,
) -> bool:
    """Block until the target server is accepting TCP connections."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            sock.connect((target_ip, port))
            sock.close()
            return True
        except socket.timeout, OSError:
            time.sleep(interval)
    return False
