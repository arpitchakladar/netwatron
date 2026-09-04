from scapy.layers.inet import IP, TCP
from scapy.config import conf
from scapy.volatile import RandShort
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Any, Iterator


@dataclass
class FlowFeatureTracker:
    """Tracks state for a single bi-directional network flow."""

    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int

    first_timestamp: float = 0.0
    last_timestamp: float = 0.0

    # Packet counts
    tot_fwd_pkts: int = 0
    tot_bwd_pkts: int = 0

    # Lengths
    fwd_pkt_len_tot: int = 0
    bwd_pkt_len_tot: int = 0

    # TCP Flags (Aggregated across the flow)
    fin_flag_cnt: int = 0
    syn_flag_cnt: int = 0
    rst_flag_cnt: int = 0
    psh_flag_cnt: int = 0
    ack_flag_cnt: int = 0
    urg_flag_cnt: int = 0

    def add_packet(self, pkt, timestamp: float, direction: str):
        """Updates flow statistics with a new packet."""
        if self.first_timestamp == 0.0:
            self.first_timestamp = timestamp
        self.last_timestamp = timestamp

        # IP Length (total packet length)
        pkt_len = int(pkt[IP].len) if IP in pkt else 0

        if direction == "fwd":
            self.tot_fwd_pkts += 1
            self.fwd_pkt_len_tot += pkt_len
        else:
            self.tot_bwd_pkts += 1
            self.bwd_pkt_len_tot += pkt_len

        # Extract TCP Flags if present
        if TCP in pkt:
            flags = pkt[TCP].flags
            if "F" in flags:
                self.fin_flag_cnt += 1
            if "S" in flags:
                self.syn_flag_cnt += 1
            if "R" in flags:
                self.rst_flag_cnt += 1
            if "P" in flags:
                self.psh_flag_cnt += 1
            if "A" in flags:
                self.ack_flag_cnt += 1
            if "U" in flags:
                self.urg_flag_cnt += 1

    def to_cic_dict(self) -> Dict[str, Any]:
        """Exports the flow data matching the CIC-IDS2018 CSV columns."""
        flow_duration = int(
            (self.last_timestamp - self.first_timestamp) * 1_000_000
        )  # Microseconds

        return {
            "Src IP": self.src_ip,
            "Dst IP": self.dst_ip,
            "Src Port": self.src_port,
            "Dst Port": self.dst_port,
            "Protocol": self.protocol,
            "Timestamp": self.first_timestamp,
            "Flow Duration": flow_duration,
            "Tot Fwd Pkts": self.tot_fwd_pkts,
            "Tot Bwd Pkts": self.tot_bwd_pkts,
            "TotLen Fwd Pkts": self.fwd_pkt_len_tot,
            "TotLen Bwd Pkts": self.bwd_pkt_len_tot,
            "FIN Flag Cnt": self.fin_flag_cnt,
            "SYN Flag Cnt": self.syn_flag_cnt,
            "RST Flag Cnt": self.rst_flag_cnt,
            "PSH Flag Cnt": self.psh_flag_cnt,
            "ACK Flag Cnt": self.ack_flag_cnt,
            "URG Flag Cnt": self.urg_flag_cnt,
        }


class PacketToFlowParser:
    """Traps packets and aggregates them into CIC-IDS2018 formatted flows."""

    def __init__(self, flow_timeout_sec: int = 120):
        self.active_flows: Dict[str, FlowFeatureTracker] = {}
        self.flow_timeout_sec = flow_timeout_sec

    def _generate_flow_key(self, pkt) -> tuple:
        """Generates a bi-directional key and determines packet direction."""
        if IP not in pkt or TCP not in pkt:
            return None, None

        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst
        src_port = pkt[TCP].sport
        dst_port = pkt[TCP].dport
        proto = pkt[IP].proto

        # Bi-directional logic: Sort endpoints so A->B and B->A create the same key
        endpoint1 = f"{src_ip}:{src_port}"
        endpoint2 = f"{dst_ip}:{dst_port}"

        if endpoint1 < endpoint2:
            key = (src_ip, dst_ip, src_port, dst_port, proto)
            direction = "fwd"
        else:
            key = (dst_ip, src_ip, dst_port, src_port, proto)
            direction = "bwd"

        return key, direction

    def process_packet(self, pkt) -> None:
        """Processes a single trapped packet."""
        key, direction = self._generate_flow_key(pkt)
        if not key:
            return  # Skip non-TCP/IP packets for now

        timestamp = float(pkt.time)

        # Initialize flow if it doesn't exist
        if key not in self.active_flows:
            self.active_flows[key] = FlowFeatureTracker(
                src_ip=pkt[IP].src,
                dst_ip=pkt[IP].dst,
                src_port=pkt[TCP].sport,
                dst_port=pkt[TCP].dport,
                protocol=key[4],
            )

        # Update flow stats
        self.active_flows[key].add_packet(pkt, timestamp, direction)

    def extract_finished_flows(
        self, current_time: float
    ) -> Iterator[Dict[str, Any]]:
        """Yields flows that have exceeded the timeout (e.g., for live trapping)."""
        expired_keys = []
        for key, flow in self.active_flows.items():
            if (current_time - flow.last_timestamp) > self.flow_timeout_sec:
                yield flow.to_cic_dict()
                expired_keys.append(key)

        for key in expired_keys:
            del self.active_flows[key]

    def dump_all_flows(self) -> Iterator[Dict[str, Any]]:
        """Yields all active flows immediately (useful for PCAP processing)."""
        for flow in self.active_flows.values():
            yield flow.to_cic_dict()
