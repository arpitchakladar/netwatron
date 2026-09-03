#!/usr/bin/env python3

import sys
import os
import json
from scapy.all import sniff

current_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.join(current_dir, "..", "lib")
sys.path.append(lib_dir)

from flow_generator import PacketToFlowParser


def process_pcap(pcap_file: str):
    """Parses a static PCAP file into ML-ready flow data."""
    print(f"Processing {pcap_file}...")
    parser = PacketToFlowParser()

    sniff(offline=pcap_file, prn=parser.process_packet, store=False)

    for flow_data in parser.dump_all_flows():
        print(json.dumps(flow_data, indent=2))


def trap_live_traffic(interface: str):
    """Traps live packets and prints flows as they timeout."""
    print(f"Listening on {interface}... Press Ctrl+C to stop.")

    # TODO: LOWERED to 5 seconds for testing (change back to 30 or 120 for real ML data)
    parser = PacketToFlowParser(flow_timeout_sec=5)

    packet_count = 0

    def packet_callback(pkt):
        nonlocal packet_count
        packet_count += 1

        # Print a small dot for every packet so you know it's working
        # (Prints every 10th packet to avoid completely flooding your screen)
        if packet_count % 10 == 0:
            print(".", end="", flush=True)

        parser.process_packet(pkt)

        for finished_flow in parser.extract_finished_flows(float(pkt.time)):
            print("\n\n--- ML Model Input Ready ---")
            print(json.dumps(finished_flow, indent=2))

    sniff(iface=interface, filter="tcp", prn=packet_callback, store=False)


if __name__ == "__main__":
    trap_live_traffic("enp4s0")
