"""
Parses the raw pcaps from `capture_scenario.sh` into the same per-tick
feature shape `tick_features()` produces, so Layer 3 (Isolation Forest) can
be grounded in real captured packets alongside the synthetic generator in
`train_isolation_forest.py`.

Run `capture_scenario.sh` yourself first (it needs `sudo`, which this script
cannot supply interactively) for each of: idle, browsing, beaconing,
bulk_upload, port_scan. Then:

    python -m client.rl_agent.capture_real_traffic

writes `client/rl_agent/models/real_capture_corpus.json` (derived 6-float
ticks only — the raw pcaps under `models/real_capture/raw/` are gitignored,
since they may carry real destination hostnames/IPs from the browsing/
beaconing curl loops).

Category convention, chosen to match this codebase's EXISTING split rather
than inventing a new one: `BENIGN_PROFILES` has no upload-heavy profile at
all, precisely because sustained high `upload_share` is already treated as
attack-mix-shaped here (`ATTACK_MIXES["exfiltration"]`), not benign-shaped.
So:

  - idle, browsing      -> real BENIGN ticks (added to the training matrix
                           and held-out evaluation in train_isolation_forest.py).
  - beaconing            -> real ticks shaped like the "beaconing" attack mix
                           (a periodic curl loop at a shortened 7s interval;
                           the *action* is benign, but Layer 3 only ever sees
                           shape, and a steady low-rate near-zero-jitter
                           pattern IS that shape). Used only to *measure*
                           detection, like every synthetic attack mix — never
                           trained on.
  - bulk_upload          -> real ticks shaped like the "exfiltration" attack
                           mix (a genuine, sustained, loopback bulk transfer).
                           This is also PROGRESS.md's own stated caveat made
                           concrete: "a legitimate large upload would look
                           like exfiltration here." Measured, not trained on,
                           for the same reason bulk-upload was never a benign
                           profile to begin with.
  - port_scan            -> real ticks shaped like the "port_scan" attack mix
                           (a genuine `nmap -sS -p-` self-scan). Measured
                           only.

Known limitations, stated rather than hidden: `bulk_upload`/`port_scan` run
over loopback (`lo0`), whose ~64KB MTU and zero real-world latency/jitter are
not a real NIC path, and direction (`outbound`) on loopback is classified by
TCP port role or SYN/RST flags rather than by source IP (both sides are
127.0.0.1) -- see `_classify_outbound_loopback_nc` / `_classify_outbound_scan`
below. `beaconing`'s 7s interval is far faster than a realistic C2 cadence.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scapy.all import IP, TCP, rdpcap  # noqa: E402

from client.rl_agent.anomaly_detector import tick_features  # noqa: E402

TICK_SECONDS = 5.0
NC_PORT = 8737  # must match capture_scenario.sh's NC_PORT

RAW_DIR = Path(__file__).resolve().parent / "models" / "real_capture" / "raw"
CORPUS_PATH = Path(__file__).resolve().parent / "models" / "real_capture_corpus.json"

# scenario -> (category, attack_mix_name_or_None, how direction is classified)
_SCENARIOS = {
    "idle": ("benign", None, "nic"),
    "browsing": ("benign", None, "nic"),
    "beaconing": ("attack_shaped", "beaconing", "nic"),
    "bulk_upload": ("attack_shaped", "exfiltration", "nc_port"),
    "port_scan": ("attack_shaped", "port_scan", "tcp_flags"),
}


def _local_ip_for(iface_hint_pcap_packets) -> str | None:
    """Best-effort local IPv4 for a NIC-based capture: the IP every captured
    packet either sends from or receives at, found by majority vote over
    the capture rather than trusting one fixed interface name (robust to
    the capture having been taken on whichever interface was actually up)."""
    from collections import Counter
    ips = Counter()
    for pkt in iface_hint_pcap_packets:
        if IP in pkt:
            ips[pkt[IP].src] += 1
            ips[pkt[IP].dst] += 1
    if not ips:
        return None
    # The local machine's IP appears on nearly every packet (as src or dst);
    # a given remote IP appears only in the packets for that one connection.
    return ips.most_common(1)[0][0]


def _classify_outbound_nic(pkt, local_ip: str) -> bool:
    return IP in pkt and pkt[IP].src == local_ip


def _classify_outbound_nc_port(pkt) -> bool:
    """Loopback bulk-transfer capture: both sides are 127.0.0.1, so direction
    comes from which side of the nc listener a packet is headed. Data
    flowing TO the listener port is this machine's upload."""
    if TCP not in pkt:
        return False
    return pkt[TCP].dport == NC_PORT


def _classify_outbound_scan(pkt) -> bool:
    """Loopback port-scan capture: a SYN (no ACK) is the probe going out; a
    RST or SYN-ACK is the target's response coming back."""
    if TCP not in pkt:
        return False
    flags = pkt[TCP].flags
    is_syn_only = ("S" in flags) and ("A" not in flags)
    return is_syn_only


def _direction_classifier(mode: str, packets):
    if mode == "nic":
        local_ip = _local_ip_for(packets)
        return lambda pkt: _classify_outbound_nic(pkt, local_ip)
    if mode == "nc_port":
        return _classify_outbound_nc_port
    if mode == "tcp_flags":
        return _classify_outbound_scan
    raise ValueError(mode)


def parse_pcap(path: Path, mode: str) -> list[dict]:
    """One scenario's pcap -> a chronological list of per-tick feature dicts
    (ticks with fewer than LAYER3_MIN_PACKETS packets are dropped, exactly
    as the live path drops them)."""
    packets = rdpcap(str(path))
    if not packets:
        return []
    classify = _direction_classifier(mode, packets)

    start = float(packets[0].time)
    buckets: dict[int, list] = {}
    for pkt in packets:
        if IP not in pkt:
            continue
        idx = int((float(pkt.time) - start) // TICK_SECONDS)
        buckets.setdefault(idx, []).append(pkt)

    rows = []
    for idx in sorted(buckets):
        pkts = buckets[idx]
        sizes = [len(p) for p in pkts]
        times = [float(p.time) for p in pkts]
        outbound = [classify(p) for p in pkts]
        f = tick_features(sizes, times, outbound, tick_seconds=TICK_SECONDS)
        if f is not None:
            rows.append(f)
    return rows


def main() -> int:
    scenarios_meta = {}
    all_ticks = []

    for scenario, (category, attack_mix, mode) in _SCENARIOS.items():
        pcap_path = RAW_DIR / f"{scenario}.pcap"
        if not pcap_path.exists():
            print(f"skipping {scenario}: {pcap_path} not found "
                  f"(run ./client/rl_agent/capture_scenario.sh {scenario} first)")
            continue
        rows = parse_pcap(pcap_path, mode)
        scenarios_meta[scenario] = {
            "category": category,
            "attack_mix": attack_mix,
            "direction_mode": mode,
            "n_ticks": len(rows),
        }
        for tick_index, features in enumerate(rows):
            all_ticks.append({
                "scenario": scenario,
                "category": category,
                "attack_mix": attack_mix,
                "tick_index": tick_index,
                "n_ticks_in_scenario": len(rows),
                "features": features,
            })
        print(f"{scenario}: {len(rows)} ticks")

    if not all_ticks:
        print("no pcaps found under", RAW_DIR, "-- nothing to write")
        return 1

    corpus = {
        "schema_version": 1,
        "generated_by": "python -m client.rl_agent.capture_real_traffic",
        "notes": [
            "idle/browsing are real BENIGN ticks; beaconing/bulk_upload/"
            "port_scan are real ticks shaped like their matching "
            "ATTACK_MIXES entry and are for measurement only, never "
            "trained on -- see this module's docstring for why.",
            "bulk_upload/port_scan were captured over loopback (lo0): not "
            "a real NIC path (MTU, latency/jitter).",
            "beaconing used a 7s interval, far faster than a realistic C2 "
            "cadence, so capture finished in one sitting.",
        ],
        "scenarios": scenarios_meta,
        "ticks": all_ticks,
    }
    CORPUS_PATH.write_text(json.dumps(corpus, indent=2) + "\n")
    print(f"wrote {CORPUS_PATH} ({len(all_ticks)} total ticks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
