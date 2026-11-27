#!/usr/bin/env bash
# client/rl_agent/capture_scenario.sh — real local packet capture, Week 8.
#
# This generates the raw, real pcaps that `capture_real_traffic.py` parses
# into `real_capture_corpus.json` to ground Layer 3 (Isolation Forest) in
# captured traffic instead of only the synthetic generator in
# `train_isolation_forest.py`.
#
# YOU run this, not Claude: it needs `sudo` for `tcpdump`/`nmap`, and an
# automated tool call has no way to answer an interactive password prompt.
# One password prompt covers a whole invocation (sudo caches it briefly).
#
# Everything attack-shaped stays on loopback (127.0.0.1) only — never a
# third-party host. `browsing`/`beaconing` touch a handful of real public
# pages, which carries no authorization concern (plain GETs).
#
# Usage:
#   ./client/rl_agent/capture_scenario.sh idle        [duration_seconds=300]
#   ./client/rl_agent/capture_scenario.sh browsing    [duration_seconds=300]
#   ./client/rl_agent/capture_scenario.sh beaconing   [duration_seconds=300]
#   ./client/rl_agent/capture_scenario.sh bulk_upload [duration_seconds=120]
#   ./client/rl_agent/capture_scenario.sh port_scan
#
# Run all five in one sitting (~20-30 min total) for a corpus big enough to
# retrain against, then: python -m client.rl_agent.capture_real_traffic
#
# Known, deliberate limitations (see train_isolation_forest.py's retrain
# notes for how these are accounted for, not hidden):
#   - bulk_upload/port_scan run over loopback (lo0), whose ~64KB MTU and
#     zero real-world latency/jitter are not a real NIC path.
#   - beaconing uses a 7s interval, not a realistic C2 cadence of minutes to
#     hours, so the capture finishes in one sitting.
set -euo pipefail

SCENARIO="${1:?usage: $0 <idle|browsing|beaconing|bulk_upload|port_scan> [duration_seconds]}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="$SCRIPT_DIR/models/real_capture/raw"
mkdir -p "$RAW_DIR"
PCAP="$RAW_DIR/${SCENARIO}.pcap"
IFACE_EXTERNAL="${IFACE:-en0}"
NC_PORT=8737   # also hardcoded in capture_real_traffic.py's direction classifier

start_capture() {
    local iface="$1"
    echo "== starting tcpdump on $iface -> $PCAP (sudo) =="
    sudo tcpdump -i "$iface" -w "$PCAP" -U >/tmp/capture_scenario_tcpdump.log 2>&1 &
    TCPDUMP_PID=$!
    sleep 2   # let tcpdump attach before any traffic starts
}

stop_capture() {
    sudo kill -INT "$TCPDUMP_PID" 2>/dev/null || true
    wait "$TCPDUMP_PID" 2>/dev/null || true
    echo "== wrote $PCAP =="
}

case "$SCENARIO" in
    idle)
        DURATION="${2:-300}"
        start_capture "$IFACE_EXTERNAL"
        echo "capturing ambient traffic for ${DURATION}s -- do nothing deliberate on this machine..."
        sleep "$DURATION"
        stop_capture
        ;;

    browsing)
        DURATION="${2:-300}"
        start_capture "$IFACE_EXTERNAL"
        URLS=(https://example.com https://www.wikipedia.org https://httpbin.org/get https://www.iana.org)
        END=$((SECONDS + DURATION))
        while [ $SECONDS -lt $END ]; do
            for u in "${URLS[@]}"; do
                curl -s -o /dev/null "$u" || true
                sleep "$(awk 'BEGIN{srand(); print 0.5 + rand()*2}')"
            done
        done
        stop_capture
        ;;

    beaconing)
        DURATION="${2:-300}"
        start_capture "$IFACE_EXTERNAL"
        END=$((SECONDS + DURATION))
        while [ $SECONDS -lt $END ]; do
            curl -s -o /dev/null https://example.com || true
            sleep 7
        done
        stop_capture
        ;;

    bulk_upload)
        DURATION="${2:-120}"
        start_capture lo0
        TMPFILE="$(mktemp)"
        RATE_MB=5
        TOTAL_BYTES=$((RATE_MB * 1024 * 1024 * DURATION))
        head -c "$TOTAL_BYTES" /dev/urandom > "$TMPFILE"
        # Plain BSD nc's stream-closing semantics on macOS were verified
        # unreliable for a one-shot bulk transfer (truncated after a few
        # KB); an HTTP PUT has explicit Content-Length framing instead.
        # An UNTHROTTLED push was also verified to overrun tcpdump's
        # capture buffer under sudo (68% of packets silently dropped by
        # the kernel) -- --limit-rate paces it to something tcpdump can
        # keep up with, which also better matches what a real sustained
        # exfiltration actually looks like (see EXFIL_UPLOAD_BPS_THRESHOLD)
        # than an instant burst would.
        python3 -c "
import http.server, socketserver
class H(http.server.BaseHTTPRequestHandler):
    def do_PUT(self):
        remaining = int(self.headers.get('Content-Length', 0))
        while remaining > 0:
            chunk = self.rfile.read(min(65536, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
        self.send_response(200)
        self.end_headers()
    def log_message(self, *a): pass
with socketserver.TCPServer(('127.0.0.1', $NC_PORT), H) as httpd:
    httpd.handle_request()
" &
        SERVER_PID=$!
        sleep 1
        curl -s -X PUT --limit-rate "${RATE_MB}M" --data-binary "@$TMPFILE" \
            "http://127.0.0.1:$NC_PORT/upload" -o /dev/null
        wait "$SERVER_PID" 2>/dev/null || true
        rm -f "$TMPFILE"
        sleep 2   # grace period for tcpdump to flush before SIGINT (see port_scan)
        stop_capture
        ;;

    port_scan)
        start_capture lo0
        echo "== running sudo nmap -sS -p- against 127.0.0.1 =="
        sudo nmap -sS -p- -T4 127.0.0.1 || true
        sleep 2
        stop_capture
        ;;

    *)
        echo "unknown scenario: $SCENARIO" >&2
        exit 1
        ;;
esac
