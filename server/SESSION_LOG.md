# Session log — schema v1

The handshake server writes one JSON object per line (**JSON Lines**) for every
session-level event: connects, rekeys, rejections, rate-limit refusals, the
WireGuard tunnel coming up and going down, and evictions. It exists for
Member 3's baseline evaluation (session-level data) and Member 1's dashboard.
Member 2, Week 9.

- **On the droplet:** `/var/lib/pqc-vpn/sessions.jsonl`. It rotates to
  `sessions.jsonl.1` at 20 MB, and both survive redeploys.
- **Local dry-run server:** next to the identity file, or `--session-log PATH`.
  Turn it off with `--no-session-log`.
- **Never logged:** PSKs, keys other than public WireGuard keys, and any other
  secret.

## Getting it

```bash
# summary (text, or --json for a machine-readable one)
ssh member2@168.144.73.113 "sudo docker exec pqc-handshake-server session-report /var/lib/pqc-vpn/sessions.jsonl"

# the raw log, to your laptop
ssh member2@168.144.73.113 "sudo cat /var/lib/pqc-vpn/sessions.jsonl" > sessions.jsonl

# follow it live during a demo
ssh member2@168.144.73.113 "sudo tail -f /var/lib/pqc-vpn/sessions.jsonl"
```

Locally: `cargo run -p handshake-server --bin session-report -- sessions.jsonl`.

```python
import pandas as pd
df = pd.read_json("sessions.jsonl", lines=True)
df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms")
hs = df[df.event.isin(["handshake", "rekey"])]
hs.groupby("algo").handshake_ms.describe()          # handshake cost per ML-KEM level
df[df.event == "tunnel_down"].up_s.describe()       # tunnel session lengths
```

## Common fields

| field | type | meaning |
|---|---|---|
| `v` | int | schema version, `1` |
| `ts_ms` | int | Unix time, milliseconds (server clock, UTC) |
| `event` | string | one of the events below |

`peer` is always the client's **WireGuard public key** (base64). It is the
stable client identity: the same laptop keeps it across reconnects, but the
desktop app currently generates a new one per run. `session_id` is the
16-byte PQC session id (hex), as in `PROTOCOL.md`. `src` is the client's
source IP, which may be shared by every laptop behind one NAT.

## Events

### `server_start`
The server started. Every later record ran under this configuration.
`version`, `psk_install` (`wg set wg0` or `dry-run`), `max_peers`,
`idle_timeout_s` (0 = no eviction), `replay_window_s`, `rate_burst`,
`rate_per_min`, `max_conns`.

### `handshake` — a client connected
A full `ClientHello` handshake completed and the PSK was installed.
`session_id`, `peer`, `assigned_ip`, `algo` (`ML-KEM-512|768|1024`), `src`,
`new_peer` (`false` = a known WireGuard key reconnecting, same `assigned_ip`),
`handshake_ms`.

`handshake_ms` is **server-side**: from receiving message 1 to sending
`ServerFinish`. It includes the server's ML-KEM encapsulation and ML-DSA-65
signing, **plus one client round trip** (ServerHello → ClientFinish). Compare
algorithms from the same client and network, or subtract a ping RTT. It is not
a client-observed connect time.

### `rekey` — fresh PSK, same tunnel
A `RekeyRequest` completed. Fields are as for `handshake`, plus `from_algo`,
`escalated` (stronger than before; `REKEY_ESCALATES`) and `rekeys` (count on
this session, including this one). A rekey never changes `assigned_ip` and
never drops the tunnel.

### `rejected` — the server answered with a protocol `Error`
`src`, `stage` (`hello` | `finish` | `registry` | `install`), `code` (the
`PROTOCOL.md` §4.7 code: 2 MALFORMED, 3 ALGO_REJECTED, 4 UNKNOWN_SESSION,
5 ALGO_MISMATCH/downgrade, 6 AUTH_FAILED, 8 INTERNAL, 9 CAPACITY), `reason`,
`rekey` (bool), and `peer` / `algo` when the hello was readable.

### `refused` — turned away by the Week 7 guard
Before any PQC work. `src`, `code` (6 = replayed `client_nonce`, 7 = rate
limited or server busy), `reason`.

### `aborted` — the client went away mid-handshake
The socket closed, timed out or sent a garbled frame after the handshake had
started. `src`, `error`.

### `tunnel_up` / `tunnel_down` — the WireGuard session itself
Protocol v1 has no disconnect message, so these come from WireGuard. The
server polls `wg show` every 15 s:

- **`tunnel_up`** fires when a registered peer has a WireGuard handshake
  newer than 180 s. Fields: `peer`, `session_id`, `assigned_ip`, `algo`.
- **`tunnel_down`** fires once there has been **no WireGuard handshake for
  180 s** (WireGuard's `REJECT_AFTER_TIME`, after which its keys are dead).
  Fields: `peer`, `session_id`, `assigned_ip`; `idle_s` (since the last
  WireGuard handshake); `up_s` (from tunnel up to that last handshake, i.e.
  the session length); `rx_bytes` / `tx_bytes` (WireGuard's counters for the
  peer).

Every client config sets `PersistentKeepalive = 25`, so a live tunnel
re-handshakes about every 2 minutes. `tunnel_down` therefore means the client
really disconnected, or lost its tunnel because the server rebooted.
Detection lags by up to 180 s + 15 s, so `up_s` is the better session-length
measure than the time between the two events.

**Server restarts.** Liveness is tracked in memory. After a restart or
redeploy, the server logs a fresh `tunnel_up` for every tunnel that is still
live (WireGuard keeps running across a container redeploy), with no
`tunnel_down` before it. `up_s` then counts from that restart, so a session
that spans a `server_start` is longer than its `up_s`. Split sessions at
`server_start` records when measuring.

`rx_bytes` / `tx_bytes` are WireGuard's cumulative counters for that peer
since it was added to `wg0`, not just for the last up period. A rekey doesn't
reset them; a reboot or an eviction does. Take differences between successive
`tunnel_down` records of the same peer for per-period traffic.

### `evicted` — a peer was removed
`peer`, `reason`:

- `idle`: the `--idle-timeout-secs` sweep, where both the PQC and the
  WireGuard activity were older than the timeout. Also logs `session_id`,
  `assigned_ip`, `session_age_s` and `rekeys`.
- `unknown_on_startup`: on `wg0` but not in the registry.

## Compatibility

New fields may be added to an event without bumping `v`; readers should ignore
unknown fields and events. Renaming or removing a field, or changing its
meaning, bumps `v`. The field names are pinned by
`events::tests::record_shape_is_stable` and the `tests/session_log.rs` suite.
