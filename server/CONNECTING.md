# Connecting to the team server

How to get a real PQC-VPN tunnel to the team droplet from your own laptop.
Takes about 10 minutes the first time. Tested on Windows 11, on university
eduroam and on a mobile hotspot (2026-11-28).

| | |
|---|---|
| Server | `168.144.73.113` |
| PQC handshake | TCP `51821` |
| WireGuard-over-TCP relay | TCP `443` |
| Server key (pinned) | `server-vk.hex` in the repo root |
| Tunnel subnet | `10.8.0.0/24`, server is `10.8.0.1` |

## Why there's a relay

On the networks we've tested from (eduroam, a mobile hotspot), **WireGuard's
own UDP traffic is blocked**: the firewall recognises WireGuard and drops it on
any port. Plain TCP gets through. So WireGuard's packets are carried inside a
TCP connection to port 443 by `wg-relay`, with the stream scrambled so it
doesn't look like WireGuard either. The relay is transport only — your traffic
is still protected by WireGuard keyed with the PQC-derived pre-shared key.
Details: `PROGRESS.md`, Member 2 Week 8.

If you're on a network that doesn't block WireGuard you can skip the relay
(leave out `--relay` in step 2 and skip step 3), but using it always is
simplest.

## You need

- The repo, pulled to latest `main`, and a Rust toolchain (`cargo`).
- The official WireGuard app: <https://www.wireguard.com/install/>
  (Windows installer, or the Mac App Store).

## 1. Build the two tools (once)

From the repo root:

```bash
cargo build -p handshake-server --bin test-client --bin wg-relay
```

They land in `target/debug/` (`.exe` on Windows).

## 2. Do a PQC handshake and get your tunnel config

```bash
# Windows
target\debug\test-client.exe 168.144.73.113:51821 @server-vk.hex --relay --out pqc0.conf
# macOS / Linux
./target/debug/test-client 168.144.73.113:51821 @server-vk.hex --relay --out pqc0.conf
```

You should see `server signature verified`, `HANDSHAKE OK` and your
`assigned IP` (`10.8.0.x`). This is the real post-quantum handshake (ML-KEM-768
by default; `--algo 512|1024` to choose). The server has now installed the
derived key for your laptop.

`pqc0.conf` contains **your private WireGuard key and pre-shared key** — keep
it to yourself and don't commit it. Everyone runs step 2 on their own laptop;
never share a config.

## 3. Start the relay — and leave it running

In a separate terminal:

```bash
target\debug\wg-relay.exe client --server 168.144.73.113:443      # Windows
./target/debug/wg-relay client --server 168.144.73.113:443        # macOS / Linux
```

It prints `connected to 168.144.73.113:443`. Keep this terminal open for as
long as you want the tunnel. If the network changes it reconnects by itself.

## 4. Bring the tunnel up

WireGuard app → **Import tunnel(s) from file** → `pqc0.conf` → **Activate**.

## 5. Check it

```bash
ping 10.8.0.1
```

Replies mean the full chain works: PQC handshake → derived key → WireGuard →
relay → server. The WireGuard app's **Log** tab should show `Receiving
handshake response` and then a `keepalive` every 25 s. Over the relay on
campus Wi-Fi, ~300–500 ms ping and a few Mbit/s are normal.

The config only routes `10.8.0.0/24` into the tunnel, so the rest of your
internet is unaffected.

## When you're done

Deactivate the tunnel and stop the relay (Ctrl+C). Peers that stay idle for
an hour are removed from the server automatically, so a later session just
repeats steps 2–5.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `test-client`: connection refused / timed out | Network blocks TCP 51821, or the server is down — tell Member 2 |
| `test-client`: `AuthFailed` / signature did not verify | Your `server-vk.hex` is stale — pull `main` |
| `test-client`: `Error 0x07 rate limited` | Too many handshakes from one IP (shared campus NAT); wait a minute |
| WireGuard log: `Handshake did not complete after 5 seconds` | The relay isn't running, or the tunnel's `Endpoint` isn't `127.0.0.1:51820` |
| Relay keeps printing `connection lost` | Network is interfering with TCP 443 too — try another network and tell Member 2 |
| `wg-relay`: address already in use | Another relay is already running, or the WireGuard app's *own* listen port is 51820 — stop the other one |
| Ping worked, then stopped (relay still says `connected`) | The server machine was rebooted, or your peer was idle for over an hour (a routine redeploy doesn't do this). The server never stores tunnel keys, so it can't restore them: redo step 2, then in the WireGuard app delete the old tunnel and import the new `pqc0.conf` |

## The desktop app

Rimsha's app (and `cargo run -p desktop --example vertical_slice`) already
does the real PQC handshake, the RL agent and the anomaly layers against this
server, and works on eduroam with no relay — it uses TCP 51821 only. It
doesn't bring the WireGuard interface up yet (Member 1, Week 9); when it does,
it should start the relay and point WireGuard's Endpoint at it exactly as in
steps 3–4.
