# syntax=docker/dockerfile:1
#
# PQC-VPN handshake server (server/handshake-server).
# Build context is the repo root:  docker build -f server/Dockerfile .
#
# Multi-stage: a Rust builder, then a slim Debian runtime that carries only the
# binary plus `wireguard-tools` (the server shells out to `wg set` / `wg show`).

# ---- builder ----------------------------------------------------------------
FROM rust:1-slim-trixie AS builder
# Since Week 7 vpn_core links ONNX Runtime (the `ort` crate, for the client's RL
# policy). The server never runs the policy, but it depends on vpn_core, and the
# prebuilt static ONNX Runtime is C++ — the link needs libstdc++ (from g++).
# Debian 13 (trixie), not bookworm: that prebuilt library needs glibc >= 2.38
# (`__isoc23_strtol`) and GCC >= 13's libstdc++; bookworm has 2.36 / GCC 12.
RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build

# Copy the whole workspace. cargo needs every member manifest to resolve the
# workspace, but only `handshake-server` and its deps are compiled.
COPY . .

# BuildKit cache mounts keep the crate registry and target dir out of the image
# layer and make rebuilds fast. Copy the binary out before the mounts unmount.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    cargo build --release -p handshake-server --bin handshake-server --bin wg-relay \
    && cp target/release/handshake-server /handshake-server \
    && cp target/release/wg-relay /wg-relay

# ---- runtime --------------------------------------------------------------
FROM debian:trixie-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends wireguard-tools ca-certificates libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /handshake-server /usr/local/bin/handshake-server
# UDP-over-TCP relay for WireGuard (Week 8) — run by the `wg-relay` compose service.
COPY --from=builder /wg-relay /usr/local/bin/wg-relay

# Identity seed + derived .pub live here; mount a volume so they survive
# container recreation (the .pub is pinned in every client).
VOLUME ["/var/lib/pqc-vpn"]

EXPOSE 51821/tcp 443/tcp
ENTRYPOINT ["/usr/local/bin/handshake-server"]
# --idle-timeout-secs: evict a peer only when BOTH its last PQC handshake and
# its last WireGuard handshake are over an hour old. A live tunnel re-handshakes
# every ~2 min, so it is never evicted; this stops one-off clients (the desktop
# app currently makes a fresh WireGuard key per run) filling the /24.
CMD ["--bind", "0.0.0.0:51821", \
     "--identity", "/var/lib/pqc-vpn/server-identity.seed", \
     "--wg-interface", "wg0", \
     "--wg-port", "51820", \
     "--tunnel-cidr", "10.8.0.0/24", \
     "--idle-timeout-secs", "3600"]
