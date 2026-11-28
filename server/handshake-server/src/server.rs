//! TCP server loop, per-connection handler, and multi-peer housekeeping.
//!
//! Blocking, one thread per connection, one handshake (or rekey) per connection.
//! A completed handshake updates the [`Registry`] and installs the derived PSK
//! into WireGuard.
//!
//! Since Week 7 the accept loop runs every connection through a [`Guard`]
//! (`PROTOCOL.md` §7): a concurrency cap and per-IP rate limit at accept time,
//! and a `client_nonce` replay check on message 1 — all before any PQC work.
//!
//! Since Week 9 every session-level outcome is also written to an
//! [`EventLog`] (`server/SESSION_LOG.md`). The un-logged entry points
//! ([`serve`], [`handle_connection`], [`handle_connection_guarded`]) keep their
//! signatures for existing callers, including `vpn_core`'s loopback tests.

use std::cell::Cell;
use std::collections::HashSet;
use std::net::{IpAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;

use crate::error::HandshakeError;
use crate::events::{Event, EventLog, Liveness, LIVENESS_INTERVAL};
use crate::guard::{Guard, Rejection};
use crate::handshake;
use crate::identity::ServerIdentity;
use crate::kdf;
use crate::registry::{PeerSlot, Registry, RegistryError};
use crate::tunnel::{InstallError, PskInstaller, TunnelSettings};
use crate::wire::{ClientHello, ErrorMsg, FrameError, Message, ServerFinish};

pub const CONN_TIMEOUT: Duration = Duration::from_secs(10);
const SWEEP_INTERVAL: Duration = Duration::from_secs(60);

/// Everything a connection handler needs, shared across threads.
pub struct ServerContext {
    pub identity: ServerIdentity,
    pub registry: Registry,
    pub installer: Box<dyn PskInstaller>,
    pub tunnel: TunnelSettings,
}

fn b64(k: &[u8; 32]) -> String {
    B64.encode(k)
}

fn secs_between(later: SystemTime, earlier: SystemTime) -> u64 {
    later.duration_since(earlier).map(|d| d.as_secs()).unwrap_or(0)
}

/// Remove `wg0` peers the registry does not know about (e.g. leftovers from a
/// previous run). Registered peers missing from `wg0` are kept — their clients
/// re-handshake.
pub fn reconcile_wg_peers(ctx: &ServerContext, log: &EventLog) {
    let installed = match ctx.installer.list_installed_peers() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("reconcile: could not list wg peers: {e}");
            return;
        }
    };
    let known: HashSet<[u8; 32]> = ctx.registry.pubkeys().into_iter().collect();
    let installed_set: HashSet<[u8; 32]> = installed.iter().copied().collect();

    let mut removed = 0;
    for pk in &installed {
        if !known.contains(pk) {
            match ctx.installer.remove_peer(pk) {
                Ok(()) => {
                    removed += 1;
                    log.emit(Event::Evicted {
                        peer: b64(pk),
                        reason: "unknown_on_startup".into(),
                        session_id: None,
                        assigned_ip: None,
                        session_age_s: None,
                        rekeys: None,
                    });
                }
                Err(e) => eprintln!("reconcile: failed to remove {}: {e}", b64(pk)),
            }
        }
    }
    let absent = known.iter().filter(|k| !installed_set.contains(*k)).count();
    println!(
        "reconcile: {} known / {} on wg0 / {removed} stale removed / {absent} awaiting re-handshake",
        known.len(),
        installed.len()
    );
}

/// One idle-sweep pass: evict peers whose last activity *and* last WireGuard
/// handshake are both older than `timeout`. Returns how many were evicted.
pub fn sweep_once(ctx: &ServerContext, log: &EventLog, timeout: Duration, now: SystemTime) -> usize {
    let wg_hs = ctx.installer.last_handshakes().unwrap_or_default();
    let mut evicted = 0;
    for pk in ctx.registry.idle_peers(timeout, now) {
        // If WireGuard still shows a recent handshake, the tunnel is live —
        // keep it and just refresh our clock.
        if let Some(hs) = wg_hs.get(&pk) {
            if now.duration_since(*hs).map(|d| d <= timeout).unwrap_or(false) {
                ctx.registry.touch(&pk);
                continue;
            }
        }
        let Some(peer) = ctx.registry.remove(&pk) else { continue };
        let _ = ctx.installer.remove_peer(&pk);
        println!("idle-sweep: evicted peer {}", b64(&pk));
        log.emit_at(
            Event::Evicted {
                peer: b64(&pk),
                reason: "idle".into(),
                session_id: Some(hex::encode(peer.session_id)),
                assigned_ip: Some(peer.assigned_ip.to_string()),
                session_age_s: Some(secs_between(now, peer.created)),
                rekeys: Some(peer.rekeys),
            },
            now,
        );
        evicted += 1;
    }
    if evicted > 0 {
        let _ = ctx.registry.persist();
    }
    evicted
}

/// Background thread: [`sweep_once`] every minute.
pub fn spawn_idle_sweep(ctx: Arc<ServerContext>, log: Arc<EventLog>, timeout: Duration) {
    std::thread::spawn(move || loop {
        std::thread::sleep(SWEEP_INTERVAL);
        sweep_once(&ctx, &log, timeout, SystemTime::now());
    });
}

/// Background thread: poll WireGuard every [`LIVENESS_INTERVAL`] and log
/// `tunnel_up` / `tunnel_down` transitions (the session log's "disconnect").
pub fn spawn_liveness_monitor(ctx: Arc<ServerContext>, log: Arc<EventLog>) {
    std::thread::spawn(move || {
        let live = Liveness::new();
        loop {
            live.tick(&ctx.registry, ctx.installer.as_ref(), &log, SystemTime::now());
            std::thread::sleep(LIVENESS_INTERVAL);
        }
    });
}

/// Background thread: every minute, drop expired replay-cache entries and
/// fully-refilled rate-limit buckets so both tables stay small.
pub fn spawn_guard_prune(guard: Arc<Guard>) {
    std::thread::spawn(move || loop {
        std::thread::sleep(SWEEP_INTERVAL);
        guard.prune(Instant::now());
    });
}

/// How long the accept loop will spend writing an `Error` frame to a refused
/// connection. The frame is a few dozen bytes and fits in the socket buffer,
/// so this only matters for a peer that never reads — it must not stall
/// `accept` for everyone else.
const REFUSAL_WRITE_TIMEOUT: Duration = Duration::from_millis(200);

/// Accept loop without a session log. See [`serve_logged`].
pub fn serve(listener: TcpListener, ctx: Arc<ServerContext>, guard: Arc<Guard>) {
    serve_logged(listener, ctx, guard, Arc::new(EventLog::null()))
}

/// Accept loop. Returns only if the listener errors fatally.
///
/// Each connection must pass [`Guard::admit`] (concurrency cap, then the
/// per-IP rate limit) before a handler thread is spawned for it. A refused
/// connection gets `Error(RATE_LIMITED)` and is closed from the accept thread.
pub fn serve_logged(listener: TcpListener, ctx: Arc<ServerContext>, guard: Arc<Guard>, log: Arc<EventLog>) {
    for stream in listener.incoming() {
        let s = match stream {
            Ok(s) => s,
            Err(e) => {
                eprintln!("accept error: {e}");
                continue;
            }
        };
        let ip = match s.peer_addr() {
            Ok(a) => a.ip(),
            Err(e) => {
                eprintln!("accept: no peer address: {e}");
                continue;
            }
        };
        let permit = match guard.admit(ip, Instant::now()) {
            Ok(p) => p,
            Err(r) => {
                refuse(s, ip, r, &log);
                continue;
            }
        };
        let ctx = Arc::clone(&ctx);
        let guard = Arc::clone(&guard);
        let log = Arc::clone(&log);
        std::thread::spawn(move || {
            let _permit = permit; // concurrency slot held until this thread ends
            if let Err(e) = handle_connection_logged(s, &ctx, Some(&guard), &log) {
                eprintln!("connection error from {ip}: {e}");
            }
        });
    }
}

fn refuse(mut s: TcpStream, ip: IpAddr, r: Rejection, log: &EventLog) {
    eprintln!("REFUSED    {ip}: {r}");
    log.emit(Event::Refused { src: ip.to_string(), code: r.protocol_code(), reason: r.message().into() });
    let _ = s.set_write_timeout(Some(REFUSAL_WRITE_TIMEOUT));
    let _ = reject(&mut s, r.protocol_code(), r.message());
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnResult {
    Handshake { session_id: [u8; 16] },
    Rekey { session_id: [u8; 16] },
    Rejected { code: u8 },
}

fn registry_reject_code(e: &RegistryError) -> (u8, &'static str) {
    match e {
        RegistryError::AtCapacity => (0x09, "server at peer capacity"),
        RegistryError::UnknownPeer | RegistryError::SessionMismatch => {
            (0x04, "unknown session for this peer")
        }
        RegistryError::Downgrade => (0x05, "rekey may not downgrade the algorithm"),
        RegistryError::AddressPoolExhausted => (0x08, "tunnel address pool exhausted"),
    }
}

/// Handle one connection with no [`Guard`] and no log — no replay or
/// rate-limit checks. Kept for callers (tests, `vpn_core`'s loopback tests)
/// that drive a single connection directly.
pub fn handle_connection(stream: TcpStream, ctx: &ServerContext) -> Result<ConnResult, ConnError> {
    handle_connection_guarded(stream, ctx, None)
}

/// Handle one connection with an optional [`Guard`] and no log.
pub fn handle_connection_guarded(
    stream: TcpStream,
    ctx: &ServerContext,
    guard: Option<&Guard>,
) -> Result<ConnResult, ConnError> {
    handle_connection_logged(stream, ctx, guard, &EventLog::null())
}

/// Handle one connection. With a `guard`, the `client_nonce` of message 1 is
/// checked against the replay cache before any ML-KEM / ML-DSA work is done.
/// Every outcome — success, protocol rejection, or the client vanishing
/// mid-handshake — is written to `log`.
pub fn handle_connection_logged(
    stream: TcpStream,
    ctx: &ServerContext,
    guard: Option<&Guard>,
    log: &EventLog,
) -> Result<ConnResult, ConnError> {
    let src = stream.peer_addr().map(|a| a.ip().to_string()).unwrap_or_default();
    let logged = Cell::new(false);
    let r = handle_inner(stream, ctx, guard, log, &src, &logged);
    if let Err(e) = &r {
        if !logged.get() {
            log.emit(Event::Aborted { src, error: e.to_string() });
        }
    }
    r
}

fn handle_inner(
    mut stream: TcpStream,
    ctx: &ServerContext,
    guard: Option<&Guard>,
    log: &EventLog,
    src: &str,
    logged: &Cell<bool>,
) -> Result<ConnResult, ConnError> {
    stream.set_read_timeout(Some(CONN_TIMEOUT))?;
    stream.set_write_timeout(Some(CONN_TIMEOUT))?;

    // Every explicit rejection goes to the log exactly once.
    let rejected = |stage: &str, code: u8, reason: &str, hello: Option<&ClientHello>, rekey: bool| {
        logged.set(true);
        log.emit(Event::Rejected {
            src: src.to_string(),
            stage: stage.to_string(),
            code,
            reason: reason.to_string(),
            peer: hello.map(|h| b64(&h.client_wg_pubkey)),
            algo: hello.map(|h| h.algo.name().to_string()),
            rekey,
        });
    };

    // --- message 1: ClientHello or RekeyRequest ---
    let (hello, rekey_session): (ClientHello, Option<[u8; 16]>) = match Message::read(&mut stream) {
        Ok(Message::ClientHello(h)) => (h, None),
        Ok(Message::RekeyRequest(r)) => (r.hello, Some(r.session_id)),
        Ok(other) => {
            rejected("hello", 0x02, "expected ClientHello or RekeyRequest", None, false);
            reject(&mut stream, 0x02, "expected ClientHello or RekeyRequest")?;
            return Err(ConnError::Protocol(format!("unexpected first message: {other:?}")));
        }
        Err(e) => {
            rejected("hello", 0x02, &e.to_string(), None, false);
            let _ = reject(&mut stream, 0x02, &e.to_string());
            return Err(e.into());
        }
    };
    let started = Instant::now();
    let is_rekey_req = rekey_session.is_some();

    // --- replay check (PROTOCOL.md §7) — before any expensive work ---
    if let Some(g) = guard {
        if let Err(r) = g.check_nonce(&hello.client_nonce, Instant::now()) {
            eprintln!("REFUSED    {src}: {r}");
            logged.set(true);
            log.emit(Event::Refused { src: src.to_string(), code: r.protocol_code(), reason: r.message().into() });
            let code = r.protocol_code();
            reject(&mut stream, code, r.message())?;
            return Ok(ConnResult::Rejected { code });
        }
    }

    let session_id = rekey_session.unwrap_or_else(|| rand::random());
    let server_nonce: [u8; 32] = rand::random();

    let outcome = match handshake::respond(&ctx.identity, &hello, session_id, server_nonce) {
        Ok(o) => o,
        Err(e) => {
            let code = e.protocol_code();
            rejected("hello", code, &e.to_string(), Some(&hello), is_rekey_req);
            reject(&mut stream, code, &e.to_string())?;
            return Ok(ConnResult::Rejected { code });
        }
    };

    Message::ServerHello(outcome.server_hello.clone()).write(&mut stream)?;

    // --- message 3: ClientFinish ---
    let fin = match Message::read(&mut stream)? {
        Message::ClientFinish(f) => f,
        other => {
            rejected("finish", 0x02, "expected ClientFinish", Some(&hello), is_rekey_req);
            reject(&mut stream, 0x02, "expected ClientFinish")?;
            return Err(ConnError::Protocol(format!("expected ClientFinish, got {other:?}")));
        }
    };
    if fin.session_id != session_id
        || !kdf::verify_client_tag(&outcome.keys.confirm_key, &outcome.transcript, &fin.client_tag)
    {
        let code = HandshakeError::AuthFailed.protocol_code();
        rejected("finish", code, "client finish tag did not verify", Some(&hello), is_rekey_req);
        reject(&mut stream, code, "client finish tag did not verify")?;
        return Ok(ConnResult::Rejected { code: 0x06 });
    }

    // --- update the registry ---
    let before = ctx.registry.get(&hello.client_wg_pubkey);
    let (slot, is_rekey): (PeerSlot, bool) = if let Some(sid) = rekey_session {
        match ctx.registry.rekey(&hello.client_wg_pubkey, &sid, hello.algo) {
            Ok(s) => (s, true),
            Err(e) => {
                let (code, msg) = registry_reject_code(&e);
                rejected("registry", code, msg, Some(&hello), true);
                reject(&mut stream, code, msg)?;
                return Ok(ConnResult::Rejected { code });
            }
        }
    } else {
        match ctx.registry.upsert_handshake(hello.client_wg_pubkey, hello.algo, session_id) {
            Ok(s) => (s, false),
            Err(e) => {
                let (code, msg) = registry_reject_code(&e);
                rejected("registry", code, msg, Some(&hello), false);
                reject(&mut stream, code, msg)?;
                return Ok(ConnResult::Rejected { code });
            }
        }
    };

    // --- install the PSK ---
    if let Err(e) = ctx.installer.install(
        &hello.client_wg_pubkey,
        &outcome.keys.psk,
        slot.assigned_ip,
        hello.algo,
    ) {
        rejected("install", 0x08, &e.to_string(), Some(&hello), is_rekey);
        reject(&mut stream, 0x08, &e.to_string())?;
        return Err(ConnError::Install(e));
    }
    if let Err(e) = ctx.registry.persist() {
        eprintln!("registry persist failed: {e}");
    }

    let peer_hex = b64(&hello.client_wg_pubkey);
    let kind = if is_rekey { "REKEY    " } else { "HANDSHAKE" };
    println!(
        "{kind}  session={}  algo={}  peer {peer_hex} @ {}  (peers: {})",
        hex::encode(slot.session_id),
        hello.algo.name(),
        slot.assigned_ip,
        ctx.registry.len(),
    );

    // --- message 4: ServerFinish ---
    let server_tag = kdf::server_tag(&outcome.keys.confirm_key, &outcome.transcript);
    Message::ServerFinish(ServerFinish {
        session_id: slot.session_id,
        server_tag,
        server_wg_pubkey: ctx.tunnel.server_wg_pubkey,
        assigned_ip: slot.assigned_ip.octets(),
        wg_port: ctx.tunnel.wg_port,
    })
    .write(&mut stream)?;

    let handshake_ms = (started.elapsed().as_secs_f64() * 1000.0 * 100.0).round() / 100.0;
    logged.set(true);
    log.emit(if is_rekey {
        let from = before.as_ref().map(|p| p.algo).unwrap_or(hello.algo);
        Event::Rekey {
            session_id: hex::encode(slot.session_id),
            peer: peer_hex,
            assigned_ip: slot.assigned_ip.to_string(),
            from_algo: from.name().to_string(),
            algo: hello.algo.name().to_string(),
            escalated: hello.algo > from,
            rekeys: ctx.registry.get(&hello.client_wg_pubkey).map(|p| p.rekeys).unwrap_or(0),
            src: src.to_string(),
            handshake_ms,
        }
    } else {
        Event::Handshake {
            session_id: hex::encode(slot.session_id),
            peer: peer_hex,
            assigned_ip: slot.assigned_ip.to_string(),
            algo: hello.algo.name().to_string(),
            src: src.to_string(),
            new_peer: before.is_none(),
            handshake_ms,
        }
    });

    Ok(if is_rekey {
        ConnResult::Rekey { session_id: slot.session_id }
    } else {
        ConnResult::Handshake { session_id: slot.session_id }
    })
}

fn reject<W: std::io::Write>(w: &mut W, code: u8, message: &str) -> std::io::Result<()> {
    Message::Error(ErrorMsg { code, message: message.to_string() }).write(w)
}

#[derive(Debug)]
pub enum ConnError {
    Io(std::io::Error),
    Frame(FrameError),
    Protocol(String),
    Install(InstallError),
}

impl std::fmt::Display for ConnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConnError::Io(e) => write!(f, "io: {e}"),
            ConnError::Frame(e) => write!(f, "frame: {e}"),
            ConnError::Protocol(m) => write!(f, "protocol: {m}"),
            ConnError::Install(e) => write!(f, "psk install: {e}"),
        }
    }
}
impl std::error::Error for ConnError {}
impl From<std::io::Error> for ConnError {
    fn from(e: std::io::Error) -> Self {
        ConnError::Io(e)
    }
}
impl From<FrameError> for ConnError {
    fn from(e: FrameError) -> Self {
        ConnError::Frame(e)
    }
}
