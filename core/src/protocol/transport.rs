//! Real network I/O for the handshake.
//!
//! This is the Week 6 deliverable itself
//! (`TEAM_TIMELINE_PROPOSAL.md`, Member 1 Week 6): *"Integrate `protocol/`
//! client side against Member 2's server ... First real handshake over the
//! network, Rust client <-> Rust-or-mock server."* Everything in
//! [`wire`](super::wire), [`client`](super::client) and
//! [`tags`](super::tags) is pure logic with no I/O, exercised against
//! `server/handshake-vectors.json` and an in-process simulated server; this
//! module is the one place that opens a real `TcpStream` and drives that
//! logic against whatever is actually listening on the other end —
//! `server/handshake-server` today, a mock in a test, a mobile port's own
//! socket layer later.

use std::io;
use std::net::{Ipv4Addr, SocketAddr, TcpStream, ToSocketAddrs};
use std::time::Duration;

use crate::crypto::MlKemLevel;
use crate::state::TunnelConfig;

use super::client::{ClientHandshake, HandshakeError};
use super::tags;
use super::wire::{FrameError, Message, RekeyRequest};

/// How long to wait for any single read or write during the handshake.
/// Matches the server's own idle timeout (`server/PROTOCOL.md` §2) — if the
/// server hasn't replied by then, something else is wrong and retrying
/// blindly won't help.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);

/// Everything that can go wrong running the handshake over the network.
#[derive(Debug)]
pub enum ProtocolError {
    /// A transport-level failure (connect, read, write, timeout).
    Io(io::Error),
    /// A frame did not decode (`server/PROTOCOL.md` §3).
    Wire(super::wire::WireError),
    /// The handshake logic itself rejected the exchange (bad signature,
    /// algorithm mismatch, ...).
    Handshake(HandshakeError),
    /// The server replied with an `Error` message (`PROTOCOL.md` §4.7).
    Server { code: u8, message: String },
    /// A structurally valid message arrived, but not the one expected next.
    UnexpectedMessage(&'static str),
}

impl std::fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProtocolError::Io(e) => write!(f, "network error: {e}"),
            ProtocolError::Wire(e) => write!(f, "malformed message: {e}"),
            ProtocolError::Handshake(e) => write!(f, "{e}"),
            ProtocolError::Server { code, message } => {
                write!(f, "server rejected the handshake ({code:#04x}): {message}")
            }
            ProtocolError::UnexpectedMessage(what) => write!(f, "{what}"),
        }
    }
}

impl std::error::Error for ProtocolError {}

impl From<io::Error> for ProtocolError {
    fn from(e: io::Error) -> Self {
        ProtocolError::Io(e)
    }
}

impl From<FrameError> for ProtocolError {
    fn from(e: FrameError) -> Self {
        match e {
            FrameError::Io(e) => ProtocolError::Io(e),
            FrameError::Wire(e) => ProtocolError::Wire(e),
        }
    }
}

impl From<HandshakeError> for ProtocolError {
    fn from(e: HandshakeError) -> Self {
        ProtocolError::Handshake(e)
    }
}

/// A completed handshake (or rekey): the tunnel parameters, plus what the
/// client needs to rekey this session later.
///
/// Week 7: the RL control loop (`core::session`) has to send `RekeyRequest`
/// with the session's id (`PROTOCOL.md` §4.6), so [`connect`]'s bare
/// [`TunnelConfig`] is no longer enough on its own.
#[derive(Debug)]
pub struct Established {
    /// Server-assigned session id. Stays the same across rekeys
    /// (`PROTOCOL.md` §4.6); a fresh `ClientHello` gets a new one.
    pub session_id: [u8; 16],
    /// The algorithm this handshake actually ran at.
    pub algo: MlKemLevel,
    /// Everything [`TunnelHandle::bring_up`](crate::state::TunnelHandle::bring_up)
    /// or [`rotate_psk`](crate::state::TunnelHandle::rotate_psk) needs.
    pub tunnel: TunnelConfig,
}

/// Runs one full handshake against `addr` and returns everything
/// [`TunnelHandle::bring_up`](crate::state::TunnelHandle::bring_up) needs.
///
/// `pinned_verifying_key` is the server's 1952-byte ML-DSA-65 public key,
/// shipped with the client (`server/PROTOCOL.md` §5.2) — this function
/// never fetches it and never trusts one it wasn't given. `client_wg_pubkey`
/// is this client's own WireGuard public key, which becomes the server's
/// peer identity for the session (`PROTOCOL.md` §4.2, §8 item 5).
///
/// Four messages, one TCP connection, matching `PROTOCOL.md` §1's diagram:
/// `ClientHello -> ServerHello -> ClientFinish -> ServerFinish`. The
/// connection is closed on return (success or failure) either way, per
/// `PROTOCOL.md` §2 ("one handshake per TCP connection, then close").
pub fn connect<A: ToSocketAddrs>(
    addr: A,
    algo: MlKemLevel,
    client_wg_pubkey: [u8; 32],
    pinned_verifying_key: &[u8],
    timeout: Duration,
) -> Result<TunnelConfig, ProtocolError> {
    handshake(addr, algo, client_wg_pubkey, pinned_verifying_key, timeout).map(|e| e.tunnel)
}

/// [`connect`], but also returns the session id needed for a later
/// [`rekey`]. Also how the client changes algorithm mid-session: a fresh
/// `ClientHello` is the only way to step *down* (`PROTOCOL.md` §6), and the
/// server keeps the same peer (and tunnel address) for the same
/// `client_wg_pubkey`.
pub fn handshake<A: ToSocketAddrs>(
    addr: A,
    algo: MlKemLevel,
    client_wg_pubkey: [u8; 32],
    pinned_verifying_key: &[u8],
    timeout: Duration,
) -> Result<Established, ProtocolError> {
    run(addr, algo, client_wg_pubkey, pinned_verifying_key, timeout, None)
}

/// Rotates key material for an existing session (`PROTOCOL.md` §4.6, §6):
/// `RekeyRequest` carrying `session_id`, then the same three remaining
/// messages as a handshake. The server swaps the peer's PSK without
/// dropping it.
///
/// `algo` may be equal to or stronger than the session's in-force
/// algorithm (`REKEY_ESCALATES`); a weaker one comes back as
/// `ProtocolError::Server { code: 0x05 }` (`ALGO_MISMATCH`). The decision
/// gate never asks for a weaker one, so that error means the client and
/// server disagree about what is in force.
pub fn rekey<A: ToSocketAddrs>(
    addr: A,
    session_id: [u8; 16],
    algo: MlKemLevel,
    client_wg_pubkey: [u8; 32],
    pinned_verifying_key: &[u8],
    timeout: Duration,
) -> Result<Established, ProtocolError> {
    run(addr, algo, client_wg_pubkey, pinned_verifying_key, timeout, Some(session_id))
}

fn run<A: ToSocketAddrs>(
    addr: A,
    algo: MlKemLevel,
    client_wg_pubkey: [u8; 32],
    pinned_verifying_key: &[u8],
    timeout: Duration,
    rekey_session: Option<[u8; 16]>,
) -> Result<Established, ProtocolError> {
    let mut stream = connect_with_timeout(addr, timeout)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    let peer_ip = stream.peer_addr()?.ip();

    let handshake = ClientHandshake::start(algo, client_wg_pubkey);

    match rekey_session {
        None => Message::ClientHello(handshake.hello().clone()).write(&mut stream)?,
        Some(session_id) => Message::RekeyRequest(RekeyRequest {
            session_id,
            hello: handshake.hello().clone(),
        })
        .write(&mut stream)?,
    }

    let server_hello = match Message::read(&mut stream)? {
        Message::ServerHello(sh) => sh,
        Message::Error(e) => {
            return Err(ProtocolError::Server { code: e.code, message: e.message });
        }
        _ => return Err(ProtocolError::UnexpectedMessage("expected ServerHello")),
    };

    if let Some(session_id) = rekey_session {
        if server_hello.session_id != session_id {
            // A rekey keeps the session id (§4.6). A different one means the
            // server treated this as some other session; don't install keys
            // for it.
            return Err(ProtocolError::UnexpectedMessage(
                "ServerHello session id did not match the rekeyed session",
            ));
        }
    }

    let (keys, finish, transcript) = handshake.finish(&server_hello, pinned_verifying_key)?;

    Message::ClientFinish(finish.clone()).write(&mut stream)?;

    let server_finish = match Message::read(&mut stream)? {
        Message::ServerFinish(sf) => sf,
        Message::Error(e) => {
            return Err(ProtocolError::Server { code: e.code, message: e.message });
        }
        _ => return Err(ProtocolError::UnexpectedMessage("expected ServerFinish")),
    };

    if server_finish.session_id != finish.session_id {
        return Err(ProtocolError::UnexpectedMessage("ServerFinish session id did not match ClientFinish"));
    }

    if !tags::verify_server_tag(&keys.confirm_key, &transcript, &server_finish.server_tag) {
        // The server proved nothing about its own derived secret — never
        // trust the tunnel parameters that follow.
        return Err(ProtocolError::Handshake(HandshakeError::AuthFailed));
    }

    Ok(Established {
        session_id: server_finish.session_id,
        algo,
        tunnel: TunnelConfig {
            assigned_ip: Ipv4Addr::from(server_finish.assigned_ip),
            server_wg_pubkey: server_finish.server_wg_pubkey,
            server_endpoint: SocketAddr::new(peer_ip, server_finish.wg_port),
            psk: keys.psk,
        },
    })
}

/// `TcpStream::connect` has no timeout, and an unreachable server would
/// otherwise stall a control-loop tick for the OS default (tens of seconds).
/// Tries each resolved address in turn with `timeout`.
fn connect_with_timeout<A: ToSocketAddrs>(addr: A, timeout: Duration) -> io::Result<TcpStream> {
    let mut last_err = None;
    for a in addr.to_socket_addrs()? {
        match TcpStream::connect_timeout(&a, timeout) {
            Ok(s) => return Ok(s),
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.unwrap_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "address resolved to nothing")))
}

#[cfg(test)]
mod tests {
    use std::net::TcpListener;
    use std::sync::Arc;

    use handshake_server::identity::ServerIdentity;
    use handshake_server::registry::{Registry, DEFAULT_MAX_PEERS};
    use handshake_server::server::{self, ServerContext};
    use handshake_server::tunnel::{DryRun, TunnelSettings};

    use super::*;

    /// The Week 6 proof: a real `core::protocol::connect` client, over a
    /// real loopback `TcpStream`, against the real
    /// `server/handshake-server` connection handler (dry-run PSK install,
    /// so it needs no `CAP_NET_ADMIN` / no actual WireGuard interface) —
    /// "Rust client <-> Rust-or-mock server" from the delivery plan.
    #[test]
    fn real_handshake_over_a_loopback_socket() {
        let identity = ServerIdentity::generate();
        let pinned_vk = identity.verifying_key_bytes();

        let ctx = Arc::new(ServerContext {
            identity,
            registry: Registry::new([10, 8, 0, 0], DEFAULT_MAX_PEERS),
            installer: Box::new(DryRun),
            tunnel: TunnelSettings {
                server_wg_pubkey: [0x5a; 32],
                wg_port: 51820,
                subnet_base: [10, 8, 0, 0],
            },
        });

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback listener");
        let addr = listener.local_addr().unwrap();

        let server_ctx = Arc::clone(&ctx);
        let server_thread = std::thread::spawn(move || {
            let (stream, _) = listener.accept().expect("accept one connection");
            server::handle_connection(stream, &server_ctx).expect("server-side handshake")
        });

        let client_wg_pubkey = [0x11u8; 32];
        let tunnel = connect(
            addr,
            MlKemLevel::MlKem768,
            client_wg_pubkey,
            &pinned_vk,
            Duration::from_secs(5),
        )
        .expect("client-side handshake over the network");

        server_thread.join().expect("server thread panicked");

        assert_eq!(tunnel.server_wg_pubkey, [0x5a; 32]);
        assert_eq!(tunnel.server_endpoint.port(), 51820);
        assert_eq!(tunnel.assigned_ip, Ipv4Addr::new(10, 8, 0, 2));
        assert_eq!(tunnel.psk.len(), 32);
        assert!(ctx.registry.contains(&client_wg_pubkey));
    }

    /// A real `handshake-server` handler serving `n` sequential connections
    /// on loopback. Returns the address, the pinned key, the shared context
    /// (to inspect the registry) and the server thread.
    fn serve(n: usize) -> (SocketAddr, Vec<u8>, Arc<ServerContext>, std::thread::JoinHandle<()>) {
        let identity = ServerIdentity::generate();
        let pinned_vk = identity.verifying_key_bytes();
        let ctx = Arc::new(ServerContext {
            identity,
            registry: Registry::new([10, 8, 0, 0], DEFAULT_MAX_PEERS),
            installer: Box::new(DryRun),
            tunnel: TunnelSettings {
                server_wg_pubkey: [0x5a; 32],
                wg_port: 51820,
                subnet_base: [10, 8, 0, 0],
            },
        });
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server_ctx = Arc::clone(&ctx);
        let handle = std::thread::spawn(move || {
            for _ in 0..n {
                let (stream, _) = listener.accept().unwrap();
                // Errors are the client's to observe; the server just
                // moves on to the next connection, like the real accept loop.
                let _ = server::handle_connection(stream, &server_ctx);
            }
        });
        (addr, pinned_vk, ctx, handle)
    }

    /// Week 7: the rekey path the RL loop drives. Same session id, same
    /// tunnel address, fresh PSK, and an escalation is accepted.
    #[test]
    fn rekey_over_the_network_keeps_the_session_and_rotates_the_psk() {
        let (addr, vk, _ctx, server) = serve(2);
        let wg = [0x33u8; 32];
        let t = Duration::from_secs(5);

        let first = handshake(addr, MlKemLevel::MlKem512, wg, &vk, t).expect("handshake");
        let second = rekey(addr, first.session_id, MlKemLevel::MlKem1024, wg, &vk, t)
            .expect("rekey with escalation");
        server.join().unwrap();

        assert_eq!(second.session_id, first.session_id);
        assert_eq!(second.algo, MlKemLevel::MlKem1024);
        assert_eq!(second.tunnel.assigned_ip, first.tunnel.assigned_ip);
        assert_ne!(*second.tunnel.psk, *first.tunnel.psk);
    }

    #[test]
    fn rekey_downgrade_is_refused_by_the_server() {
        let (addr, vk, _ctx, server) = serve(2);
        let wg = [0x44u8; 32];
        let t = Duration::from_secs(5);

        let first = handshake(addr, MlKemLevel::MlKem1024, wg, &vk, t).unwrap();
        let result = rekey(addr, first.session_id, MlKemLevel::MlKem512, wg, &vk, t);
        server.join().unwrap();

        assert!(matches!(result, Err(ProtocolError::Server { code: 0x05, .. })), "{result:?}");
    }

    /// A step down in strength is a fresh handshake for the same peer
    /// (`PROTOCOL.md` §6): new session id, same tunnel address.
    #[test]
    fn fresh_handshake_can_step_down_for_the_same_peer() {
        let (addr, vk, ctx, server) = serve(2);
        let wg = [0x55u8; 32];
        let t = Duration::from_secs(5);

        let first = handshake(addr, MlKemLevel::MlKem1024, wg, &vk, t).unwrap();
        let second = handshake(addr, MlKemLevel::MlKem512, wg, &vk, t).unwrap();
        server.join().unwrap();

        assert_ne!(second.session_id, first.session_id);
        assert_eq!(second.tunnel.assigned_ip, first.tunnel.assigned_ip);
        assert!(ctx.registry.contains(&wg));
    }

    #[test]
    fn unreachable_server_fails_within_the_timeout() {
        // Bind then drop, so the port is very likely closed.
        let addr = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap();
        let start = std::time::Instant::now();
        let r = handshake(addr, MlKemLevel::MlKem768, [0u8; 32], &[0u8; 1952], Duration::from_secs(2));
        assert!(matches!(r, Err(ProtocolError::Io(_))));
        assert!(start.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn wrong_pinned_key_over_the_network_is_rejected_client_side() {
        let identity = ServerIdentity::generate();
        let attacker_vk = ServerIdentity::generate().verifying_key_bytes();

        let ctx = Arc::new(ServerContext {
            identity,
            registry: Registry::new([10, 8, 0, 0], DEFAULT_MAX_PEERS),
            installer: Box::new(DryRun),
            tunnel: TunnelSettings {
                server_wg_pubkey: [0x5a; 32],
                wg_port: 51820,
                subnet_base: [10, 8, 0, 0],
            },
        });

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let server_thread = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            // The server completes its side; only the client's signature
            // check should fail.
            let _ = server::handle_connection(stream, &ctx);
        });

        let result = connect(
            addr,
            MlKemLevel::MlKem768,
            [0x22u8; 32],
            &attacker_vk,
            Duration::from_secs(5),
        );

        server_thread.join().unwrap();

        assert!(matches!(
            result,
            Err(ProtocolError::Handshake(HandshakeError::AuthFailed))
        ));
    }
}
