//! UDP-over-TCP relay for the WireGuard data plane (Week 8 finding).
//!
//! **Why this exists.** Live testing from Pakistani networks (university
//! eduroam and a mobile hotspot, 2026-11-28) showed WireGuard's UDP being
//! dropped in transit while plain TCP to the same server — including this
//! project's own PQC handshake on 51821 — passed. A first, plain version of
//! this relay (length-prefixed WireGuard packets over TCP) then had its TCP
//! connections reset *toward the client only* as soon as a real WireGuard
//! exchange flowed through it, while random payloads on the same port were
//! left alone: the middlebox recognises WireGuard's message structure inside
//! the stream, not just on UDP.
//!
//! So the stream is **obfuscated**: after a 16-byte random nonce from each
//! side, every byte (lengths included) is XORed with a per-direction keystream,
//! and every datagram carries random padding so WireGuard's fixed 148/92-byte
//! handshake sizes don't show either. On the wire it is uniformly random bytes.
//!
//! **This is camouflage, not security.** The keystream key is a public
//! constant; anyone reading this file can undo it. Confidentiality and
//! integrity come, unchanged, from WireGuard keyed by the PQC-derived PSK —
//! the relay only ever carries WireGuard ciphertext. The server side forwards
//! to exactly one fixed address (the local WireGuard port); it is not a proxy.
//!
//! ```text
//!   WireGuard app ──UDP──► relay client ══TCP (obfuscated)══► relay server ──UDP──► wg0
//!   (Endpoint = 127.0.0.1:51820)            (e.g. :443)                (127.0.0.1:51820)
//! ```
//!
//! **Stream format.** `client → server: nonce_c (16)`, `server → client:
//! nonce_s (16)`, then in each direction a sequence of frames, each XORed with
//! that direction's keystream: `u16 BE len ‖ u8 pad ‖ datagram (len) ‖ pad
//! random bytes`. Keystream block `i` = `SHA-256(LABEL ‖ nonce ‖ dir ‖ i)`.
//!
//! Each TCP connection gets its own UDP socket on the server, so WireGuard
//! sees each client as a distinct source port and its endpoint roaming
//! handles reconnects.

use std::io::{self, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream, UdpSocket};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

/// Largest datagram the relay accepts. WireGuard packets are at most
/// MTU-sized (~1500 B); this leaves headroom and stays within `u16`.
pub const MAX_DATAGRAM: usize = 65_535;

/// Random padding per frame is `0..=MAX_PAD` bytes.
pub const MAX_PAD: u8 = 63;

/// Nonce each side sends when the connection opens.
pub const NONCE_LEN: usize = 16;

/// Domain-separation label for the keystream. Public by design (see module docs).
const LABEL: &[u8] = b"pqc-vpn wg-relay obfuscation v1";

/// Keystream direction tags.
const DIR_UP: u8 = 0; // client -> server
const DIR_DOWN: u8 = 1; // server -> client

/// A server-side connection with no traffic in either direction for this
/// long is closed. WireGuard's `PersistentKeepalive = 25` keeps a live tunnel
/// well inside it.
pub const IDLE_TIMEOUT: Duration = Duration::from_secs(180);

/// How often blocked reads wake up to check whether the other half has quit.
const POLL: Duration = Duration::from_millis(500);

/// How long the server waits for the client's nonce.
const OPEN_TIMEOUT: Duration = Duration::from_secs(10);

// ---------------------------------------------------------------------------
// Keystream (obfuscation only)
// ---------------------------------------------------------------------------

/// SHA-256 counter-mode keystream for one direction of one connection.
pub struct Keystream {
    nonce: [u8; NONCE_LEN],
    dir: u8,
    counter: u64,
    block: [u8; 32],
    pos: usize,
}

impl Keystream {
    pub fn new(nonce: [u8; NONCE_LEN], dir: u8) -> Self {
        Self { nonce, dir, counter: 0, block: [0; 32], pos: 32 }
    }

    /// XOR `data` in place with the next `data.len()` keystream bytes.
    pub fn apply(&mut self, data: &mut [u8]) {
        for b in data {
            if self.pos == 32 {
                let mut h = Sha256::new();
                h.update(LABEL);
                h.update(self.nonce);
                h.update([self.dir]);
                h.update(self.counter.to_be_bytes());
                self.block = h.finalize().into();
                self.counter += 1;
                self.pos = 0;
            }
            *b ^= self.block[self.pos];
            self.pos += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Framing
// ---------------------------------------------------------------------------

/// Write one obfuscated, padded frame carrying `datagram`.
pub fn write_frame<W: Write>(w: &mut W, ks: &mut Keystream, datagram: &[u8]) -> io::Result<()> {
    if datagram.is_empty() || datagram.len() > MAX_DATAGRAM {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "datagram size out of range"));
    }
    let pad = rand::random::<u8>() % (MAX_PAD + 1);
    let mut buf = Vec::with_capacity(3 + datagram.len() + pad as usize);
    buf.extend_from_slice(&(datagram.len() as u16).to_be_bytes());
    buf.push(pad);
    buf.extend_from_slice(datagram);
    buf.extend((0..pad).map(|_| rand::random::<u8>()));
    ks.apply(&mut buf);
    w.write_all(&buf)
}

/// Read one frame into `buf`, returning the datagram length.
/// `Ok(None)` on a clean EOF at a frame boundary.
pub fn read_frame<R: Read>(r: &mut R, ks: &mut Keystream, buf: &mut [u8]) -> io::Result<Option<usize>> {
    let mut hdr = [0u8; 3];
    match r.read_exact(&mut hdr[..1]) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    r.read_exact(&mut hdr[1..])?;
    ks.apply(&mut hdr);
    let n = u16::from_be_bytes([hdr[0], hdr[1]]) as usize;
    let pad = hdr[2] as usize;
    if n == 0 || n > buf.len() || pad > MAX_PAD as usize {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("bad frame (len {n}, pad {pad})")));
    }
    r.read_exact(&mut buf[..n])?;
    ks.apply(&mut buf[..n]);
    let mut padding = [0u8; MAX_PAD as usize];
    r.read_exact(&mut padding[..pad])?;
    ks.apply(&mut padding[..pad]);
    Ok(Some(n))
}

fn is_timeout(e: &io::Error) -> bool {
    matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut)
}

/// A UDP socket reporting that an *earlier* datagram bounced (ICMP port
/// unreachable). Windows surfaces this as `WSAECONNRESET` (10054, "An
/// existing connection was forcibly closed by the remote host") on the next
/// `recv_from`; Linux as `ECONNREFUSED` on a connected socket. It happens
/// whenever the WireGuard app restarts on a new port, or wg0 is briefly down.
/// It says nothing about the TCP connection and must not end the session.
fn is_udp_bounce(e: &io::Error) -> bool {
    matches!(e.kind(), io::ErrorKind::ConnectionReset | io::ErrorKind::ConnectionRefused)
}

/// Exchange nonces: `ours` is sent, the peer's is returned.
fn swap_nonces(tcp: &mut TcpStream, ours: &[u8; NONCE_LEN]) -> io::Result<[u8; NONCE_LEN]> {
    tcp.write_all(ours)?;
    let mut theirs = [0u8; NONCE_LEN];
    tcp.read_exact(&mut theirs)?;
    Ok(theirs)
}

// ---------------------------------------------------------------------------
// Server: TCP in, UDP to the local WireGuard
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Where WireGuard listens — the *only* destination this relay forwards to.
    pub forward_to: SocketAddr,
    /// Concurrent client connections.
    pub max_conns: usize,
    pub idle_timeout: Duration,
}

/// Accept loop. Returns only if the listener fails fatally.
pub fn run_server(listener: TcpListener, cfg: ServerConfig) {
    let active = Arc::new(AtomicUsize::new(0));
    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(e) => {
                eprintln!("relay accept error: {e}");
                continue;
            }
        };
        let peer = stream.peer_addr().map(|a| a.to_string()).unwrap_or_default();
        if active.fetch_add(1, Ordering::AcqRel) >= cfg.max_conns {
            active.fetch_sub(1, Ordering::AcqRel);
            eprintln!("relay REFUSED {peer}: at {} connections", cfg.max_conns);
            continue; // dropping the stream closes it
        }
        let (active, cfg) = (Arc::clone(&active), cfg.clone());
        std::thread::spawn(move || {
            println!("relay OPEN   {peer}");
            if let Err(e) = serve_one(stream, &cfg) {
                eprintln!("relay {peer}: {e}");
            }
            println!("relay CLOSE  {peer}");
            active.fetch_sub(1, Ordering::AcqRel);
        });
    }
}

/// Shuttle one client connection until either side closes or it idles out.
fn serve_one(mut tcp: TcpStream, cfg: &ServerConfig) -> io::Result<()> {
    tcp.set_nodelay(true)?;
    tcp.set_read_timeout(Some(OPEN_TIMEOUT))?;
    let nonce_s: [u8; NONCE_LEN] = rand::random();
    let nonce_c = swap_nonces(&mut tcp, &nonce_s)?;
    let mut ks_up = Keystream::new(nonce_c, DIR_UP);
    let ks_down = Keystream::new(nonce_s, DIR_DOWN);

    let local = if cfg.forward_to.is_ipv4() { "127.0.0.1:0" } else { "[::1]:0" };
    let udp = UdpSocket::bind(local)?;
    udp.connect(cfg.forward_to)?;
    udp.set_read_timeout(Some(POLL))?;
    tcp.set_read_timeout(Some(cfg.idle_timeout))?;

    let done = Arc::new(AtomicBool::new(false));
    let last_rx = Arc::new(Mutex::new(Instant::now()));

    // WireGuard → client
    let down = {
        let (udp, mut tcp_w, done, last_rx) =
            (udp.try_clone()?, tcp.try_clone()?, Arc::clone(&done), Arc::clone(&last_rx));
        let mut ks = ks_down;
        let idle = cfg.idle_timeout;
        std::thread::spawn(move || {
            let mut buf = vec![0u8; MAX_DATAGRAM];
            while !done.load(Ordering::Acquire) {
                match udp.recv(&mut buf) {
                    Ok(n) if n > 0 => {
                        *last_rx.lock().unwrap() = Instant::now();
                        if write_frame(&mut tcp_w, &mut ks, &buf[..n]).is_err() {
                            break;
                        }
                    }
                    Ok(_) => {}
                    Err(e) if is_timeout(&e) => {
                        if last_rx.lock().unwrap().elapsed() > idle {
                            break;
                        }
                    }
                    Err(e) if is_udp_bounce(&e) => {}
                    Err(_) => break,
                }
            }
            done.store(true, Ordering::Release);
            let _ = tcp_w.shutdown(Shutdown::Both);
        })
    };

    // client → WireGuard
    let mut tcp_r = tcp;
    let mut buf = vec![0u8; MAX_DATAGRAM];
    let result = loop {
        if done.load(Ordering::Acquire) {
            break Ok(());
        }
        match read_frame(&mut tcp_r, &mut ks_up, &mut buf) {
            Ok(Some(n)) => {
                *last_rx.lock().unwrap() = Instant::now();
                match udp.send(&buf[..n]) {
                    Ok(_) => {}
                    Err(e) if is_udp_bounce(&e) => {}
                    Err(e) => break Err(e),
                }
            }
            Ok(None) => break Ok(()),
            Err(e) if is_timeout(&e) => break Ok(()), // idle: nothing from the client
            Err(e) => break Err(e),
        }
    };
    done.store(true, Ordering::Release);
    let _ = tcp_r.shutdown(Shutdown::Both);
    let _ = down.join();
    result
}

// ---------------------------------------------------------------------------
// Client: local UDP in, TCP to the relay server
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// Where the local WireGuard sends to (its tunnel `Endpoint`).
    pub listen: SocketAddr,
    /// The relay server.
    pub server: String,
    /// Pause before reconnecting after the TCP connection drops.
    pub reconnect_delay: Duration,
}

/// Run the client forever: one TCP connection at a time, reconnecting when it
/// drops. The UDP socket stays bound across reconnects, so the WireGuard app
/// never notices.
pub fn run_client(cfg: ClientConfig) -> io::Result<()> {
    let udp = UdpSocket::bind(cfg.listen)?;
    println!("relay client: WireGuard Endpoint = {}  ->  tcp {}", cfg.listen, cfg.server);
    // The local WireGuard's source address, learned from its datagrams.
    let wg_src: Arc<Mutex<Option<SocketAddr>>> = Arc::new(Mutex::new(None));
    loop {
        match TcpStream::connect(&cfg.server) {
            Ok(tcp) => {
                println!("relay client: connected to {}", cfg.server);
                let started = Instant::now();
                match client_session(&udp, tcp, &wg_src) {
                    Err(e) => eprintln!(
                        "relay client: connection lost after {:.1}s: {e}",
                        started.elapsed().as_secs_f32()
                    ),
                    Ok(()) => eprintln!("relay client: server closed the connection"),
                }
            }
            Err(e) => eprintln!("relay client: connect {} failed: {e}", cfg.server),
        }
        std::thread::sleep(cfg.reconnect_delay);
    }
}

fn client_session(
    udp: &UdpSocket,
    mut tcp: TcpStream,
    wg_src: &Arc<Mutex<Option<SocketAddr>>>,
) -> io::Result<()> {
    tcp.set_nodelay(true)?;
    tcp.set_read_timeout(Some(OPEN_TIMEOUT))?;
    let nonce_c: [u8; NONCE_LEN] = rand::random();
    let nonce_s = swap_nonces(&mut tcp, &nonce_c)?;
    tcp.set_read_timeout(None)?;
    let mut ks_up = Keystream::new(nonce_c, DIR_UP);
    let ks_down = Keystream::new(nonce_s, DIR_DOWN);
    udp.set_read_timeout(Some(POLL))?;
    let done = Arc::new(AtomicBool::new(false));

    // server → local WireGuard
    let down = {
        let (udp, mut tcp_r, done, wg_src) =
            (udp.try_clone()?, tcp.try_clone()?, Arc::clone(&done), Arc::clone(wg_src));
        let mut ks = ks_down;
        std::thread::spawn(move || -> io::Result<()> {
            let mut buf = vec![0u8; MAX_DATAGRAM];
            let r = loop {
                match read_frame(&mut tcp_r, &mut ks, &mut buf) {
                    Ok(Some(n)) => {
                        if let Some(dst) = *wg_src.lock().unwrap() {
                            let _ = udp.send_to(&buf[..n], dst);
                        }
                    }
                    Ok(None) => break Ok(()),
                    Err(e) => break Err(e),
                }
            };
            done.store(true, Ordering::Release);
            r
        })
    };

    // local WireGuard → server
    let mut tcp_w = tcp;
    let mut buf = vec![0u8; MAX_DATAGRAM];
    let up: io::Result<()> = loop {
        if done.load(Ordering::Acquire) {
            break Ok(());
        }
        match udp.recv_from(&mut buf) {
            Ok((n, src)) if n > 0 => {
                *wg_src.lock().unwrap() = Some(src);
                if let Err(e) = write_frame(&mut tcp_w, &mut ks_up, &buf[..n]) {
                    break Err(e);
                }
            }
            Ok(_) => {}
            Err(e) if is_timeout(&e) || is_udp_bounce(&e) => {}
            Err(e) => break Err(e),
        }
    };
    let _ = tcp_w.shutdown(Shutdown::Both);
    let down_r = down.join().unwrap_or(Ok(()));
    up.and(down_r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn pair() -> (Keystream, Keystream) {
        (Keystream::new([9; NONCE_LEN], DIR_UP), Keystream::new([9; NONCE_LEN], DIR_UP))
    }

    #[test]
    fn frame_round_trip() {
        let (mut w_ks, mut r_ks) = pair();
        let mut wire = Vec::new();
        write_frame(&mut wire, &mut w_ks, b"hello").unwrap();
        write_frame(&mut wire, &mut w_ks, &[7u8; 1500]).unwrap();
        let mut r = Cursor::new(wire);
        let mut buf = vec![0u8; MAX_DATAGRAM];
        assert_eq!(read_frame(&mut r, &mut r_ks, &mut buf).unwrap(), Some(5));
        assert_eq!(&buf[..5], b"hello");
        assert_eq!(read_frame(&mut r, &mut r_ks, &mut buf).unwrap(), Some(1500));
        assert!(buf[..1500].iter().all(|&b| b == 7));
        assert_eq!(read_frame(&mut r, &mut r_ks, &mut buf).unwrap(), None, "clean EOF");
    }

    #[test]
    fn wireguard_structure_is_not_visible_on_the_wire() {
        // A WireGuard handshake initiation starts 01 00 00 00 and is 148 bytes;
        // the plain framing exposed exactly `00 94 01 00 00 00`. After
        // obfuscation neither the header nor the payload appears in clear, and
        // the frame length varies from send to send.
        let mut init = vec![0u8; 148];
        init[0] = 1;
        let mut lens = std::collections::HashSet::new();
        for i in 0..32u8 {
            let mut ks = Keystream::new([i; NONCE_LEN], DIR_UP);
            let mut wire = Vec::new();
            write_frame(&mut wire, &mut ks, &init).unwrap();
            assert!(!wire.windows(4).any(|w| w == [0x01, 0x00, 0x00, 0x00]), "header in clear");
            assert!(!wire.windows(2).any(|w| w == [0x00, 0x94]), "length 148 in clear");
            lens.insert(wire.len());
        }
        assert!(lens.len() > 4, "padding should vary frame sizes, got {lens:?}");
    }

    #[test]
    fn keystreams_differ_by_nonce_and_direction() {
        let take = |mut k: Keystream| {
            let mut v = [0u8; 64];
            k.apply(&mut v);
            v
        };
        let a = take(Keystream::new([1; NONCE_LEN], DIR_UP));
        assert_ne!(a, take(Keystream::new([2; NONCE_LEN], DIR_UP)));
        assert_ne!(a, take(Keystream::new([1; NONCE_LEN], DIR_DOWN)));
        assert_eq!(a, take(Keystream::new([1; NONCE_LEN], DIR_UP)), "deterministic");
    }

    #[test]
    fn rejects_empty_oversized_and_garbage() {
        let (mut ks, _) = pair();
        assert!(write_frame(&mut Vec::new(), &mut ks, b"").is_err());
        assert!(write_frame(&mut Vec::new(), &mut ks, &vec![0u8; MAX_DATAGRAM + 1]).is_err());
        // garbage that decodes to a too-long frame for the buffer
        let mut buf = vec![0u8; 16];
        let mut ks = Keystream::new([0; NONCE_LEN], DIR_UP);
        let mut hdr = [0x00, 0x40, 0x00]; // len 64 > 16
        Keystream::new([0; NONCE_LEN], DIR_UP).apply(&mut hdr);
        assert!(read_frame(&mut Cursor::new(hdr.to_vec()), &mut ks, &mut buf).is_err());
    }

    #[test]
    fn truncated_frame_is_an_error_not_eof() {
        let (mut w_ks, mut r_ks) = pair();
        let mut wire = Vec::new();
        write_frame(&mut wire, &mut w_ks, b"abcdef").unwrap();
        wire.truncate(5);
        let mut buf = vec![0u8; 64];
        assert!(read_frame(&mut Cursor::new(wire), &mut r_ks, &mut buf).is_err());
    }
}
