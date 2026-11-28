//! End-to-end tests for the UDP-over-TCP relay (`handshake_server::relay`):
//! fake WireGuard app ─UDP─► relay client ═TCP═► relay server ─UDP─► fake wg0
//! (an echo socket), and back.

use std::net::{SocketAddr, TcpListener, UdpSocket};
use std::time::Duration;

use handshake_server::relay::{self, ClientConfig, ServerConfig};

/// A UDP echo socket standing in for wg0. Returns its address.
fn fake_wg0() -> SocketAddr {
    let s = UdpSocket::bind("127.0.0.1:0").unwrap();
    let addr = s.local_addr().unwrap();
    std::thread::spawn(move || {
        let mut buf = vec![0u8; 65_535];
        loop {
            let (n, src) = s.recv_from(&mut buf).unwrap();
            // reply with a recognisable transform so we know it went the whole way
            let reply: Vec<u8> = buf[..n].iter().map(|b| b.wrapping_add(1)).collect();
            s.send_to(&reply, src).unwrap();
        }
    });
    addr
}

fn free_udp_port() -> SocketAddr {
    UdpSocket::bind("127.0.0.1:0").unwrap().local_addr().unwrap()
}

/// Start a relay server forwarding to `wg0` and a relay client pointing at
/// it. Returns the client's UDP listen address (what WireGuard's Endpoint
/// would be).
fn chain(wg0: SocketAddr) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let server_addr = listener.local_addr().unwrap();
    let cfg = ServerConfig { forward_to: wg0, max_conns: 8, idle_timeout: Duration::from_secs(30) };
    std::thread::spawn(move || relay::run_server(listener, cfg));

    let listen = free_udp_port();
    let ccfg = ClientConfig {
        listen,
        server: server_addr.to_string(),
        reconnect_delay: Duration::from_millis(100),
    };
    std::thread::spawn(move || relay::run_client(ccfg).unwrap());
    listen
}

/// The "WireGuard app": send `msg` to the relay client, wait for the reply.
/// Retries briefly while the client binds and connects.
fn round_trip(app: &UdpSocket, endpoint: SocketAddr, msg: &[u8]) -> Vec<u8> {
    let mut buf = vec![0u8; 65_535];
    for _ in 0..50 {
        let _ = app.send_to(msg, endpoint);
        match app.recv_from(&mut buf) {
            Ok((n, _)) => return buf[..n].to_vec(),
            // A timeout waits 200 ms by itself, but if the relay client has
            // not bound yet, Windows reports the bounce (10054) instantly —
            // pause so 50 retries actually span a few seconds.
            Err(_) => std::thread::sleep(Duration::from_millis(100)),
        }
    }
    panic!("no reply through the relay");
}

fn app_socket() -> UdpSocket {
    let s = UdpSocket::bind("127.0.0.1:0").unwrap();
    s.set_read_timeout(Some(Duration::from_millis(200))).unwrap();
    s
}

#[test]
fn datagrams_cross_the_relay_both_ways() {
    let endpoint = chain(fake_wg0());
    let app = app_socket();

    // WireGuard's real message sizes: handshake init 148, response 92,
    // keepalive 32, a full-MTU data packet ~1420.
    for size in [148usize, 92, 32, 1420, 1] {
        let msg: Vec<u8> = (0..size).map(|i| i as u8).collect();
        let reply = round_trip(&app, endpoint, &msg);
        let expect: Vec<u8> = msg.iter().map(|b| b.wrapping_add(1)).collect();
        assert_eq!(reply, expect, "size {size}");
    }
}

#[test]
fn datagram_boundaries_are_preserved_under_a_burst() {
    // TCP is a byte stream; the framing must hand WireGuard back exactly the
    // datagrams it sent, never merged or split.
    let endpoint = chain(fake_wg0());
    let app = app_socket();
    round_trip(&app, endpoint, b"warm-up"); // make sure the chain is connected

    let sizes = [148usize, 92, 1420, 32, 600, 1, 1420, 148];
    for (i, &size) in sizes.iter().enumerate() {
        app.send_to(&vec![i as u8; size], endpoint).unwrap();
    }
    let mut got = Vec::new();
    let mut buf = vec![0u8; 65_535];
    while got.len() < sizes.len() {
        let (n, _) = app.recv_from(&mut buf).expect("reply for every datagram");
        got.push(buf[..n].to_vec());
    }
    got.sort_by_key(|d| d[0]); // UDP gives no ordering guarantee on the local leg
    for (i, d) in got.iter().enumerate() {
        assert_eq!(d.len(), sizes[i], "datagram {i} length");
        assert!(d.iter().all(|&b| b == (i as u8).wrapping_add(1)), "datagram {i} content");
    }
}

#[test]
fn relay_survives_wireguard_restarting() {
    // Toggling the tunnel in the WireGuard app gives it a new local port. The
    // relay client then sends a reply to the old, now-closed port; on Windows
    // the resulting ICMP "port unreachable" surfaces as WSAECONNRESET (10054)
    // on the client's *UDP* socket. That must not be mistaken for the TCP
    // connection dying (the first live test against the droplet did exactly
    // that, reconnecting in a loop).
    let endpoint = chain(fake_wg0());

    let old = app_socket();
    assert_eq!(round_trip(&old, endpoint, b"before"), b"cfgpsf");
    // The app goes away, and a reply is still in flight to its old port.
    old.send_to(b"orphan", endpoint).unwrap();
    drop(old);
    std::thread::sleep(Duration::from_millis(300));

    // The "restarted" app on a new port is served without delay.
    let new = app_socket();
    for _ in 0..5 {
        assert_eq!(round_trip(&new, endpoint, b"after"), b"bgufs");
    }
}

#[test]
fn two_clients_are_kept_apart() {
    // One relay server, two independent relay clients (two laptops): each
    // gets its own replies only.
    let wg0 = fake_wg0();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let server_addr = listener.local_addr().unwrap();
    let cfg = ServerConfig { forward_to: wg0, max_conns: 8, idle_timeout: Duration::from_secs(30) };
    std::thread::spawn(move || relay::run_server(listener, cfg));

    let mut endpoints = Vec::new();
    for _ in 0..2 {
        let listen = free_udp_port();
        let ccfg = ClientConfig {
            listen,
            server: server_addr.to_string(),
            reconnect_delay: Duration::from_millis(100),
        };
        std::thread::spawn(move || relay::run_client(ccfg).unwrap());
        endpoints.push(listen);
    }
    let (a, b) = (app_socket(), app_socket());
    assert_eq!(round_trip(&a, endpoints[0], b"AAAA"), b"BBBB");
    assert_eq!(round_trip(&b, endpoints[1], b"xyz"), b"yz{");
    assert_eq!(round_trip(&a, endpoints[0], b"A"), b"B");
}
