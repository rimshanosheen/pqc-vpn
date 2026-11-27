//! Desktop's real `vpn_core::state::TrafficSignals` (Member 1, Week 8): the
//! live inputs the anomaly layers need, read from this machine.
//!
//! | Field | Windows | Linux |
//! |---|---|---|
//! | packet rate / size, up/down bytes | `sysinfo` network counters | same |
//! | new connections, distinct ports, failures | `GetExtendedTcpTable` + `GetTcp6Table` | `/proc/net/tcp{,6}` |
//! | retransmit rate | `GetTcpStatisticsEx` (v4 + v6) | `/proc/net/snmp` |
//! | DNS server | `GetNetworkParams` | `/etc/resolv.conf` |
//!
//! Other platforms (macOS is out of scope per the delivery plan) get the
//! `sysinfo` numbers and nothing else: zero connections, zero retransmits,
//! no DNS server. A missing reading is always "nothing seen", never a
//! made-up value.
//!
//! **Known limits, stated plainly:**
//! - Connections are seen by snapshotting the TCP table once a tick. A
//!   connection that opens and closes between two snapshots is invisible,
//!   which includes a scan probe to a *closed* port (refused within one
//!   round trip). Probes to filtered ports sit in `SYN_SENT` and are seen.
//!   Catching every probe needs OS event tracing or packet capture.
//! - No packet capture, so [`TrafficSample::packets`] stays empty and
//!   Layer 3 reports not-ready (it contributes nothing rather than a guess).
//!   Capture needs admin rights or Npcap/libpcap; that's a separate decision.

use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::Mutex;
use std::time::Instant;

use sysinfo::Networks;
use vpn_core::state::{TrafficSample, TrafficSignals};

/// TCP connection state, reduced to what the port-scan check needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TcpState {
    Listen,
    SynSent,
    Established,
    Other,
}

/// One row of the OS TCP table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TcpConn {
    pub local: SocketAddr,
    pub remote: SocketAddr,
    pub state: TcpState,
}

/// What one tick's connection-table diff found.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ConnStats {
    pub attempts: u32,
    pub failed: u32,
    pub distinct_ports: u32,
}

/// Diffs consecutive TCP-table snapshots into per-tick connection counts.
#[derive(Debug, Default)]
pub struct ConnTracker {
    /// Connections in the previous snapshot, keyed by (local, remote).
    previous: HashMap<(SocketAddr, SocketAddr), TcpState>,
    primed: bool,
}

impl ConnTracker {
    /// Feeds one snapshot; returns the counts since the previous one.
    ///
    /// - An *attempt* is an outbound connection that appeared since the last
    ///   snapshot. Outbound means its local port is not one of our
    ///   listening ports (those are inbound connections to a server here).
    /// - A *failure* is an attempt still in `SYN_SENT`, plus a connection
    ///   that was `SYN_SENT` last tick and vanished without ever being seen
    ///   established. Capped at `attempts` so the failure ratio stays a
    ///   ratio. A new connection already closing (`TIME_WAIT` etc.) did
    ///   connect, so it is an attempt but not a failure.
    /// - *Distinct ports* counts remote ports among this tick's attempts.
    ///
    /// The first snapshot only primes the tracker: everything already open
    /// at startup would otherwise look like a burst of new connections.
    pub fn update(&mut self, snapshot: &[TcpConn]) -> ConnStats {
        let listening: HashSet<u16> =
            snapshot.iter().filter(|c| c.state == TcpState::Listen).map(|c| c.local.port()).collect();

        let current: HashMap<(SocketAddr, SocketAddr), TcpState> = snapshot
            .iter()
            .filter(|c| c.state != TcpState::Listen && !c.remote.ip().is_unspecified())
            .map(|c| ((c.local, c.remote), c.state))
            .collect();

        let mut stats = ConnStats::default();
        if self.primed {
            let mut ports = HashSet::new();
            for (&(local, remote), &state) in &current {
                let is_new = !self.previous.contains_key(&(local, remote));
                if !is_new || listening.contains(&local.port()) {
                    continue;
                }
                stats.attempts += 1;
                ports.insert(remote.port());
                // Only SYN_SENT means "not connected (yet)". A connection
                // already in FIN_WAIT / TIME_WAIT / CLOSE_WAIT connected and
                // finished: a short, ordinary request, not a failure.
                if state == TcpState::SynSent {
                    stats.failed += 1;
                }
            }
            let vanished_pending = self
                .previous
                .iter()
                .filter(|(key, state)| **state == TcpState::SynSent && !current.contains_key(key))
                .count() as u32;
            stats.failed = (stats.failed + vanished_pending).min(stats.attempts);
            stats.distinct_ports = ports.len() as u32;
        }

        self.previous = current;
        self.primed = true;
        stats
    }
}

/// Retransmitted / sent segments between two counter readings, in `[0, 1]`.
/// `None` on either side, a counter reset, or nothing sent gives 0.
pub fn retransmit_rate(prev: Option<(u64, u64)>, now: Option<(u64, u64)>) -> f64 {
    let (Some((out0, re0)), Some((out1, re1))) = (prev, now) else { return 0.0 };
    if out1 < out0 || re1 < re0 {
        return 0.0; // counters wrapped or reset
    }
    let sent = out1 - out0;
    if sent == 0 {
        return 0.0;
    }
    ((re1 - re0) as f64 / sent as f64).clamp(0.0, 1.0)
}

/// Desktop's [`TrafficSignals`].
pub struct SystemTrafficSignals {
    inner: Mutex<Inner>,
}

struct Inner {
    networks: Networks,
    last: Instant,
    conns: ConnTracker,
    tcp_counters: Option<(u64, u64)>,
}

impl SystemTrafficSignals {
    /// Takes a baseline reading so the first `sample` covers a real
    /// interval.
    pub fn new() -> Self {
        let mut conns = ConnTracker::default();
        conns.update(&os::tcp_connections());
        Self {
            inner: Mutex::new(Inner {
                networks: Networks::new_with_refreshed_list(),
                last: Instant::now(),
                conns,
                tcp_counters: os::tcp_segment_counters(),
            }),
        }
    }
}

impl Default for SystemTrafficSignals {
    fn default() -> Self {
        Self::new()
    }
}

impl TrafficSignals for SystemTrafficSignals {
    fn sample(&self) -> TrafficSample {
        let mut inner = self.inner.lock().expect("traffic mutex poisoned");

        inner.networks.refresh(true);
        let elapsed = inner.last.elapsed().as_secs_f64().max(1e-6);
        inner.last = Instant::now();

        let (mut bytes_up, mut bytes_down, mut pkts) = (0u64, 0u64, 0u64);
        for (name, data) in inner.networks.iter() {
            if is_loopback(name) {
                continue;
            }
            bytes_up += data.transmitted();
            bytes_down += data.received();
            pkts += data.packets_transmitted() + data.packets_received();
        }

        let conn = inner.conns.update(&os::tcp_connections());
        let counters = os::tcp_segment_counters();
        let retransmit = retransmit_rate(inner.tcp_counters, counters);
        inner.tcp_counters = counters;

        TrafficSample {
            packet_rate: pkts as f64 / elapsed,
            mean_packet_size: if pkts > 0 { (bytes_up + bytes_down) as f64 / pkts as f64 } else { 0.0 },
            upload_bytes_per_sec: bytes_up as f64 / elapsed,
            download_bytes_per_sec: bytes_down as f64 / elapsed,
            distinct_dest_ports: conn.distinct_ports,
            connection_attempts: conn.attempts,
            failed_attempts: conn.failed,
            retransmit_rate: retransmit,
            dns_server: os::dns_server(),
            packets: Vec::new(),
        }
    }
}

fn is_loopback(name: &str) -> bool {
    let n = name.to_lowercase();
    n == "lo" || n.contains("loopback")
}

// ---------------------------------------------------------------- parsers
// Pure functions over the Linux /proc text formats. Compiled everywhere so
// they're tested on every platform, not only on Linux CI.

/// Parses `/proc/net/tcp` or `/proc/net/tcp6`.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub fn parse_proc_net_tcp(text: &str) -> Vec<TcpConn> {
    text.lines()
        .skip(1)
        .filter_map(|line| {
            let mut cols = line.split_whitespace();
            let _slot = cols.next()?;
            let local = parse_proc_addr(cols.next()?)?;
            let remote = parse_proc_addr(cols.next()?)?;
            let state = match u8::from_str_radix(cols.next()?, 16).ok()? {
                0x01 => TcpState::Established,
                0x02 => TcpState::SynSent,
                0x0A => TcpState::Listen,
                _ => TcpState::Other,
            };
            Some(TcpConn { local, remote, state })
        })
        .collect()
}

/// `0100007F:1F90` (IPv4) or a 32-hex-digit IPv6 address, each 32-bit word
/// in host (little-endian on x86/ARM) order, then a big-endian hex port.
fn parse_proc_addr(field: &str) -> Option<SocketAddr> {
    let (addr, port) = field.split_once(':')?;
    let port = u16::from_str_radix(port, 16).ok()?;
    match addr.len() {
        8 => {
            let word = u32::from_str_radix(addr, 16).ok()?;
            Some(SocketAddr::from((word.to_le_bytes(), port)))
        }
        32 => {
            let mut bytes = [0u8; 16];
            for i in 0..4 {
                let word = u32::from_str_radix(&addr[i * 8..i * 8 + 8], 16).ok()?;
                bytes[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
            }
            Some(SocketAddr::from((bytes, port)))
        }
        _ => None,
    }
}

/// `(OutSegs, RetransSegs)` from `/proc/net/snmp`'s two `Tcp:` lines.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub fn parse_proc_net_snmp(text: &str) -> Option<(u64, u64)> {
    let mut tcp = text.lines().filter(|l| l.starts_with("Tcp:"));
    let names: Vec<&str> = tcp.next()?.split_whitespace().collect();
    let values: Vec<&str> = tcp.next()?.split_whitespace().collect();
    let get = |key: &str| -> Option<u64> {
        let i = names.iter().position(|n| *n == key)?;
        values.get(i)?.parse().ok()
    };
    Some((get("OutSegs")?, get("RetransSegs")?))
}

/// First `nameserver` in `/etc/resolv.conf`.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub fn parse_resolv_conf(text: &str) -> Option<String> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.starts_with('#'))
        .find_map(|l| l.strip_prefix("nameserver").map(|rest| rest.trim().to_string()))
        .filter(|s| !s.is_empty())
}

// ---------------------------------------------------------------- per-OS

#[cfg(target_os = "linux")]
mod os {
    use super::*;

    pub fn tcp_connections() -> Vec<TcpConn> {
        ["/proc/net/tcp", "/proc/net/tcp6"]
            .iter()
            .filter_map(|p| std::fs::read_to_string(p).ok())
            .flat_map(|t| parse_proc_net_tcp(&t))
            .collect()
    }

    pub fn tcp_segment_counters() -> Option<(u64, u64)> {
        parse_proc_net_snmp(&std::fs::read_to_string("/proc/net/snmp").ok()?)
    }

    pub fn dns_server() -> Option<String> {
        parse_resolv_conf(&std::fs::read_to_string("/etc/resolv.conf").ok()?)
    }
}

#[cfg(windows)]
mod os {
    use std::ffi::CStr;
    use std::net::{Ipv4Addr, Ipv6Addr};

    use windows_sys::Win32::Foundation::{ERROR_BUFFER_OVERFLOW, ERROR_INSUFFICIENT_BUFFER, NO_ERROR};
    use windows_sys::Win32::NetworkManagement::IpHelper::{
        GetExtendedTcpTable, GetNetworkParams, GetTcp6Table, GetTcpStatisticsEx, FIXED_INFO_W2KSP1,
        MIB_TCP6TABLE, MIB_TCPSTATS_LH, MIB_TCPTABLE, MIB_TCP_STATE_ESTAB, MIB_TCP_STATE_LISTEN,
        MIB_TCP_STATE_SYN_SENT, TCP_TABLE_BASIC_ALL,
    };
    use windows_sys::Win32::Networking::WinSock::{AF_INET, AF_INET6};

    use super::*;

    fn state(raw: i32) -> TcpState {
        match raw {
            MIB_TCP_STATE_LISTEN => TcpState::Listen,
            MIB_TCP_STATE_SYN_SENT => TcpState::SynSent,
            MIB_TCP_STATE_ESTAB => TcpState::Established,
            _ => TcpState::Other,
        }
    }

    /// Ports are stored in network byte order in the low 16 bits.
    fn port(raw: u32) -> u16 {
        u16::from_be(raw as u16)
    }

    /// Calls a size-probing IP Helper function: first to learn the buffer
    /// size, then for real. A u64-aligned buffer, since the tables are read
    /// as C structs. Retries once if the table grew in between.
    fn fill(mut call: impl FnMut(*mut u8, &mut u32) -> u32) -> Option<Vec<u64>> {
        let mut size = 0u32;
        for _ in 0..3 {
            let mut buf = vec![0u64; (size as usize).div_ceil(8).max(1)];
            let rc = call(buf.as_mut_ptr().cast(), &mut size);
            if rc == NO_ERROR {
                return Some(buf);
            }
            if rc != ERROR_INSUFFICIENT_BUFFER && rc != ERROR_BUFFER_OVERFLOW {
                return None;
            }
        }
        None
    }

    pub fn tcp_connections() -> Vec<TcpConn> {
        let mut out = Vec::new();

        if let Some(buf) = fill(|p, size| unsafe {
            GetExtendedTcpTable(p.cast(), size, 0, AF_INET as u32, TCP_TABLE_BASIC_ALL, 0)
        }) {
            // SAFETY: the call succeeded, so `buf` holds a MIB_TCPTABLE with
            // `dwNumEntries` rows laid out after the count.
            unsafe {
                let table = buf.as_ptr() as *const MIB_TCPTABLE;
                let rows = std::slice::from_raw_parts((*table).table.as_ptr(), (*table).dwNumEntries as usize);
                for r in rows {
                    out.push(TcpConn {
                        local: SocketAddr::from((Ipv4Addr::from(u32::from_be(r.dwLocalAddr)), port(r.dwLocalPort))),
                        remote: SocketAddr::from((Ipv4Addr::from(u32::from_be(r.dwRemoteAddr)), port(r.dwRemotePort))),
                        state: state(r.Anonymous.State),
                    });
                }
            }
        }

        if let Some(buf) = fill(|p, size| unsafe { GetTcp6Table(p.cast(), size, 0) }) {
            // SAFETY: as above, for MIB_TCP6TABLE.
            unsafe {
                let table = buf.as_ptr() as *const MIB_TCP6TABLE;
                let rows = std::slice::from_raw_parts((*table).table.as_ptr(), (*table).dwNumEntries as usize);
                for r in rows {
                    out.push(TcpConn {
                        local: SocketAddr::from((Ipv6Addr::from(r.LocalAddr.u.Byte), port(r.dwLocalPort))),
                        remote: SocketAddr::from((Ipv6Addr::from(r.RemoteAddr.u.Byte), port(r.dwRemotePort))),
                        state: state(r.State),
                    });
                }
            }
        }

        out
    }

    pub fn tcp_segment_counters() -> Option<(u64, u64)> {
        let mut total: Option<(u64, u64)> = None;
        for family in [AF_INET, AF_INET6] {
            // SAFETY: a plain out-parameter struct, zero-initialized.
            let mut stats: MIB_TCPSTATS_LH = unsafe { std::mem::zeroed() };
            if unsafe { GetTcpStatisticsEx(&mut stats, family as u32) } == NO_ERROR {
                let (o, r) = total.unwrap_or((0, 0));
                total = Some((o + stats.dwOutSegs as u64, r + stats.dwRetransSegs as u64));
            }
        }
        total
    }

    pub fn dns_server() -> Option<String> {
        let buf = fill(|p, size| unsafe { GetNetworkParams(p.cast(), size) })?;
        // SAFETY: the call succeeded, so `buf` starts with a FIXED_INFO
        // whose DnsServerList is an inline, NUL-terminated ASCII string.
        let ip = unsafe {
            let info = buf.as_ptr() as *const FIXED_INFO_W2KSP1;
            CStr::from_ptr((*info).DnsServerList.IpAddress.String.as_ptr()).to_string_lossy().into_owned()
        };
        Some(ip).filter(|s| !s.is_empty())
    }
}

#[cfg(not(any(target_os = "linux", windows)))]
mod os {
    use super::TcpConn;

    pub fn tcp_connections() -> Vec<TcpConn> {
        Vec::new()
    }
    pub fn tcp_segment_counters() -> Option<(u64, u64)> {
        None
    }
    pub fn dns_server() -> Option<String> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conn(local: &str, remote: &str, state: TcpState) -> TcpConn {
        TcpConn { local: local.parse().unwrap(), remote: remote.parse().unwrap(), state }
    }

    #[test]
    fn first_snapshot_only_primes() {
        let mut t = ConnTracker::default();
        let snap = vec![conn("10.0.0.2:50000", "1.1.1.1:443", TcpState::Established)];
        assert_eq!(t.update(&snap), ConnStats::default());
        assert_eq!(t.update(&snap), ConnStats::default(), "nothing new on the second tick");
    }

    #[test]
    fn a_scan_shows_many_ports_mostly_pending() {
        let mut t = ConnTracker::default();
        t.update(&[]);
        let mut snap: Vec<TcpConn> = (0..20)
            .map(|i| conn(&format!("10.0.0.2:{}", 50000 + i), &format!("10.0.0.9:{}", 20 + i), TcpState::SynSent))
            .collect();
        snap.push(conn("10.0.0.2:49999", "10.0.0.9:80", TcpState::Established));
        let s = t.update(&snap);
        assert_eq!(s, ConnStats { attempts: 21, failed: 20, distinct_ports: 21 });
    }

    /// Found running the slice on a real laptop: short HTTPS requests that
    /// finish between two snapshots show up already in TIME_WAIT, and were
    /// being counted as failed, tripping the port-scan check on normal use.
    #[test]
    fn short_completed_connections_are_not_failures() {
        let mut t = ConnTracker::default();
        t.update(&[]);
        let snap: Vec<TcpConn> = (0..20)
            .map(|i| conn(&format!("10.0.0.2:{}", 50000 + i), &format!("10.0.0.{}:443", 10 + i), TcpState::Other))
            .collect();
        let s = t.update(&snap);
        assert_eq!(s.attempts, 20);
        assert_eq!(s.failed, 0);
    }

    #[test]
    fn inbound_and_listening_sockets_are_not_attempts() {
        let mut t = ConnTracker::default();
        t.update(&[]);
        let s = t.update(&[
            conn("0.0.0.0:22", "0.0.0.0:0", TcpState::Listen),
            conn("10.0.0.2:22", "10.0.0.7:51000", TcpState::Established),
        ]);
        assert_eq!(s.attempts, 0);
    }

    #[test]
    fn a_pending_connection_that_vanishes_counts_as_failed() {
        let mut t = ConnTracker::default();
        t.update(&[]);
        t.update(&[conn("10.0.0.2:50000", "10.0.0.9:25", TcpState::SynSent)]);
        let s = t.update(&[conn("10.0.0.2:50001", "10.0.0.9:443", TcpState::Established)]);
        assert_eq!(s, ConnStats { attempts: 1, failed: 1, distinct_ports: 1 });
    }

    #[test]
    fn retransmit_rate_handles_resets_and_gaps() {
        assert_eq!(retransmit_rate(Some((100, 1)), Some((200, 11))), 0.1);
        assert_eq!(retransmit_rate(None, Some((200, 11))), 0.0);
        assert_eq!(retransmit_rate(Some((100, 1)), Some((100, 1))), 0.0, "nothing sent");
        assert_eq!(retransmit_rate(Some((100, 50)), Some((10, 1))), 0.0, "counter reset");
    }

    #[test]
    fn parses_proc_net_tcp_v4_and_v6() {
        let v4 = "  sl  local_address rem_address   st tx_queue rx_queue\n\
                  0: 0100007F:1F90 00000000:0000 0A 00000000:00000000\n\
                  1: 0200000A:C350 0101A8C0:01BB 01 00000000:00000000\n\
                  2: 0200000A:C351 0101A8C0:0019 02 00000000:00000000\n";
        let c = parse_proc_net_tcp(v4);
        assert_eq!(c[0], conn("127.0.0.1:8080", "0.0.0.0:0", TcpState::Listen));
        assert_eq!(c[1], conn("10.0.0.2:50000", "192.168.1.1:443", TcpState::Established));
        assert_eq!(c[2].state, TcpState::SynSent);

        let v6 = "header\n 0: 00000000000000000000000001000000:0016 00000000000000000000000000000000:0000 0A x\n";
        assert_eq!(parse_proc_net_tcp(v6)[0].local, "[::1]:22".parse().unwrap());
    }

    #[test]
    fn parses_snmp_and_resolv_conf() {
        let snmp = "Ip: Forwarding\nIp: 1\nTcp: RtoAlgorithm OutSegs RetransSegs InErrs\nTcp: 1 5000 25 0\n";
        assert_eq!(parse_proc_net_snmp(snmp), Some((5000, 25)));
        assert_eq!(parse_resolv_conf("# c\nsearch lan\nnameserver 127.0.0.53\nnameserver 8.8.8.8\n"), Some("127.0.0.53".into()));
        assert_eq!(parse_resolv_conf("search lan\n"), None);
    }

    /// Reads this machine for real. Asserts only what must hold on any
    /// machine: the numbers are finite and the ratio is a ratio.
    #[test]
    fn live_sample_is_well_formed() {
        let t = SystemTrafficSignals::new();
        std::thread::sleep(std::time::Duration::from_millis(300));
        let s = t.sample();
        assert!(s.packet_rate.is_finite() && s.packet_rate >= 0.0);
        assert!((0.0..=1.0).contains(&s.retransmit_rate));
        assert!(s.failed_attempts <= s.connection_attempts);
        eprintln!("live traffic sample: {s:?}");
    }
}
