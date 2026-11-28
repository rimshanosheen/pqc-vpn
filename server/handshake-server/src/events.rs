//! Session event log — Week 9.
//!
//! One JSON object per line (JSON Lines) for every session-level event the
//! server sees: handshakes, rekeys, rejections, guard refusals, the tunnel
//! coming up and going down, and evictions. It feeds Member 1's dashboard and
//! Member 3's baseline evaluation, which needs session-level data
//! (`TEAM_TIMELINE_PROPOSAL.md`, Member 2 Week 9). Schema and field meanings:
//! `server/SESSION_LOG.md`.
//!
//! Every record has `v` (schema version, currently 1), `ts_ms` (Unix
//! milliseconds) and `event`, plus the event's own fields. Keys and session
//! ids are logged; **PSKs and any other secret never are**.
//!
//! **"Disconnect"** — protocol v1 has no disconnect message, so the tunnel's
//! state is inferred from WireGuard itself (see [`Liveness`]): `tunnel_down`
//! fires once a peer has gone [`TUNNEL_DOWN_AFTER`] without a WireGuard
//! handshake — WireGuard's own `REJECT_AFTER_TIME`, after which its session
//! keys are dead. With `PersistentKeepalive` set (every client config does),
//! a live tunnel re-handshakes every ~2 minutes, so this means the client is
//! gone.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use serde::Serialize;
use serde_json::Value;

use crate::registry::Registry;
use crate::tunnel::PskInstaller;

/// Bumped on any incompatible change to a record's fields.
pub const SCHEMA_VERSION: u32 = 1;

/// No WireGuard handshake for this long ⇒ the tunnel is down. WireGuard's
/// `REJECT_AFTER_TIME` (180 s): past it, the peer's session keys are rejected.
pub const TUNNEL_DOWN_AFTER: Duration = Duration::from_secs(180);

/// How often the liveness monitor polls `wg show`.
pub const LIVENESS_INTERVAL: Duration = Duration::from_secs(15);

/// Default size at which the log rotates to `<path>.1`.
pub const DEFAULT_MAX_BYTES: u64 = 20 * 1024 * 1024;

/// One session-level event. Serialised with `"event": "<snake_case name>"`.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    /// The server started; records the configuration the rest of the log ran under.
    ServerStart {
        version: String,
        psk_install: String,
        max_peers: usize,
        idle_timeout_s: u64,
        replay_window_s: u64,
        rate_burst: u32,
        rate_per_min: u32,
        max_conns: usize,
    },
    /// A full handshake (`ClientHello`) completed: a client connected, or
    /// reconnected with a new session.
    Handshake {
        session_id: String,
        peer: String,
        assigned_ip: String,
        algo: String,
        src: String,
        /// `false` when this WireGuard key was already registered (a reconnect).
        new_peer: bool,
        /// Server side: message 1 received → `ServerFinish` sent. Includes one
        /// client round trip (ServerHello → ClientFinish).
        handshake_ms: f64,
    },
    /// A `RekeyRequest` completed: fresh PSK on the same session, tunnel kept.
    Rekey {
        session_id: String,
        peer: String,
        assigned_ip: String,
        from_algo: String,
        algo: String,
        escalated: bool,
        /// Rekeys on this session so far, including this one.
        rekeys: u32,
        src: String,
        handshake_ms: f64,
    },
    /// The server answered with a protocol `Error` (`PROTOCOL.md` §4.7).
    Rejected {
        src: String,
        /// `hello` | `finish` | `registry` | `install`
        stage: String,
        code: u8,
        reason: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        peer: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        algo: Option<String>,
        /// `true` for a `RekeyRequest`.
        rekey: bool,
    },
    /// The Week 7 guard turned a connection away before any PQC work.
    Refused { src: String, code: u8, reason: String },
    /// The client went away mid-handshake (socket closed, timed out, garbled frame).
    Aborted { src: String, error: String },
    /// WireGuard started handshaking with a registered peer: the tunnel is
    /// carrying traffic.
    TunnelUp { peer: String, session_id: String, assigned_ip: String, algo: String },
    /// No WireGuard handshake for [`TUNNEL_DOWN_AFTER`]: the client is gone.
    TunnelDown {
        peer: String,
        session_id: String,
        assigned_ip: String,
        /// Since the last WireGuard handshake.
        idle_s: u64,
        /// From `tunnel_up` to the last WireGuard handshake.
        up_s: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        rx_bytes: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tx_bytes: Option<u64>,
    },
    /// A peer was removed from the registry and/or `wg0`.
    Evicted {
        peer: String,
        /// `idle` (the `--idle-timeout-secs` sweep) | `unknown_on_startup`
        /// (on `wg0` but not in the registry).
        reason: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        session_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        assigned_ip: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        session_age_s: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        rekeys: Option<u32>,
    },
}

impl Event {
    /// The `event` tag this variant serialises to.
    pub fn name(&self) -> &'static str {
        match self {
            Event::ServerStart { .. } => "server_start",
            Event::Handshake { .. } => "handshake",
            Event::Rekey { .. } => "rekey",
            Event::Rejected { .. } => "rejected",
            Event::Refused { .. } => "refused",
            Event::Aborted { .. } => "aborted",
            Event::TunnelUp { .. } => "tunnel_up",
            Event::TunnelDown { .. } => "tunnel_down",
            Event::Evicted { .. } => "evicted",
        }
    }
}

pub fn key_b64(k: &[u8; 32]) -> String {
    B64.encode(k)
}

fn unix_ms(t: SystemTime) -> u64 {
    t.duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

/// `{"v":1,"ts_ms":…,"event":…, …fields}` for `ev` at time `at`.
pub fn record(ev: &Event, at: SystemTime) -> Value {
    let mut obj = serde_json::Map::new();
    obj.insert("v".into(), SCHEMA_VERSION.into());
    obj.insert("ts_ms".into(), unix_ms(at).into());
    if let Value::Object(fields) = serde_json::to_value(ev).expect("events always serialise") {
        obj.extend(fields);
    }
    Value::Object(obj)
}

enum Sink {
    Null,
    Memory(Vec<Value>),
    File { path: PathBuf, file: File, written: u64, max_bytes: u64 },
}

/// Where events go. Cheap to call from any thread.
pub struct EventLog {
    sink: Mutex<Sink>,
}

impl EventLog {
    /// Discard everything.
    pub fn null() -> Self {
        Self { sink: Mutex::new(Sink::Null) }
    }

    /// Keep records in memory (tests; read them back with [`EventLog::records`]).
    pub fn in_memory() -> Self {
        Self { sink: Mutex::new(Sink::Memory(Vec::new())) }
    }

    /// Append to `path` (created if missing). When it passes `max_bytes` it is
    /// renamed to `<path>.1` (replacing an older one) and a fresh file started.
    pub fn to_file(path: impl Into<PathBuf>, max_bytes: u64) -> io::Result<Self> {
        let path = path.into();
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        let written = file.metadata()?.len();
        Ok(Self { sink: Mutex::new(Sink::File { path, file, written, max_bytes }) })
    }

    pub fn emit(&self, ev: Event) {
        self.emit_at(ev, SystemTime::now());
    }

    pub fn emit_at(&self, ev: Event, at: SystemTime) {
        let rec = record(&ev, at);
        let mut sink = self.sink.lock().unwrap();
        match &mut *sink {
            Sink::Null => {}
            Sink::Memory(v) => v.push(rec),
            Sink::File { path, file, written, max_bytes } => {
                let mut line = rec.to_string();
                line.push('\n');
                if let Err(e) = file.write_all(line.as_bytes()).and_then(|_| file.flush()) {
                    eprintln!("session log: write to {} failed: {e}", path.display());
                    return;
                }
                *written += line.len() as u64;
                if *written >= *max_bytes {
                    match rotate(path) {
                        Ok(f) => {
                            *file = f;
                            *written = 0;
                        }
                        Err(e) => eprintln!("session log: rotate {} failed: {e}", path.display()),
                    }
                }
            }
        }
    }

    /// Records captured by an [`EventLog::in_memory`] log (empty otherwise).
    pub fn records(&self) -> Vec<Value> {
        match &*self.sink.lock().unwrap() {
            Sink::Memory(v) => v.clone(),
            _ => Vec::new(),
        }
    }

    /// The events of one type captured in memory.
    pub fn records_of(&self, event: &str) -> Vec<Value> {
        self.records().into_iter().filter(|r| r["event"] == event).collect()
    }
}

fn rotate(path: &Path) -> io::Result<File> {
    let mut old = path.as_os_str().to_owned();
    old.push(".1");
    std::fs::rename(path, &old)?;
    OpenOptions::new().create(true).append(true).open(path)
}

// ---------------------------------------------------------------------------
// Tunnel liveness (the "disconnect" signal)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct LiveState {
    up_since: SystemTime,
}

/// Tracks which registered peers currently have a live WireGuard session and
/// emits `tunnel_up` / `tunnel_down` on transitions.
#[derive(Default)]
pub struct Liveness {
    up: Mutex<HashMap<[u8; 32], LiveState>>,
}

impl Liveness {
    pub fn new() -> Self {
        Self::default()
    }

    /// One poll: compare each registered peer's last WireGuard handshake with
    /// `now`. Returns the number of transitions logged.
    pub fn tick(
        &self,
        registry: &Registry,
        installer: &dyn PskInstaller,
        log: &EventLog,
        now: SystemTime,
    ) -> usize {
        let Ok(handshakes) = installer.last_handshakes() else { return 0 };
        let mut up = self.up.lock().unwrap();
        let mut transitions = 0;
        let mut transfer: Option<HashMap<[u8; 32], (u64, u64)>> = None;

        for pk in registry.pubkeys() {
            let Some(peer) = registry.get(&pk) else { continue };
            let last = handshakes.get(&pk).copied();
            let alive = last
                .map(|t| now.duration_since(t).map(|d| d < TUNNEL_DOWN_AFTER).unwrap_or(true))
                .unwrap_or(false);
            match (alive, up.get(&pk).copied()) {
                (true, None) => {
                    up.insert(pk, LiveState { up_since: last.unwrap_or(now) });
                    log.emit_at(
                        Event::TunnelUp {
                            peer: key_b64(&pk),
                            session_id: hex::encode(peer.session_id),
                            assigned_ip: peer.assigned_ip.to_string(),
                            algo: peer.algo.name().to_string(),
                        },
                        now,
                    );
                    transitions += 1;
                }
                (false, Some(state)) => {
                    up.remove(&pk);
                    let tr = transfer
                        .get_or_insert_with(|| installer.transfer().unwrap_or_default())
                        .get(&pk)
                        .copied();
                    let secs = |a: SystemTime, b: SystemTime| a.duration_since(b).map(|d| d.as_secs()).unwrap_or(0);
                    let last = last.unwrap_or(state.up_since);
                    log.emit_at(
                        Event::TunnelDown {
                            peer: key_b64(&pk),
                            session_id: hex::encode(peer.session_id),
                            assigned_ip: peer.assigned_ip.to_string(),
                            idle_s: secs(now, last),
                            up_s: secs(last, state.up_since),
                            rx_bytes: tr.map(|t| t.0),
                            tx_bytes: tr.map(|t| t.1),
                        },
                        now,
                    );
                    transitions += 1;
                }
                _ => {}
            }
        }
        // Peers that left the registry (evicted) stop being tracked.
        let known: std::collections::HashSet<_> = registry.pubkeys().into_iter().collect();
        up.retain(|pk, _| known.contains(pk));
        transitions
    }

    pub fn is_up(&self, pk: &[u8; 32]) -> bool {
        self.up.lock().unwrap().contains_key(pk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Event {
        Event::Handshake {
            session_id: "ab".into(),
            peer: "pk".into(),
            assigned_ip: "10.8.0.2".into(),
            algo: "ML-KEM-768".into(),
            src: "203.0.113.5".into(),
            new_peer: true,
            handshake_ms: 12.5,
        }
    }

    #[test]
    fn record_shape_is_stable() {
        // Pins the wire names Member 3's evaluation and the dashboard read.
        let r = record(&sample(), UNIX_EPOCH + Duration::from_millis(1_700_000_000_123));
        assert_eq!(r["v"], 1);
        assert_eq!(r["ts_ms"], 1_700_000_000_123u64);
        assert_eq!(r["event"], "handshake");
        for k in ["session_id", "peer", "assigned_ip", "algo", "src", "new_peer", "handshake_ms"] {
            assert!(r.get(k).is_some(), "missing {k}");
        }
        assert_eq!(r.as_object().unwrap().len(), 10);
    }

    #[test]
    fn name_matches_the_serialised_tag() {
        let events = [
            sample(),
            Event::Refused { src: "x".into(), code: 7, reason: "r".into() },
            Event::Aborted { src: "x".into(), error: "e".into() },
            Event::Evicted {
                peer: "p".into(),
                reason: "idle".into(),
                session_id: None,
                assigned_ip: None,
                session_age_s: None,
                rekeys: None,
            },
        ];
        for e in events {
            assert_eq!(record(&e, SystemTime::now())["event"], e.name());
        }
    }

    #[test]
    fn optional_fields_are_omitted_not_null() {
        let r = record(
            &Event::Evicted {
                peer: "p".into(),
                reason: "unknown_on_startup".into(),
                session_id: None,
                assigned_ip: None,
                session_age_s: None,
                rekeys: None,
            },
            SystemTime::now(),
        );
        assert!(r.get("session_id").is_none());
        assert!(r.get("rekeys").is_none());
    }

    #[test]
    fn file_sink_appends_lines_and_rotates() {
        let dir = std::env::temp_dir().join(format!("pqc-evlog-{}", rand::random::<u64>()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sessions.jsonl");
        {
            let log = EventLog::to_file(&path, 10_000).unwrap();
            log.emit(sample());
            log.emit(sample());
        }
        // Reopening appends rather than truncating; set the limit so the next
        // record crosses it.
        let existing = std::fs::metadata(&path).unwrap().len();
        let log = EventLog::to_file(&path, existing + 1).unwrap();
        log.emit(sample());
        let rotated = std::fs::read_to_string(dir.join("sessions.jsonl.1")).unwrap();
        let lines: Vec<&str> = rotated.lines().collect();
        assert_eq!(lines.len(), 3, "all 3 records kept, in the rotated file");
        for l in lines {
            let v: Value = serde_json::from_str(l).unwrap();
            assert_eq!(v["event"], "handshake");
        }
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "", "fresh live file after rotation");
        log.emit(sample());
        assert_eq!(std::fs::read_to_string(&path).unwrap().lines().count(), 1, "fresh file after rotation");
        std::fs::remove_dir_all(&dir).ok();
    }
}
