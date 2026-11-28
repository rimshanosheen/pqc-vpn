//! The Week 8 vertical slice on the desktop (Member 1): real device
//! readings + real traffic signals → anomaly layers → RL agent → real
//! handshakes against the server, one tick every 5 s on a background
//! thread.
//!
//! Used by the Tauri commands in `lib.rs` and by
//! `examples/vertical_slice.rs` (the same loop, headless, for checking the
//! slice from a terminal).
//!
//! **What is not real yet:** the WireGuard interface. [`PendingTunnel`]
//! records the tunnel config and counts PSK rotations but configures no
//! interface; `wg set` from Rust is Week 9. Everything up to and including
//! the server installing the PSK on its side is real.

use std::net::{Ipv4Addr, SocketAddr};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use rand::rngs::OsRng;
use serde::Serialize;
use vpn_core::protocol::VERIFYING_KEY_LEN;
use vpn_core::rl::{Action, Policy, PolicyAction};
use vpn_core::rl::gate::TICK_SECONDS;
use vpn_core::session::{Outcome, ServerEndpoint, TickReport, VpnSession};
use vpn_core::state::{SessionPsk, TunnelConfig, TunnelError, TunnelHandle, TunnelStatus};
use x25519_dalek::{PublicKey, StaticSecret};

use crate::device_state::SysinfoDeviceState;
use crate::traffic::SystemTrafficSignals;

/// Member 2's droplet (`PROGRESS.md`, Member 2 Week 1), handshake port.
pub const DEFAULT_SERVER_ADDR: &str = "168.144.73.113:51821";
/// That server's ML-DSA-65 key, pinned in the client
/// (`server/PROTOCOL.md` §5.2: shipped with the installer, never fetched).
pub const DEFAULT_VERIFYING_KEY_HEX: &str = include_str!("../../../server-vk.hex");

/// Where to connect.
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct ClientConfig {
    pub server_addr: String,
    /// Hex of the server's 1952-byte verifying key.
    pub verifying_key_hex: String,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            server_addr: DEFAULT_SERVER_ADDR.to_string(),
            verifying_key_hex: DEFAULT_VERIFYING_KEY_HEX.trim().to_string(),
        }
    }
}

impl ClientConfig {
    fn verifying_key(&self) -> Result<Vec<u8>, String> {
        let bytes = hex::decode(self.verifying_key_hex.trim())
            .map_err(|e| format!("verifying key is not hex: {e}"))?;
        if bytes.len() != VERIFYING_KEY_LEN {
            return Err(format!(
                "verifying key is {} bytes, an ML-DSA-65 key is {VERIFYING_KEY_LEN}",
                bytes.len()
            ));
        }
        Ok(bytes)
    }
}

/// Records what a real tunnel would be configured with. Week 9 replaces
/// this with a `wg`-backed `TunnelHandle`.
#[derive(Debug, Default)]
pub struct PendingTunnel {
    config: Mutex<Option<(Ipv4Addr, SocketAddr)>>,
    rotations: AtomicU32,
}

impl PendingTunnel {
    pub fn rotations(&self) -> u32 {
        self.rotations.load(Ordering::Relaxed)
    }
}

impl TunnelHandle for PendingTunnel {
    fn bring_up(&self, config: &TunnelConfig) -> Result<(), TunnelError> {
        *self.config.lock().unwrap() = Some((config.assigned_ip, config.server_endpoint));
        Ok(())
    }

    fn rotate_psk(&self, _psk: &SessionPsk) -> Result<(), TunnelError> {
        if self.config.lock().unwrap().is_none() {
            return Err(TunnelError::InterfaceUnavailable("rotate_psk before bring_up".into()));
        }
        self.rotations.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn tear_down(&self) -> Result<(), TunnelError> {
        *self.config.lock().unwrap() = None;
        Ok(())
    }

    fn status(&self) -> Result<TunnelStatus, TunnelError> {
        let c = *self.config.lock().unwrap();
        Ok(TunnelStatus { up: c.is_some(), assigned_ip: c.map(|(ip, _)| ip), last_handshake_age: None })
    }
}

/// One tick, shaped for the frontend and the terminal.
#[derive(Debug, Clone, Serialize)]
pub struct TickView {
    pub tick: u64,
    pub in_force: &'static str,
    /// The policy's raw choice, before the gate.
    pub raw_choice: &'static str,
    /// Softmax over ML-KEM-512, -768, -1024, rekey-now.
    pub probs: [f32; 4],
    pub gate_reason: &'static str,
    pub action: String,
    pub outcome: String,
    /// The 7-dim state the policy saw.
    pub state: [f32; 7],
    pub threat: f64,
    pub anomalous: bool,
    pub layer1_ready: bool,
    pub layer2_active: bool,
    pub layer3_active: bool,
    /// Layer 2's live inputs this tick, so a threat score can be traced.
    pub new_connections: u32,
    pub failed_connections: u32,
    pub distinct_ports: u32,
    pub retransmit_rate: f64,
    pub dns_server: Option<String>,
    pub session_id: String,
    pub psk_rotations: u32,
}

fn action_name(a: PolicyAction) -> &'static str {
    match a {
        PolicyAction::Kem(l) => l.name(),
        PolicyAction::RekeyNow => "rekey-now",
    }
}

fn describe_action(a: Action) -> String {
    match a {
        Action::Stay => "stay".into(),
        Action::Renegotiate(l) => format!("renegotiate to {}", l.name()),
        Action::Rekey(l) => format!("rekey at {}", l.name()),
    }
}

fn describe_outcome(o: &Outcome) -> String {
    match o {
        Outcome::Unchanged => "unchanged".into(),
        Outcome::Renegotiated(l) => format!("renegotiated to {}", l.name()),
        Outcome::Rekeyed(l) => format!("rekeyed at {}", l.name()),
        Outcome::Failed { error, .. } => format!("failed: {error}"),
    }
}

impl TickView {
    fn new(tick: u64, r: &TickReport, session_id: [u8; 16], psk_rotations: u32) -> Self {
        Self {
            tick,
            in_force: r.in_force.name(),
            raw_choice: action_name(r.step.raw_choice),
            probs: r.step.probs,
            gate_reason: r.step.decision.reason,
            action: describe_action(r.step.action),
            outcome: describe_outcome(&r.outcome),
            state: r.step.state,
            threat: r.threat.threat_score,
            anomalous: r.threat.anomalous,
            layer1_ready: r.threat.ready,
            layer2_active: r.threat.layer2_active,
            layer3_active: r.threat.layer3_active,
            new_connections: r.traffic.connection_attempts,
            failed_connections: r.traffic.failed_attempts,
            distinct_ports: r.traffic.distinct_dest_ports,
            retransmit_rate: r.traffic.retransmit_rate,
            dns_server: r.traffic.dns_server.clone(),
            session_id: hex::encode(session_id),
            psk_rotations,
        }
    }
}

/// Everything the background session reports.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SessionEvent {
    Connected { in_force: &'static str, session_id: String, assigned_ip: String },
    Tick(TickView),
    Error { message: String },
    Stopped,
}

/// A running session thread.
pub struct Running {
    stop: Arc<AtomicBool>,
    handle: JoinHandle<()>,
}

impl Running {
    /// Asks the loop to stop and waits for it (at most one tick's work).
    pub fn stop(self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.handle.join();
    }
}

/// Starts a session on a background thread. Returns an error only for bad
/// configuration; connection failures arrive as [`SessionEvent::Error`].
///
/// The client's WireGuard key pair is generated fresh for each session.
/// Week 9 has to persist it, since `wg` needs the private half and a rekey
/// must come from the same peer identity.
pub fn start(
    config: ClientConfig,
    on_event: impl Fn(SessionEvent) + Send + 'static,
) -> Result<Running, String> {
    let pinned = config.verifying_key()?;
    let wg_secret = StaticSecret::random_from_rng(OsRng);
    let endpoint = ServerEndpoint {
        addr: config.server_addr.clone(),
        pinned_verifying_key: pinned,
        client_wg_pubkey: PublicKey::from(&wg_secret).to_bytes(),
        timeout: Duration::from_secs(10),
    };

    let stop = Arc::new(AtomicBool::new(false));
    let stop_flag = Arc::clone(&stop);

    let handle = std::thread::Builder::new()
        .name("vpn-session".into())
        .spawn(move || run(endpoint, stop_flag, on_event))
        .map_err(|e| format!("could not start the session thread: {e}"))?;

    Ok(Running { stop, handle })
}

fn run(endpoint: ServerEndpoint, stop: Arc<AtomicBool>, on_event: impl Fn(SessionEvent)) {
    let policy = match Policy::from_embedded() {
        Ok(p) => p,
        Err(e) => return on_event(SessionEvent::Error { message: e.to_string() }),
    };

    let mut session = match VpnSession::connect(
        endpoint,
        SysinfoDeviceState::new(),
        SystemTrafficSignals::new(),
        PendingTunnel::default(),
        policy,
    ) {
        Ok(s) => s,
        Err(e) => return on_event(SessionEvent::Error { message: format!("connect failed: {e}") }),
    };

    on_event(SessionEvent::Connected {
        in_force: session.in_force().name(),
        session_id: hex::encode(session.session_id()),
        assigned_ip: session.assigned_ip().to_string(),
    });

    let period = Duration::from_secs(TICK_SECONDS as u64);
    let mut last = Instant::now();
    let mut tick = 0u64;

    'outer: loop {
        // Sleep in short slices so a disconnect is noticed quickly.
        while last.elapsed() < period {
            if stop.load(Ordering::Relaxed) {
                break 'outer;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        let elapsed = last.elapsed();
        last = Instant::now();
        tick += 1;

        match session.tick(elapsed) {
            Ok(report) => on_event(SessionEvent::Tick(TickView::new(
                tick,
                &report,
                session.session_id(),
                session.tunnel().rotations(),
            ))),
            Err(e) => {
                on_event(SessionEvent::Error { message: format!("tick failed: {e}") });
                break;
            }
        }
    }

    let _ = session.disconnect();
    on_event(SessionEvent::Stopped);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_carries_a_valid_pinned_key() {
        assert_eq!(ClientConfig::default().verifying_key().unwrap().len(), VERIFYING_KEY_LEN);
    }

    #[test]
    fn bad_keys_are_rejected_before_any_network() {
        let short = ClientConfig { verifying_key_hex: "abcd".into(), ..ClientConfig::default() };
        assert!(start(short, |_| {}).is_err());
        let not_hex = ClientConfig { verifying_key_hex: "zz".into(), ..ClientConfig::default() };
        assert!(start(not_hex, |_| {}).is_err());
    }
}
