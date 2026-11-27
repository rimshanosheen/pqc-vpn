//! End-to-end control-loop tests: real ONNX policy, real decision gate,
//! real `server/handshake-server` connection handler over loopback TCP
//! (dry-run PSK install, so no WireGuard or admin rights needed), mock
//! device and tunnel so the test controls the inputs.
//!
//! The fixture states were picked by probing the shipped policy over
//! states the mock device can produce; each is chosen where the policy is
//! decisive (top probability ≈ 1.00), so the tests don't hinge on a
//! near-tie that a retrain could flip.

use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use handshake_server::identity::ServerIdentity;
use handshake_server::registry::{Registry, DEFAULT_MAX_PEERS};
use handshake_server::server::{self, ServerContext};
use handshake_server::tunnel::{DryRun, TunnelSettings};

use super::*;
use crate::rl::gate::{CONFIRM_TICKS, REKEY_COOLDOWN_TICKS};
use crate::state::mock::{MockDeviceState, MockTunnelHandle};
use crate::state::{ConnectionType, TrafficSample, THREAT, TIME_SINCE_REKEY};

const TICK: Duration = Duration::from_secs(5);

/// Wired, quiet, fresh keys: the policy picks ML-KEM-512.
fn low_need() -> MockDeviceState {
    MockDeviceState {
        cpu_load: 0.3,
        ram_available_fraction: 0.8,
        latency: Some(Duration::from_millis(30)),
        upload_rate_bytes_per_sec: 250_000.0,
        connection_type: ConnectionType::Wired,
    }
}

/// Cellular, idle CPU. With stale keys and threat 1.0 the policy picks
/// ML-KEM-1024.
fn high_need_idle_cpu() -> MockDeviceState {
    MockDeviceState { cpu_load: 0.05, connection_type: ConnectionType::Cellular, ..low_need() }
}

/// Cellular, busy CPU (60%: busy, but still under Layer 2's 70% gate so the
/// threat stays visible). With stale keys and threat 1.0 the policy asks for
/// rekey-now (p = 0.98) instead of the heavier KEM.
fn high_need_busy_cpu() -> MockDeviceState {
    MockDeviceState { cpu_load: 0.6, connection_type: ConnectionType::Cellular, ..low_need() }
}

/// Ordinary traffic: the anomaly layers score it 0.
fn quiet_traffic() -> TrafficSample {
    TrafficSample {
        packet_rate: 50.0,
        mean_packet_size: 500.0,
        upload_bytes_per_sec: 10_000.0,
        download_bytes_per_sec: 50_000.0,
        distinct_dest_ports: 2,
        connection_attempts: 3,
        failed_attempts: 0,
        retransmit_rate: 0.0,
        dns_server: Some("1.1.1.1".into()),
        packets: Vec::new(),
    }
}

/// A sustained 20% TCP retransmit rate: Layer 2's retransmission
/// signature scores it 1.0 (twice its 10% threshold) whenever the CPU gate
/// lets Layer 2 run.
fn hostile_traffic() -> TrafficSample {
    TrafficSample { retransmit_rate: 0.2, ..quiet_traffic() }
}

/// Traffic the test can change between ticks.
#[derive(Clone)]
struct SharedTraffic(Arc<Mutex<TrafficSample>>);

impl SharedTraffic {
    fn new(sample: TrafficSample) -> Self {
        Self(Arc::new(Mutex::new(sample)))
    }
    fn set(&self, sample: TrafficSample) {
        *self.0.lock().unwrap() = sample;
    }
}

impl TrafficSignals for SharedTraffic {
    fn sample(&self) -> TrafficSample {
        self.0.lock().unwrap().clone()
    }
}

/// A device whose readings the test can change between ticks.
#[derive(Clone)]
struct SharedDevice(Arc<Mutex<MockDeviceState>>);

impl SharedDevice {
    fn new(state: MockDeviceState) -> Self {
        Self(Arc::new(Mutex::new(state)))
    }
    fn set(&self, state: MockDeviceState) {
        *self.0.lock().unwrap() = state;
    }
}

impl DeviceState for SharedDevice {
    fn cpu_load(&self) -> f32 {
        self.0.lock().unwrap().cpu_load
    }
    fn ram_available_fraction(&self) -> f32 {
        self.0.lock().unwrap().ram_available_fraction
    }
    fn latency(&self) -> Option<Duration> {
        self.0.lock().unwrap().latency
    }
    fn upload_rate_bytes_per_sec(&self) -> f64 {
        self.0.lock().unwrap().upload_rate_bytes_per_sec
    }
    fn connection_type(&self) -> ConnectionType {
        self.0.lock().unwrap().connection_type
    }
}

struct TestServer {
    endpoint: ServerEndpoint,
    ctx: Arc<ServerContext>,
}

/// The real handshake-server handler, accepting connections until the
/// test process exits.
fn server() -> TestServer {
    let identity = ServerIdentity::generate();
    let pinned = identity.verifying_key_bytes();
    let ctx = Arc::new(ServerContext {
        identity,
        registry: Registry::new([10, 8, 0, 0], DEFAULT_MAX_PEERS),
        installer: Box::new(DryRun),
        tunnel: TunnelSettings { server_wg_pubkey: [0x5a; 32], wg_port: 51820, subnet_base: [10, 8, 0, 0] },
    });
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let server_ctx = Arc::clone(&ctx);
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let _ = server::handle_connection(stream, &server_ctx);
        }
    });
    TestServer {
        endpoint: ServerEndpoint {
            addr: addr.to_string(),
            pinned_verifying_key: pinned,
            client_wg_pubkey: [0x77; 32],
            timeout: Duration::from_secs(5),
        },
        ctx,
    }
}

impl TestServer {
    /// What the server believes is in force for our peer.
    fn server_algo(&self) -> MlKemLevel {
        let peer = self.ctx.registry.remove(&self.endpoint.client_wg_pubkey).expect("peer registered");
        // Removes the peer: callers only use this at the end of a test.
        MlKemLevel::from_wire_code(peer.algo.to_u8()).unwrap()
    }
}

type TestSession = VpnSession<SharedDevice, SharedTraffic, MockTunnelHandle>;

/// Connects on quiet traffic; returns the traffic handle so the test can
/// turn hostile later.
fn connect(srv: &TestServer, device: SharedDevice) -> (TestSession, SharedTraffic) {
    let traffic = SharedTraffic::new(quiet_traffic());
    let session = VpnSession::connect(
        srv.endpoint.clone(),
        device,
        traffic.clone(),
        MockTunnelHandle::default(),
        Policy::from_embedded().unwrap(),
    )
    .expect("connect");
    (session, traffic)
}

#[test]
fn connect_uses_the_policys_algorithm_and_brings_the_tunnel_up() {
    let srv = server();
    let (session, _) = connect(&srv, SharedDevice::new(low_need()));

    assert_eq!(session.in_force(), MlKemLevel::MlKem512);
    assert!(session.tunnel().is_up());
    assert_eq!(srv.server_algo(), MlKemLevel::MlKem512);
}

#[test]
fn a_steady_state_does_not_touch_the_network() {
    let srv = server();
    let (mut session, _) = connect(&srv, SharedDevice::new(low_need()));
    let sid = session.session_id();

    for _ in 0..5 {
        let r = session.tick(TICK).unwrap();
        assert_eq!(r.outcome, Outcome::Unchanged);
        assert!(r.step.state[THREAT] < 0.05, "quiet traffic carries almost no threat");
        assert!(!r.threat.anomalous);
    }
    assert_eq!(session.session_id(), sid);
    assert_eq!(session.tunnel().rotations(), 0);
}

/// Week 8: the THREAT dimension the policy sees is the anomaly layers'
/// combined score on live traffic, and the report says which layers ran.
#[test]
fn threat_comes_from_the_anomaly_layers() {
    let srv = server();
    let (mut session, traffic) = connect(&srv, SharedDevice::new(low_need()));

    traffic.set(hostile_traffic());
    let r = session.tick(TICK).unwrap();
    assert!(r.threat.layer2_active, "30% CPU is under the 70% Layer 2 gate");
    assert!(r.threat.layer3_active, "30% CPU is under the 40% Layer 3 gate");
    assert_eq!(r.threat.threat_score, 1.0);
    assert_eq!(r.step.state[THREAT], 1.0);

    // Same traffic on a busy device: Layer 2 is gated off and Layer 1 is
    // not ready yet, so nothing reaches the policy.
    session.pipeline.device().set(MockDeviceState { cpu_load: 0.9, ..low_need() });
    let r = session.tick(TICK).unwrap();
    assert!(!r.threat.layer2_active);
    assert_eq!(r.step.state[THREAT], 0.0);
}

/// The whole slice in one test: the situation turns hostile (cellular,
/// stale keys, sustained retransmits), the anomaly layers score it, the
/// policy wants a stronger KEM, the gate makes it wait CONFIRM_TICKS, then
/// the session renegotiates for real and the server agrees on the level.
#[test]
fn rising_need_renegotiates_to_1024_after_the_confirm_streak() {
    let srv = server();
    let device = SharedDevice::new(low_need());
    let (mut session, traffic) = connect(&srv, device.clone());
    let first_sid = session.session_id();

    device.set(high_need_idle_cpu());
    traffic.set(hostile_traffic());
    // Keys an hour old, so TIME_SINCE_REKEY saturates.
    let mut elapsed = Duration::from_secs(3600);

    for _ in 1..CONFIRM_TICKS {
        let r = session.tick(elapsed).unwrap();
        assert_eq!(r.outcome, Outcome::Unchanged, "gate should still be confirming");
        elapsed = TICK;
    }
    let r = session.tick(TICK).unwrap();
    assert_eq!(r.outcome, Outcome::Renegotiated(MlKemLevel::MlKem1024));
    assert_eq!(session.in_force(), MlKemLevel::MlKem1024);
    assert_ne!(session.session_id(), first_sid, "a renegotiation is a fresh session");
    assert_eq!(session.tunnel().rotations(), 1, "same peer and address: PSK hot-swapped");

    // The key rotation reset the rekey timer.
    let next = session.tick(TICK).unwrap();
    assert!(next.step.state[TIME_SINCE_REKEY] < 0.01);

    assert_eq!(srv.server_algo(), MlKemLevel::MlKem1024);
}

#[test]
fn rekey_now_rotates_keys_on_the_same_session_then_cools_down() {
    let srv = server();
    let device = SharedDevice::new(low_need());
    let (mut session, traffic) = connect(&srv, device.clone());
    let sid = session.session_id();

    device.set(high_need_busy_cpu());
    traffic.set(hostile_traffic());
    let r = session.tick(Duration::from_secs(3600)).unwrap();

    let Outcome::Rekeyed(level) = r.outcome else { panic!("expected a rekey, got {:?}", r.outcome) };
    assert!(level >= MlKemLevel::MlKem512, "a rekey never weakens the algorithm");
    assert_eq!(session.session_id(), sid, "a rekey keeps the session");
    assert_eq!(session.tunnel().rotations(), 1);

    // Even with keys forced stale again, the cooldown blocks a second
    // rekey for REKEY_COOLDOWN_TICKS ticks.
    for _ in 1..REKEY_COOLDOWN_TICKS {
        let r = session.tick(Duration::from_secs(3600)).unwrap();
        assert!(!matches!(r.outcome, Outcome::Rekeyed(_)), "rekey inside the cooldown");
    }
    assert_eq!(srv.server_algo(), session.in_force());
}

/// The server goes away mid-session: the action fails, the session keeps
/// its algorithm and keys, and the gate is told so.
#[test]
fn a_failed_handshake_keeps_the_old_algorithm() {
    let srv = server();
    let device = SharedDevice::new(low_need());
    let (mut session, traffic) = connect(&srv, device.clone());

    // Point the session at a closed port.
    session.endpoint.addr = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().to_string();
    session.endpoint.timeout = Duration::from_secs(1);

    device.set(high_need_idle_cpu());
    traffic.set(hostile_traffic());
    let mut elapsed = Duration::from_secs(3600);
    let mut failed = None;
    for _ in 0..CONFIRM_TICKS {
        let r = session.tick(elapsed).unwrap();
        elapsed = TICK;
        if let Outcome::Failed { attempted, .. } = r.outcome {
            failed = Some((attempted, r.in_force));
        }
    }
    let (attempted, in_force) = failed.expect("the renegotiation was attempted and failed");
    assert_eq!(attempted, Action::Renegotiate(MlKemLevel::MlKem1024));
    assert_eq!(in_force, MlKemLevel::MlKem512);
    assert_eq!(session.in_force(), MlKemLevel::MlKem512);
    assert_eq!(session.tunnel().rotations(), 0);
}

#[test]
fn disconnect_tears_the_tunnel_down() {
    let srv = server();
    let (session, _) = connect(&srv, SharedDevice::new(low_need()));
    session.disconnect().unwrap();
}
