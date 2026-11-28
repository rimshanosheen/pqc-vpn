//! Week 9 — the session log, end to end: real `server::serve_logged` over
//! loopback TCP for handshake / rekey / rejection / refusal / abort events, and
//! a controllable fake WireGuard for tunnel_up / tunnel_down / eviction.

use std::collections::HashMap;
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use handshake_server::client_handshake;
use handshake_server::events::{EventLog, Liveness, TUNNEL_DOWN_AFTER};
use handshake_server::guard::{Guard, GuardConfig};
use handshake_server::identity::ServerIdentity;
use handshake_server::registry::{Registry, DEFAULT_MAX_PEERS};
use handshake_server::server::{self, ServerContext};
use handshake_server::tunnel::{InstallError, PskInstaller, TunnelSettings};
use handshake_server::wire::{AlgoCode, Message, RekeyRequest};
use serde_json::Value;

/// A fake WireGuard whose handshake times and byte counters a test sets.
#[derive(Default, Clone)]
struct FakeWg {
    handshakes: Arc<Mutex<HashMap<[u8; 32], SystemTime>>>,
    transfer: Arc<Mutex<HashMap<[u8; 32], (u64, u64)>>>,
    removed: Arc<Mutex<Vec<[u8; 32]>>>,
}

impl PskInstaller for FakeWg {
    fn install(&self, _: &[u8; 32], _: &[u8; 32], _: Ipv4Addr, _: AlgoCode) -> Result<(), InstallError> {
        Ok(())
    }
    fn list_installed_peers(&self) -> Result<Vec<[u8; 32]>, InstallError> {
        Ok(vec![])
    }
    fn remove_peer(&self, pk: &[u8; 32]) -> Result<(), InstallError> {
        self.removed.lock().unwrap().push(*pk);
        Ok(())
    }
    fn last_handshakes(&self) -> Result<HashMap<[u8; 32], SystemTime>, InstallError> {
        Ok(self.handshakes.lock().unwrap().clone())
    }
    fn transfer(&self) -> Result<HashMap<[u8; 32], (u64, u64)>, InstallError> {
        Ok(self.transfer.lock().unwrap().clone())
    }
    fn describe(&self) -> String {
        "fake (test)".into()
    }
}

fn ctx_with(wg: FakeWg) -> Arc<ServerContext> {
    Arc::new(ServerContext {
        identity: ServerIdentity::generate(),
        registry: Registry::new([10, 8, 0, 0], DEFAULT_MAX_PEERS),
        installer: Box::new(wg),
        tunnel: TunnelSettings { server_wg_pubkey: [0x5a; 32], wg_port: 51820, subnet_base: [10, 8, 0, 0] },
    })
}

struct Rig {
    addr: SocketAddr,
    ctx: Arc<ServerContext>,
    log: Arc<EventLog>,
}

fn rig(cfg: GuardConfig) -> Rig {
    let ctx = ctx_with(FakeWg::default());
    let log = Arc::new(EventLog::in_memory());
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (c, g, l) = (Arc::clone(&ctx), Arc::new(Guard::new(cfg)), Arc::clone(&log));
    std::thread::spawn(move || server::serve_logged(listener, c, g, l));
    Rig { addr, ctx, log }
}

/// Full handshake, or a rekey of `session` when given. `Ok(session_id)` or
/// `Err(code)`.
fn connect(
    r: &Rig,
    wg_pub: [u8; 32],
    algo: AlgoCode,
    session: Option<[u8; 16]>,
) -> Result<[u8; 16], u8> {
    let state = client_handshake::start(algo, rand::random(), wg_pub);
    let mut s = TcpStream::connect(r.addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    match session {
        Some(session_id) => Message::RekeyRequest(RekeyRequest { session_id, hello: state.hello().clone() }),
        None => Message::ClientHello(state.hello().clone()),
    }
    .write(&mut s)
    .unwrap();
    let sh = match Message::read(&mut s).unwrap() {
        Message::ServerHello(sh) => sh,
        Message::Error(e) => return Err(e.code),
        other => panic!("{other:?}"),
    };
    let (_, fin, _) = state.finish(&sh, &r.ctx.identity.verifying_key_bytes()).unwrap();
    Message::ClientFinish(fin).write(&mut s).unwrap();
    match Message::read(&mut s).unwrap() {
        Message::ServerFinish(sf) => Ok(sf.session_id),
        Message::Error(e) => Err(e.code),
        other => panic!("{other:?}"),
    }
}

/// Wait until the log holds `n` records of `event` (the handler thread logs
/// just after it writes ServerFinish).
fn wait_events(log: &EventLog, event: &str, n: usize) -> Vec<Value> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let v = log.records_of(event);
        if v.len() >= n {
            return v;
        }
        assert!(Instant::now() < deadline, "expected {n} `{event}` events, have {}: {:?}", v.len(), log.records());
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn handshakes_and_reconnects_are_logged() {
    let r = rig(GuardConfig::default());
    let sid = connect(&r, [0x11; 32], AlgoCode::MlKem768, None).unwrap();
    let ev = &wait_events(&r.log, "handshake", 1)[0];
    assert_eq!(ev["v"], 1);
    assert!(ev["ts_ms"].as_u64().unwrap() > 1_700_000_000_000);
    assert_eq!(ev["session_id"], hex::encode(sid));
    assert_eq!(ev["algo"], "ML-KEM-768");
    assert_eq!(ev["assigned_ip"], "10.8.0.2");
    assert_eq!(ev["src"], "127.0.0.1");
    assert_eq!(ev["new_peer"], true);
    assert!(ev["handshake_ms"].as_f64().unwrap() > 0.0);
    assert!(ev.get("psk").is_none() && !ev.to_string().contains("psk\""), "no secrets in the log");

    // the same WireGuard key connecting again is a reconnect, same address
    connect(&r, [0x11; 32], AlgoCode::MlKem512, None).unwrap();
    let again = &wait_events(&r.log, "handshake", 2)[1];
    assert_eq!(again["new_peer"], false);
    assert_eq!(again["assigned_ip"], "10.8.0.2");
    assert_eq!(again["algo"], "ML-KEM-512");
}

#[test]
fn rekeys_and_rejections_are_logged() {
    let r = rig(GuardConfig::default());
    let pk = [0x22; 32];
    let sid = connect(&r, pk, AlgoCode::MlKem768, None).unwrap();
    assert_eq!(connect(&r, pk, AlgoCode::MlKem1024, Some(sid)), Ok(sid));
    let rk = &wait_events(&r.log, "rekey", 1)[0];
    assert_eq!(rk["from_algo"], "ML-KEM-768");
    assert_eq!(rk["algo"], "ML-KEM-1024");
    assert_eq!(rk["escalated"], true);
    assert_eq!(rk["rekeys"], 1);
    assert_eq!(rk["session_id"], hex::encode(sid));

    // a downgrade is refused with 0x05 and logged as a registry rejection
    assert_eq!(connect(&r, pk, AlgoCode::MlKem512, Some(sid)), Err(0x05));
    let rej = &wait_events(&r.log, "rejected", 1)[0];
    assert_eq!(rej["stage"], "registry");
    assert_eq!(rej["code"], 5);
    assert_eq!(rej["rekey"], true);
    assert_eq!(rej["algo"], "ML-KEM-512");

    // an unknown session is 0x04
    assert_eq!(connect(&r, [0x23; 32], AlgoCode::MlKem768, Some([9; 16])), Err(0x04));
    assert_eq!(wait_events(&r.log, "rejected", 2)[1]["code"], 4);
    assert_eq!(r.log.records_of("rekey").len(), 1, "failed rekeys are not logged as rekeys");
}

#[test]
fn guard_refusals_are_logged() {
    let r = rig(GuardConfig { rate_burst: 2, rate_per_minute: 0, ..GuardConfig::default() });
    // replay: send the same hello twice
    let state = client_handshake::start(AlgoCode::MlKem768, rand::random(), [0x33; 32]);
    for _ in 0..2 {
        let mut s = TcpStream::connect(r.addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        Message::ClientHello(state.hello().clone()).write(&mut s).unwrap();
        let _ = Message::read(&mut s);
    }
    let replay = &wait_events(&r.log, "refused", 1)[0];
    assert_eq!(replay["code"], 6);
    // the burst of 2 is spent: the third connection is rate limited at accept
    let mut s = TcpStream::connect(r.addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    let _ = Message::read(&mut s);
    let limited = &wait_events(&r.log, "refused", 2)[1];
    assert_eq!(limited["code"], 7);
    assert_eq!(limited["src"], "127.0.0.1");
}

#[test]
fn a_client_vanishing_mid_handshake_is_logged_as_aborted() {
    let r = rig(GuardConfig::default());
    let state = client_handshake::start(AlgoCode::MlKem512, rand::random(), [0x44; 32]);
    {
        let mut s = TcpStream::connect(r.addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        Message::ClientHello(state.hello().clone()).write(&mut s).unwrap();
        assert!(matches!(Message::read(&mut s).unwrap(), Message::ServerHello(_)));
    } // dropped before ClientFinish
    let ab = &wait_events(&r.log, "aborted", 1)[0];
    assert_eq!(ab["src"], "127.0.0.1");
    assert!(r.log.records_of("rejected").is_empty(), "an abort is not double-logged as a rejection");
    assert!(r.log.records_of("handshake").is_empty());
}

#[test]
fn tunnel_up_and_down_follow_wireguard_handshakes() {
    let wg = FakeWg::default();
    let ctx = ctx_with(wg.clone());
    let log = EventLog::in_memory();
    let live = Liveness::new();
    let pk = [0x55; 32];
    ctx.registry.upsert_handshake(pk, AlgoCode::MlKem768, [7; 16]).unwrap();
    let t0 = SystemTime::now();

    // PQC handshake done, but WireGuard hasn't handshaken yet: nothing
    assert_eq!(live.tick(&ctx.registry, ctx.installer.as_ref(), &log, t0), 0);

    // WireGuard handshakes: tunnel_up, once
    wg.handshakes.lock().unwrap().insert(pk, t0);
    assert_eq!(live.tick(&ctx.registry, ctx.installer.as_ref(), &log, t0 + Duration::from_secs(1)), 1);
    assert_eq!(live.tick(&ctx.registry, ctx.installer.as_ref(), &log, t0 + Duration::from_secs(16)), 0);
    let up = &log.records_of("tunnel_up")[0];
    assert_eq!(up["session_id"], hex::encode([7u8; 16]));
    assert_eq!(up["algo"], "ML-KEM-768");
    assert!(live.is_up(&pk));

    // keepalive re-handshakes keep it up across many minutes
    let last = t0 + Duration::from_secs(600);
    wg.handshakes.lock().unwrap().insert(pk, last);
    wg.transfer.lock().unwrap().insert(pk, (123_456, 7_890));
    assert_eq!(live.tick(&ctx.registry, ctx.installer.as_ref(), &log, last + Duration::from_secs(170)), 0);

    // silence past WireGuard's REJECT_AFTER_TIME: tunnel_down
    let n = live.tick(&ctx.registry, ctx.installer.as_ref(), &log, last + TUNNEL_DOWN_AFTER + Duration::from_secs(5));
    assert_eq!(n, 1);
    let down = &log.records_of("tunnel_down")[0];
    assert_eq!(down["up_s"], 600);
    assert_eq!(down["idle_s"], 185);
    assert_eq!(down["rx_bytes"], 123_456);
    assert_eq!(down["tx_bytes"], 7_890);
    assert!(!live.is_up(&pk));

    // it comes back
    wg.handshakes.lock().unwrap().insert(pk, last + Duration::from_secs(400));
    assert_eq!(live.tick(&ctx.registry, ctx.installer.as_ref(), &log, last + Duration::from_secs(401)), 1);
    assert_eq!(log.records_of("tunnel_up").len(), 2);
}

#[test]
fn idle_eviction_is_logged_and_live_tunnels_are_kept() {
    let wg = FakeWg::default();
    let ctx = ctx_with(wg.clone());
    let log = EventLog::in_memory();
    let (idle, live) = ([0x66; 32], [0x67; 32]);
    ctx.registry.upsert_handshake(idle, AlgoCode::MlKem512, [1; 16]).unwrap();
    ctx.registry.upsert_handshake(live, AlgoCode::MlKem768, [2; 16]).unwrap();
    let later = SystemTime::now() + Duration::from_secs(7200);
    // `live` still handshakes in WireGuard; `idle` never did
    wg.handshakes.lock().unwrap().insert(live, later - Duration::from_secs(30));

    assert_eq!(server::sweep_once(&ctx, &log, Duration::from_secs(3600), later), 1);
    let ev = &log.records_of("evicted")[0];
    assert_eq!(ev["reason"], "idle");
    assert_eq!(ev["session_id"], hex::encode([1u8; 16]));
    assert_eq!(ev["assigned_ip"], "10.8.0.2");
    assert!(ev["session_age_s"].as_u64().unwrap() >= 7199);
    assert!(ctx.registry.contains(&live) && !ctx.registry.contains(&idle));
    assert_eq!(*wg.removed.lock().unwrap(), vec![idle]);
}
