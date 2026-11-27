//! The Week 8 vertical slice, headless: the exact loop the desktop app's
//! Connect button runs (`desktop_lib::vpn`), printed to the terminal.
//!
//! Live device readings and traffic signals → anomaly layers → RL agent →
//! real handshakes against a real `handshake-server`.
//!
//! ```text
//! cargo run -p desktop --example vertical_slice -- [HOST:PORT] [VK-HEX | @PATH] [TICKS]
//! ```
//!
//! Defaults: the team droplet and its pinned key (`server-vk.hex`), 6
//! ticks (30 s). Against a local server, see `desktop/README.md`.

use std::sync::mpsc;

use desktop_lib::vpn::{self, ClientConfig, SessionEvent};

fn main() {
    let mut args = std::env::args().skip(1);
    let mut config = ClientConfig::default();
    if let Some(addr) = args.next() {
        config.server_addr = addr;
    }
    if let Some(vk) = args.next() {
        config.verifying_key_hex = match vk.strip_prefix('@') {
            Some(path) => std::fs::read_to_string(path).expect("read verifying key file"),
            None => vk,
        };
    }
    let ticks: u64 = args.next().map(|t| t.parse().expect("TICKS is a number")).unwrap_or(6);

    println!("connecting to {} (the agent picks the algorithm)...", config.server_addr);
    let (tx, rx) = mpsc::channel();
    let running = vpn::start(config, move |e| {
        let _ = tx.send(e);
    })
    .unwrap_or_else(|e| panic!("bad config: {e}"));

    for event in rx {
        match event {
            SessionEvent::Connected { in_force, session_id, assigned_ip } => {
                println!("CONNECTED  {in_force}  session {session_id}  tunnel ip {assigned_ip}");
            }
            SessionEvent::Tick(t) => {
                let fmt = |xs: &[f32]| xs.iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>().join(" ");
                println!(
                    "tick {:>2}  state [{}]  threat {:.3}{}  (L1 {}, L2 {}, L3 {})",
                    t.tick,
                    fmt(&t.state),
                    t.threat,
                    if t.anomalous { " ANOMALOUS" } else { "" },
                    if t.layer1_ready { "ready" } else { "warming up" },
                    if t.layer2_active { "on" } else { "off" },
                    if t.layer3_active { "on" } else { "off" },
                );
                println!(
                    "          traffic: {} new conns ({} failed, {} ports), retransmits {:.1}%, dns {}",
                    t.new_connections,
                    t.failed_connections,
                    t.distinct_ports,
                    t.retransmit_rate * 100.0,
                    t.dns_server.as_deref().unwrap_or("?"),
                );
                println!("          policy {:<11} p=[{}]  gate: {}", t.raw_choice, fmt(&t.probs), t.gate_reason);
                println!(
                    "          -> {}  => {}  (in force {}, psk rotations {})",
                    t.action, t.outcome, t.in_force, t.psk_rotations
                );
                if t.tick >= ticks {
                    break;
                }
            }
            SessionEvent::Error { message } => {
                eprintln!("ERROR  {message}");
                std::process::exit(1);
            }
            SessionEvent::Stopped => break,
        }
    }
    running.stop();
    println!("stopped");
}
