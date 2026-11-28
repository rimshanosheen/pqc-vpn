//! PQC-VPN handshake server — `PROTOCOL.md` v1.
//!
//! Usage:
//!   handshake-server [--bind ADDR] [--identity PATH]
//!                    [--wg-interface NAME] [--wg-port N] [--tunnel-cidr CIDR]
//!                    [--wg-pubkey BASE64]
//!                    [--peers-file PATH] [--max-peers N] [--idle-timeout-secs N]
//!                    [--replay-window-secs N] [--rate-burst N] [--rate-per-min N]
//!                    [--max-conns N]
//!                    [--session-log PATH | --no-session-log] [--session-log-max-mb N]
//!
//! defaults: --bind 0.0.0.0:51821  --identity ./server-identity.seed
//!           --wg-port 51820       --tunnel-cidr 10.8.0.0/24
//!           --peers-file <identity dir>/peers.json  --max-peers 64
//!           --idle-timeout-secs 0  (0 = no idle eviction)
//!           --replay-window-secs 120  --rate-burst 20  --rate-per-min 30
//!           --max-conns 64           (Week 7, PROTOCOL.md §7)
//!           --session-log <identity dir>/sessions.jsonl  --session-log-max-mb 20
//!                                    (Week 9, server/SESSION_LOG.md)
//!
//! With `--wg-interface` the derived PSK is installed via `wg set` (needs
//! CAP_NET_ADMIN). Without it the server runs in dry-run.

use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};

use handshake_server::events::{Event, EventLog};
use handshake_server::guard::{Guard, GuardConfig};
use handshake_server::identity::ServerIdentity;
use handshake_server::registry::{Registry, DEFAULT_MAX_PEERS};
use handshake_server::server::{self, ServerContext};
use handshake_server::tunnel::{self, DryRun, PskInstaller, TunnelSettings, WgCli};

fn main() -> Result<()> {
    let mut bind = "0.0.0.0:51821".to_string();
    let mut identity_path = PathBuf::from("server-identity.seed");
    let mut wg_interface: Option<String> = None;
    let mut wg_port: u16 = 51820;
    let mut tunnel_cidr = "10.8.0.0/24".to_string();
    let mut wg_pubkey_b64: Option<String> = None;
    let mut peers_file: Option<PathBuf> = None;
    let mut max_peers = DEFAULT_MAX_PEERS;
    let mut idle_timeout_secs: u64 = 0;
    let mut guard_cfg = GuardConfig::default();
    let mut session_log: Option<PathBuf> = None;
    let mut no_session_log = false;
    let mut session_log_max_mb: u64 = 20;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || -> Result<String> { args.next().context("missing argument value") };
        match a.as_str() {
            "--bind" => bind = val()?,
            "--identity" => identity_path = PathBuf::from(val()?),
            "--wg-interface" => wg_interface = Some(val()?),
            "--wg-port" => wg_port = val()?.parse().context("--wg-port")?,
            "--tunnel-cidr" => tunnel_cidr = val()?,
            "--wg-pubkey" => wg_pubkey_b64 = Some(val()?),
            "--peers-file" => peers_file = Some(PathBuf::from(val()?)),
            "--max-peers" => max_peers = val()?.parse().context("--max-peers")?,
            "--idle-timeout-secs" => idle_timeout_secs = val()?.parse().context("--idle-timeout-secs")?,
            "--replay-window-secs" => {
                guard_cfg.replay_window = Duration::from_secs(val()?.parse().context("--replay-window-secs")?)
            }
            "--rate-burst" => guard_cfg.rate_burst = val()?.parse().context("--rate-burst")?,
            "--rate-per-min" => guard_cfg.rate_per_minute = val()?.parse().context("--rate-per-min")?,
            "--max-conns" => guard_cfg.max_concurrent = val()?.parse().context("--max-conns")?,
            "--session-log" => session_log = Some(PathBuf::from(val()?)),
            "--no-session-log" => no_session_log = true,
            "--session-log-max-mb" => session_log_max_mb = val()?.parse().context("--session-log-max-mb")?,
            "-h" | "--help" => {
                println!("handshake-server [--bind ADDR] [--identity PATH] [--wg-interface NAME]");
                println!("                 [--wg-port N] [--tunnel-cidr CIDR] [--wg-pubkey BASE64]");
                println!("                 [--peers-file PATH] [--max-peers N] [--idle-timeout-secs N]");
                println!("                 [--replay-window-secs N] [--rate-burst N] [--rate-per-min N]");
                println!("                 [--max-conns N]");
                println!("                 [--session-log PATH | --no-session-log] [--session-log-max-mb N]");
                return Ok(());
            }
            other => anyhow::bail!("unknown argument: {other}"),
        }
    }

    if guard_cfg.rate_burst == 0 {
        anyhow::bail!("--rate-burst must be at least 1 (0 would refuse every handshake)");
    }
    if guard_cfg.max_concurrent == 0 {
        anyhow::bail!("--max-conns must be at least 1");
    }

    let identity = ServerIdentity::load_or_create(&identity_path)
        .with_context(|| format!("loading identity from {}", identity_path.display()))?;

    let vk = identity.verifying_key_bytes();
    let vk_hex = hex::encode(&vk);
    let pub_path = identity_path.with_extension("pub");
    std::fs::write(&pub_path, &vk_hex).ok();

    let subnet_base = TunnelSettings::parse_cidr(&tunnel_cidr).map_err(anyhow::Error::msg)?;

    let peers_file = peers_file.unwrap_or_else(|| {
        identity_path
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join("peers.json")
    });
    let registry = Registry::load(peers_file.clone(), subnet_base, max_peers)
        .with_context(|| format!("loading registry from {}", peers_file.display()))?;

    let (installer, server_wg_pubkey): (Box<dyn PskInstaller>, [u8; 32]) = match &wg_interface {
        Some(iface) => {
            let pk = WgCli::query_pubkey(iface)
                .with_context(|| format!("`wg show {iface} public-key` (is the interface up?)"))?;
            (Box::new(WgCli { interface: iface.clone() }), pk)
        }
        None => {
            let pk = wg_pubkey_b64
                .as_deref()
                .and_then(tunnel::wg_key_from_base64)
                .unwrap_or([0u8; 32]);
            (Box::new(DryRun), pk)
        }
    };

    let ctx = Arc::new(ServerContext {
        identity,
        registry,
        tunnel: TunnelSettings { server_wg_pubkey, wg_port, subnet_base },
        installer,
    });

    println!("identity seed  : {}", identity_path.display());
    println!("verifying key  : {} ({} bytes)", pub_path.display(), vk.len());
    println!("               : {vk_hex}");
    println!("peers file     : {} ({} known)", peers_file.display(), ctx.registry.len());
    println!("psk install    : {}", ctx.installer.describe());
    println!("tunnel         : {tunnel_cidr}, wg port {wg_port}, max peers {max_peers}");
    if idle_timeout_secs > 0 {
        println!("idle eviction  : after {idle_timeout_secs}s");
    }
    println!(
        "guard          : replay window {}s, {} burst + {}/min per IP, max {} open conns",
        guard_cfg.replay_window.as_secs(),
        guard_cfg.rate_burst,
        guard_cfg.rate_per_minute,
        guard_cfg.max_concurrent
    );

    // Week 9: session event log (JSON Lines), next to the identity by default.
    let log = if no_session_log {
        println!("session log    : off");
        EventLog::null()
    } else {
        let path = session_log.unwrap_or_else(|| {
            identity_path
                .parent()
                .unwrap_or_else(|| std::path::Path::new("."))
                .join("sessions.jsonl")
        });
        let log = EventLog::to_file(&path, session_log_max_mb.max(1) * 1024 * 1024)
            .with_context(|| format!("opening session log {}", path.display()))?;
        println!("session log    : {} (rotates at {} MB)", path.display(), session_log_max_mb.max(1));
        log
    };
    let log = Arc::new(log);
    log.emit(Event::ServerStart {
        version: env!("CARGO_PKG_VERSION").to_string(),
        psk_install: ctx.installer.describe(),
        max_peers,
        idle_timeout_s: idle_timeout_secs,
        replay_window_s: guard_cfg.replay_window.as_secs(),
        rate_burst: guard_cfg.rate_burst,
        rate_per_min: guard_cfg.rate_per_minute,
        max_conns: guard_cfg.max_concurrent,
    });

    // reconcile the registry against the live interface
    server::reconcile_wg_peers(&ctx, &log);

    if idle_timeout_secs > 0 {
        server::spawn_idle_sweep(Arc::clone(&ctx), Arc::clone(&log), Duration::from_secs(idle_timeout_secs));
    }
    // tunnel_up / tunnel_down from WireGuard's own handshakes
    server::spawn_liveness_monitor(Arc::clone(&ctx), Arc::clone(&log));

    let listener = TcpListener::bind(&bind).with_context(|| format!("bind {bind}"))?;
    println!("listening on   : {}\n", listener.local_addr()?);

    let guard = Arc::new(Guard::new(guard_cfg));
    server::spawn_guard_prune(Arc::clone(&guard));
    server::serve_logged(listener, ctx, guard, log);
    Ok(())
}
