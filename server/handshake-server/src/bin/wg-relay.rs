//! `wg-relay` — carry WireGuard's UDP over one TCP connection (see
//! `handshake_server::relay` for why).
//!
//! ```text
//! # on the server (forwards only to the local WireGuard):
//! wg-relay server [--listen 0.0.0.0:443] [--forward 127.0.0.1:51820] [--max-conns 64]
//!
//! # on each client, then set the WireGuard tunnel's Endpoint = 127.0.0.1:51820
//! wg-relay client --server HOST:443 [--listen 127.0.0.1:51820]
//! ```

use std::net::{SocketAddr, TcpListener};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use handshake_server::relay::{self, ClientConfig, ServerConfig, IDLE_TIMEOUT};

fn usage() -> ! {
    eprintln!("usage:");
    eprintln!("  wg-relay server [--listen ADDR] [--forward ADDR] [--max-conns N]");
    eprintln!("      defaults: --listen 0.0.0.0:443 --forward 127.0.0.1:51820 --max-conns 64");
    eprintln!("  wg-relay client --server HOST:PORT [--listen ADDR]");
    eprintln!("      default:  --listen 127.0.0.1:51820  (set WireGuard's Endpoint to this)");
    std::process::exit(2);
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| usage());
    let mut listen: Option<String> = None;
    let mut forward = "127.0.0.1:51820".to_string();
    let mut server: Option<String> = None;
    let mut max_conns = 64usize;
    while let Some(a) = args.next() {
        let mut val = || args.next().context("missing argument value");
        match a.as_str() {
            "--listen" => listen = Some(val()?),
            "--forward" => forward = val()?,
            "--server" => server = Some(val()?),
            "--max-conns" => max_conns = val()?.parse().context("--max-conns")?,
            "-h" | "--help" => usage(),
            other => bail!("unknown argument: {other}"),
        }
    }

    match mode.as_str() {
        "server" => {
            let listen = listen.unwrap_or_else(|| "0.0.0.0:443".into());
            let forward_to: SocketAddr = forward.parse().context("--forward must be IP:PORT")?;
            let listener = TcpListener::bind(&listen).with_context(|| format!("bind {listen}"))?;
            println!(
                "wg-relay server: tcp {} -> udp {forward_to} (max {max_conns} conns, idle {}s)",
                listener.local_addr()?,
                IDLE_TIMEOUT.as_secs()
            );
            relay::run_server(listener, ServerConfig { forward_to, max_conns, idle_timeout: IDLE_TIMEOUT });
            Ok(())
        }
        "client" => {
            let server = server.unwrap_or_else(|| usage());
            let listen: SocketAddr = listen
                .unwrap_or_else(|| "127.0.0.1:51820".into())
                .parse()
                .context("--listen must be IP:PORT")?;
            relay::run_client(ClientConfig { listen, server, reconnect_delay: Duration::from_secs(2) })?;
            Ok(())
        }
        _ => usage(),
    }
}
