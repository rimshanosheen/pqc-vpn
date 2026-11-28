//! The seam between a completed handshake and the WireGuard tunnel —
//! `PROTOCOL.md` §1 and §6.
//!
//! `PskInstaller` installs/updates a peer's preshared key and can list/remove
//! peers (for Week 5 startup reconciliation). `WgCli` shells out to `wg`;
//! `DryRun` only logs and is the default when `--wg-interface` is not given.

use std::collections::HashMap;
use std::io::Write;
use std::net::Ipv4Addr;
use std::process::{Command, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;

use crate::wire::AlgoCode;

fn b64(bytes: &[u8]) -> String {
    B64.encode(bytes)
}

/// Decode a 32-byte WireGuard key from base64.
pub fn wg_key_from_base64(s: &str) -> Option<[u8; 32]> {
    let v = B64.decode(s.trim()).ok()?;
    v.try_into().ok()
}

#[derive(Clone)]
pub struct TunnelSettings {
    /// The server's own WireGuard public key (sent to the client in `ServerFinish`).
    pub server_wg_pubkey: [u8; 32],
    /// UDP port the server's WireGuard listens on.
    pub wg_port: u16,
    /// Tunnel network, assumed /24.
    pub subnet_base: [u8; 4],
}

impl TunnelSettings {
    /// Parse `10.8.0.0/24` (only /24 is supported for now).
    pub fn parse_cidr(cidr: &str) -> Result<[u8; 4], String> {
        let (addr, prefix) = cidr.split_once('/').ok_or("expected addr/prefix")?;
        if prefix.trim() != "24" {
            return Err("only /24 tunnel subnets are supported for now".into());
        }
        let octets: Vec<u8> = addr
            .split('.')
            .map(|o| o.parse::<u8>().map_err(|_| "bad octet".to_string()))
            .collect::<Result<_, _>>()?;
        octets.try_into().map_err(|_| "expected 4 octets".to_string())
    }
}

#[derive(Debug)]
pub enum InstallError {
    Spawn(String),
    WgFailed { code: Option<i32>, stderr: String },
    Parse(String),
}

impl std::fmt::Display for InstallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InstallError::Spawn(e) => write!(f, "could not run `wg`: {e}"),
            InstallError::WgFailed { code, stderr } => {
                write!(f, "`wg` failed (exit {code:?}): {}", stderr.trim())
            }
            InstallError::Parse(m) => write!(f, "parsing `wg` output: {m}"),
        }
    }
}
impl std::error::Error for InstallError {}

pub trait PskInstaller: Send + Sync {
    /// Install or replace the peer's preshared key and pin its `allowed-ips` to
    /// `<assigned_ip>/32`. Idempotent — a rekey re-sets the same values.
    fn install(
        &self,
        peer_wg_pubkey: &[u8; 32],
        psk: &[u8; 32],
        assigned_ip: Ipv4Addr,
        algo: AlgoCode,
    ) -> Result<(), InstallError>;

    /// The WireGuard public keys currently configured on the interface.
    fn list_installed_peers(&self) -> Result<Vec<[u8; 32]>, InstallError>;

    /// Remove a peer from the interface.
    fn remove_peer(&self, peer_wg_pubkey: &[u8; 32]) -> Result<(), InstallError>;

    /// Each peer's most recent WireGuard handshake time (the tunnel's own
    /// liveness signal). Empty for `DryRun`.
    fn last_handshakes(&self) -> Result<HashMap<[u8; 32], SystemTime>, InstallError>;

    /// Each peer's `(rx_bytes, tx_bytes)` counters as WireGuard reports them
    /// (Week 9, for the session log). Defaults to empty, so existing
    /// implementations need no change.
    fn transfer(&self) -> Result<HashMap<[u8; 32], (u64, u64)>, InstallError> {
        Ok(HashMap::new())
    }

    fn describe(&self) -> String;
}

/// Logs actions without touching WireGuard.
pub struct DryRun;

impl PskInstaller for DryRun {
    fn install(
        &self,
        peer_wg_pubkey: &[u8; 32],
        _psk: &[u8; 32],
        assigned_ip: Ipv4Addr,
        algo: AlgoCode,
    ) -> Result<(), InstallError> {
        println!(
            "[dry-run] would set PSK + allowed-ips {assigned_ip}/32 for wg peer {} ({})",
            b64(peer_wg_pubkey),
            algo.name(),
        );
        Ok(())
    }
    fn list_installed_peers(&self) -> Result<Vec<[u8; 32]>, InstallError> {
        Ok(Vec::new())
    }
    fn remove_peer(&self, peer_wg_pubkey: &[u8; 32]) -> Result<(), InstallError> {
        println!("[dry-run] would remove wg peer {}", b64(peer_wg_pubkey));
        Ok(())
    }
    fn last_handshakes(&self) -> Result<HashMap<[u8; 32], SystemTime>, InstallError> {
        Ok(HashMap::new())
    }
    fn describe(&self) -> String {
        "dry-run (no --wg-interface)".into()
    }
}

/// Shells out to `wg`.
pub struct WgCli {
    pub interface: String,
}

impl WgCli {
    /// `wg show <iface> public-key` → 32 raw bytes.
    pub fn query_pubkey(interface: &str) -> Result<[u8; 32], InstallError> {
        let out = run("wg", &["show", interface, "public-key"])?;
        wg_key_from_base64(String::from_utf8_lossy(&out).trim())
            .ok_or_else(|| InstallError::Parse("public key".into()))
    }

    /// argv for [`install`](Self::install), exposed for testing.
    pub fn install_argv(&self, peer: &[u8; 32], assigned_ip: Ipv4Addr) -> Vec<String> {
        vec![
            "set".into(),
            self.interface.clone(),
            "peer".into(),
            b64(peer),
            "preshared-key".into(),
            "/dev/stdin".into(),
            "allowed-ips".into(),
            format!("{assigned_ip}/32"),
        ]
    }
}

impl PskInstaller for WgCli {
    fn install(
        &self,
        peer_wg_pubkey: &[u8; 32],
        psk: &[u8; 32],
        assigned_ip: Ipv4Addr,
        _algo: AlgoCode,
    ) -> Result<(), InstallError> {
        let mut child = Command::new("wg")
            .args(self.install_argv(peer_wg_pubkey, assigned_ip))
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| InstallError::Spawn(e.to_string()))?;
        child
            .stdin
            .take()
            .expect("piped")
            .write_all(format!("{}\n", b64(psk)).as_bytes())
            .map_err(|e| InstallError::Spawn(e.to_string()))?;
        let out = child
            .wait_with_output()
            .map_err(|e| InstallError::Spawn(e.to_string()))?;
        if !out.status.success() {
            return Err(InstallError::WgFailed {
                code: out.status.code(),
                stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
            });
        }
        Ok(())
    }

    fn list_installed_peers(&self) -> Result<Vec<[u8; 32]>, InstallError> {
        let out = run("wg", &["show", &self.interface, "peers"])?;
        String::from_utf8_lossy(&out)
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                wg_key_from_base64(l.trim())
                    .ok_or_else(|| InstallError::Parse(format!("peer key {l:?}")))
            })
            .collect()
    }

    fn remove_peer(&self, peer_wg_pubkey: &[u8; 32]) -> Result<(), InstallError> {
        run(
            "wg",
            &["set", &self.interface, "peer", &b64(peer_wg_pubkey), "remove"],
        )
        .map(|_| ())
    }

    fn last_handshakes(&self) -> Result<HashMap<[u8; 32], SystemTime>, InstallError> {
        let out = run("wg", &["show", &self.interface, "latest-handshakes"])?;
        let mut map = HashMap::new();
        for line in String::from_utf8_lossy(&out).lines() {
            let mut cols = line.split_whitespace();
            let (Some(pk), Some(ts)) = (cols.next(), cols.next()) else { continue };
            let (Some(pk), Ok(ts)) = (wg_key_from_base64(pk), ts.parse::<u64>()) else { continue };
            if ts > 0 {
                map.insert(pk, UNIX_EPOCH + Duration::from_secs(ts));
            }
        }
        Ok(map)
    }

    fn transfer(&self) -> Result<HashMap<[u8; 32], (u64, u64)>, InstallError> {
        let out = run("wg", &["show", &self.interface, "transfer"])?;
        Ok(parse_transfer(&String::from_utf8_lossy(&out)))
    }

    fn describe(&self) -> String {
        format!("wg set {}", self.interface)
    }
}

/// Parse `wg show <iface> transfer` output: `<pubkey>\t<rx>\t<tx>` per line.
pub fn parse_transfer(out: &str) -> HashMap<[u8; 32], (u64, u64)> {
    let mut map = HashMap::new();
    for line in out.lines() {
        let mut cols = line.split_whitespace();
        let (Some(pk), Some(rx), Some(tx)) = (cols.next(), cols.next(), cols.next()) else { continue };
        if let (Some(pk), Ok(rx), Ok(tx)) = (wg_key_from_base64(pk), rx.parse(), tx.parse()) {
            map.insert(pk, (rx, tx));
        }
    }
    map
}

fn run(cmd: &str, args: &[&str]) -> Result<Vec<u8>, InstallError> {
    let out = Command::new(cmd)
        .args(args)
        .output()
        .map_err(|e| InstallError::Spawn(e.to_string()))?;
    if !out.status.success() {
        return Err(InstallError::WgFailed {
            code: out.status.code(),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        });
    }
    Ok(out.stdout)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn install_argv_always_sets_allowed_ips() {
        let cli = WgCli { interface: "wg0".into() };
        let v = cli.install_argv(&[0xAB; 32], Ipv4Addr::new(10, 8, 0, 5));
        assert_eq!(v[0..3], ["set", "wg0", "peer"]);
        assert_eq!(v[3], B64.encode([0xABu8; 32]));
        assert_eq!(v[4..6], ["preshared-key", "/dev/stdin"]);
        assert_eq!(v[6..8], ["allowed-ips", "10.8.0.5/32"]);
    }

    #[test]
    fn cidr_parsing() {
        assert_eq!(TunnelSettings::parse_cidr("10.8.0.0/24").unwrap(), [10, 8, 0, 0]);
        assert!(TunnelSettings::parse_cidr("10.8.0.0/16").is_err());
    }

    #[test]
    fn dry_run_is_inert() {
        let d = DryRun;
        assert!(d.install(&[0; 32], &[0; 32], Ipv4Addr::new(10, 8, 0, 2), AlgoCode::MlKem768).is_ok());
        assert!(d.list_installed_peers().unwrap().is_empty());
        assert!(d.remove_peer(&[0; 32]).is_ok());
        assert!(d.transfer().unwrap().is_empty());
    }

    #[test]
    fn transfer_parsing() {
        let k = B64.encode([7u8; 32]);
        let out = format!("{k}\t3727\t2180\nnot-a-key\t1\t2\n{}\tx\t1\n", B64.encode([8u8; 32]));
        let m = parse_transfer(&out);
        assert_eq!(m.len(), 1);
        assert_eq!(m[&[7u8; 32]], (3727, 2180));
    }
}
