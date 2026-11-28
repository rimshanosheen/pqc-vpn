//! `session-report` — summarise the Week 9 session log.
//!
//! ```text
//! session-report [--json] FILE [FILE ...]     # e.g. sessions.jsonl.1 sessions.jsonl
//! ```
//!
//! On the droplet:
//! `sudo docker exec pqc-handshake-server session-report /var/lib/pqc-vpn/sessions.jsonl`

use anyhow::{bail, Context, Result};
use handshake_server::report;

fn main() -> Result<()> {
    let mut json = false;
    let mut files = Vec::new();
    for a in std::env::args().skip(1) {
        match a.as_str() {
            "--json" => json = true,
            "-h" | "--help" => {
                println!("session-report [--json] FILE [FILE ...]");
                return Ok(());
            }
            _ => files.push(a),
        }
    }
    if files.is_empty() {
        bail!("usage: session-report [--json] FILE [FILE ...]");
    }
    let mut records = Vec::new();
    let mut skipped = 0;
    for f in &files {
        let text = std::fs::read_to_string(f).with_context(|| format!("reading {f}"))?;
        let (recs, bad) = report::parse_lines(&text);
        records.extend(recs);
        skipped += bad;
    }
    let mut r = report::summarize(&records);
    r.skipped = skipped;
    if json {
        println!("{}", serde_json::to_string_pretty(&r)?);
    } else {
        print!("{}", report::render(&r));
    }
    Ok(())
}
