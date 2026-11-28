//! Summaries of the Week 9 session log, for demos and Member 3's evaluation.
//!
//! [`summarize`] folds JSON-Lines records (see [`crate::events`]) into a
//! [`Report`]; `session-report` prints it as text or JSON.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Default, Clone, PartialEq, Serialize)]
pub struct Stats {
    pub n: usize,
    pub mean: f64,
    pub p50: f64,
    pub p95: f64,
    pub max: f64,
}

impl Stats {
    pub fn of(mut xs: Vec<f64>) -> Stats {
        if xs.is_empty() {
            return Stats::default();
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pick = |q: f64| xs[((xs.len() - 1) as f64 * q).round() as usize];
        let r2 = |x: f64| (x * 100.0).round() / 100.0;
        Stats {
            n: xs.len(),
            mean: r2(xs.iter().sum::<f64>() / xs.len() as f64),
            p50: r2(pick(0.5)),
            p95: r2(pick(0.95)),
            max: r2(*xs.last().unwrap()),
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Serialize)]
pub struct Report {
    /// Records read, and records skipped as unparseable.
    pub records: usize,
    pub skipped: usize,
    pub first_ts_ms: Option<u64>,
    pub last_ts_ms: Option<u64>,
    pub server_starts: usize,
    pub events: BTreeMap<String, usize>,
    pub distinct_peers: usize,
    /// Full handshakes per algorithm.
    pub handshakes_by_algo: BTreeMap<String, usize>,
    /// Server-side handshake time per algorithm, ms (full handshakes and rekeys).
    pub handshake_ms_by_algo: BTreeMap<String, Stats>,
    pub reconnects: usize,
    pub rekeys: usize,
    /// `"ML-KEM-768 -> ML-KEM-1024"` → count.
    pub rekey_transitions: BTreeMap<String, usize>,
    pub escalations: usize,
    /// `"0x05 rekey may not downgrade…"` → count.
    pub rejected: BTreeMap<String, usize>,
    pub refused: BTreeMap<String, usize>,
    pub aborted: usize,
    /// Completed tunnel sessions (`tunnel_down` events): seconds up.
    pub tunnel_sessions: Stats,
    pub tunnel_rx_bytes: u64,
    pub tunnel_tx_bytes: u64,
    /// Peers whose last liveness event was `tunnel_up`.
    pub tunnels_up_at_end: usize,
    pub evicted: BTreeMap<String, usize>,
}

fn s(v: &Value, k: &str) -> String {
    v.get(k).and_then(Value::as_str).unwrap_or("?").to_string()
}

/// Parse JSON Lines; blank lines are ignored, bad lines counted as skipped.
pub fn parse_lines(text: &str) -> (Vec<Value>, usize) {
    let mut out = Vec::new();
    let mut bad = 0;
    for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
        match serde_json::from_str::<Value>(line) {
            Ok(v) if v.get("event").is_some() => out.push(v),
            _ => bad += 1,
        }
    }
    (out, bad)
}

pub fn summarize(records: &[Value]) -> Report {
    let mut r = Report { records: records.len(), ..Default::default() };
    let mut peers = BTreeSet::new();
    let mut hs_ms: HashMap<String, Vec<f64>> = HashMap::new();
    let mut up_secs = Vec::new();
    let mut live: HashMap<String, bool> = HashMap::new();

    for v in records {
        let ev = s(v, "event");
        *r.events.entry(ev.clone()).or_default() += 1;
        if let Some(ts) = v.get("ts_ms").and_then(Value::as_u64) {
            r.first_ts_ms = Some(r.first_ts_ms.map_or(ts, |f| f.min(ts)));
            r.last_ts_ms = Some(r.last_ts_ms.map_or(ts, |l| l.max(ts)));
        }
        if let Some(p) = v.get("peer").and_then(Value::as_str) {
            peers.insert(p.to_string());
        }
        let ms = v.get("handshake_ms").and_then(Value::as_f64);
        match ev.as_str() {
            "server_start" => r.server_starts += 1,
            "handshake" => {
                let algo = s(v, "algo");
                *r.handshakes_by_algo.entry(algo.clone()).or_default() += 1;
                if v.get("new_peer") == Some(&Value::Bool(false)) {
                    r.reconnects += 1;
                }
                if let Some(ms) = ms {
                    hs_ms.entry(algo).or_default().push(ms);
                }
            }
            "rekey" => {
                r.rekeys += 1;
                let (from, to) = (s(v, "from_algo"), s(v, "algo"));
                *r.rekey_transitions.entry(format!("{from} -> {to}")).or_default() += 1;
                if v.get("escalated") == Some(&Value::Bool(true)) {
                    r.escalations += 1;
                }
                if let Some(ms) = ms {
                    hs_ms.entry(to).or_default().push(ms);
                }
            }
            "rejected" => {
                let code = v.get("code").and_then(Value::as_u64).unwrap_or(0);
                *r.rejected.entry(format!("0x{code:02x} {}", s(v, "reason"))).or_default() += 1;
            }
            "refused" => *r.refused.entry(s(v, "reason")).or_default() += 1,
            "aborted" => r.aborted += 1,
            "tunnel_up" => {
                live.insert(s(v, "peer"), true);
            }
            "tunnel_down" => {
                live.insert(s(v, "peer"), false);
                if let Some(up) = v.get("up_s").and_then(Value::as_f64) {
                    up_secs.push(up);
                }
                r.tunnel_rx_bytes += v.get("rx_bytes").and_then(Value::as_u64).unwrap_or(0);
                r.tunnel_tx_bytes += v.get("tx_bytes").and_then(Value::as_u64).unwrap_or(0);
            }
            "evicted" => *r.evicted.entry(s(v, "reason")).or_default() += 1,
            _ => {}
        }
    }
    r.distinct_peers = peers.len();
    r.handshake_ms_by_algo = hs_ms.into_iter().map(|(k, v)| (k, Stats::of(v))).collect();
    r.tunnel_sessions = Stats::of(up_secs);
    r.tunnels_up_at_end = live.values().filter(|&&up| up).count();
    r
}

/// Human-readable rendering of a [`Report`].
pub fn render(r: &Report) -> String {
    use std::fmt::Write;
    let mut o = String::new();
    let span = match (r.first_ts_ms, r.last_ts_ms) {
        (Some(a), Some(b)) => format!("{:.1} h", (b - a) as f64 / 3_600_000.0),
        _ => "-".into(),
    };
    let _ = writeln!(o, "records        : {} ({} unparseable), spanning {span}, {} server start(s)", r.records, r.skipped, r.server_starts);
    let _ = writeln!(o, "distinct peers : {}", r.distinct_peers);
    let _ = writeln!(o, "handshakes     : {} ({} reconnects of a known peer)", r.handshakes_by_algo.values().sum::<usize>(), r.reconnects);
    for (algo, n) in &r.handshakes_by_algo {
        let _ = writeln!(o, "  {algo:<12} {n}");
    }
    let _ = writeln!(o, "handshake time (server side, ms)");
    for (algo, st) in &r.handshake_ms_by_algo {
        let _ = writeln!(o, "  {algo:<12} n={:<4} mean {:>8.2}  p50 {:>8.2}  p95 {:>8.2}  max {:>8.2}", st.n, st.mean, st.p50, st.p95, st.max);
    }
    let _ = writeln!(o, "rekeys         : {} ({} escalations)", r.rekeys, r.escalations);
    for (t, n) in &r.rekey_transitions {
        let _ = writeln!(o, "  {t:<28} {n}");
    }
    let _ = writeln!(
        o,
        "tunnels        : {} completed sessions (up mean {:.0}s, p50 {:.0}s, max {:.0}s), {} up at end of log",
        r.tunnel_sessions.n, r.tunnel_sessions.mean, r.tunnel_sessions.p50, r.tunnel_sessions.max, r.tunnels_up_at_end
    );
    let _ = writeln!(o, "  traffic in completed sessions: rx {} B, tx {} B", r.tunnel_rx_bytes, r.tunnel_tx_bytes);
    let mut section = |title: &str, m: &BTreeMap<String, usize>| {
        let _ = writeln!(o, "{title:<15}: {}", m.values().sum::<usize>());
        for (k, n) in m {
            let _ = writeln!(o, "  {k:<52} {n}");
        }
    };
    section("rejected", &r.rejected);
    section("refused (guard)", &r.refused);
    section("evicted", &r.evicted);
    let _ = writeln!(o, "aborted        : {}", r.aborted);
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOG: &str = r#"
{"v":1,"ts_ms":1000,"event":"server_start","version":"0.1.0","psk_install":"dry-run","max_peers":64,"idle_timeout_s":0,"replay_window_s":120,"rate_burst":20,"rate_per_min":30,"max_conns":64}
{"v":1,"ts_ms":2000,"event":"handshake","session_id":"aa","peer":"P1","assigned_ip":"10.8.0.2","algo":"ML-KEM-768","src":"1.2.3.4","new_peer":true,"handshake_ms":10.0}
{"v":1,"ts_ms":3000,"event":"handshake","session_id":"bb","peer":"P2","assigned_ip":"10.8.0.3","algo":"ML-KEM-512","src":"1.2.3.4","new_peer":true,"handshake_ms":4.0}
{"v":1,"ts_ms":3500,"event":"tunnel_up","peer":"P1","session_id":"aa","assigned_ip":"10.8.0.2","algo":"ML-KEM-768"}
{"v":1,"ts_ms":4000,"event":"rekey","session_id":"aa","peer":"P1","assigned_ip":"10.8.0.2","from_algo":"ML-KEM-768","algo":"ML-KEM-1024","escalated":true,"rekeys":1,"src":"1.2.3.4","handshake_ms":20.0}
{"v":1,"ts_ms":4500,"event":"rejected","src":"1.2.3.4","stage":"registry","code":5,"reason":"rekey may not downgrade the algorithm","peer":"P1","algo":"ML-KEM-512","rekey":true}
not json
{"v":1,"ts_ms":5000,"event":"refused","src":"9.9.9.9","code":7,"reason":"rate limited: too many handshakes from this address"}
{"v":1,"ts_ms":6000,"event":"handshake","session_id":"cc","peer":"P2","assigned_ip":"10.8.0.3","algo":"ML-KEM-512","src":"1.2.3.4","new_peer":false,"handshake_ms":6.0}
{"v":1,"ts_ms":9000,"event":"tunnel_down","peer":"P1","session_id":"aa","assigned_ip":"10.8.0.2","idle_s":180,"up_s":300,"rx_bytes":1000,"tx_bytes":500}
{"v":1,"ts_ms":9500,"event":"tunnel_up","peer":"P2","session_id":"cc","assigned_ip":"10.8.0.3","algo":"ML-KEM-512"}
{"v":1,"ts_ms":9900,"event":"aborted","src":"1.2.3.4","error":"io: timed out"}
"#;

    #[test]
    fn summarizes_a_known_log() {
        let (recs, bad) = parse_lines(LOG);
        assert_eq!(bad, 1);
        let mut r = summarize(&recs);
        r.skipped = bad;
        assert_eq!(r.records, 11);
        assert_eq!(r.server_starts, 1);
        assert_eq!(r.distinct_peers, 2);
        assert_eq!(r.handshakes_by_algo["ML-KEM-512"], 2);
        assert_eq!(r.handshakes_by_algo["ML-KEM-768"], 1);
        assert_eq!(r.reconnects, 1);
        assert_eq!(r.rekeys, 1);
        assert_eq!(r.escalations, 1);
        assert_eq!(r.rekey_transitions["ML-KEM-768 -> ML-KEM-1024"], 1);
        assert_eq!(r.handshake_ms_by_algo["ML-KEM-512"].n, 2);
        assert_eq!(r.handshake_ms_by_algo["ML-KEM-512"].mean, 5.0);
        assert_eq!(r.handshake_ms_by_algo["ML-KEM-1024"].n, 1, "rekey time counted under its new algo");
        assert_eq!(r.rejected.values().sum::<usize>(), 1);
        assert!(r.rejected.keys().next().unwrap().starts_with("0x05 "));
        assert_eq!(r.refused.values().sum::<usize>(), 1);
        assert_eq!(r.aborted, 1);
        assert_eq!(r.tunnel_sessions.n, 1);
        assert_eq!(r.tunnel_sessions.max, 300.0);
        assert_eq!(r.tunnel_rx_bytes, 1000);
        assert_eq!(r.tunnels_up_at_end, 1, "P1 went down, P2 is still up");
        assert_eq!((r.first_ts_ms, r.last_ts_ms), (Some(1000), Some(9900)));
        let text = render(&r);
        assert!(text.contains("ML-KEM-768 -> ML-KEM-1024"));
    }

    #[test]
    fn stats_percentiles() {
        let st = Stats::of((1..=100).map(f64::from).collect());
        assert_eq!((st.n, st.p50, st.p95, st.max), (100, 51.0, 95.0, 100.0));
        assert_eq!(Stats::of(vec![]), Stats::default());
    }
}
