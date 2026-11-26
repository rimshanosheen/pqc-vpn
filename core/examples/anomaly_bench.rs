//! Local CPU-gate sanity check for Week 8 (Member 3): a tight loop of
//! synthetic ticks spanning benign/attack/mimicry shapes through
//! `AnomalyCombiner` (Layers 1-3 together), printing an aggregate so the
//! optimizer can't eliminate the loop, plus wall-clock per tick.
//!
//! Run under `samply` (or `cargo flamegraph` if `samply` hits a permission
//! wall) with the `profiling` profile for readable symbols:
//!
//!   cargo build --profile profiling --example anomaly_bench -p vpn_core
//!   samply record ./target/profiling/examples/anomaly_bench
//!
//! Explicit caveat, not hidden: this runs on Apple Silicon/macOS, not the
//! deployment droplet's Linux/x86 1vCPU -- a directional sanity check on
//! whether Layer 3's cost is in the right ballpark for a <40% CPU gate to
//! make sense at all, not a deployment-architecture validation.
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use vpn_core::anomaly::{AnomalyCombiner, CombinerInput};

const TICKS: usize = 1_000_000;

/// One of a few traffic shapes, cycled through so the loop isn't scoring
/// the same tick a million times (which an optimizer, or a cache, could
/// exploit in ways a profile of real varied traffic would not see).
fn make_tick(rng: &mut StdRng, shape: usize) -> ([f64; 32], [f64; 32], [bool; 32], f64) {
    let (base_size, p_out, cpu_load): (f64, f64, f64) = match shape % 4 {
        0 => (120.0, 0.5, 0.1),   // idle-ish, both gates open
        1 => (900.0, 0.35, 0.1),  // browsing-ish, both gates open
        2 => (58.0, 0.6, 0.1),    // scan-shaped, both gates open
        _ => (1450.0, 0.8, 0.1),  // exfiltration-shaped, both gates open
    };
    let mut sizes = [0.0f64; 32];
    let mut times = [0.0f64; 32];
    let mut outbound = [false; 32];
    let mut t = 0.0;
    for i in 0..32 {
        sizes[i] = (base_size + rng.gen_range(-20.0..20.0)).clamp(40.0, 1500.0);
        t += rng.gen_range(0.01..0.2);
        times[i] = t;
        outbound[i] = rng.gen_bool(p_out);
    }
    (sizes, times, outbound, cpu_load)
}

fn main() {
    let mut rng = StdRng::seed_from_u64(0);
    let mut combiner = AnomalyCombiner::new(50);

    let mut anomalous_count = 0u64;
    let mut score_sum = 0.0f64;

    let start = Instant::now();
    for i in 0..TICKS {
        let (sizes, times, outbound, cpu_load) = make_tick(&mut rng, i);
        let r = combiner.update(CombinerInput {
            cpu_load,
            latency: 20.0 + rng.gen_range(-5.0..5.0),
            packet_rate: 50.0,
            packet_size: 500.0,
            distinct_ports: 2,
            connection_attempts: 5,
            failed_attempts: 0,
            retransmit_rate: 0.0,
            latency_ms: 20.0,
            upload_bytes_per_sec: 1000.0,
            download_bytes_per_sec: 5000.0,
            dns_server: "1.1.1.1",
            packet_sizes: &sizes,
            arrival_times: &times,
            outbound: &outbound,
        });
        if r.anomalous {
            anomalous_count += 1;
        }
        score_sum += r.threat_score;
    }
    let elapsed = start.elapsed();

    println!("{TICKS} ticks in {elapsed:?} ({:.2} us/tick)", elapsed.as_secs_f64() * 1e6 / TICKS as f64);
    println!("anomalous_count={anomalous_count} score_sum={score_sum:.3}");
}
