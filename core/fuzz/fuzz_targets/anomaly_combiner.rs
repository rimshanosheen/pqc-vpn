//! Fuzzes `AnomalyCombiner::update` with arbitrary numeric inputs,
//! including NaN/Inf metrics and mismatched-length packet arrays (the
//! packet_sizes/arrival_times/outbound slices need not agree in length --
//! `tick_features` only guards `packet_sizes.len()`, matching the Python
//! source of truth exactly). The invariant under test: no input panics,
//! and the reported threat_score always stays in [0,1].
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use vpn_core::anomaly::{AnomalyCombiner, CombinerInput};

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    cpu_load: f64,
    latency: f64,
    packet_rate: f64,
    packet_size: f64,
    distinct_ports: u32,
    connection_attempts: u32,
    failed_attempts: u32,
    retransmit_rate: f64,
    latency_ms: f64,
    upload_bytes_per_sec: f64,
    download_bytes_per_sec: f64,
    dns_server: String,
    // Capped so one fuzz input can't force an enormous allocation; the
    // invariants under test don't depend on array length.
    packet_sizes: Vec<f64>,
    arrival_times: Vec<f64>,
    outbound: Vec<bool>,
}

fuzz_target!(|input: FuzzInput| {
    if input.packet_sizes.len() > 1000 || input.arrival_times.len() > 1000 || input.outbound.len() > 1000 {
        return;
    }
    let mut combiner = AnomalyCombiner::new(50);
    let r = combiner.update(CombinerInput {
        cpu_load: input.cpu_load,
        latency: input.latency,
        packet_rate: input.packet_rate,
        packet_size: input.packet_size,
        distinct_ports: input.distinct_ports,
        connection_attempts: input.connection_attempts,
        failed_attempts: input.failed_attempts,
        retransmit_rate: input.retransmit_rate,
        latency_ms: input.latency_ms,
        upload_bytes_per_sec: input.upload_bytes_per_sec,
        download_bytes_per_sec: input.download_bytes_per_sec,
        dns_server: &input.dns_server,
        packet_sizes: &input.packet_sizes,
        arrival_times: &input.arrival_times,
        outbound: &input.outbound,
    });
    assert!(
        r.threat_score.is_finite() && (0.0..=1.0).contains(&r.threat_score),
        "threat_score {} out of [0,1] for {input:?}", r.threat_score
    );
});
