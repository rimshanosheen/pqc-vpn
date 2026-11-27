//! Live readings → Member 3's anomaly combiner → the `THREAT` dimension
//! (Member 1, Week 8).
//!
//! `core::anomaly` takes plain numbers on purpose (its module docs: "wiring
//! psutil-equivalent/packet-capture sources into these inputs is separate,
//! later work"). This is that wiring: it maps one tick's
//! [`DeviceReadings`] and [`TrafficSample`] onto a [`CombinerInput`], in the
//! units the Python reference and `contracts/anomaly_vectors.json` use
//! (latency in ms, packet rate in packets/s, sizes in bytes).

use crate::anomaly::{AnomalyCombiner, CombinedReading, CombinerInput};
use crate::state::{DeviceReadings, TrafficSample};

/// Layer 1's rolling window, in ticks: the Python default
/// (`AnomalyCombiner(window=50)`) and what the contract vectors replay at.
/// At 5 s per tick, Layer 1 is ready about 4 minutes into a session.
pub const LAYER1_WINDOW: usize = 50;

/// Latency fed to the anomaly layers when the probe target was
/// unreachable: the desktop ping's own 1 s timeout. The state vector
/// saturates unreachable to 1.0 instead (`StatePipeline`); the anomaly
/// layers need a number in ms, and "at least the timeout" is the honest one.
pub const UNREACHABLE_LATENCY_MS: f64 = 1000.0;

/// Placeholder DNS identity before any DNS server has ever been read.
const DNS_UNKNOWN: &str = "unknown";

/// Owns the per-session anomaly state.
pub struct ThreatMonitor {
    combiner: AnomalyCombiner,
    /// Last DNS server successfully read. A failed read reuses it, so a
    /// transient read failure is never mistaken for a DNS-server change
    /// (Layer 2's DNS signature fires on *any* change).
    last_dns: Option<String>,
}

impl Default for ThreatMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl ThreatMonitor {
    pub fn new() -> Self {
        Self { combiner: AnomalyCombiner::new(LAYER1_WINDOW), last_dns: None }
    }

    /// One tick of anomaly detection.
    pub fn update(&mut self, readings: &DeviceReadings, sample: &TrafficSample) -> CombinedReading {
        if let Some(dns) = &sample.dns_server {
            self.last_dns = Some(dns.clone());
        }
        let dns = self.last_dns.as_deref().unwrap_or(DNS_UNKNOWN);

        let latency_ms = readings
            .latency
            .map(|d| d.as_secs_f64() * 1000.0)
            .unwrap_or(UNREACHABLE_LATENCY_MS);

        let sizes: Vec<f64> = sample.packets.iter().map(|p| p.size).collect();
        let times: Vec<f64> = sample.packets.iter().map(|p| p.time).collect();
        let outbound: Vec<bool> = sample.packets.iter().map(|p| p.outbound).collect();

        self.combiner.update(CombinerInput {
            cpu_load: readings.cpu_load as f64,
            latency: latency_ms,
            packet_rate: sample.packet_rate,
            packet_size: sample.mean_packet_size,
            distinct_ports: sample.distinct_dest_ports,
            connection_attempts: sample.connection_attempts,
            failed_attempts: sample.failed_attempts,
            retransmit_rate: sample.retransmit_rate,
            latency_ms,
            upload_bytes_per_sec: sample.upload_bytes_per_sec,
            download_bytes_per_sec: sample.download_bytes_per_sec,
            dns_server: dns,
            packet_sizes: &sizes,
            arrival_times: &times,
            outbound: &outbound,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::anomaly::{LAYER2_CPU_GATE, LAYER3_CPU_GATE};
    use crate::state::{ConnectionType, PacketRecord};

    fn readings(cpu: f32) -> DeviceReadings {
        DeviceReadings {
            cpu_load: cpu,
            ram_available_fraction: 0.8,
            latency: Some(Duration::from_millis(20)),
            upload_rate_bytes_per_sec: 1000.0,
            connection_type: ConnectionType::Wired,
        }
    }

    fn quiet() -> TrafficSample {
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

    #[test]
    fn quiet_traffic_scores_zero() {
        let mut m = ThreatMonitor::new();
        for _ in 0..5 {
            let r = m.update(&readings(0.2), &quiet());
            // Graded, not binary: Layer 2's exfiltration check gives a
            // small score to any upload, so quiet is near zero, not zero.
            assert!(r.threat_score < 0.05, "{}", r.threat_score);
            assert!(!r.anomalous);
        }
    }

    #[test]
    fn port_scan_inputs_reach_layer_2() {
        let mut m = ThreatMonitor::new();
        let scan = TrafficSample { distinct_dest_ports: 40, connection_attempts: 40, failed_attempts: 38, ..quiet() };
        let r = m.update(&readings(0.2), &scan);
        assert!(r.layer2_active);
        assert!(r.anomalous);
        assert!(r.threat_score > 0.5);
    }

    #[test]
    fn layer_2_is_gated_off_under_cpu_load() {
        let mut m = ThreatMonitor::new();
        let spike = TrafficSample { retransmit_rate: 0.5, ..quiet() };
        let r = m.update(&readings(LAYER2_CPU_GATE as f32 + 0.05), &spike);
        assert!(!r.layer2_active);
        // Layer 1 isn't ready yet, so nothing contributes.
        assert_eq!(r.threat_score, 0.0);
    }

    #[test]
    fn a_failed_dns_read_is_not_a_dns_change() {
        let mut m = ThreatMonitor::new();
        m.update(&readings(0.2), &quiet());
        let r = m.update(&readings(0.2), &TrafficSample { dns_server: None, ..quiet() });
        assert!(!r.anomalous);
        let r = m.update(&readings(0.2), &quiet());
        assert!(!r.anomalous, "same server after a failed read is no change");
        let r = m.update(&readings(0.2), &TrafficSample { dns_server: Some("9.9.9.9".into()), ..quiet() });
        assert!(r.anomalous, "a real change still fires");
    }

    #[test]
    fn packets_feed_layer_3_when_present() {
        let mut m = ThreatMonitor::new();
        let packets: Vec<PacketRecord> = (0..20)
            .map(|i| PacketRecord { size: 400.0 + i as f64, time: i as f64 * 0.2, outbound: i % 2 == 0 })
            .collect();
        let r = m.update(&readings(LAYER3_CPU_GATE as f32 - 0.1), &TrafficSample { packets, ..quiet() });
        assert!(r.layer3_active);
        assert!((0.0..=1.0).contains(&r.threat_score));
    }
}
