//! Anomaly detection pipeline (proposal section 3.3).
//!
//! Reference port of `client/rl_agent/anomaly_detector.py` (Member 3),
//! verified tick-for-tick against `contracts/anomaly_vectors.json`. The
//! Python module is the source of truth; change it there, regenerate the
//! vectors, and this file's tests fail until the port follows.
//!
//! Layer 1 (always-active Z-score baseline), Layer 2 (five rule-based
//! signatures, gated below [`LAYER2_CPU_GATE`]), Layer 3 (Isolation Forest
//! over per-tick traffic shape, gated below the stricter [`LAYER3_CPU_GATE`]
//! -- see [`forest`]), and the combiner that folds all three together are
//! implemented here. Every type takes plain numeric inputs and is
//! decoupled from platform/network sourcing, the same boundary
//! `state::StatePipeline::observe` documents for the `threat_score`
//! handoff: wiring psutil-equivalent/packet-capture sources into these
//! inputs is separate, later work.

pub mod forest;

use std::collections::VecDeque;

pub const Z_SCORE_THRESHOLD: f64 = 3.0;

fn mean(xs: &VecDeque<f64>) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn std_dev(xs: &VecDeque<f64>, m: f64) -> f64 {
    (xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64).sqrt()
}

struct MetricHistory {
    window: usize,
    values: VecDeque<f64>,
}

impl MetricHistory {
    fn new(window: usize) -> Self {
        Self { window, values: VecDeque::with_capacity(window) }
    }

    /// Z-score of `value` against the history *so far* (not including
    /// `value` itself) -- the sample doesn't inflate its own baseline.
    /// Needs at least 2 prior samples; 0.0 (not an error) below that or
    /// when the history has zero variance, mirroring Python's stance that
    /// "not enough information" and "definitely not anomalous" return the
    /// same thing here.
    fn z_score(&self, value: f64) -> f64 {
        if self.values.len() < 2 {
            return 0.0;
        }
        let m = mean(&self.values);
        let s = std_dev(&self.values, m);
        if s == 0.0 {
            return 0.0;
        }
        ((value - m) / s).abs()
    }

    fn push(&mut self, value: f64) {
        if self.values.len() == self.window {
            self.values.pop_front();
        }
        self.values.push_back(value);
    }
}

/// Shared by every layer below: the module's usual {anomalous, threat_score,
/// ready} shape (Python's `detail` is debug/test-only there and has no
/// consumer on the Rust side, so it is not ported).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerResult {
    pub anomalous: bool,
    pub threat_score: f64,
    pub ready: bool,
}

/// Layer 1: rolling mean/std baseline per metric, flags any sample whose
/// Z-score exceeds [`Z_SCORE_THRESHOLD`]. Not `ready` until `window`
/// samples have been seen.
pub struct ZScoreBaseline {
    window: usize,
    latency: MetricHistory,
    packet_rate: MetricHistory,
    packet_size: MetricHistory,
}

impl ZScoreBaseline {
    pub fn new(window: usize) -> Self {
        Self {
            window,
            latency: MetricHistory::new(window),
            packet_rate: MetricHistory::new(window),
            packet_size: MetricHistory::new(window),
        }
    }

    pub fn update(&mut self, latency: f64, packet_rate: f64, packet_size: f64) -> LayerResult {
        let z_latency = self.latency.z_score(latency);
        let z_rate = self.packet_rate.z_score(packet_rate);
        let z_size = self.packet_size.z_score(packet_size);

        self.latency.push(latency);
        self.packet_rate.push(packet_rate);
        self.packet_size.push(packet_size);

        let max_z = z_latency.max(z_rate).max(z_size);
        let anomalous = max_z > Z_SCORE_THRESHOLD;
        // Squash to [0,1]: anything at/above 2x threshold saturates to 1.0.
        let threat_score = (max_z / (Z_SCORE_THRESHOLD * 2.0)).clamp(0.0, 1.0);
        let ready = self.latency.values.len() >= self.window;

        LayerResult { anomalous, threat_score, ready }
    }
}

// ------------------------------------------------------------- Layer 2

/// Squash a rule-based metric to [0,1]: 0 at zero, saturating to 1.0 once
/// `value` reaches `ceiling`x `threshold`. Same shape [`ZScoreBaseline`]
/// uses for its Z-score.
fn ratio_score(value: f64, threshold: f64, ceiling: f64) -> f64 {
    if threshold <= 0.0 {
        return 0.0;
    }
    (value / (threshold * ceiling)).clamp(0.0, 1.0)
}

pub const PORT_SCAN_DISTINCT_PORTS: u32 = 15;
pub const PORT_SCAN_FAILURE_RATIO: f64 = 0.7;

/// Fanout + failure rate: many distinct destination ports touched in one
/// tick, most of which never complete. Per-tick, no window: `ready` is
/// always true.
#[derive(Default)]
pub struct PortScanSignature;

impl PortScanSignature {
    pub fn update(&self, distinct_ports: u32, connection_attempts: u32, failed_attempts: u32) -> LayerResult {
        let failure_ratio = if connection_attempts > 0 {
            failed_attempts as f64 / connection_attempts as f64
        } else {
            0.0
        };
        let port_score = ratio_score(distinct_ports as f64, PORT_SCAN_DISTINCT_PORTS as f64, 2.0);
        let failure_score = ratio_score(failure_ratio, PORT_SCAN_FAILURE_RATIO, 2.0);
        let anomalous = distinct_ports >= PORT_SCAN_DISTINCT_PORTS && failure_ratio >= PORT_SCAN_FAILURE_RATIO;
        LayerResult { anomalous, threat_score: port_score.min(failure_score), ready: true }
    }
}

pub const RETRANS_RATE_THRESHOLD: f64 = 0.10;
pub const RETRANS_CONSECUTIVE_TICKS: u32 = 2;

/// Sustained elevated TCP retransmission rate: a streak counter, not a
/// window of values.
pub struct RetransmissionSpikeSignature {
    consecutive_ticks: u32,
    streak: u32,
}

impl RetransmissionSpikeSignature {
    pub fn new() -> Self {
        Self { consecutive_ticks: RETRANS_CONSECUTIVE_TICKS, streak: 0 }
    }

    pub fn update(&mut self, retransmit_rate: f64) -> LayerResult {
        self.streak = if retransmit_rate >= RETRANS_RATE_THRESHOLD { self.streak + 1 } else { 0 };
        let anomalous = self.streak >= self.consecutive_ticks;
        LayerResult {
            anomalous,
            threat_score: ratio_score(retransmit_rate, RETRANS_RATE_THRESHOLD, 2.0),
            ready: true,
        }
    }
}

pub const MITM_JITTER_MS_THRESHOLD: f64 = 80.0;
pub const MITM_WINDOW_TICKS: usize = 3;
pub const MITM_CONSECUTIVE_TICKS: u32 = 2;

/// Jitter (max - min over a short window of raw latencies), not absolute
/// level -- Layer 1 already flags an absolute-latency outlier.
pub struct MitMLatencySignature {
    history: VecDeque<f64>,
    window_ticks: usize,
    consecutive_ticks: u32,
    streak: u32,
}

impl MitMLatencySignature {
    pub fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(MITM_WINDOW_TICKS),
            window_ticks: MITM_WINDOW_TICKS,
            consecutive_ticks: MITM_CONSECUTIVE_TICKS,
            streak: 0,
        }
    }

    pub fn update(&mut self, latency_ms: f64) -> LayerResult {
        if self.history.len() == self.window_ticks {
            self.history.pop_front();
        }
        self.history.push_back(latency_ms);

        let ready = self.history.len() >= self.window_ticks;
        let jitter = if ready {
            let max = self.history.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let min = self.history.iter().copied().fold(f64::INFINITY, f64::min);
            max - min
        } else {
            0.0
        };
        self.streak = if ready && jitter >= MITM_JITTER_MS_THRESHOLD { self.streak + 1 } else { 0 };
        let anomalous = self.streak >= self.consecutive_ticks;
        LayerResult { anomalous, threat_score: ratio_score(jitter, MITM_JITTER_MS_THRESHOLD, 2.0), ready }
    }
}

pub const EXFIL_UPLOAD_BPS_THRESHOLD: f64 = 2_000_000.0;
pub const EXFIL_UPLOAD_DOWNLOAD_RATIO: f64 = 3.0;
pub const EXFIL_CONSECUTIVE_TICKS: u32 = 2;

/// Upload/download asymmetry: a host pushing far more up than down is
/// atypical for ordinary consumption.
pub struct BandwidthExfiltrationSignature {
    consecutive_ticks: u32,
    streak: u32,
}

impl BandwidthExfiltrationSignature {
    pub fn new() -> Self {
        Self { consecutive_ticks: EXFIL_CONSECUTIVE_TICKS, streak: 0 }
    }

    pub fn update(&mut self, upload_bytes_per_sec: f64, download_bytes_per_sec: f64) -> LayerResult {
        let ratio = upload_bytes_per_sec / download_bytes_per_sec.max(1.0);
        let clears_line = upload_bytes_per_sec >= EXFIL_UPLOAD_BPS_THRESHOLD && ratio >= EXFIL_UPLOAD_DOWNLOAD_RATIO;
        self.streak = if clears_line { self.streak + 1 } else { 0 };
        let anomalous = self.streak >= self.consecutive_ticks;
        let threat_score = ratio_score(upload_bytes_per_sec, EXFIL_UPLOAD_BPS_THRESHOLD, 2.0)
            .min(ratio_score(ratio, EXFIL_UPLOAD_DOWNLOAD_RATIO, 2.0));
        LayerResult { anomalous, threat_score, ready: true }
    }
}

/// Edge-triggered on a DNS-server change. Known, explicit limitation
/// (ported, not fixed here): fires on ANY change, benign ones included.
pub struct DNSServerChangeSignature {
    last: Option<String>,
}

impl DNSServerChangeSignature {
    pub fn new() -> Self {
        Self { last: None }
    }

    pub fn update(&mut self, dns_server: &str) -> LayerResult {
        let ready = self.last.is_some();
        let anomalous = ready && self.last.as_deref() != Some(dns_server);
        self.last = Some(dns_server.to_string());
        LayerResult { anomalous, threat_score: if anomalous { 1.0 } else { 0.0 }, ready }
    }
}

// ------------------------------------------------------------- Layer 3 glue

pub const LAYER3_MIN_PACKETS: usize = 3;
const TICK_SECONDS: f64 = 5.0;

fn slice_mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn slice_std(xs: &[f64]) -> f64 {
    let m = slice_mean(xs);
    (xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64).sqrt()
}

/// Reduce one tick's packets to the Layer 3 feature vector (frozen order:
/// out_size_mean, in_size_mean, pkt_size_std, log_pkt_rate, iat_cv,
/// upload_share). `None` when the tick has fewer than
/// [`LAYER3_MIN_PACKETS`] packets -- too sparse for `iat_cv` to be defined.
pub fn tick_features(
    packet_sizes: &[f64],
    arrival_times: &[f64],
    outbound: &[bool],
) -> Option<[f64; forest::FEATURE_COUNT]> {
    if packet_sizes.len() < LAYER3_MIN_PACKETS {
        return None;
    }
    let mut times: Vec<f64> = arrival_times.to_vec();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let gaps: Vec<f64> = times.windows(2).map(|w| w[1] - w[0]).collect();
    let gap_mean = slice_mean(&gaps);
    let total: f64 = packet_sizes.iter().sum();

    let out_sizes: Vec<f64> = packet_sizes.iter().copied().zip(outbound.iter().copied())
        .filter(|&(_, o)| o).map(|(s, _)| s).collect();
    let in_sizes: Vec<f64> = packet_sizes.iter().copied().zip(outbound.iter().copied())
        .filter(|&(_, o)| !o).map(|(s, _)| s).collect();
    let out_sum: f64 = out_sizes.iter().sum();

    Some([
        if out_sizes.is_empty() { 0.0 } else { slice_mean(&out_sizes) },
        if in_sizes.is_empty() { 0.0 } else { slice_mean(&in_sizes) },
        slice_std(packet_sizes),
        (1.0 + packet_sizes.len() as f64 / TICK_SECONDS).log10(),
        if gap_mean > 0.0 { slice_std(&gaps) / gap_mean } else { 0.0 },
        if total > 0.0 { out_sum / total } else { 0.0 },
    ])
}

// ------------------------------------------------------------- Combiner

pub const LAYER2_CPU_GATE: f64 = 0.70;
pub const LAYER3_CPU_GATE: f64 = 0.40;

/// One tick's input to [`AnomalyCombiner::update`]. A plain struct stands
/// in for Python's keyword arguments -- every field is a plain number (or
/// slice), same boundary every type above holds to.
pub struct CombinerInput<'a> {
    pub cpu_load: f64,
    pub latency: f64,
    pub packet_rate: f64,
    pub packet_size: f64,
    pub distinct_ports: u32,
    pub connection_attempts: u32,
    pub failed_attempts: u32,
    pub retransmit_rate: f64,
    pub latency_ms: f64,
    pub upload_bytes_per_sec: f64,
    pub download_bytes_per_sec: f64,
    pub dns_server: &'a str,
    pub packet_sizes: &'a [f64],
    pub arrival_times: &'a [f64],
    pub outbound: &'a [bool],
}

/// The combiner's one-tick output. `ready` mirrors Layer 1's own; the
/// per-layer `detail` Python returns for debugging has no Rust consumer and
/// is not ported.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CombinedReading {
    pub anomalous: bool,
    pub threat_score: f64,
    pub ready: bool,
    pub layer2_active: bool,
    pub layer3_active: bool,
}

/// Owns one [`ZScoreBaseline`] (Layer 1), the five Layer 2 signatures, and
/// one [`forest::IsolationForest`] (Layer 3), and combines whichever are
/// active into one reading.
pub struct AnomalyCombiner {
    layer1: ZScoreBaseline,
    port_scan: PortScanSignature,
    retransmission: RetransmissionSpikeSignature,
    mitm_latency: MitMLatencySignature,
    exfiltration: BandwidthExfiltrationSignature,
    dns_change: DNSServerChangeSignature,
    forest: forest::IsolationForest,
}

impl AnomalyCombiner {
    pub fn new(window: usize) -> Self {
        Self {
            layer1: ZScoreBaseline::new(window),
            port_scan: PortScanSignature,
            retransmission: RetransmissionSpikeSignature::new(),
            mitm_latency: MitMLatencySignature::new(),
            exfiltration: BandwidthExfiltrationSignature::new(),
            dns_change: DNSServerChangeSignature::new(),
            forest: forest::IsolationForest::from_embedded(),
        }
    }

    /// One tick. Layer 1 always runs. The five Layer 2 signatures run only
    /// when `cpu_load < LAYER2_CPU_GATE`; Layer 3 only when
    /// `cpu_load < LAYER3_CPU_GATE` (the stricter gate) -- when gated off, a
    /// layer is simply not called this tick, so a transient CPU spike
    /// doesn't corrupt its streak/window state. `threat_score` is the max
    /// (not average) over every active layer whose own `ready` is true, so
    /// a not-yet-ready layer contributes nothing rather than a masking 0.0.
    pub fn update(&mut self, input: CombinerInput) -> CombinedReading {
        let layer1 = self.layer1.update(input.latency, input.packet_rate, input.packet_size);
        let layer2_active = input.cpu_load < LAYER2_CPU_GATE;
        let layer3_active = input.cpu_load < LAYER3_CPU_GATE;

        let mut contributing: Vec<LayerResult> = vec![layer1];

        if layer2_active {
            contributing.push(self.port_scan.update(
                input.distinct_ports,
                input.connection_attempts,
                input.failed_attempts,
            ));
            contributing.push(self.retransmission.update(input.retransmit_rate));
            contributing.push(self.mitm_latency.update(input.latency_ms));
            contributing.push(
                self.exfiltration.update(input.upload_bytes_per_sec, input.download_bytes_per_sec),
            );
            contributing.push(self.dns_change.update(input.dns_server));
        }

        if layer3_active {
            let features = tick_features(input.packet_sizes, input.arrival_times, input.outbound);
            contributing.push(self.forest.update(features));
        }

        let ready_layers = contributing.iter().filter(|r| r.ready);
        let threat_score =
            ready_layers.clone().map(|r| r.threat_score).fold(0.0f64, f64::max);
        let anomalous = ready_layers.clone().any(|r| r.anomalous);

        CombinedReading { anomalous, threat_score, ready: layer1.ready, layer2_active, layer3_active }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const VECTORS: &str = include_str!("../../../contracts/anomaly_vectors.json");

    fn vectors() -> Value {
        serde_json::from_str(VECTORS).expect("anomaly_vectors.json parses")
    }

    fn f64_vec(v: &Value) -> Vec<f64> {
        v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
    }

    fn bool_vec(v: &Value) -> Vec<bool> {
        v.as_array().unwrap().iter().map(|x| x.as_bool().unwrap()).collect()
    }

    #[test]
    fn replays_every_contract_vector_tick_for_tick() {
        let payload = vectors();
        let cases = payload["cases"].as_array().unwrap();
        assert!(!cases.is_empty(), "no cases in the shipped vector file");

        for case in cases {
            let name = case["name"].as_str().unwrap();
            let mut combiner = AnomalyCombiner::new(50);

            for (i, tick) in case["ticks"].as_array().unwrap().iter().enumerate() {
                let input = &tick["input"];
                let packet_sizes = f64_vec(&input["packet_sizes"]);
                let arrival_times = f64_vec(&input["arrival_times"]);
                let outbound = bool_vec(&input["outbound"]);

                let r = combiner.update(CombinerInput {
                    cpu_load: input["cpu_load"].as_f64().unwrap(),
                    latency: input["latency"].as_f64().unwrap(),
                    packet_rate: input["packet_rate"].as_f64().unwrap(),
                    packet_size: input["packet_size"].as_f64().unwrap(),
                    distinct_ports: input["distinct_ports"].as_u64().unwrap() as u32,
                    connection_attempts: input["connection_attempts"].as_u64().unwrap() as u32,
                    failed_attempts: input["failed_attempts"].as_u64().unwrap() as u32,
                    retransmit_rate: input["retransmit_rate"].as_f64().unwrap(),
                    latency_ms: input["latency_ms"].as_f64().unwrap(),
                    upload_bytes_per_sec: input["upload_bytes_per_sec"].as_f64().unwrap(),
                    download_bytes_per_sec: input["download_bytes_per_sec"].as_f64().unwrap(),
                    dns_server: input["dns_server"].as_str().unwrap(),
                    packet_sizes: &packet_sizes,
                    arrival_times: &arrival_times,
                    outbound: &outbound,
                });

                let want = &tick["expect"];
                let at = format!("{name} tick {i}");
                assert_eq!(r.ready, want["ready"].as_bool().unwrap(), "{at}: ready");
                assert_eq!(r.layer2_active, want["layer2_active"].as_bool().unwrap(), "{at}: layer2_active");
                assert_eq!(r.layer3_active, want["layer3_active"].as_bool().unwrap(), "{at}: layer3_active");
                assert_eq!(r.anomalous, want["anomalous"].as_bool().unwrap(), "{at}: anomalous");
                let want_score = want["threat_score"].as_f64().unwrap();
                assert!(
                    (r.threat_score - want_score).abs() <= 1e-5,
                    "{at}: threat_score {} vs expected {want_score}", r.threat_score
                );
            }
        }
    }

    #[test]
    fn constants_match_the_contract() {
        let c = &vectors()["constants"];
        assert_eq!(c["layer2_cpu_gate"].as_f64(), Some(LAYER2_CPU_GATE));
        assert_eq!(c["layer3_cpu_gate"].as_f64(), Some(LAYER3_CPU_GATE));
    }

    #[test]
    fn layer3_gate_is_stricter_than_layer2s() {
        assert!(LAYER3_CPU_GATE < LAYER2_CPU_GATE);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Gating a layer off never increases the reported score relative
        /// to the same tick with every gate open: the combiner's score is
        /// a max over active+ready layers, so removing a layer from that
        /// set can only drop the max, never raise it.
        #[test]
        fn gating_a_layer_off_never_increases_the_score(
            latency in 0.0f64..10_000.0,
            packet_rate in 0.0f64..5_000.0,
            packet_size in 0.0f64..1500.0,
        ) {
            let mut gated = AnomalyCombiner::new(50);
            let mut open = AnomalyCombiner::new(50);
            let empty_sizes: [f64; 0] = [];
            let empty_times: [f64; 0] = [];
            let empty_outbound: [bool; 0] = [];

            for cpu_load in [0.9, 0.9, 0.9] {
                let input_gated = CombinerInput {
                    cpu_load, latency, packet_rate, packet_size,
                    distinct_ports: 0, connection_attempts: 0, failed_attempts: 0,
                    retransmit_rate: 0.0, latency_ms: latency,
                    upload_bytes_per_sec: 0.0, download_bytes_per_sec: 1.0,
                    dns_server: "1.1.1.1",
                    packet_sizes: &empty_sizes, arrival_times: &empty_times, outbound: &empty_outbound,
                };
                let r_gated = gated.update(input_gated);

                let input_open = CombinerInput {
                    cpu_load: 0.0, latency, packet_rate, packet_size,
                    distinct_ports: 0, connection_attempts: 0, failed_attempts: 0,
                    retransmit_rate: 0.0, latency_ms: latency,
                    upload_bytes_per_sec: 0.0, download_bytes_per_sec: 1.0,
                    dns_server: "1.1.1.1",
                    packet_sizes: &empty_sizes, arrival_times: &empty_times, outbound: &empty_outbound,
                };
                let r_open = open.update(input_open);

                prop_assert!(r_gated.threat_score <= r_open.threat_score + 1e-9);
            }
        }

        /// threat_score is always in [0,1], across any cpu_load and any
        /// Layer 1 metrics -- the squash functions all clamp, and the
        /// combiner only ever takes a max over clamped values.
        #[test]
        fn threat_score_is_always_in_unit_range(
            cpu_load in 0.0f64..1.0,
            latency in 0.0f64..1e5,
            packet_rate in 0.0f64..1e5,
            packet_size in 0.0f64..1e5,
        ) {
            let mut combiner = AnomalyCombiner::new(10);
            let empty_sizes: [f64; 0] = [];
            let empty_times: [f64; 0] = [];
            let empty_outbound: [bool; 0] = [];
            let r = combiner.update(CombinerInput {
                cpu_load, latency, packet_rate, packet_size,
                distinct_ports: 0, connection_attempts: 0, failed_attempts: 0,
                retransmit_rate: 0.0, latency_ms: latency,
                upload_bytes_per_sec: 0.0, download_bytes_per_sec: 1.0,
                dns_server: "1.1.1.1",
                packet_sizes: &empty_sizes, arrival_times: &empty_times, outbound: &empty_outbound,
            });
            prop_assert!((0.0..=1.0).contains(&r.threat_score));
        }

        /// tick_features never panics or returns a non-finite feature,
        /// across randomized packet counts/sizes/times, including
        /// simultaneous-arrival ticks (zero gaps) and all-one-direction
        /// ticks.
        #[test]
        fn tick_features_is_finite_whenever_it_is_some(
            sizes in prop::collection::vec(0.0f64..2000.0, 3..50),
            seed in 0u64..10_000,
        ) {
            let times: Vec<f64> = (0..sizes.len()).map(|i| i as f64 + (seed as f64 * 0.0)).collect();
            let outbound: Vec<bool> = (0..sizes.len()).map(|i| (i + seed as usize) % 2 == 0).collect();
            if let Some(f) = tick_features(&sizes, &times, &outbound) {
                for v in f {
                    prop_assert!(v.is_finite());
                }
            }
        }
    }
}
