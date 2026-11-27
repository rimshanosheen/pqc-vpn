//! Fuzzes `IsolationForest::raw_score` directly with arbitrary 6-float
//! vectors (libFuzzer's `arbitrary` crate happily produces NaN/Inf/extreme
//! floats from raw bytes, which is exactly the malformed-input space a
//! fixed example-list test never reaches). The tree walk is a `<=`
//! comparison with no bound on the input's range, so the invariant under
//! test is purely structural: it always reaches a leaf and returns a
//! finite, positive score, for any `[f64; 6]` whatsoever.
#![no_main]

use libfuzzer_sys::fuzz_target;
use std::sync::OnceLock;
use vpn_core::anomaly::forest::{IsolationForest, FEATURE_COUNT};

static FOREST: OnceLock<IsolationForest> = OnceLock::new();

fuzz_target!(|data: [f64; FEATURE_COUNT]| {
    let forest = FOREST.get_or_init(IsolationForest::from_embedded);
    let score = forest.raw_score(&data);
    assert!(score.is_finite() && score > 0.0, "non-finite/non-positive score for {data:?}: {score}");
});
