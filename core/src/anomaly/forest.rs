//! Layer 3: Isolation Forest tree-walker.
//!
//! Reference port of the forest `client/rl_agent/export_forest_trees.py`
//! exports from the trained `isolation_forest.joblib`
//! (`client/rl_agent/train_isolation_forest.py`). Walks the exact trees
//! sklearn trained -- feature/threshold/children arrays plus a
//! precomputed per-leaf path-length correction `c(n)` -- rather than
//! reimplementing sklearn's path-length-correction math itself; see
//! `isolation_forest_trees.json`'s own `purpose` field for the formula.
//! Verified numerically against `contracts/isolation_forest_test_vectors.json`.
//!
//! `score_median`/`score_threshold` (the calibration constants) come from
//! `isolation_forest_calibration.json`, embedded the same way. Both files
//! are regenerated together by `train_isolation_forest.py` +
//! `export_forest_trees.py`; change one without the other and this
//! module's own parity test fails.

const TREES_JSON: &str =
    include_str!("../../../client/rl_agent/models/isolation_forest_trees.json");
const CALIBRATION_JSON: &str =
    include_str!("../../../client/rl_agent/models/isolation_forest_calibration.json");

/// Width of a Layer 3 feature vector (`LAYER3_FEATURES` in
/// `client/rl_agent/anomaly_detector.py`): out_size_mean, in_size_mean,
/// pkt_size_std, log_pkt_rate, iat_cv, upload_share, in that frozen order.
pub const FEATURE_COUNT: usize = 6;

struct Tree {
    feature: Vec<i32>,
    threshold: Vec<f64>,
    children_left: Vec<i32>,
    children_right: Vec<i32>,
    /// Precomputed `c(n_node_samples[node])` for every node; only the
    /// value at the leaf a query lands on is ever read.
    leaf_correction: Vec<f64>,
}

impl Tree {
    /// Path length for one feature vector: edges from the root, plus the
    /// landed leaf's precomputed correction. `children_left[node] == -1`
    /// marks a leaf (sklearn's `TREE_LEAF` sentinel).
    fn path_length(&self, x: &[f64; FEATURE_COUNT]) -> f64 {
        let mut node = 0usize;
        let mut depth = 0.0f64;
        loop {
            let left = self.children_left[node];
            if left == -1 {
                return depth + self.leaf_correction[node];
            }
            let go_left = x[self.feature[node] as usize] <= self.threshold[node];
            node = if go_left { left as usize } else { self.children_right[node] as usize };
            depth += 1.0;
        }
    }
}

pub struct IsolationForest {
    trees: Vec<Tree>,
    global_path_length_correction: f64,
    score_median: f64,
    score_threshold: f64,
}

impl IsolationForest {
    /// Parses the two embedded JSON artifacts once. No disk I/O: unlike
    /// Python's `IsolationForestLayer()`, there is nothing to inject for
    /// tests here -- the model is baked into the binary, so a Rust test
    /// always exercises the exact shipped forest.
    pub fn from_embedded() -> Self {
        let trees_v: serde_json::Value =
            serde_json::from_str(TREES_JSON).expect("isolation_forest_trees.json parses");
        let calibration_v: serde_json::Value = serde_json::from_str(CALIBRATION_JSON)
            .expect("isolation_forest_calibration.json parses");

        let trees = trees_v["trees"]
            .as_array()
            .expect("trees is an array")
            .iter()
            .map(|t| Tree {
                feature: arr_i32(&t["feature"]),
                threshold: arr_f64(&t["threshold"]),
                children_left: arr_i32(&t["children_left"]),
                children_right: arr_i32(&t["children_right"]),
                leaf_correction: arr_f64(&t["leaf_correction"]),
            })
            .collect();

        Self {
            trees,
            global_path_length_correction: trees_v["global_path_length_correction"]
                .as_f64()
                .expect("global_path_length_correction is a float"),
            score_median: calibration_v["score_median"].as_f64().expect("score_median is a float"),
            score_threshold: calibration_v["score_threshold"]
                .as_f64()
                .expect("score_threshold is a float"),
        }
    }

    /// Anomaly score, higher = more anomalous (matches `-model.score_samples(x)`
    /// in Python; sklearn's own convention is reversed, see
    /// `IsolationForestLayer.raw_score`). Terminates for any finite input,
    /// including values at or beyond a split threshold, since the walk is
    /// a plain `<=` comparison with no bound on `x`'s range.
    pub fn raw_score(&self, x: &[f64; FEATURE_COUNT]) -> f64 {
        let mean_path =
            self.trees.iter().map(|t| t.path_length(x)).sum::<f64>() / self.trees.len() as f64;
        2f64.powf(-mean_path / self.global_path_length_correction)
    }

    /// One tick: `None` for a too-sparse tick (mirrors
    /// `tick_features`/`LAYER3_MIN_PACKETS` upstream), `Some(features)`
    /// otherwise.
    pub fn update(&self, features: Option<[f64; FEATURE_COUNT]>) -> super::LayerResult {
        match features {
            None => super::LayerResult { anomalous: false, threat_score: 0.0, ready: false },
            Some(x) => {
                let raw = self.raw_score(&x);
                let span = self.score_threshold - self.score_median;
                let threat = ((raw - self.score_median) / (2.0 * span)).clamp(0.0, 1.0);
                super::LayerResult { anomalous: raw >= self.score_threshold, threat_score: threat, ready: true }
            }
        }
    }
}

fn arr_i32(v: &serde_json::Value) -> Vec<i32> {
    v.as_array().expect("expected a JSON array").iter().map(|x| x.as_i64().unwrap() as i32).collect()
}

fn arr_f64(v: &serde_json::Value) -> Vec<f64> {
    v.as_array().expect("expected a JSON array").iter().map(|x| x.as_f64().unwrap()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const VECTORS: &str = include_str!("../../../contracts/isolation_forest_test_vectors.json");

    fn vectors() -> Value {
        serde_json::from_str(VECTORS).expect("isolation_forest_test_vectors.json parses")
    }

    fn features_of(v: &Value) -> [f64; FEATURE_COUNT] {
        const NAMES: [&str; FEATURE_COUNT] =
            ["out_size_mean", "in_size_mean", "pkt_size_std", "log_pkt_rate", "iat_cv", "upload_share"];
        std::array::from_fn(|i| v["features"][NAMES[i]].as_f64().unwrap())
    }

    #[test]
    fn raw_score_matches_python_score_samples_for_every_vector() {
        let forest = IsolationForest::from_embedded();
        let payload = vectors();
        let tolerance = payload["tolerance"].as_f64().unwrap();
        let cases = payload["vectors"].as_array().unwrap();
        assert!(!cases.is_empty(), "no vectors in the shipped test-vector file");

        for case in cases {
            let name = case["name"].as_str().unwrap();
            let x = features_of(case);
            let got = forest.raw_score(&x);
            let want = case["expect_raw_score"].as_f64().unwrap();
            assert!(
                (got - want).abs() <= tolerance,
                "{name}: raw_score {got} vs expected {want} (tolerance {tolerance})"
            );
        }
    }

    #[test]
    fn calibrated_update_matches_python_for_every_vector() {
        let forest = IsolationForest::from_embedded();
        let payload = vectors();
        let cases = payload["vectors"].as_array().unwrap();

        for case in cases {
            let name = case["name"].as_str().unwrap();
            let x = features_of(case);
            let r = forest.update(Some(x));
            let want_score = case["expect_threat_score"].as_f64().unwrap();
            let want_anomalous = case["expect_anomalous"].as_bool().unwrap();
            assert!(
                (r.threat_score - want_score).abs() <= 1e-5,
                "{name}: threat_score {} vs expected {want_score}", r.threat_score
            );
            assert_eq!(r.anomalous, want_anomalous, "{name}: anomalous mismatch");
            assert!(r.ready, "{name}: a Some(features) tick must be ready");
        }
    }

    #[test]
    fn a_none_tick_is_not_ready_and_contributes_nothing() {
        let forest = IsolationForest::from_embedded();
        let r = forest.update(None);
        assert!(!r.ready);
        assert!(!r.anomalous);
        assert_eq!(r.threat_score, 0.0);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// The walk always reaches a leaf and returns a finite score, for
        /// any finite feature vector -- including values far outside any
        /// training range, and values that land exactly on a split
        /// threshold (handled by the plain `<=` comparison, not a special
        /// case). This is the tree-walker's core safety property: nothing
        /// about a malformed or extreme vector can make it loop, index out
        /// of bounds, or return NaN.
        #[test]
        fn raw_score_is_finite_for_any_finite_input(
            x in prop::array::uniform6(-1e6f64..1e6f64),
        ) {
            let forest = IsolationForest::from_embedded();
            let score = forest.raw_score(&x);
            prop_assert!(score.is_finite());
            prop_assert!(score > 0.0);
        }

        /// Boundary inputs: every coordinate exactly 0, exactly at a
        /// representative extreme, or NaN-adjacent (±infinity is excluded
        /// deliberately -- see `forest_handles_infinite_input` below,
        /// which checks it separately rather than folding it into this
        /// property and asserting less).
        #[test]
        fn raw_score_is_finite_at_feature_extremes(
            which in 0..FEATURE_COUNT,
            extreme in prop::sample::select(vec![0.0f64, -1e9, 1e9]),
        ) {
            let mut x = [1.0f64; FEATURE_COUNT];
            x[which] = extreme;
            let forest = IsolationForest::from_embedded();
            prop_assert!(forest.raw_score(&x).is_finite());
        }
    }

    #[test]
    fn forest_handles_infinite_input() {
        let forest = IsolationForest::from_embedded();
        let mut x = [1.0f64; FEATURE_COUNT];
        x[0] = f64::INFINITY;
        x[1] = f64::NEG_INFINITY;
        // The walk terminates (every comparison against +-inf is
        // well-defined) and the score stays in sklearn's own range: not
        // claiming it is a *meaningful* score for nonsense input, only
        // that nothing panics or diverges.
        let score = forest.raw_score(&x);
        assert!(score.is_finite() && score > 0.0);
    }
}
