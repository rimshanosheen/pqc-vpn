//! ONNX inference over Member 3's trained policy (Member 1, Week 7).
//!
//! Loads `client/rl_agent/models/ppo_vpn_agent.onnx` through the `ort`
//! crate (ONNX Runtime) and turns one state vector into the four raw action
//! logits. `contracts/INTEGRATION.md` Step 2 is the spec:
//!
//! - one input `state: float32[batch, 7]`, one output
//!   `action_logits: float32[batch, 4]`; on-device the batch is always 1;
//! - read the action count from the model's output shape, never assume it;
//! - verify the port against `contracts/policy_test_vectors.json`: logits
//!   within the file's tolerance, argmax exactly equal.
//!
//! The model bytes are embedded at compile time, the same way
//! `anomaly::forest` embeds its trees, so the binary can never pair a
//! policy with the wrong contract files at runtime. The `manifest_sha256_*`
//! test checks the embedded bytes against `contracts/manifest.json`.

use std::sync::OnceLock;

use ort::session::Session;
use ort::value::{TensorRef, ValueType};

use crate::state::STATE_DIM;

use super::gate::ACTION_COUNT;

/// The shipped policy, embedded at build time.
pub const EMBEDDED_MODEL: &[u8] =
    include_bytes!("../../../client/rl_agent/models/ppo_vpn_agent.onnx");

/// Input and output tensor names fixed by `contracts/INTEGRATION.md`.
pub const INPUT_NAME: &str = "state";
pub const OUTPUT_NAME: &str = "action_logits";

/// Why the policy could not be loaded or run.
#[derive(Debug)]
pub enum PolicyError {
    /// ONNX Runtime failed to load the model or run it.
    Runtime(String),
    /// The model's output width is not what this client is built for.
    ///
    /// `INTEGRATION.md` says to read the count from the model, so that
    /// enabling a fifth action is a re-export rather than a code change.
    /// The decision gate's arrays are sized for [`ACTION_COUNT`], so a
    /// different width is refused at load time, loudly, instead of having
    /// action indices silently mean something else.
    ActionCount { model: usize, client: usize },
    /// The model's input or output does not have the contract's shape.
    Shape(String),
}

impl std::fmt::Display for PolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PolicyError::Runtime(m) => write!(f, "onnx runtime: {m}"),
            PolicyError::ActionCount { model, client } => write!(
                f,
                "policy has {model} actions but this client handles {client}; \
                 re-copy the .onnx, policy_test_vectors.json and manifest.json together"
            ),
            PolicyError::Shape(m) => write!(f, "policy shape: {m}"),
        }
    }
}

impl std::error::Error for PolicyError {}

impl From<ort::Error> for PolicyError {
    fn from(e: ort::Error) -> Self {
        PolicyError::Runtime(e.to_string())
    }
}

/// ONNX Runtime allows one environment per process, so every [`Policy`]
/// (one per VPN session, plus any in tests running in parallel) shares it.
/// Built on first use; a failure is cached, since retrying cannot help.
fn environment() -> Result<&'static ort::environment::Environment, PolicyError> {
    static ENV: OnceLock<Result<ort::environment::Environment, String>> = OnceLock::new();
    ENV.get_or_init(|| ort::init().with_name("vpn_core").build().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| PolicyError::Runtime(e.clone()))
}

/// The trained policy, ready to run.
pub struct Policy {
    session: Session,
    action_count: usize,
}

impl std::fmt::Debug for Policy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Policy").field("action_count", &self.action_count).finish()
    }
}

impl Policy {
    /// Loads the policy embedded in this binary.
    pub fn from_embedded() -> Result<Self, PolicyError> {
        Self::from_bytes(EMBEDDED_MODEL)
    }

    /// Loads a policy from ONNX bytes and checks it against the contract.
    pub fn from_bytes(model: &[u8]) -> Result<Self, PolicyError> {
        // One intra-op thread: the model is a 7→64→64→4 MLP run once every
        // 5 s. A thread pool would cost more than the inference.
        let session = Session::builder(environment()?)?
            .with_intra_threads(1)
            .map_err(ort::Error::from)?
            .commit_from_memory(model)?;

        if session.inputs().len() != 1 || session.inputs()[0].name() != INPUT_NAME {
            return Err(PolicyError::Shape(format!(
                "expected exactly one input named {INPUT_NAME:?}"
            )));
        }

        let output = session
            .outputs()
            .iter()
            .find(|o| o.name() == OUTPUT_NAME)
            .ok_or_else(|| PolicyError::Shape(format!("no output named {OUTPUT_NAME:?}")))?;

        let action_count = match output.dtype() {
            ValueType::Tensor { shape, .. } => match shape.last() {
                Some(&n) if n > 0 => n as usize,
                other => {
                    return Err(PolicyError::Shape(format!(
                        "output's last dimension is {other:?}, expected a fixed action count"
                    )));
                }
            },
            other => return Err(PolicyError::Shape(format!("output is {other:?}, not a tensor"))),
        };

        if action_count != ACTION_COUNT {
            return Err(PolicyError::ActionCount { model: action_count, client: ACTION_COUNT });
        }

        Ok(Self { session, action_count })
    }

    /// Number of actions, as read from the model's output shape.
    pub fn action_count(&self) -> usize {
        self.action_count
    }

    /// Runs the policy on one state vector and returns the raw logits.
    ///
    /// `state` must already be normalized and clipped to `[0, 1]` in the
    /// frozen index order, which is exactly what
    /// [`StatePipeline`](crate::state::StatePipeline) produces.
    pub fn logits(&mut self, state: &[f32; STATE_DIM]) -> Result<[f32; ACTION_COUNT], PolicyError> {
        let input = TensorRef::from_array_view(([1usize, STATE_DIM], &state[..]))?;
        let outputs = self.session.run(ort::inputs![INPUT_NAME => input])?;
        let (_, data) = outputs[OUTPUT_NAME].try_extract_tensor::<f32>()?;

        <[f32; ACTION_COUNT]>::try_from(data).map_err(|_| {
            PolicyError::Shape(format!("expected {ACTION_COUNT} logits, got {}", data.len()))
        })
    }
}

/// Index of the largest value in `xs[range]`. Ties go to the lowest index,
/// matching `numpy.argmax` (and the gate's own argmax).
pub fn argmax(xs: &[f32], range: std::ops::Range<usize>) -> usize {
    let mut best = range.start;
    for i in range {
        if xs[i] > xs[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use sha2::{Digest, Sha256};

    const VECTORS: &str = include_str!("../../../contracts/policy_test_vectors.json");
    const MANIFEST: &str = include_str!("../../../contracts/manifest.json");
    const REGISTRY: &str = include_str!("../../../contracts/algo_registry.json");

    /// `INTEGRATION.md`: "A stale `ppo_vpn_agent.onnx` paired with a current
    /// `algo_registry.json` does not throw — it silently returns action
    /// indices that mean something different". So check the embedded model
    /// is the one the manifest describes.
    #[test]
    fn manifest_sha256_matches_the_embedded_model() {
        let manifest: Value = serde_json::from_str(MANIFEST).unwrap();
        let entry = &manifest["artifacts"]["client/rl_agent/models/ppo_vpn_agent.onnx"];
        let want = entry["sha256"].as_str().expect("manifest lists the model");
        let got: String = Sha256::digest(EMBEDDED_MODEL).iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(got, want, "embedded .onnx is not the one contracts/manifest.json describes");
        assert_eq!(entry["bytes"].as_u64(), Some(EMBEDDED_MODEL.len() as u64));
    }

    #[test]
    fn action_count_is_read_from_the_model_and_matches_the_registry() {
        let policy = Policy::from_embedded().expect("policy loads");
        let registry: Value = serde_json::from_str(REGISTRY).unwrap();
        let vectors: Value = serde_json::from_str(VECTORS).unwrap();
        assert_eq!(policy.action_count(), ACTION_COUNT);
        assert_eq!(
            vectors["action_names"].as_array().unwrap().len(),
            policy.action_count(),
        );
        // Every action index the model can emit has a registry entry.
        let text = registry.to_string();
        for i in 0..policy.action_count() {
            assert!(text.contains(&format!("\"action_index\":{i}")), "registry has no action_index {i}");
        }
    }

    /// `INTEGRATION.md` Step 2's parity check: every one of the 63 vectors,
    /// logits within tolerance, argmax exactly equal.
    #[test]
    fn matches_every_python_policy_vector() {
        let payload: Value = serde_json::from_str(VECTORS).unwrap();
        let tolerance = payload["tolerance"].as_f64().unwrap() as f32;
        let vectors = payload["vectors"].as_array().unwrap();
        assert!(!vectors.is_empty());

        let mut policy = Policy::from_embedded().unwrap();
        let mut worst = 0.0f32;

        for (i, v) in vectors.iter().enumerate() {
            let state: Vec<f32> =
                v["state"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
            let state: [f32; STATE_DIM] = state.try_into().unwrap();
            let want: Vec<f32> = v["action_logits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect();

            let got = policy.logits(&state).unwrap();

            for (g, w) in got.iter().zip(&want) {
                let diff = (g - w).abs();
                worst = worst.max(diff);
                assert!(diff <= tolerance, "vector {i}: logit {g} vs {w} (|diff| {diff} > {tolerance})");
            }
            let want_index = v["action_index"].as_u64().unwrap() as usize;
            assert_eq!(argmax(&got, 0..ACTION_COUNT), want_index, "vector {i}: argmax differs");
        }
        eprintln!("policy parity: worst |logit diff| {worst:e} over {} vectors", vectors.len());
    }

    #[test]
    fn wrong_bytes_are_an_error_not_a_panic() {
        assert!(matches!(Policy::from_bytes(b"not an onnx model"), Err(PolicyError::Runtime(_))));
    }
}
