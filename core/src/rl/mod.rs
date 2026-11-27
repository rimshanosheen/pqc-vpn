//! RL decision layer: turns the policy's output into client actions.
//!
//! - [`policy`] — ONNX inference over `ppo_vpn_agent.onnx` via `ort`
//!   (Member 1, Week 7), verified against `contracts/policy_test_vectors.json`.
//! - [`gate`] — the decision gate (Member 3, Week 7), ported from
//!   `client/rl_agent/decision_gate.py` and verified against
//!   `contracts/decision_gate_vectors.json`.
//! - [`agent`] — the two joined: policy → softmax → gate → one [`Action`]
//!   per tick (Member 1, Week 7). Carrying the action out is `core::session`.

pub mod agent;
pub mod gate;
pub mod policy;

pub use agent::{Action, Agent, AgentStep};
pub use gate::{softmax, Decision, DecisionGate, PolicyAction, ACTION_COUNT, REKEY_ACTION_INDEX};
pub use policy::{Policy, PolicyError};
