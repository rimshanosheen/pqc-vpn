//! The RL agent as the client runs it (Member 1, Week 7): ONNX policy →
//! softmax → decision gate → one concrete [`Action`] per tick.
//!
//! `contracts/INTEGRATION.md` Steps 2–4 in one place. The agent decides;
//! carrying the decision out over the network is `core::session`'s job, so
//! this module stays free of I/O and is testable on fixed states.

use crate::crypto::MlKemLevel;
use crate::state::STATE_DIM;

use super::gate::{softmax, Decision, DecisionGate, PolicyAction, ACTION_COUNT, REKEY_ACTION_INDEX};
use super::policy::{argmax, Policy, PolicyError};

/// What the client must do this tick, derived from a gate [`Decision`].
///
/// This is where `rl/`'s output becomes `crypto/`'s algorithm selection:
/// the level carried here is the one the next handshake runs at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Keep the current session and keys.
    Stay,
    /// Fresh handshake (`ClientHello`) at this level: the gate confirmed a
    /// different algorithm.
    Renegotiate(MlKemLevel),
    /// `RekeyRequest` at this level: rotate keys. Equal to or stronger than
    /// what was in force (`REKEY_ESCALATES`), never weaker.
    Rekey(MlKemLevel),
}

impl Action {
    /// Maps a gate decision to an action. The gate guarantees
    /// `change_algorithm` and `rekey` are never both true.
    pub fn from_decision(d: &Decision) -> Self {
        if d.change_algorithm {
            Action::Renegotiate(d.in_force)
        } else if d.rekey {
            Action::Rekey(d.in_force)
        } else {
            Action::Stay
        }
    }
}

/// Everything one agent tick produced, for the session log and dashboard.
#[derive(Debug, Clone, Copy)]
pub struct AgentStep {
    pub state: [f32; STATE_DIM],
    pub logits: [f32; ACTION_COUNT],
    pub probs: [f32; ACTION_COUNT],
    /// The policy's raw argmax, before the gate debounced it.
    pub raw_choice: PolicyAction,
    pub decision: Decision,
    pub action: Action,
}

/// Policy plus the per-session decision gate.
#[derive(Debug)]
pub struct Agent {
    policy: Policy,
    gate: DecisionGate,
}

impl Agent {
    /// Picks the algorithm to connect with from the current state, and
    /// builds the agent around it.
    ///
    /// The gate is constructed "with the algorithm negotiated at connect
    /// time" (`INTEGRATION.md` Step 3); this is how that first algorithm
    /// is chosen. Rekey-now is not a connect algorithm, so it takes the
    /// policy's best-ranked KEM, the same rule `REKEY_ESCALATES` uses.
    pub fn for_connect(mut policy: Policy, state: &[f32; STATE_DIM]) -> Result<(Self, MlKemLevel), PolicyError> {
        let logits = policy.logits(state)?;
        let level = best_kem(&logits);
        Ok((Self::new(policy, level), level))
    }

    /// An agent whose session is already running at `in_force`.
    pub fn new(policy: Policy, in_force: MlKemLevel) -> Self {
        Self { policy, gate: DecisionGate::new(in_force) }
    }

    /// The algorithm the gate considers in force.
    pub fn in_force(&self) -> MlKemLevel {
        self.gate.in_force()
    }

    /// One tick: run the policy on `state` and debounce it through the gate.
    pub fn step(&mut self, state: &[f32; STATE_DIM]) -> Result<AgentStep, PolicyError> {
        let logits = self.policy.logits(state)?;
        let probs = softmax(&logits);
        let raw_choice = PolicyAction::from_index(argmax(&logits, 0..ACTION_COUNT))
            .expect("argmax is always below ACTION_COUNT");
        let decision = self.gate.update(&probs);
        Ok(AgentStep {
            state: *state,
            logits,
            probs,
            raw_choice,
            decision,
            action: Action::from_decision(&decision),
        })
    }

    /// The client could not carry out the last action; `actual` is what is
    /// really in force. See [`DecisionGate::reset_in_force`].
    pub fn action_failed(&mut self, actual: MlKemLevel) {
        self.gate.reset_in_force(actual);
    }
}

/// The policy's top-ranked KEM, ignoring rekey-now.
pub fn best_kem(logits: &[f32; ACTION_COUNT]) -> MlKemLevel {
    let idx = argmax(logits, 0..REKEY_ACTION_INDEX);
    match PolicyAction::from_index(idx) {
        Some(PolicyAction::Kem(level)) => level,
        _ => unreachable!("indices below REKEY_ACTION_INDEX are KEMs"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::gate::{CONFIRM_TICKS, REASON_AGREES};

    fn policy() -> Policy {
        Policy::from_embedded().expect("embedded policy loads")
    }

    // First vector in contracts/policy_test_vectors.json: the model picks
    // ML-KEM-512 with a large margin (logits 25.3 / 6.9 / -18.9 / -27.0).
    const LOW_NEED: [f32; STATE_DIM] = [0.625096, 0.897214, 0.775686, 0.225207, 0.300166, 0.873553, 0.005265];

    #[test]
    fn action_from_decision_covers_all_three_cases() {
        let base = Decision { in_force: MlKemLevel::MlKem768, change_algorithm: false, rekey: false, reason: "" };
        assert_eq!(Action::from_decision(&base), Action::Stay);
        assert_eq!(
            Action::from_decision(&Decision { change_algorithm: true, ..base }),
            Action::Renegotiate(MlKemLevel::MlKem768)
        );
        assert_eq!(Action::from_decision(&Decision { rekey: true, ..base }), Action::Rekey(MlKemLevel::MlKem768));
    }

    #[test]
    fn connect_algorithm_comes_from_the_policy() {
        let (agent, level) = Agent::for_connect(policy(), &LOW_NEED).unwrap();
        assert_eq!(level, MlKemLevel::MlKem512);
        assert_eq!(agent.in_force(), MlKemLevel::MlKem512);
    }

    #[test]
    fn steady_state_agrees_and_stays() {
        let (mut agent, _) = Agent::for_connect(policy(), &LOW_NEED).unwrap();
        let step = agent.step(&LOW_NEED).unwrap();
        assert_eq!(step.raw_choice, PolicyAction::Kem(MlKemLevel::MlKem512));
        assert_eq!(step.decision.reason, REASON_AGREES);
        assert_eq!(step.action, Action::Stay);
        assert!((step.probs.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    /// A session wrongly running at 1024 on a state the policy clearly
    /// wants 512 for: the gate holds for CONFIRM_TICKS-1 ticks, then the
    /// agent asks for a renegotiation down to 512.
    #[test]
    fn a_confirmed_challenger_becomes_a_renegotiation() {
        let mut agent = Agent::new(policy(), MlKemLevel::MlKem1024);
        for _ in 1..CONFIRM_TICKS {
            assert_eq!(agent.step(&LOW_NEED).unwrap().action, Action::Stay);
        }
        let step = agent.step(&LOW_NEED).unwrap();
        assert_eq!(step.action, Action::Renegotiate(MlKemLevel::MlKem512));
        assert_eq!(agent.in_force(), MlKemLevel::MlKem512);
    }

    #[test]
    fn a_failed_action_restores_the_real_algorithm() {
        let mut agent = Agent::new(policy(), MlKemLevel::MlKem1024);
        for _ in 0..CONFIRM_TICKS {
            agent.step(&LOW_NEED).unwrap();
        }
        assert_eq!(agent.in_force(), MlKemLevel::MlKem512);
        agent.action_failed(MlKemLevel::MlKem1024);
        assert_eq!(agent.in_force(), MlKemLevel::MlKem1024);
        // The streak was cleared, so it takes a full confirm run again.
        assert_eq!(agent.step(&LOW_NEED).unwrap().action, Action::Stay);
    }

    #[test]
    fn best_kem_ignores_rekey() {
        assert_eq!(best_kem(&[0.0, 1.0, 0.5, 9.0]), MlKemLevel::MlKem768);
        assert_eq!(best_kem(&[0.0, 1.0, 2.0, 9.0]), MlKemLevel::MlKem1024);
    }
}
