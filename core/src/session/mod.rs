//! The client control loop (Member 1, Weeks 7–8): one tick every
//! [`TICK_SECONDS`](crate::rl::gate::TICK_SECONDS) seconds.
//!
//! ```text
//!   DeviceState ──► StatePipeline ──► rl::Agent ──► Action ──► protocol ──► TunnelHandle
//!                     (7-dim)        (policy+gate)  (Stay /     (handshake    (bring_up /
//!                                                    Renegotiate / or rekey)    rotate_psk)
//!                                                    Rekey)
//! ```
//!
//! This is where `rl/`'s output becomes `crypto/`'s algorithm selection:
//! the level in each [`Action`] is the level the next handshake runs at.
//!
//! `VpnSession` owns no clock and no thread. The caller (the desktop app,
//! or a test) decides when to call [`VpnSession::tick`] and reports how
//! much time passed, which keeps the loop testable without waiting on wall
//! time, the same stance `StatePipeline::advance` takes.

use std::fmt;
use std::time::Duration;

use crate::crypto::MlKemLevel;
use crate::protocol::{self, Established, ProtocolError};
use crate::rl::{Action, Agent, AgentStep, Policy, PolicyError};
use crate::state::{DeviceState, StatePipeline, TunnelError, TunnelHandle};

/// Where the handshake server is and how to trust it.
#[derive(Clone)]
pub struct ServerEndpoint {
    /// `host:port` of the PQC handshake responder (TCP).
    pub addr: String,
    /// The server's 1952-byte ML-DSA-65 key, pinned in the client
    /// (`server/PROTOCOL.md` §5.2).
    pub pinned_verifying_key: Vec<u8>,
    /// This client's WireGuard public key: the server's peer identity.
    pub client_wg_pubkey: [u8; 32],
    /// Per-read/write and connect timeout for each handshake.
    pub timeout: Duration,
}

impl fmt::Debug for ServerEndpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ServerEndpoint").field("addr", &self.addr).field("timeout", &self.timeout).finish()
    }
}

impl ServerEndpoint {
    fn handshake(&self, algo: MlKemLevel) -> Result<Established, ProtocolError> {
        protocol::handshake(
            self.addr.as_str(),
            algo,
            self.client_wg_pubkey,
            &self.pinned_verifying_key,
            self.timeout,
        )
    }

    fn rekey(&self, session_id: [u8; 16], algo: MlKemLevel) -> Result<Established, ProtocolError> {
        protocol::rekey(
            self.addr.as_str(),
            session_id,
            algo,
            self.client_wg_pubkey,
            &self.pinned_verifying_key,
            self.timeout,
        )
    }
}

/// Why a session could not be started, or a tick could not run at all.
///
/// A failed handshake *during* a tick is not an error here: the session is
/// still up on its old keys, so it is reported in [`Outcome::Failed`] and
/// the loop carries on.
#[derive(Debug)]
pub enum SessionError {
    Policy(PolicyError),
    Protocol(ProtocolError),
    Tunnel(TunnelError),
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SessionError::Policy(e) => write!(f, "policy: {e}"),
            SessionError::Protocol(e) => write!(f, "handshake: {e}"),
            SessionError::Tunnel(e) => write!(f, "tunnel: {e}"),
        }
    }
}

impl std::error::Error for SessionError {}

impl From<PolicyError> for SessionError {
    fn from(e: PolicyError) -> Self {
        SessionError::Policy(e)
    }
}
impl From<ProtocolError> for SessionError {
    fn from(e: ProtocolError) -> Self {
        SessionError::Protocol(e)
    }
}
impl From<TunnelError> for SessionError {
    fn from(e: TunnelError) -> Self {
        SessionError::Tunnel(e)
    }
}

/// What carrying out a tick's [`Action`] actually did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// The action was [`Action::Stay`].
    Unchanged,
    /// A fresh handshake completed at this level.
    Renegotiated(MlKemLevel),
    /// A rekey completed at this level.
    Rekeyed(MlKemLevel),
    /// The handshake or the tunnel update failed. The session stays on its
    /// previous keys and algorithm.
    Failed { attempted: Action, error: String },
}

/// One tick's full record, for the session log and the dashboard.
#[derive(Debug, Clone)]
pub struct TickReport {
    pub step: AgentStep,
    pub outcome: Outcome,
    /// The algorithm actually in use after this tick.
    pub in_force: MlKemLevel,
}

/// A connected client session driven by the RL agent.
pub struct VpnSession<D: DeviceState, T: TunnelHandle> {
    endpoint: ServerEndpoint,
    pipeline: StatePipeline<D>,
    tunnel: T,
    agent: Agent,
    session_id: [u8; 16],
    in_force: MlKemLevel,
    assigned_ip: std::net::Ipv4Addr,
}

impl<D: DeviceState, T: TunnelHandle> VpnSession<D, T> {
    /// Reads the device once, lets the policy pick the connect algorithm,
    /// runs the first handshake, and brings the tunnel up.
    ///
    /// `threat_score` is the anomaly score for this first reading.
    pub fn connect(
        endpoint: ServerEndpoint,
        device: D,
        tunnel: T,
        policy: Policy,
        threat_score: f32,
    ) -> Result<Self, SessionError> {
        let pipeline = StatePipeline::new(device);
        let state = pipeline.observe(threat_score);
        let (agent, level) = Agent::for_connect(policy, &state)?;

        let established = endpoint.handshake(level)?;
        tunnel.bring_up(&established.tunnel)?;

        Ok(Self {
            endpoint,
            pipeline,
            tunnel,
            agent,
            session_id: established.session_id,
            in_force: established.algo,
            assigned_ip: established.tunnel.assigned_ip,
        })
    }

    /// The algorithm in use.
    pub fn in_force(&self) -> MlKemLevel {
        self.in_force
    }

    /// The server-assigned session id (changes on a renegotiation, not on a
    /// rekey).
    pub fn session_id(&self) -> [u8; 16] {
        self.session_id
    }

    /// Borrows the tunnel handle.
    pub fn tunnel(&self) -> &T {
        &self.tunnel
    }

    /// One control-loop tick. `elapsed` is the time since the previous
    /// tick (or since connect); `threat_score` is this tick's anomaly score.
    ///
    /// Only a policy failure is an `Err`: a network or tunnel failure is
    /// reported in the returned [`TickReport::outcome`].
    pub fn tick(&mut self, elapsed: Duration, threat_score: f32) -> Result<TickReport, SessionError> {
        self.pipeline.advance(elapsed);
        let state = self.pipeline.observe(threat_score);
        let step = self.agent.step(&state)?;
        let outcome = self.carry_out(step.action);
        Ok(TickReport { step, outcome, in_force: self.in_force })
    }

    fn carry_out(&mut self, action: Action) -> Outcome {
        let result = match action {
            Action::Stay => return Outcome::Unchanged,
            Action::Renegotiate(level) => self.endpoint.handshake(level),
            Action::Rekey(level) => self.endpoint.rekey(self.session_id, level),
        };

        let established = match result {
            Ok(e) => e,
            Err(e) => {
                // Nothing changed on either side; tell the gate.
                self.agent.action_failed(self.in_force);
                return Outcome::Failed { attempted: action, error: e.to_string() };
            }
        };

        // The server has switched to the new keys whatever happens next, so
        // record what it now holds before touching the local tunnel.
        self.session_id = established.session_id;
        self.in_force = established.algo;
        // Both a renegotiation and a rekey rotate the key material, which is
        // what TIME_SINCE_REKEY measures (`contracts/state_vector.json`:
        // "wall-clock since last key rotation").
        self.pipeline.mark_rekey();

        // Same peer, same address: hot-swap the PSK. A different address
        // (only if the server dropped and re-added the peer) needs the
        // interface reconfigured.
        let tunnel_result = if established.tunnel.assigned_ip == self.assigned_ip {
            self.tunnel.rotate_psk(&established.tunnel.psk)
        } else {
            self.tunnel.bring_up(&established.tunnel)
        };
        self.assigned_ip = established.tunnel.assigned_ip;
        if let Err(e) = tunnel_result {
            return Outcome::Failed {
                attempted: action,
                error: format!("server switched keys but the local tunnel was not updated: {e}"),
            };
        }

        match action {
            Action::Renegotiate(_) => Outcome::Renegotiated(established.algo),
            Action::Rekey(_) => Outcome::Rekeyed(established.algo),
            Action::Stay => unreachable!(),
        }
    }

    /// Tears the tunnel down and ends the session.
    pub fn disconnect(self) -> Result<(), TunnelError> {
        self.tunnel.tear_down()
    }
}

#[cfg(test)]
mod tests;
