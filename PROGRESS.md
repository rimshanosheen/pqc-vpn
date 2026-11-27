# RL-PQC-VPN — Progress Report

Status against the project proposal (`RL_PQC_VPN_Proposal.pdf`). This
reflects what is actually implemented and passing tests as of this
submission, organized by phase rather than by date.

## Phase 1 — Core Cryptographic Layer (complete)

Implements proposal section 3.1 (hybrid PQC key exchange) and 3.2's
authentication requirement.

| File | What it does |
|---|---|
| `client/vpn_daemon/hybrid_kem.py` | Hybrid X25519 + ML-KEM-768 key exchange. Shared secret = SHA-256(x25519_secret \|\| mlkem_secret), so the session stays secure even if one of the two algorithms is later broken. |
| `client/vpn_daemon/auth.py` | ML-DSA-65 (FIPS 204) server authentication — server signs its ML-KEM public key, client verifies before accepting it, closing the MitM substitution gap. |
| `client/vpn_daemon/key_store.py` | In-memory key store with explicit `wipe()`/`wipe_all()` that zeroes key bytes before deallocation — no private key ever touches disk. |
| `client/vpn_daemon/algo_registry.py` | The four-action registry (ML-KEM-512/768/1024, rekey-now) the RL agent selects from. HQC-256 is defined but disabled pending NIST finalization, per the proposal's algorithm table. |

**Tests:** `tests/test_phase1.py` — 10/10 passing. Covers ML-KEM-512/768/1024
roundtrips, ML-DSA-65 sign/verify, the full hybrid KEM handshake, and
environment sanity checks (PyTorch MPS backend, psutil, Gymnasium,
Stable-Baselines3).

## Phase 2 — RL Decision Engine (initial implementation)

Implements the MDP formulation from proposal section 7: state space =
device/network conditions, action space = NIST algorithm choice or
rekey, reward = security strength earned vs. computational cost paid.

| File | What it does |
|---|---|
| `client/rl_agent/vpn_env.py` | Custom Gymnasium environment (`VPNEnv`) — the 7-dim state space and 4-action space wired directly to `algo_registry.ACTIVE_ACTIONS`, so the registry stays the single source of truth. Validated against Gymnasium's `check_env`. |
| `client/rl_agent/state_observer.py` | Live state reader — real `psutil` CPU/RAM/traffic readings, an actual ICMP ping for latency, and interface-name heuristics for connection type. This is the deployment-time counterpart to `vpn_env.py`'s simulator. |
| `client/rl_agent/anomaly_detector.py` | Layer 1 of the anomaly pipeline (proposal 3.3) — rolling Z-score baseline over latency/packet-rate/packet-size, always active. |
| `client/rl_agent/train.py` | Trains PPO (Stable-Baselines3) on `VPNEnv`. |
| `client/rl_agent/models/ppo_vpn_agent.zip` | Saved model from an initial training run (50k timesteps). |

**Architecture check against the proposal:** the proposal specifies a PPO
policy with two 64-neuron hidden layers, ~4,932 parameters, under 20KB on
disk. The policy-only network here (the part actually needed for on-device
inference — the value network is training-only) comes out to exactly
**4,932 parameters** (≈19.7KB in fp32), confirming the 64×64 architecture
matches spec. `tests/test_phase2.py::test_ppo_policy_param_count_matches_proposal_spec`
pins this.

**Tests:** `tests/test_phase2.py` — 9/9 passing. Covers the env against the
Gymnasium API contract, reset/step shape and bounds checks, episode
truncation, the state observer's output range, rekey-timer reset, the
anomaly detector flagging an injected outlier, and the trained model
loading and producing a valid action.

**Combined: 19/19 tests passing** (`pytest tests/ -v`).

**Live demo:** `demo.py` chains the two phases together — real device state
(CPU, RAM, a live ping, a live Z-score anomaly check) feeds the trained
agent, which picks an algorithm, which then runs a real, timed hybrid
handshake. Two of the seven state dimensions (connection type, session age)
are staged per scenario and printed as `[STAGED]`, since a short demo can't
actually hop onto cellular or run a 45-minute session — everything else is
live. Run with `python demo.py`.

## Reward-shaping bugs found and fixed while building the demo

Worth documenting because they're the kind of thing that only shows up once
you actually test the trained policy against varied states, not just check
that training runs without crashing:

1. **First reward draft was symmetric** (`1 - |algo.security - need|`).
   Checked the trained policy against 2,000 random states and found it
   picked ML-KEM-512 100% of the time, including at maximum threat — a
   completely degenerate, state-blind policy. Root cause: with a uniform
   state distribution, the average security need lands near 0.5, and 0.7
   (ML-KEM-512's security level) is the closest available constant to that
   average — so the symmetric reward taught the agent to minimize distance
   to the *average* case rather than adapt to the *actual* case. Fixed by
   making the reward asymmetric: heavily penalize under-provisioning
   (`shortfall = max(0, need - algo.security)`, weighted 3x) and only
   lightly penalize over-provisioning (via cost). This matches how security
   actually works — falling short is a real risk, using more than strictly
   necessary just costs efficiency.

2. **`rekey-now`'s reward was on a different scale** than the three
   algorithm-choice actions (max ~2.0 vs. max ~1.0), so it dominated
   whenever threat was merely nonzero, and the agent never learned to pick
   ML-KEM-768 or ML-KEM-1024 at all. Rescaled it to the same range and tied
   it to the proposal's actual stated justification for rekeying (stale
   session keys sitting in RAM — `time_since_rekey`), not threat alone.

**Current state, verified two ways:**
- *Offline, exhaustively*: brute-forced the reward function over a
  6-value grid across all 5 relevant state dimensions (7,776 combinations).
  All four actions are the analytically optimal choice somewhere — ML-KEM-512
  in ~86%, ML-KEM-768 in ~11%, rekey-now in ~1%, ML-KEM-1024 in ~1.3% of the
  grid — so the reward function itself is sound; the rarer actions just have
  narrow optimal regions.

  **Correction (Week 2):** those percentages are properties of the *grid*,
  not of the state space, and the grid overstates the rare actions. A
  product grid places 1/6 of its mass at exactly 1.0 in every dimension,
  which massively over-represents the corner where threat, connection type
  and session age are simultaneously maximal — precisely where ML-KEM-1024
  wins. Measured against the continuous uniform distribution the environment
  actually samples from, the true figures are ML-KEM-512 **90.5%**,
  ML-KEM-768 **8.9%**, ML-KEM-1024 **0.39%**, rekey-now **0.14%**. So
  ML-KEM-1024's optimal region is ~3.4x smaller than previously recorded and
  rekey-now's is ~7x smaller. The grid numbers reproduce exactly
  (86.47 / 11.32 / 1.31 / 0.90) — they were not wrong, they were answering a
  different question. This matters because it is the correct denominator for
  judging whether the agent is under-selecting an action.
- *On the actual trained policy*: bucketing 4,000 random states by
  computed security need shows a clear, monotonic trend — ML-KEM-512
  dominant at low need, ML-KEM-768 share rising through the middle, rekey-now
  dominant at high need. The ML-KEM-768 boundary is still narrow enough that
  it can flip to rekey-now under realistic background CPU load (30-60%,
  i.e. a normal dev laptop) rather than holding a wide margin — a 50k-step
  run on a small network is not enough to carve out a robust boundary for an
  action whose optimal region is only ~11% of the state space. More
  training steps, reward curriculum, or explicit exploration bonuses for
  under-visited actions are the next thing to try here — not something to
  paper over with a longer run picked because it happened to look better on
  one random seed.

## Important caveat on the RL training run

The 50k-timestep run bundled here is a **smoke run** proving the pipeline
end-to-end (env → PPO → saved model → reloaded and queried) — it is not a
converged, evaluated policy. The proposal's Expected Outcomes call for the
trained agent to be benchmarked against three baselines (classical
WireGuard, static ML-KEM-768, rule-based policy) across defined scenarios.
That evaluation hasn't been run yet and no performance claims are being
made about this model.

The training environment (`VPNEnv`) is also a **simulator** with
hand-designed reward weights and state-transition dynamics — reasonable
first-pass choices, not fit to any measured traffic or threat data. That's
flagged directly in the file's docstring as the first thing to revisit
once real traces are available.

## Fixed along the way

- `torch` was broken in the venv (missing `libtorch_cpu.dylib`, blocking
  2 of the 10 Phase 1 tests) — reinstalled clean.
- `.gitignore` had a comment stating model files under
  `client/rl_agent/models/` should be tracked, but the actual `*.zip` /
  `**/*.zip` rules were still silently excluding them. Added a negation
  rule so the trained model can actually be committed.
- Added `requirements.txt` (wasn't present) with pinned versions from the
  working venv; noted that `liboqs-python` isn't on PyPI and has to be
  built from source per the OQS project's instructions.

## Not yet built (per the proposal's remaining scope)

- Anomaly detection Layers 2 (rule-based signatures, CPU-gated <70%) and
  3 (Isolation Forest, CPU-gated <40%).
- WireGuard PSK injection (`wg set`) tying the hybrid KEM output into an
  actual tunnel.
- FastAPI daemon + WebSocket state pushes (section 3.4).
- Tauri/React desktop application and dashboard.
- Baseline comparison and evaluation (latency overhead, CPU usage,
  security level vs. the three defined baselines).
- Windows support (explicitly phase 2 in the proposal's own timeline).

## Week 1 (Member 3 track, 2026-08-05) — quantifying the decision-boundary weak point

Per `TEAM_TIMELINE_PROPOSAL.md`, Member 3's Week 1 task is to "start
addressing the known weak point" from Section 6 above — not to fix it
yet (that's Week 2's entropy-bonus/longer-training work), but to turn
"we noticed this in the demo" into actual numbers. New script:
`client/rl_agent/analyze_boundary.py` (`python -m
client.rl_agent.analyze_boundary`). It reads the existing trained model
only — no retraining, no changes to `vpn_env.py`.

**Method:** fixed a synthetic state deliberately inside ML-KEM-768's
narrow analytically-optimal region (moderate threat=0.5, cellular
connection, mid session age=0.5, ram_avail=0.7), then swept `CPU_LOAD`
and read the policy's actual action *probabilities* (via
`model.policy.get_distribution(...).distribution.probs`), not just the
argmax — so the margin between the top and runner-up action is a real
number, not a guess.

**Finding 1 — the ML-KEM-768 window is real but thin.** In this
scenario, ML-KEM-768 is the top action only for `cpu ∈ [0.32, 0.46]`
across the sampled grid, flipping away at `cpu=0.48` — so the window is
14 percentage points wide measured to the last winning sample, or 16
measured to the flip point (the sweep steps in 0.02 increments through
this range, so the true edge sits somewhere between). The margin over
the runner-up peaks at just 0.030 (probability mass) around `cpu=0.38`
and shrinks to 0.006 and 0.004 at the two edges of the window. Below `cpu≈0.32` the
policy prefers `rekey-now` instead (with a much more decisive margin —
up to 0.164 at `cpu=0`); above `cpu≈0.48` it drops to ML-KEM-512. A
±0.05 CPU perturbation from ordinary background load is enough to flip
the decision at either edge — consistent with what `demo.py` already
had to work around by staging CPU/RAM per scenario.

**Finding 2 — ML-KEM-1024 is more unreachable than previously
characterized.** The brute-force reward-grid analysis above found
ML-KEM-1024 analytically optimal in ~1.3% of the grid — a statement about
the reward function, and one the Week 2 correction above revises down to
0.39% of the actual state space. Checking the *trained
policy* itself against 4,000 random states: ML-KEM-1024 is never the
argmax choice — **0/4000 (0.00%)** — and its probability mass never
exceeds **0.1576** anywhere in that sample (mean 0.095), while the
other three actions each reach 0.6–0.79 somewhere. The network hasn't
just under-visited this action, it currently has no state where it can
structurally win. This is a materially larger gap than "narrow optimal
region" alone implies, and sharpens what Week 2's exploration bonus
needs to target — it needs to lift ML-KEM-1024's probability ceiling,
not just its visit frequency.

**Finding 3 — 2D (CPU, RAM) sweep.** Mapping the same scenario over a
CPU×RAM grid shows `rekey-now` dominating most of the low-CPU /
low-RAM-available region, and the ML-KEM-768/1024-vs-512 boundary only
opening up once both CPU and available RAM are moderate-to-high. This
means the fragile boundary isn't purely a 1D CPU effect — RAM pressure
shifts it too, which the original demo-staging workaround (CPU/RAM
jitter-checked together) already implicitly accounted for without it
being written down as a number until now.

**Baseline for Week 2/3:** these numbers (14–16-point-wide ML-KEM-768
window, ≤0.030 margin, 0.00% ML-KEM-1024 reachability, 0.1576 probability
ceiling) are the explicit "before" state. `analyze_boundary.py` prints all
four together in a `BASELINE SUMMARY` block at the end of its run, so a
post-retraining re-run can be diffed against this section directly rather
than the numbers having to be recomputed by hand. Week 2's longer training run
+ targeted exploration bonus should be checked against them directly —
e.g. re-running `analyze_boundary.py` after retraining and confirming
the window widens, the margin grows, and ML-KEM-1024 starts winning on
at least some non-trivial slice of state space — rather than relying on
a qualitative "seems better" impression.

## Week 2 (Member 3 track, 2026-08-15) — unblocking Members 1 & 2, and retraining

Two workstreams. The assigned task was the retraining; the contracts work
was added because Member 1's Week 7 dependency on the ONNX policy is the
top item in `TEAM_TIMELINE_PROPOSAL.md`'s own risk table, and the
interface — unlike the weights — was already stable enough to ship.

### Shipped for Members 1 and 2 (pulled forward from Week 4)

- **`client/rl_agent/models/ppo_vpn_agent.onnx`** — the policy, exported
  and ready for Member 1's `ort` wrapper. Input `float32[batch,7]`, output
  `float32[batch,4]` raw logits with `argmax` as the chosen action, dynamic
  batch, 4,932 parameters.

  One real bug caught here: torch's exporter writes weights to a sibling
  `.onnx.data` file by default, so the `.onnx` alone is a 2.2 KB graph with
  no weights in it. That loads fine as long as both files sit together and
  fails the moment someone copies just the model into a Rust crate's
  assets. The export now inlines the weights and asserts there are no
  external references left, producing one self-contained 29 KB file.

- **`contracts/`** — `algo_registry.json`, `state_vector.json`, and
  `policy_test_vectors.json`, all *generated* from the Python source of
  truth by `python -m client.rl_agent.export_contracts`. The algorithm
  table was about to exist in four places (Python, the freeze proposal's
  prose, Member 1's Rust core, Member 2's server), three of them
  hand-maintained. `tests/test_phase3.py` fails if the generated files and
  the Python disagree, so drift is caught by CI rather than at integration.

- **PyTorch-vs-ONNX parity check**, run as part of every export and
  currently `max|logit diff| = 9.5e-07` with 31/31 argmax agreement. The 31
  states and their expected logits ship as the test-vector fixture so the
  Rust side can run the same check — the same technique Member 1 is already
  using to validate the Phase 1 crypto port.

- **`rekey-now` semantics pinned.** The action space never said which
  algorithm a rekey handshake uses; `demo.py` was hardcoding ML-KEM-768
  while the registry lists the action with `security: 1.00`. Resolved: a
  rekey re-runs the handshake at the algorithm currently in force —
  rotation and strength stay independent decisions, matching the stated
  reason to rekey (stale keys in RAM, not threat escalation). Member 2's
  Week 6 task is server-side rekey for an existing peer and could not have
  been implemented against an undefined answer. `demo.py` now threads the
  in-force algorithm through and rekeys at it.

### Evaluation harness (`client/rl_agent/evaluate.py`)

New, and it changed the interpretation of the problem. Scoring the policy
by how often it agrees with the single-step reward optimum makes it look
broken — 57% agreement, where a constant "always ML-KEM-512" policy scores
91%. On **episode return**, which is what PPO actually optimises, the same
policy is comfortably ahead of every constant baseline:

| Policy | Mean return (120 episodes x 200 steps) |
|---|---|
| greedy oracle (reference, reads the reward function) | 162.30 ± 1.06 |
| **trained PPO (50k smoke run)** | **155.65 ± 1.22** |
| rule-based baseline | 149.25 ± 1.61 |
| static ML-KEM-768 | 132.39 ± 3.76 |
| always ML-KEM-1024 | 99.57 ± 3.23 |
| always ML-KEM-512 | 98.83 ± 7.40 |

**Success criteria therefore key off return; greedy agreement is a
diagnostic only.** Three of these baselines are the ones the proposal's
Expected Outcomes call for in Week 9, so that milestone now starts from
working code.

What return does *not* excuse, and an earlier draft of this section got
wrong: it is tempting to read the policy's heavy `rekey-now` use (14% of
states against a greedy optimum of 0.16%) as long-run reasoning that
per-step scoring cannot see — rekeying does reset the session timer and
cut threat by 0.5. **The measurement contradicts that.** The greedy
per-step oracle rekeys ~0.1% of the time and still scores 162.30 against
the trained policy's 155.65, so on this environment the myopic policy is
the better one over full episodes. The over-rekeying is a training
artifact, not a strategy. That also says something useful about the
problem: if a myopic oracle is near-optimal, this is close to a
contextual bandit, which is what justifies the short-horizon settings in
the training section below rather than them being a tuning hack.

Two further gaps. ML-KEM-1024 is selected in 0.00% of states, and on
targeted high-need states where it is optimal in 23% of cases the policy
picks `rekey-now` 75% of the time instead. And the margin over a
four-line rule-based baseline is only +4.3% — worth stating plainly ahead
of Week 9 rather than discovering it there.

### The Week 1 boundary metric was measuring the wrong thing

`analyze_boundary.py`'s `BASE_STATE` is documented as sitting "inside
ML-KEM-768's narrow optimal region". It does not. Its `security_need` is
**0.650**, below ML-KEM-512's security of 0.70, so both 512 and 768 have
zero shortfall and 512 wins on cost at *every* CPU value on that sweep.

So Week 1's headline — "ML-KEM-768 is the top action only for
cpu ∈ [0.32, 0.46], peak margin 0.030" — describes a band where the
policy is **wrong**, and the Week 2 target derived from it ("widen the
window, grow the margin") would have been a target for making it more
wrong. Caught before the retraining was scored against it.

Two replacement probes are now swept alongside it, each verified so that
the named algorithm is optimal across the *whole* CPU range, with the
oracle's choice printed next to the policy's:

| probe | `need` | oracle over cpu∈[0,1] | committed policy |
|---|---|---|---|
| `PROBE_768` | 0.805 | ML-KEM-768 everywhere | **0.0% agreement** — picks `rekey-now`/512 |
| `PROBE_1024` | 0.942 | ML-KEM-1024 everywhere | **0.0% agreement** — picks `rekey-now`/768 |

The true "before" is therefore worse than Week 1 recorded. `BASE_STATE`
is kept for continuity but now prints a warning explaining why its window
number should not be used as a target.

### Training changes

- **Curriculum.** `VPNEnv(curriculum=p)` draws a fraction `p` of episode
  resets from a high-security-need / low-resource-pressure mixture
  (Beta(4,1) on threat, connection type and session age; Beta(1,3) on CPU,
  Beta(3,1) on available RAM). At `p=0.3` the share of states where
  ML-KEM-1024 is optimal goes from **0.39% to 7.2%** — an 18x increase in
  learning signal — while 70% of training stays on the true deployment
  distribution so the policy is not warped everywhere else. `p=0.0` is the
  default and short-circuits before touching the RNG, so the default
  environment produces a byte-identical state sequence to the pre-change
  version; verified against the committed file, and pinned by
  `test_phase3.py::test_default_env_reset_distribution_is_unchanged`.
- **Entropy bonus.** `ent_coef` was sitting at SB3's default of 0.0 — there
  was no exploration bonus at all. Now swept over {0, 0.005, 0.01, 0.02}.
- **Seeding bug.** All four vector envs were constructed with the *same*
  seed, so they generated four correlated copies of the same trajectory
  noise instead of independent experience. Now offset per rank.
- **Model selection.** Added `EvalCallback`/checkpointing/tensorboard.
  Evaluation always runs on `curriculum=0.0` even when training uses a
  curriculum, so a model cannot win selection by overfitting the
  oversampled corner. Both the best-by-eval and final models are kept,
  because comparing them is how you spot an early fluke.
- **Unchanged on purpose:** `net_arch=[64,64]`, `STATE_DIM`, and the active
  action count — the three things the 4,932-parameter pin depends on. The
  reward function and the registry's security/cost values were also left
  alone, so any improvement below is attributable to training, not to
  moving the target.

### What actually fixed it — an ablation, not a bundle

Two measurements reframed the problem before any of this was run:

- **The curriculum washes out.** At `episode_len=200` it lifts the share of
  ML-KEM-1024-optimal *reset* states from 0.39% to 7.2%, but only 0.57% of
  *visited* states — the episode dynamics erase the bias within a few steps.
- **`CPU_LOAD` is a one-way ratchet.** Every action adds load and nothing
  removes it, against zero-mean drift, so CPU saturates at 1.0 by about step
  10 and stays there: mean 0.994 over an episode, 96.5% of steps above 0.95.
  The 30-60% band Week 1 found fragile is a band the agent is almost never
  trained in. `cpu_relax` mean-reverts toward the episode's starting load and
  is off by default.

Seven configurations x 3 seeds x 300k timesteps, each adding one lever:

| config | macro-recall | ML-KEM-1024 recall | regret | seeds passing |
|---|---|---|---|---|
| A status quo | 48.1 ± 1.5 | 0.0% | 0.0812 | 0/3 |
| B + curriculum (`ep=200`) | 48.6 ± 0.3 | 0.0% | 0.0739 | 0/3 |
| C + short horizon (γ=0.9) | 59.8 ± 0.5 | 0.0% | 0.0182 | 0/3 |
| **D + γ=0.5** | **84.0 ± 2.1** | **44.8%** | 0.0006 | **3/3** |
| E + `cpu_relax=0.5` | 85.3 ± 2.6 | 50.8% | 0.0007 | 3/3 |
| **F `ep=4`, full curriculum** | **87.6 ± 0.9** | **55.0%** | 0.0006 | **3/3** |
| G + `ent_coef=0.01` | 86.6 ± 1.1 | 52.7% | 0.0005 | 3/3 |

Read A→B→C→D: **the assigned Week 2 levers are not the ones that work.**
The curriculum alone changes nothing (A ≈ B, and both leave ML-KEM-1024 at
0.0% even at 300k with decisiveness 0.96, so this is a converged policy, not
an undertrained one). Shortening the horizon helps accuracy but still never
reaches ML-KEM-1024. **The discount factor is the decisive lever** — C→D, one
change, takes macro-recall 59.8 → 84.0 and ML-KEM-1024 recall 0% → 44.8%.
And the entropy bonus the timeline named as *the* mechanism makes things
slightly **worse** (F → G), which is consistent with what it actually does:
raising a probability ceiling by flattening the softmax is not the same as
improving a decision.

Why γ matters so much: the reward difference between ML-KEM-1024 and
ML-KEM-768 where the former wins is ~0.075 per step, against a return
standard deviation of ~5.8 over a 200-step episode at γ=0.99. The signal is
buried. Shortening the effective horizon is what makes it visible. This is
also a modelling stance worth stating rather than hiding: it says the
decision is close to a per-tick reactive one, which matches the proposal's
own 5-second control loop and is corroborated by the myopic oracle being
near-optimal on this environment.

### Results — promoted model `F_ep4_full_s1`

| metric | 50k smoke model | **promoted** | reference |
|---|---|---|---|
| Mean episode return | 155.65 | **164.31** | greedy oracle 162.30 |
| …vs best baseline (rule-based) | +4.3% | **+10.1%** | |
| Macro-recall over the 4 actions | 45.4% | **88.5%** | 100% |
| ML-KEM-1024 recall (targeted) | 0.0% | **86.5%** | 100% |
| ML-KEM-1024 argmax share | 0.00% | **0.31%** | oracle 0.47% |
| ML-KEM-1024 probability ceiling | 0.1576 | **0.9999** | 1.0 |
| ML-KEM-768 recall | 22.5% | **94.4%** | 100% |
| Mean per-step regret vs oracle | 0.1009 | **0.0007** | 0 |
| Oracle agreement | 56.1% | **99.3%** | 100% |
| `PROBE_768` agreement over cpu∈[0,1] | 0.0% | **100.0%** | 100% |
| `PROBE_1024` agreement over cpu∈[0,1] | 0.0% | **75.6%** | 100% |

**The trained policy now exceeds the greedy oracle** (164.31 vs 162.30). The
50k model sat 4.1% below it. That is the sequential advantage RL is supposed
to provide finally showing up, and it is the cleanest single answer to "does
the agent beat a rule-based policy" — +10.1%, up from +4.3%.

All acceptance criteria set at the start of the week are met, except the one
that had to be withdrawn because it was measuring the wrong thing (the
`BASE_STATE` window; replaced by the two probe agreements above).

`tests/test_phase3.py` pins macro-recall ≥ 75%, ML-KEM-1024 recall ≥ 55%,
argmax share within [0.10%, 0.85%], regret ≤ 0.010 and decisiveness ≥ 0.60 —
all comfortably below what was achieved so seed noise doesn't fail the suite,
and all far above the smoke model so a regression to it cannot pass. The
share bound is deliberately two-sided: over-selecting ML-KEM-1024 is as wrong
as never selecting it, and a one-sided "nonzero" target is trivially gamed.

### Honest caveats

- The evaluation environment is the same simulator the agent trains in, with
  hand-designed reward weights. Every number above is "correct with respect
  to a reward function we wrote", not "better VPN security". That does not
  change until there are real traffic traces.
- `cpu_relax` and the short horizon change the simulator, not the agent's
  interface. The CPU ratchet looks like a simulator bug rather than a
  modelling choice; it is fixed here only for training runs, and the default
  dynamics are untouched and still pinned by a golden-trajectory test.
- ML-KEM-1024 recall is 86.5% on targeted states but 63% on a uniform sample,
  and `PROBE_1024` agreement is 75.6%, not 100%. The rarest action is
  learned, not mastered.
- The demo's five scenarios are staged. They are now each verified
  analytically optimal before being wired in, which the previous four were
  not — two of them encoded the smoke model's mistakes as expected behaviour.

## Week 3 (Member 3 track, 2026-08-26) — re-verification, and what it found

The assigned task (`TEAM_TIMELINE_PROPOSAL.md`) was to re-verify the retrained
policy the same way it was verified before — bucket-by-security-need
distribution check, jitter-stability check — and confirm all four actions now
have a *robust* margin rather than a technically-correct-on-average one.

The verification is built (`client/rl_agent/verify_policy.py`, `python -m
client.rl_agent.verify_policy`). Its answer is split: **stability is fine, and
it found a defect the Week 2 aggregates could not see.**

### The checks, and why they are shaped this way

- **Bucket-by-security-need.** Policy vs oracle action distribution in ten
  `security_need` buckets. Bucketed rather than pooled because the pooled
  number is dominated by low-need states: ~76% of a uniform sample has
  `need <= 0.7`, where ML-KEM-512 is trivially correct, so a policy that is
  perfect there and useless above 0.9 still scores >90% pooled. Which is
  exactly what happened.
- **Jitter stability.** Perturb the state, count decision flips. The metric
  reported is the *unjustified* flip rate — the policy changing its mind while
  the oracle does not. A flip the oracle also makes is the policy correctly
  tracking a real boundary, and counting that as instability would penalise it
  for being right.
- **Robust margin.** Of the decisions the policy gets right, how many are held
  by enough probability mass to survive noise. "All four actions reachable"
  (Week 2) is about the argmax; fragility is about the gap underneath it, and
  the second does not follow from the first.

### Result 1 — the margins are robust

| metric | measured |
|---|---|
| unjustified flip rate, +/-0.05 jitter on CPU/RAM | 0.13% |
| ...+/-0.10 | 0.19% |
| ...+/-0.15 | 0.23% |
| fragile share of correct decisions (margin < 0.10) | 0.20% |
| median margin on correct decisions | 0.89 – 1.00 per action |

So the Week 1 fragility concern is genuinely resolved: where the policy is
right, it is decisively right, and ordinary sensor noise does not move it. The
worst case is ML-KEM-1024-optimal states at 2.9% unjustified flips under
+/-0.05 — the highest of the four, and the number the decision gate below is
sized against.

### Result 2 — a dead band at the top of the security-need range

| need bucket | n | policy modal | oracle modal | agreement | regret |
|---|---|---|---|---|---|
| [0.00, 0.70) | 34,756 | ML-KEM-512 | ML-KEM-512 | 100.0% | 0.0000 |
| [0.70, 0.80) | 3,674 | ML-KEM-768 | ML-KEM-768 | 97.8% | 0.0001 |
| [0.80, 0.90) | 1,368 | ML-KEM-768 | ML-KEM-768 | 89.0% | 0.0146 |
| **[0.90, 1.00)** | **202** | **rekey-now** | **ML-KEM-1024** | **46.5%** | **0.0504** |

Above `need = 0.90` the policy asks for `rekey-now` on 75.3% of states. On a
6,000-state sample of that band its agreement is 50.9%, and the confusion is
entirely one-directional:

```
oracle \ policy    512    768   1024   rekey
   ML-KEM-768        0      0      0     914
  ML-KEM-1024        0      0   1479    2034
    rekey-now        0      0      0    1573
```

It never picks something too weak. It picks `rekey-now` instead of escalating.
This band is ~0.5% of a uniform sample, which is why every Week 2 aggregate
stayed clean while it was broken — 99.3% agreement, 0.0007 mean regret, and
all four actions reachable are all still true.

### Root cause: the reward model, not the training

The obvious hypothesis was training coverage. The Week 2 curriculum draws its
high-need corner at *low* resource pressure (median 0.234), while the
ML-KEM-1024 / rekey-now boundary sits near pressure 0.5 — so only 20% of those
resets land in the [0.35, 0.60] band, and the oracle labels there run 2900:41
in ML-KEM-1024's favour. Barely any counterexample showing the agent where to
stop.

So a `curriculum_pressure="spread"` mode was added (uniform CPU/RAM at high
need), which raises in-band coverage 20.3% -> 43.7% and the rekey-now
counterexamples 41 -> 831. Ablated at 300k timesteps x 3 seeds, F re-run as a
control so both arms are scored by the same code:

| config | macro-recall | ML-KEM-1024 recall | high-need agreement |
|---|---|---|---|
| F control (Week 2 model) | 87.6 +/- 0.9 | 55.0% | **49.9%** |
| H curriculum spread | 86.2 +/- 3.6 | 54.4% | **52.7%** |

**It is not a coverage problem.** 49.9% -> 52.7% is inside seed noise (sd 3.6).
This is the second time an assigned lever has turned out not to be the
mechanism — Week 2's entropy bonus was the first — and for the same underlying
reason: no amount of exploration fixes an objective that is specified wrongly.

The actual cause is in `action_reward`. The `rekey-now` branch scores purely on
urgency and never references `algo["security"]`:

```python
urgency = 0.5 * state[THREAT] + 0.5 * state[TIME_SINCE_REKEY]
reward  = urgency - 0.5 * (1.0 - urgency) - 0.3
```

But Week 2 pinned the *semantics* of that action as "re-run the handshake using
the algorithm currently in force" — so a rekey does not raise the security
level at all. The reward function credits it as though it satisfies security
need; the semantics say it does nothing of the kind. Where the policy rekeys at
high need, mean `security_need` is 0.926, so if ML-KEM-512 is the algorithm in
force the real shortfall is 0.226 — an uncharged reward loss of 0.678 per step,
against a reward function that charges zero.

**The agent is correctly optimising a wrong objective.** This is the same class
of bug as the two reward-shaping bugs in Section 6, and it is the third.

Fixing it properly is not a Week 3 decision. The clean fix needs the agent to
know which algorithm is currently in force, and that is not one of the seven
frozen state dimensions — Members 1 and 2 are building against that contract
right now, and changing it unilaterally is precisely the wrong move. It is
written up for the Week 4 checkpoint, which exists for this ("everyone
re-confirms the contracts held up under actual implementation").

### The mitigation that does not touch the contract

There is one available, and it needs no interface change at all: when the
policy asks for a rekey, rekey at the **stronger** of {algorithm in force, the
policy's top-ranked KEM}. The information is already in the logits Member 1
receives.

| what the client rekeys at | mean security shortfall | states left short |
|---|---|---|
| the algorithm in force (literal Week 2 rule), ML-KEM-512 case | 0.2260 | 100.0% |
| **the policy's top-ranked KEM** | **0.0222** | 36.5% |
| the oracle's choice (floor) | 0.0115 | 20.2% |

Stated honestly: that is a decision-quality measurement on the high-need
states. On full simulated sessions the effect is much smaller (mean shortfall
0.0083 -> 0.0076), because the simulator's CPU ratchet rarely keeps a session
in the region where it applies. Both numbers are in
`models/decision_gate_sizing.json`; neither is quoted without the other.

This is a proposed amendment to frozen semantics, so it ships as a flag
(`rekey_escalates`, default on, `False` restores Week 2 exactly) and is flagged
for Member 2's sign-off rather than adopted unilaterally.

### Shipped for Members 1 and 2

The user-facing half of the week, and the reason the jitter measurement is not
a report-only diagnostic — it sizes a rule that ships.

- **`client/rl_agent/decision_gate.py`** — the debounce between the policy's
  argmax and the crypto layer. The policy is a pure function of a noisy state,
  and a *change* of algorithm costs a full handshake, so acting on the raw
  argmax means paying a handshake for sensor noise. Measured over 133
  client-hours at +/-0.05 noise: **71.0 handshakes/hour raw, 34.5 gated** —
  about half of all renegotiations were the client arguing with `psutil`.
  Sized from the jitter numbers above: a challenger must win `confirm_ticks=3`
  consecutive ticks and beat the incumbent by `min_margin=0.15`.

  The cost is stated with the benefit: legitimate escalations are delayed by a
  median of 2 ticks (10s), p95 4 ticks (20s). `python -m
  client.rl_agent.decision_gate` re-derives both sides across
  `confirm_ticks in {1,2,4}` so the constants can be re-checked rather than
  trusted.

  Two honest notes. The rekey cooldown never binds in simulation — rekeys are
  already spaced further apart than 12 ticks — so it is a safety rail for the
  correlated-noise case the simulator does not model, not a measured win. And
  the gate slightly *increases* mean security shortfall (0.0045 -> 0.0070),
  which is the escalation delay showing up; halving the handshake rate is what
  is being bought with it.

- **`contracts/decision_gate_vectors.json`** — eight named tick sequences with
  the expected decision after every tick, generated from the reference
  implementation, so the Rust port is verified exactly the way the ONNX wrapper
  and the Phase 1 crypto port already are. `tests/test_phase4.py` replays them
  against the Python, so a hand-edit or an un-regenerated change fails CI.

- **`contracts/manifest.json`** — SHA-256 of every artifact plus the git commit
  they came from. The artifacts are only correct *as a set*: a stale
  `.onnx` paired with a current `algo_registry.json` does not throw, it returns
  action indices meaning something different from what the registry says. That
  deserves a checksum, not a convention.

- **`contracts/INTEGRATION.md`** — the step-by-step for both members, including
  which parts Member 2 can ignore (all of it except `algo_registry.json`).

- **`tests/test_phase4.py`** — 21 tests. The gate behaviour and the manifest are
  pinned hard; the policy floors are set below what was measured so seed noise
  cannot fail the suite, and above pre-Week-2 behaviour so a regression cannot
  pass. `test_high_need_agreement_does_not_regress_below_measured` pins the
  known defect at 0.45 against a measured 0.51 — a floor to stop it worsening
  while the reward question is open, to be raised rather than deleted once it
  is settled.

### Corrections to earlier write-ups

- `policy_test_vectors.json` carries **63** vectors, not 31. The Week 2 text
  described the pre-retrain export and was not updated when the model was
  promoted. `contracts/README.md` and `INTERFACE_FREEZE_PROPOSAL.md` are fixed.
- Re-running `export_contracts` produced byte-identical `.onnx` and test
  vectors, confirming the shipped artifacts already matched the promoted model
  and that the export is deterministic.

### Status against the Week 3 mandate

"Confirm all four actions now have a robust margin." Three of the four do.
ML-KEM-1024's margin is robust where the policy selects it, but it is
under-selected in the band it exists for, and that is a reward-model defect
rather than a margin one. Reported rather than papered over, with the cause
diagnosed, an ablation showing the obvious fix does not work, a
contract-preserving mitigation shipped, and the real fix put in front of the
Week 4 checkpoint where the interface decision belongs.

## Week 4 (Member 3 track, 2026-09-07) — closing the checkpoint before Member 1's Rust port starts

The assigned task (`TEAM_TIMELINE_PROPOSAL.md`) — export the trained policy
to ONNX, verify parity, sync the interface with Member 1 — was already done:
it shipped in Week 2/3 as "pulled forward from Week 4" (see above) and was
re-verified this session (parity `max|logit diff| = 7.629e-06`, 63/63 argmax
agreement, manifest digests current). What was left is the other half of
Week 4, Section 6's own framing: *"everyone re-confirms the contracts held up
under actual implementation."* Three things the codebase itself had already
flagged as belonging here.

### Bugs found and fixed before the freeze, not after it

A review pass over the Week 3 diff surfaced two bugs inside
`client/rl_agent/decision_gate.py` — the exact module Member 1 is about to
transliterate into Rust. Finding these now is a diff; finding them after the
port would have been two fixes in two languages.

- **Stale confirm-streak survives an escalation.** After a rekey escalates
  `in_force` to a stronger algorithm, the `_candidate`/`_streak` counters
  from whatever challenger was building against the *old* incumbent were
  left in place. If that challenger reappeared on the very next tick, its
  streak looked one tick from confirming instead of three, and could
  downgrade the algorithm the escalation had just raised — a direct
  violation of the module's own stated invariant. Reproduced as a failing
  test first (`test_escalation_resets_the_confirm_streak`, confirmed to fail
  against the pre-fix code with `reason='challenger confirmed for 3 ticks'`
  after a single real post-escalation tick), then fixed by resetting the
  streak counters at the point of escalation. `contracts/decision_gate_vectors.json`
  is byte-identical after the fix — none of the eight shipped cases happened
  to exercise this sequence, so Member 1's existing port-verification target
  is unaffected; the new test is now part of what a correct Rust port has to
  reproduce.
- **Escalation-delay measurement undercounted its own sample.** In
  `simulate()`, `pending_since[i]` was cleared and then read in the same
  loop iteration, in that order — so a delay was recorded only on the rare
  tick where clearing didn't already null it out (50 of 1613 available
  samples in a repro run). Fixed by recording the delay before refreshing
  the pending state. Re-ran the sizing measurement afterward: the headline
  numbers `INTEGRATION.md` quotes (median 2 ticks / p95 4 ticks, at the
  default `+/-0.05` noise and `confirm_ticks=3`) are unchanged — they were
  under-sampled, not wrong. Other configurations' p95 shifted somewhat now
  that the estimator sees the real sample (e.g. `+/-0.10` noise: median moved
  0 → 2 ticks); the full diff is in `client/rl_agent/models/decision_gate_sizing.json`.
- **`contracts/INTEGRATION.md`** stated a rekey happens "without changing
  strength" six lines above the section documenting that it does, by
  default, escalate. Reworded to point at that section instead of
  contradicting it.

### Two decisions written up for Member 2 (and the team), not resolved unilaterally

`contracts/DECISIONS.md` (new) carries both in full, with the measured
trade-offs and a direct ask:

1. **`REKEY_ESCALATES`** — sign-off on the Week 3 amendment to the Week 2
   rekey semantics (approve as shipped / reject / request a tighter margin
   on the escalation path).
2. **The ML-KEM-1024 dead band's actual fix**, as opposed to the mitigation
   already shipped. The reward function can't charge a rekey for leaving a
   weak algorithm in place because the environment tracks the in-force
   algorithm nowhere, not even internally — confirmed by reading
   `VPNEnv._evolve_state`, which mutates `TIME_SINCE_REKEY` and `THREAT` on a
   rekey but has no equivalent state for "which algorithm is this." Two
   options are written up: give the reward that memory (the clean fix, but
   it touches the oracle every other Week 1-3 measurement is defined against,
   and likely means retraining and re-exporting the ONNX policy in the same
   week Member 1 starts building against it), or a contract-preserving
   reward penalty keyed on the same `security_need > 0.90` threshold already
   used elsewhere in the codebase (a calibrated band-aid, not the fix, but
   costs nothing else). Recommendation: ship the patch now if a training-side
   improvement is wanted at all, since the escalation rule above already
   recovers most of the practical harm, and save the clean fix for a
   deliberate retrain cycle rather than one landing mid-freeze.

As housekeeping for decision 2 specifically: `train.py`'s
`MIN_HIGH_NEED_AGREEMENT` promotion gate was recalibrated from 0.75 (set
against a comment claiming a 0.64 baseline that no recorded run — including
the actual Week 2/3 runs in `models/sweep_results_week3.json`, which top out
at 0.551 — has ever matched) down to 0.45, matching the floor
`tests/test_phase4.py` already pins the defect at. The old value made
`eligible` permanently empty in `summarise()`, and its
`pool = eligible or results` fallback silently disabled the other two
promotion gates (`decisiveness`, `mlkem1024_share`) along with it — not a
policy call, just a threshold that had never once been checked against the
data it was supposed to gate.

### Status against the Week 4 mandate

The literal task (ONNX export, parity, sync) was complete before the week
began. What the week actually needed — re-confirming the contracts held up,
per Section 6 — is done: two real bugs in the module Member 1 is about to
port are fixed and test-covered, the one documentation contradiction found is
corrected, and both open policy questions are written up as concrete,
measured decisions for the team rather than left as comments in code or
resolved by fiat. `pytest tests/ -v` — 67/67 (66 prior + the new regression
test) — and `contracts/manifest.json` regenerated and current.

## Week 5 (Member 3 track, 2026-09-07) — anomaly detection Layer 2

`TEAM_TIMELINE_PROPOSAL.md`'s Week 5 task: build five rule-based anomaly
signatures — port scan, retransmission spike, MitM-latency, bandwidth
exfiltration, DNS-server change — each "an independent stateful check over
the rolling 5-second metric window." Wiring them into the CPU-gated combiner
alongside Layer 1, and the combiner-level integration tests, is Week 6's job.

### Design

Layer 1 (`ZScoreBaseline`) was the model to follow: a small stateful class
taking plain numeric inputs, decoupled from `psutil`/live sourcing — the only
thing that has ever called it is `demo.py`, with hardcoded demo values, since
the live daemon doesn't exist yet. All five Layer 2 classes follow the same
shape and were appended to the same module (`client/rl_agent/anomaly_detector.py`)
rather than a new file — its own docstring already framed itself as Layer
2/3's eventual home, and the Rust target `core/src/anomaly/mod.rs` is one
crate module for all three layers plus the combiner, so keeping both sides'
module boundaries identical matters for the eventual port.

The distinction the proposal itself draws — Layer 1 is a *statistical
baseline* (learns per-session normal, flags deviation from it), Layer 2 is a
*rule-based signature* (a fixed rule standing in for a named attack pattern,
independent of any baseline) — shaped every constant below: these are fixed
thresholds, not adaptive statistics.

A shared `_ratio_score(value, threshold, ceiling=2.0)` helper generalizes the
squash `ZScoreBaseline` already did inline for its Z-score, so all six
detectors' `threat_score` values are calibrated the same way, and all six
return the same key set — `{anomalous, threat_score, ready, detail}` — for
Week 6's combiner to fold together without caring which rule fired.

- **Port scan** — fanout (distinct destination ports) *and* a high failure
  ratio, both required (`>=15` ports, `>=70%` failed — SYN scans never
  complete the handshake; high fanout alone isn't a scan if every connection
  also succeeds). Per-tick, no window — the fanout is already visible within
  one 5s tick.
- **Retransmission spike** — sustained (2 consecutive ticks) retransmit rate
  `>=10%`. The consecutive-ticks gate is what makes this a sustained-
  disruption signature rather than a re-run of Layer 1's single-sample
  Z-score on a new metric.
- **MitM-latency** — checks *jitter* (max-min over a 3-tick window of raw
  ms), not absolute level, since Layer 1 already flags an absolute outlier.
  A relay/interception hop's signature is variably-added latency tick to
  tick, not one clean step. `>=80ms` jitter, sustained 2 ticks.
- **Bandwidth exfiltration** — upload rate alone can't distinguish
  exfiltration from a legitimate large upload; the heuristic is asymmetry
  (upload `>=2MB/s`, chosen as a fraction of `state_observer.py`'s existing
  `UPLOAD_CAP_BYTES_PER_SEC = 5_000_000.0` so "a lot of upload" means the
  same thing in both subsystems) *and* an upload:download ratio `>=3:1`
  (legitimate heavy-upload traffic like a video call sits near 1:1),
  sustained 2 ticks so a single large-file send doesn't fire.
- **DNS-server change** — deliberately the odd one out: edge-triggered on a
  discrete change, not windowed statistics (an already-fully-informative
  event on the first observation after it happens). `threat_score` is `1.0`
  or `0.0` — no in-between makes sense for a binary match. **Known,
  explicit limitation**: this fires on *any* DNS change, including benign
  ones (switching networks). Properly suppressing benign changes needs an
  explicit "expect a reconfiguration" hook, analogous to
  `StateObserver.mark_rekey()` — out of scope for this week, flagged rather
  than silently accepted.

### Tests

`tests/test_phase5.py` (new, 18 tests): a fires/doesn't-fire pair per
signature, streak-gate tests proving the consecutive-ticks requirement
actually gates (one bad tick alone must not fire, for retransmission and
exfiltration), a direct test of `_ratio_score`, and a structural drift-guard
test that pins all five signatures' return shape against a future edit that
silently changes it — the same role `test_gate_constants_in_vectors_match_the_module`
plays for the decision gate.

Nothing in `contracts/`, `core/src/anomaly/mod.rs`, `demo.py`, or
`requirements.txt` changed — Layer 2's output isn't wired into the frozen
state vector's THREAT dimension until the Week 6/8 combiner exists, and no
new dependency was needed (every input above is a plain int/float/str).
Confirmed by re-running `python -m client.rl_agent.export_contracts`: every
artifact regenerated byte-identical except the manifest's timestamp/head
metadata.

`pytest tests/ -v` — 87/87 (69 prior + 18 new).

## Week 6 (Member 3 track, 2026-09-11) — the anomaly combiner

`TEAM_TIMELINE_PROPOSAL.md`'s Week 6 task: "Wire Layer 2 into the CPU-gated
combiner (< 70% CPU) alongside the existing always-on Layer 1. Unit tests per
signature (inject a synthetic port-scan pattern, confirm it fires; inject
normal traffic, confirm it doesn't)." The per-signature half of that already
shipped early, in Week 5's `test_phase5.py` — what was still outstanding, and
flagged explicitly in two places (`anomaly_detector.py`'s own module
docstring, and `contracts/state_vector.json`'s `THREAT` field description) is
the combiner itself.

### `AnomalyCombiner` (`client/rl_agent/anomaly_detector.py`)

Owns one `ZScoreBaseline` (Layer 1, always active) and the five Layer 2
signatures, gated by a new `LAYER2_CPU_GATE = 0.70` constant — normalized
`[0,1]`, matching `StateObserver.cpu_load()`'s units, not raw 0–100.

- `update(cpu_load, **metrics)`: Layer 1 always runs. The five Layer 2
  signatures run only when `cpu_load < LAYER2_CPU_GATE`; when gated off they
  are simply not called that tick, so a transient CPU spike doesn't reset or
  falsely advance a streak-based signature's internal state — it just sits
  the tick out.
- Combines every *active and `ready`* layer's `threat_score` with `max()`,
  not an average — one confirmed signature shouldn't get diluted by four
  quiet ones, and `max` is what makes the THREAT contract's own promise
  ("the consumer never needs to know which are active") actually hold
  regardless of how many layers are on.
- Returns the module's usual `{anomalous, threat_score, ready, detail}`
  shape (`ready` mirrors Layer 1's own `ready`, the always-on floor every
  prior direct-`ZScoreBaseline` caller relied on); `detail` additionally
  carries `layer2_active` and a per-layer result breakdown for debugging and
  tests.

### `StateObserver.cpu_load()` (`client/rl_agent/state_observer.py`)

Promoted from `_cpu_load()` to public, since the combiner's CPU gate needs
the same live reading *before* `read_state()` runs. `read_state()` also
gained an optional `cpu_load` parameter so a caller that already sampled it
for the gate doesn't pay for a second, differently-timed
`psutil.cpu_percent()` call — defaults to sampling fresh if omitted, so every
existing caller is unaffected.

### `demo.py`

`run_round()` now drives `AnomalyCombiner` instead of a bare `ZScoreBaseline`,
sharing one live CPU sample between the gate and the state vector. A sixth
scenario, appended after the original five (so it can't perturb the
`current_algo` chain they thread through each other), stages a synthetic
port-scan pattern and lets Layer 2's real `PortScanSignature` drive `THREAT`
live rather than a staged number — module docstring's "what's genuinely live"
list updated to match.

### Contracts

`python -m client.rl_agent.export_contracts` re-run after bumping the
`THREAT` field's `produced_by` text (was "Layer 1 today; Layers 2/3 land
Weeks 5-8", now reflects Layer 2 landing). Diff against the prior
`contracts/` is exactly that one string plus the manifest's
timestamp/head/hash metadata — nothing else moved.

### Tests — `tests/test_phase6.py` (new, 7 tests)

Combiner-level, as `test_phase5.py` explicitly deferred: CPU-gate boundary
(active just below the gate, inert at/above it), combined `threat_score` is
the `max` across two simultaneously-firing signatures (not a sum, which would
either overshoot `[0,1]` or under-report), a gated-off tick neither resets
nor advances a streak-based signature's counter, a not-yet-`ready` Layer 2
signature doesn't suppress a real Layer 1 hit, a structural shape check, and
a regression-parity test — `AnomalyCombiner` with CPU held below the gate and
no Layer 2 signature triggering reproduces a bare `ZScoreBaseline`'s output
exactly, which is what makes the `demo.py` swap safe.

`pytest tests/ -v` — 94/94 (87 prior + 7 new). `python demo.py` — full run;
all five original scenarios still "matches the optimal choice" (the
regression-parity guarantee holding end to end, not just in the unit test),
sixth scenario shows `threat=0.64` driven live by the port-scan signature.

### Deliberately not done (later weeks)

- No live network/packet-capture sourcing for the Layer 2 inputs (distinct
  ports, retransmit rate, DNS server, etc.) — they stay plain caller-supplied
  numbers, the same decoupling-from-`psutil` stance Layer 1 has had since
  Week 1. A real packet-capture/`netstat` source is materially bigger and
  platform-specific, and — like `StateObserver`'s own live psutil reads —
  has no live daemon to call it yet.
- Layer 3 (Isolation Forest) — Week 7 per the timeline.
- Rust-side integration of the combiner's output into `core`'s state
  pipeline — a later, separate checkpoint item (Member 1's "Integrate Member
  3's anomaly `threat_score` combiner into the state pipeline",
  `TEAM_TIMELINE_PROPOSAL.md`). `core/` and `server/` untouched this week.

### Status against the Week 6 mandate

Layer 2 wired into a CPU-gated combiner alongside Layer 1, combiner-level
integration tests written (not just per-signature ones, which already
existed), and the whole thing exercised end-to-end live in `demo.py` with no
change to the five existing scenarios' agent decisions.

## Week 1 (Member 2 track, 2026-09-08) — orientation, Rust crypto validation, protocol draft

Member 2's track (WireGuard + PQC handshake server, deployment) starts here.
Per `TEAM_TIMELINE_PROPOSAL.md` the Week 1 tasks are: provision the VPS,
install WireGuard, and draft the handshake wire protocol for the Week 1 sync
(`TEAM_TIMELINE_PROPOSAL.md` §2 item 1).

### What was already in place

- **Member 1's `core/` crate already ports the crypto to Rust** — `core/src/crypto/`
  has hybrid X25519+ML-KEM (512/768/1024), ML-DSA-65 auth, a zeroizing key
  store, and `build_handshake_transcript` with the label `PQC-VPN-HANDSHAKE-v1`.
  Built and tested in WSL: `cargo test -p core` → 20/20. It uses pure-Rust
  crates (`ml-kem`, `ml-dsa`, `x25519-dalek`), **not liboqs** — so the server
  shares this crate rather than linking a C library.
- `core/src/protocol/`, `state/`, `rl/`, `anomaly/` are empty stubs. The wire
  protocol is genuinely unbuilt and is Member 2's to define.
- `contracts/INTEGRATION.md` confirms Member 2's only dependency on the RL track
  is `algo_registry.json` (algorithm identifiers + rekey meaning).

### `server/crypto-spike/` — toolchain validation (throwaway)

A workspace-member binary that drives `core::crypto` the way the handshake
responder will and prints the encoded size of every value that crosses the
wire. `cargo run -p crypto-spike`. Confirms Member 2 can build against `core`,
and produces the framing inputs for the protocol draft:

| value | ML-KEM-512 | ML-KEM-768 | ML-KEM-1024 |
|---|--:|--:|--:|
| X25519 public (each way) | 32 | 32 | 32 |
| ML-KEM public (client→server) | 800 | 1184 | 1568 |
| ML-KEM ciphertext (server→client) | 768 | 1088 | 1568 |
| ML-DSA-65 signature | 3309 | 3309 | 3309 |
| signed transcript | 1662 | 2366 | 3231 |

The 3.3 KB fixed ML-DSA-65 signature dominates `ServerHello` and rules out a
single UDP datagram → the protocol uses TCP.

### `server/PROTOCOL.md` — wire protocol v1 DRAFT

For sign-off at the Week 1 sync. Defines: TCP on `51821/tcp` beside WireGuard's
`51820/udp`; a 4-byte length-prefixed message header; five handshake messages
(`ClientHello` / `ServerHello` / `ClientFinish` / `ServerFinish` / `RekeyRequest`)
plus `Error`; the signed transcript (extends `core::build_handshake_transcript`
with both nonces); ML-DSA-65 server identity pinned in the client; and
`HKDF-SHA256(hybrid_secret, client_nonce‖server_nonce)` → `{psk, confirm_key}`
so the WireGuard PSK and the finish-MAC key are independent.

Open items carried to the freeze meeting (`PROTOCOL.md` §8): adding the nonces
to the signed transcript (small `core` change), where HKDF lives (proposed:
`core`), the proposal-vs-code disagreement on who holds the ML-KEM keypair
(code is stronger, correct the proposal), the port number, and an explicit
sign-off on `REKEY_ESCALATES` (`contracts/DECISIONS.md` Decision 1 — a rekey may
return a stronger algorithm; recommend approve).

### `server/deploy/` — VPS provisioning

Scripts + guide to take a fresh DigitalOcean Ubuntu droplet to a hardened host
running a plain (non-PQC) WireGuard tunnel — the baseline Week 2 builds the PQC
handshake onto. `provision.sh` (user creation, sshd hardening, ufw, auto
updates, WireGuard, IP forwarding), `wireguard-baseline.sh` (wg0 up, NAT),
`add-peer.sh` (per-client config). `wsl-setup.sh` sets up the local dev
environment. The droplet itself is not yet created (needs the DigitalOcean
account + $200 student credit).

### Status against the Week 1 mandate

Wire-protocol draft: done, ready for review. VPS: scripts and runbook done,
provisioning pending the cloud account. WireGuard install: automated, pending
the droplet. No changes to any other member's code; `core` and `contracts`
untouched.

## Week 2 (Member 2 track, 2026-09-08) — the handshake responder

Assigned task (`TEAM_TIMELINE_PROPOSAL.md`): implement the server-side PQC
handshake responder matching Member 1's client — same algorithm set, same
ML-DSA-65 logic. Done, and running live on the droplet.

### Week 1 close-out first

- DigitalOcean droplet (`139.59.62.4`, Ubuntu 24.04, 1 GB) provisioned via
  `server/deploy/provision.sh`: non-root `member2` user, root SSH disabled,
  ufw (SSH + `51820/udp` + `51821/tcp`), unattended-upgrades, 2 GB swap.
- Baseline WireGuard up (`server/deploy/wireguard-baseline.sh`) — plain
  non-PQC `wg0`, `10.8.0.0/24`. Peer added, tunnel smoke-tested from WSL.

### `server/handshake-server/` — new workspace crate

Built on `vpn_core::crypto` (Member 1's port — pure-Rust `ml-kem`/`ml-dsa`, no
liboqs). This crate is only the wire protocol around it.

| module | role |
|---|---|
| `wire` | frame + message encode/decode (`PROTOCOL.md` §3–4), 8 unit tests |
| `transcript` | the signed byte string (§5.1) — `core::build_handshake_transcript` plus the two nonces |
| `kdf` | `HKDF-SHA256(hybrid_secret, nonces)` → `{psk, confirm_key}`, HMAC finish tags (§5.3) |
| `identity` | long-term ML-DSA-65 key, persisted as its 32-byte seed; load-or-create |
| `handshake` | server: `ClientHello` → `core::server_encapsulate` → sign → `ServerHello` |
| `client_handshake` | client side, for the test client + vector emitter |
| `session` | in-memory session table (rekey lookup; expands Week 5) |
| `server` | blocking TCP accept loop, thread per connection |

Binaries: `handshake-server` (the daemon), `test-client` (throwaway — the Week 3
"throwaway client script", pulled forward), `emit-vectors`.

### Verified

- `cargo test -p handshake-server` — 13 unit + 4 integration.
  `over_tcp_full_handshake` runs a real `TcpListener` handshake for
  ML-KEM-512/768/1024 and asserts client and server derive the **same PSK**.
  Negative tests: wrong pinned key → `AuthFailed`; tampered `ServerHello` →
  `AuthFailed`.
- **Live over the internet.** Deployed to the droplet as the `handshake-server`
  systemd service (`server/deploy/deploy-handshake-server.sh` +
  `handshake-server.service`, identity at `/var/lib/pqc-vpn/server-identity.seed`,
  survives redeploys). `test-client` from WSL completed the full
  `ClientHello → ServerHello → ClientFinish → ServerFinish` exchange for all
  three levels and both ends derived matching PSKs; a client with a bogus pinned
  key was correctly rejected.
- `cargo test --workspace --exclude desktop` — core 20/20 unchanged, no warnings.

### `server/handshake-vectors.json`

Deterministic fixtures (framing, transcript + SHA-256, KDF outputs, finish MACs)
for Member 1 to verify the client side of `core/protocol/` against the exact
same bytes — same pattern as `contracts/*_vectors.json`. The KEM/DH/signature
roundtrip is non-deterministic and is covered by `core`'s tests + the
integration test instead.

### Deliberately not done (later weeks)

- No `wg set` PSK injection — Week 3. The server records the derived PSK and
  logs "would install/swap PSK …" at the seam.
- No `RekeyRequest` CLI — the server-side rekey path (`PROTOCOL.md` §6, incl. the
  `REKEY_ESCALATES` no-downgrade rule) is coded and unit-tested; the client flag
  is Week 6.
- Replay cache / rate limiting — protocol fields exist, enforcement is Week 7.

### For the freeze meeting

`PROTOCOL.md` §8 now carries 7 items. New since the Week 1 draft: (7) add seed
persistence to `core::ServerAuthenticator` so the server drops its direct
`ml-dsa` use. Also flagged: adopting `HKDF` into `core` (§8 Q2) and the nonce
addition to the transcript (§8 Q1) are both now needed by working code, not
just proposed.

### Status against the Week 2 mandate

Responder implemented, matches the client's algorithm set and ML-DSA-65 identity
model, tested three ways (unit, in-process integration, live over the internet),
and deployed. No changes to `core/` or `contracts/`.

## Week 3 (Member 2 track, 2026-09-08) — `wg set` PSK injection

Assigned task (`TEAM_TIMELINE_PROPOSAL.md`): server-side `wg set` PSK injection
for a single test peer; one manual end-to-end handshake against a throwaway
client. Done — the `would install PSK …` stub is now a real WireGuard peer.

### `tunnel` module

`PskInstaller` is the seam between a completed handshake and the tunnel:

- `WgCli` — shells out to `wg set <iface> peer <client_wg_pubkey>
  preshared-key /dev/stdin [allowed-ips <ip>/32]`, and queries the server's own
  WireGuard public key at startup via `wg show <iface> public-key`.
- `DryRun` — logs the action, touches nothing. Default when `--wg-interface` is
  absent, so tests and no-privilege dev runs still work.
- `PeerAddresses` — allocates a stable `10.8.0.0/24` address per peer wg pubkey
  (`.1` is the server); a peer that rekeys keeps its address.

### Protocol addition — `ServerFinish` carries tunnel params

`PROTOCOL.md` §4.5: `ServerFinish` gained `server_wg_pubkey (32)`,
`assigned_ip (4)`, `wg_port (2)`. The client can now bring its tunnel up with no
out-of-band config — it learns everything except the PSK from the handshake, and
the PSK both sides derived. Protocol is still DRAFT/unfrozen; `handshake-vectors.json`
regenerated (`server_finish` frame 52 → 90 bytes).

### Server wiring

`handle_connection` now, after the client tag verifies: allocates the peer's
address, calls `installer.install(pubkey, psk, ip, algo, first_install)`, then
sends `ServerFinish` with the tunnel params. A `RekeyRequest` swaps only the
PSK (`first_install=false`, no `allowed-ips` change) — `PROTOCOL.md` §6.
`main.rs` gained `--wg-interface`, `--wg-port`, `--tunnel-cidr`, `--wg-pubkey`.

### `test-client`

Now generates a real WireGuard keypair, sends its public key in `ClientHello`,
and after the handshake prints a complete `wg-quick` client config built from
the returned tunnel params + derived PSK.

### Verified

- `cargo test -p handshake-server` — 17 unit + 4 integration (4 new `tunnel`
  tests: address allocation stability, `wg set` argv construction, CIDR parsing,
  dry-run). `over_tcp_full_handshake_returns_tunnel_params` asserts the client
  receives the right `assigned_ip` / `server_wg_pubkey` / `wg_port`.
- `cargo test --workspace --exclude desktop` — core 20/20 unchanged, no warnings.
- **Live end-to-end.** Redeployed to the droplet with
  `--wg-interface wg0` and `AmbientCapabilities=CAP_NET_ADMIN` (unit updated).
  `test-client` from WSL completed the handshake; the server ran `wg set` and
  `wg show wg0` then showed the new peer with `preshared key: (hidden)` and
  `allowed ips: 10.8.0.2/32`; the client emitted a ready `pqc0.conf`. Server
  log: `HANDSHAKE session=… algo=ML-KEM-768 peer @ 10.8.0.2`. The final
  `wg-quick up` + `ping 10.8.0.1` is a manual step (needs local sudo).

### Deliberately not done (later weeks)

- Peers are installed live via `wg set`, not written to `wg0.conf`, so they do
  not survive a `wg0` restart — fine for a test peer; persistence with
  multi-peer lifecycle is Week 5.
- `RekeyRequest` still has no `test-client` flag (Week 6); the server path is
  coded + unit-tested.
- Replay cache / rate limiting — Week 7.

### Status against the Week 3 mandate

Real `wg set` PSK injection working end to end against a throwaway client, live
on the droplet. `core/` and `contracts/` untouched.

## Week 4 (Member 2 track, 2026-09-08) — containerise the responder

Assigned task (`TEAM_TIMELINE_PROPOSAL.md`): containerise the handshake
responder (Docker); checkpoint — sync on protocol stability with Member 1.
No code changes to `handshake-server` — packaging plus the protocol sync.

### Docker

- `server/Dockerfile` — multi-stage: `rust:1-slim-bookworm` builder (BuildKit
  cache mounts) → `debian:bookworm-slim` runtime with `wireguard-tools` (the
  server shells out to `wg`). Final image ~129 MB, Rust build ~1m49s on the
  droplet.
- `server/docker-compose.yml` — `network_mode: host` (**required**: `wg set wg0`
  needs the netns wg0 lives in), `cap_drop: [ALL]` + `cap_add: [NET_ADMIN]`,
  `security_opt: no-new-privileges`, `read_only: true` + `tmpfs: /tmp`,
  `restart: unless-stopped`. Volume `/var/lib/pqc-vpn` → the ML-DSA-65 identity.
- `.dockerignore` at the repo root (build context is the repo root).
- `server/deploy/deploy-docker.sh` — installs Docker on the droplet if absent,
  ships the working tree as the build context (works uncommitted), stops the
  systemd service, `docker compose up -d --build`.

One real bug found and fixed during deploy: the container runs as root but
`cap_drop: ALL` removes `CAP_DAC_OVERRIDE`, so uid 0 no longer bypasses file
permissions — reading the `pqcvpn`-owned `0600` identity seed failed with
`EACCES`. Fix: the deploy script `chown`s `/var/lib/pqc-vpn` to `root`.

### Verified

- Redeployed to the droplet as the `pqc-handshake-server` container. Startup log
  shows `psk install: wg set wg0` and `server wg pubkey hWFuGT6A2crU…` — the
  container queried the host's wg0 public key from inside its netns, i.e.
  `CAP_NET_ADMIN` reaches the host interface.
- **Live handshake against the container** (`test-client` from WSL, ML-KEM-1024):
  full 4-message exchange, PSK derived; container log
  `HANDSHAKE session=… algo=ML-KEM-1024 peer @ 10.8.0.2`; `wg show wg0` shows the
  new peer with a preshared key and `allowed ips: 10.8.0.2/32`.
- **Identity persists across `docker compose restart`** — verifying key
  `56d0ee98…` unchanged (volume). Same key as the Week 2/3 systemd deployments,
  so no client re-pinning.
- `cargo test --workspace --exclude desktop` — unchanged (core 20, handshake-server
  17 unit + 4 integration).

Known cosmetic issue: the in-memory address allocator resets on container
recreation, so stale `wg0` peers with `allowed ips: (none)` accumulate. Peer
lifecycle / cleanup is Week 5.

### Protocol-stability sync (the Week 4 checkpoint)

`server/PROTOCOL.md` gained a **Change log** section for Member 1: `VER` is still
`0x01`; the only movements since the Week 1 draft are the Week 2 nonce addition
to the transcript (an additive `core` ask, §8 Q1) and the Week 3 append-only
`ServerFinish` fields. The client side of `core/protocol/` should now be built
against this document and verified against `server/handshake-vectors.json`.

### Status against the Week 4 mandate

Responder containerised and running live on the droplet with a hardened compose
config; protocol change log written for the Member 1 sync. `core/` and
`contracts/` untouched.

## Week 5 (Member 2 track, 2026-09-08) — multi-peer session management

Assigned task (`TEAM_TIMELINE_PROPOSAL.md`): track which WireGuard peer maps to
which active PQC session, for more than one client. Also fixes the Week 4
stale-peer accumulation.

### `registry.rs` (new — replaces `session.rs` and `tunnel::PeerAddresses`)

One `Registry`, keyed by WireGuard public key (the stable peer identity). Each
`Peer` = `{assigned_ip, session_id, algo, created, last_activity, rekeys}`.
Unifies session tracking + IP allocation (lowest free host in the /24).

- **Persistence** — `/var/lib/pqc-vpn/peers.json`, atomic temp+rename after every
  handshake/rekey. **No PSKs** — the server never re-uses a stored PSK, and
  WireGuard keeps the live one. Survives `docker compose restart`.
- **Fresh `ClientHello` from a known peer** replaces its session (new
  `session_id`, new PSK, same address, rekey count reset) — the algorithm-change
  path from `PROTOCOL.md` §6.
- **Rekey lookup** by `(wg_pubkey, session_id)` — a stale/wrong `session_id` →
  `Error(UNKNOWN_SESSION)`.
- **Capacity** — `--max-peers` (default 64, hard-capped at 253 for a /24). A new
  peer over the limit → `Error(0x09 CAPACITY)`, a new protocol code.

### Startup reconciliation

On boot the server runs `wg show <iface> peers` and removes any `wg0` peer the
loaded registry does not know (`wg set peer … remove`). Registry peers missing
from `wg0` (e.g. after a `wg0` restart) are kept — their clients re-handshake.

### `tunnel.rs` changes

`PskInstaller` grew `list_installed_peers`, `remove_peer`, `last_handshakes`;
`install` dropped the `first_install` branch (always sets `allowed-ips`,
idempotent). `WgCli` shells out to `wg show … peers` / `latest-handshakes` and
`wg set … peer … remove`.

### Optional idle sweep

`--idle-timeout-secs` (default 0 = off): a background thread that every minute
evicts peers whose registry `last_activity` **and** last WireGuard handshake are
both older than the timeout. Implemented, off by default — real use is Week
9/10.

### Verified

- `cargo test -p handshake-server` — 22 unit + 4 integration (6 new `registry`
  tests: distinct-address allocation, known-peer re-handshake keeps IP,
  rekey session/downgrade checks, capacity, persist round-trip, idle sweep;
  `over_tcp_two_peers_get_distinct_addresses_and_sessions`).
  `cargo test --workspace --exclude desktop` — core 20/20, no warnings.
- **Live on the droplet** (containerised): redeploy → startup log
  `reconcile: 0 known / 2 on wg0 / 2 stale removed` — the Week 3/4 leftovers
  cleaned. Three `test-client` runs → `10.8.0.2` / `.3` / `.4`, distinct
  sessions and algorithms, all three in `peers.json` and as distinct `wg0`
  peers with their own `/32`. `docker compose restart` →
  `reconcile: 3 known / 3 on wg0 / 0 stale removed` — registry reloaded,
  tunnels preserved.

### Status against the Week 5 mandate

Multi-peer registry with persistence, capacity, and startup reconciliation,
verified live with three concurrent peers and a restart. `core/` and
`contracts/` untouched.

## Weeks 1–5 (Member 1 track, through 2026-09-14) — Rust core and Tauri shell

Recorded here so every member's track is in one document; the detail lives in
the commit messages and `core/README.md`.

| Week | Delivered |
|---|---|
| 1 | Cargo workspace per the plan's Section 3: `core/`, `desktop/` (Tauri + React), `mobile-bindings/pqc_vpn.udl` stub. |
| 2 | `core::crypto`: hybrid X25519 + ML-KEM-512/768/1024, ML-DSA-65 `ServerAuthenticator`, signed handshake transcript, zeroizing `SecureKeyStore` (the `key_store.py` port landed here, a week early). Pure-Rust `ml-kem` / `ml-dsa` / `x25519-dalek` rather than `liboqs-rust`, so there is no C dependency and the server shares the crate. |
| 3 | `core::state`: `DeviceState` and `TunnelHandle` traits, `StatePipeline` producing the frozen 7-dim vector with the contract's normalization caps, and `mock` test doubles. `combine_shared_secrets` pinned against CPython-computed digests. |
| 4 | Tauri shell: window, tray, and `connect` / `disconnect` / `connection_status` wired to a stub. Checkpoint items for `PROTOCOL.md` §8: nonces in the transcript (Q1), shared `derive_session_keys` (Q2), `ServerAuthenticator` seed persistence (Q7). Package renamed `core` → `vpn_core`. |
| 5 | `desktop/src-tauri/src/device_state.rs`: real `DeviceState` via `sysinfo` (CPU, RAM, ping latency, upload rate, link type), exposed as the `device_snapshot` command with a Device panel in the UI. |

Tests at Week 5: `vpn_core` 34, `desktop` 6.

## Week 6 (Member 1 track, 2026-09-22) — `core::protocol` client

`TEAM_TIMELINE_PROPOSAL.md`'s Week 6 task: integrate `protocol/`'s client
side against Member 2's server and get a first real handshake over the
network, Rust client against the real (not mocked) `server/handshake-server`.

`core/src/protocol/` was an empty stub before this week; `server/PROTOCOL.md`
names it directly ("build the client side of `core/protocol/` against this
document and verify it against `server/handshake-vectors.json`"). Delivered:

- **`wire.rs`** — message framing (`ClientHello`/`ServerHello`/
  `ClientFinish`/`ServerFinish`/`RekeyRequest`/`Error`), byte-for-byte against
  `PROTOCOL.md` §3–4. `crate::crypto::MlKemLevel` gained the wire byte-codes
  and encoded lengths so the protocol layer doesn't duplicate a second
  algorithm enum.
- **`tags.rs`** — the `ClientFinish`/`ServerFinish` confirmation MACs
  (§5.3). These previously only existed server-side
  (`server/handshake-server/src/kdf.rs`); moving them here is the piece
  `PROTOCOL.md` explicitly flags as needed once `core::protocol` exists.
- **`client.rs`** — `ClientHandshake`: generates ephemeral keys, builds
  `ClientHello`, verifies the server's ML-DSA-65 signature against a pinned
  key *before* deriving any secret, decapsulates, derives session keys.
- **`transport.rs`** — `connect()`, the only place in `core` that opens a
  real `TcpStream`; drives the four-message exchange and returns a
  `TunnelConfig` for `TunnelHandle::bring_up` (Week 9).

**Verification:** framing and MACs checked field-by-field against
`server/handshake-vectors.json` (not just Rust struct equality — re-encoded
bytes compared against the vector file's exact hex). `transport.rs` includes
an over-the-network test that spins up the real `handshake-server` connection
handler on a loopback socket and runs `core::protocol::connect` against it,
plus a negative test confirming a wrong pinned key is rejected client-side.

`cargo test --workspace` — `vpn_core` 52/52 (34 prior + 18 new), all other
crates unaffected (`handshake-server` 22+4, `desktop` 6).

## Week 4 checkpoint closed (2026-09-15) — contracts frozen, server on shared crypto

The end-of-Week-4 sync (`TEAM_TIMELINE_PROPOSAL.md` §6) is recorded:

- **Frozen:** `server/PROTOCOL.md` (DRAFT → FROZEN v1, all §8 questions resolved),
  `INTERFACE_FREEZE_PROPOSAL.md` (FROZEN), and the workspace layout.
- **`contracts/DECISIONS.md`:** Decision 1 `REKEY_ESCALATES` approved as shipped.
  Decision 2 is Option B, with the retrain deferred to Member 3's Week 11 so the
  ONNX artifact stays stable through Member 1's Week 7 `ort` integration.
  `algo_registry.json`'s rekey semantics regenerated to state the approved rule.
- **Server now uses `vpn_core`** for the transcript (new
  `build_handshake_transcript_from_bytes`, which the typed builder also calls),
  the HKDF key schedule, and the ML-DSA-65 identity. `ml-dsa` and `hkdf` are
  dropped from the server's dependencies. `server/handshake-vectors.json`
  regenerates byte-identical apart from the protocol status string, and
  existing identity seeds load the same key, so the droplet's pinned key is
  unchanged.
- **Plan gaps closed** in `TEAM_TIMELINE_PROPOSAL.md`: named owners for the Rust
  ports of the decision gate and anomaly layers (Member 3 writes them against
  Python vectors, Member 1 wires them in), a `core` trait for the Layer 2 live
  inputs, and the Week 11 retrain and artifact swap.
- **Proposal v2** (`RL_PQC_VPN_Proposal_v2.md` / `.pdf`) matches the design as
  built. v1 is kept for history.

## Week 6 (Member 2 track, 2026-09-22) — live rekey: test-client support + integration proof

Assigned task (`TEAM_TIMELINE_PROPOSAL.md`): support live rekey — when a
connected client's agent triggers `rekey-now`, the server must accept a fresh
handshake for an *existing* peer and swap its PSK without dropping the
tunnel; first real integration test against Member 1's client.

### What was already in place

Week 5's registry work landed ahead of the plan: the wire protocol's
`RekeyRequest` (`PROTOCOL.md` §4.6), `Registry::rekey` (session/downgrade
checks, `REKEY_ESCALATES` escalation), and `server.rs`'s connection handler
already dispatched `ClientHello`/`RekeyRequest` end-to-end into the same
`handle_connection`, and `PskInstaller::install` is idempotent — a rekey
re-sets the same `allowed-ips` and never calls `remove_peer`, which is the
no-drop guarantee by construction. What was missing, and flagged as deferred
in the Week 3 notes, was any way to *exercise* that path: `test-client` had
no rekey mode, and nothing drove `RekeyRequest` over a real TCP connection —
only `Registry::rekey` itself was unit-tested, in isolation from the wire
dispatch and the installer. This week closes that gap.

### `test-client --rekey`

A first run now also prints a ready-to-paste `test-client ... --rekey
--session <hex> --wg-key <base64>` command alongside the `wg-quick` config —
`--session` is the session id from `ServerFinish`, `--wg-key` is the same
client's WireGuard private key (a rekey must come from the same peer
identity). Running it sends a `RekeyRequest` instead of a `ClientHello`, and
on success prints the new derived PSK plus the exact `wg set <iface> peer …
preshared-key …` one-liner to hot-swap it — no `wg-quick down` — so the
rekey can be rehearsed by hand against a tunnel that's actually up and
passing traffic: bring the tunnel up from the first run's config, `ping`
through it, run the `--rekey` command, hot-swap the PSK, confirm the `ping`
never stopped and `wg show` reflects the swap.

### Integration tests (`tests/handshake.rs`)

New `RecordingInstaller` (a `PskInstaller` that logs every `install` /
`remove_peer` call instead of touching real WireGuard) backs four new
over-the-wire tests:

- `over_tcp_rekey_swaps_psk_without_removing_peer` — full handshake, then a
  `RekeyRequest` on the same session that escalates the algorithm. Asserts
  the assigned IP is unchanged, the two `install` calls used different PSKs
  and the escalated algorithm on the second, the registry still has exactly
  one peer — and, the actual no-drop proof, that `remove_peer` was never
  called across either step.
- `over_tcp_rekey_rejects_unknown_session` / `…_session_mismatch_for_known_peer`
  — a rekey claiming a session the registry has no record of, or the wrong
  session for a known peer, both come back `Error(0x04 UNKNOWN_SESSION)` over
  the wire and never touch the installer.
- `over_tcp_rekey_rejects_downgrade` — a rekey at a weaker algorithm than
  in-force comes back `Error(0x05 ALGO_MISMATCH)` and is not installed.

These are the first tests to exercise `RekeyRequest` through the real
`TcpListener` + `handle_connection` path rather than calling
`Registry::rekey` directly.

### Verified

- `cargo test -p handshake-server` — 22 unit + 8 integration (4 new
  `over_tcp_rekey_*`), all passing.
- **Live rekey against the droplet**, native Windows client (official
  WireGuard app + a natively-built `test-client.exe` — WSL2 was ruled out
  for this one test session by an unrelated local networking quirk, not
  anything in this code). Full PQC handshake at ML-KEM-768, tunnel brought
  up, then `test-client --rekey` against the same session escalated to
  ML-KEM-1024; server accepted it, returned the same session id and
  `10.8.0.2` address, derived a new PSK. Hot-swapped the `PresharedKey` in
  the running WireGuard tunnel (no deactivate). `wg show wg0` on the
  droplet afterward: same peer entry throughout (never removed/re-added),
  `latest handshake` refreshed a few minutes post-rekey, transfer counters
  climbing rather than reset — the tunnel never dropped.
- Droplet housekeeping done along the way: confirmed the `pqc-handshake-server`
  container and `wg0` both survive a droplet power cycle (Docker
  `restart: unless-stopped` + `wg-quick@wg0` enabled on boot); the registry
  was wiped (`rm peers.json` + container restart, which reconciles `wg0`
  back to empty) partway through testing to get a clean peer for the final
  run.

### Status against the Week 6 mandate

Server-side rekey was substantially built during Week 5; this week adds the
missing proof — a `test-client` rekey mode for rehearsing it against a live
tunnel, and integration tests that drive `RekeyRequest` over real TCP and
assert the no-drop property directly, rather than trusting it by
inspection. `core/` and `contracts/` untouched — nothing here blocks
Member 1's Week 6 client-side integration; the server already accepts real
`ClientHello`/`RekeyRequest` over TCP from any conforming client.

## Week 7 (Member 3 track, 2026-09-26) — decision-gate Rust port, anomaly Layer 3

The delivery plan's Week 7 task has two parts: the Rust reference port of the
decision gate for Member 1's Week 7 wiring, verified against
`contracts/decision_gate_vectors.json`, and anomaly Layer 3 (define the
feature vector, build benign training data, train `sklearn.IsolationForest`).
The gate went first because Member 1 is blocked on it this week.

### Closing a gap in the gate vectors first

The Week 4 fix (`test_escalation_resets_the_confirm_streak`) had no matching
case in `decision_gate_vectors.json`. A port could pass every shipped case and
still keep a streak alive across an *escalating* rekey. Two changes in
`export_contracts.py`:

- a 10th case, "an escalating rekey resets a challenger streak": two ticks
  build a streak toward ML-KEM-768, a rekey escalates to ML-KEM-1024, and the
  next challenger tick must start a fresh streak rather than confirm a
  one-tick downgrade;
- every tick's `expect` now carries `reason`, so a port has to match the fixed
  reason strings exactly. All seven are exercised across the cases.

After regeneration the `.onnx` and `policy_test_vectors.json` are
byte-identical. Only the gate vectors and the manifest changed.
`test_phase4.py`'s replay now also asserts `reason`.

### `vpn_core::rl::gate` (`core/src/rl/gate.rs`)

A line-for-line port of `DecisionGate.update`:

- **Types:** `in_force` is an `MlKemLevel` rather than an integer, so the gate
  cannot hold `rekey-now` as the algorithm in force. `PolicyAction::from_index`
  maps a policy output index to `Kem(MlKemLevel)` / `RekeyNow` by reusing
  `MlKemLevel::from_wire_code`. KEM action indices equal wire codes, so there
  is still only one algorithm table.
- **Constants and reasons:** constants use the same names as the Python, and
  the reasons are `&'static str`, so a tick allocates nothing.
- **Argmax:** ties go to the first index, matching Python's `max(range, key=)`.
- **`softmax` helper:** stable against large logits, because `ort` returns
  logits and the gate needs probabilities.

**Tests (6):**
- replays all 10 vector cases tick-by-tick, checking all five `expect`
  fields;
- a constants drift guard against the vector file;
- rekey cooldown expiry, `rekey_escalates = false`, index round-trip, and
  softmax.

`serde_json` is a dev-dependency only, so there is no new runtime dependency
in `core`.

**Mutation check:** reintroducing the Week 4 bug (leaving the streak untouched
on a rekey tick) fails the vector replay at "a rekey resets a challenger
streak, even when it does not escalate", tick 3. The first mutation attempt
did *not* reproduce the bug: it fell through to the branch that reassigns the
candidate. That's a reminder that a mutation check has to reproduce the bug
itself, not a lookalike.

`contracts/INTEGRATION.md` step 3 now points at `rl::gate`. It also fixes a
bug in the pseudocode: the old version applied `softmax` to vector `probs`
that the file already stores softmaxed.

### Layer 3 — `IsolationForestLayer` (`client/rl_agent/anomaly_detector.py`)

**Features** (`LAYER3_FEATURES`, frozen order, pinned by a test for the Week 8
Rust port). One vector per 5 s tick from caller-supplied packets (sizes,
arrival times, direction), via `tick_features`:

| feature | meaning |
|---|---|
| `out_size_mean` | bytes per outbound packet |
| `in_size_mean` | bytes per inbound packet |
| `pkt_size_std` | pooled size spread; ~0 means uniform packets (a scan) |
| `log_pkt_rate` | log10(1 + packets/s) |
| `iat_cv` | std/mean of inter-arrival gaps; ~0 is clockwork, ~1 Poisson, >1 bursty |
| `upload_share` | outbound bytes / total |

Ticks with fewer than 3 packets are not `ready`, because `iat_cv` is undefined
there. Inter-arrival *mean* was left out on purpose: over a fixed tick it is
just 1/rate.

**How the feature set was arrived at.** The first set was
pooled size mean/std, rate, `iat_cv`, and upload share. It detected exfiltration
**0%** of the time. The cause is structural, not a matter of tuning: an Isolation
Forest cannot score a point *beyond* its training range any higher than the
benign points at that edge. "More upload than any benign tick" therefore
scored like a video call. Splitting size by direction turns exfiltration into
high-rate traffic with tiny inbound packets. That region lies *inside* the
feature ranges and no benign profile occupies it, so it went to 100%.
Dropping the pooled std for that change cost port-scan detection (99% → 21.5%),
because uniform packet size was what gave scans away. Restoring it as a sixth
feature brought scans back to 100%.

**Training data** (`train_isolation_forest.py`). The data is simulated, since
there is no captured traffic yet. Five benign profiles (idle, browsing,
streaming, bulk download, video call) are built packet by packet and reduced
through the real `tick_features`. Three attack mixes (exfiltration, port scan,
beaconing) are used only for measurement; the forest never sees them.
Settings: `n_estimators=100`, `random_state=0`, 5,000 training ticks. The model
is saved to `models/isolation_forest.joblib` (403 KB, compressed), and the
calibration and measurements to `models/isolation_forest_calibration.json`.

**Scoring.** The threshold is the 99.5th percentile of held-out benign scores.
`threat_score` uses the same squash as Layers 1–2: 0 at the benign median, 0.5
at the threshold, 1.0 at twice that distance.

**Measured**, on held-out data; seeds 0–3 were each retrained and recalibrated:

| | seed 0 (shipped) | seeds 1–3 |
|---|---|---|
| benign false positives, overall | 0.50% | 0.50% |
| — idle (the noisiest: 5–25 packets a tick) | 2.50% | 2.25–2.50% |
| — browsing / streaming / bulk / video call | 0.00% | 0.00% |
| exfiltration detected | 100% | 100% |
| port scan detected | 100% | 98.8–100% |
| beaconing detected | 87.5% | 81.5–88.5% |

These are numbers against simulated shapes, not captures. The profiles are
reasoned first-pass choices like `state_observer.py`'s caps, and need
re-deriving once Member 1's Week 8 live source can record real traffic. A
legitimate large upload (a cloud backup) would look like exfiltration here.
That is the same caveat Layer 2's `BandwidthExfiltrationSignature` carries,
and no benign upload-heavy profile was invented to hide it.

`LAYER3_CPU_GATE = 0.40` is defined. The combiner does not read it yet.

### Tests — `tests/test_phase7.py` (new, 19 tests)

- **Feature contract:** a feature-order pin, the calibration file matching the
  module, the layer refusing a calibration with a different order, and the
  gate ordering (L3 < L2).
- **`tick_features`:** checked on a hand-built tick with every value computed
  by hand, plus timestamp sorting, the one-direction case, and a sparse tick
  being not-`ready`.
- **Measured-behaviour floors on a fresh seed:** benign FP ≤ 2% per profile
  (idle ≤ 6%); detection ≥ 95% for exfiltration and port scan, and ≥ 70% for
  beaconing.
- **Scoring:** result shape and [0,1] range, the squash checked through
  `update()` with a stub model, and training determinism under a seed.

`pytest tests/ -v`: 113/113 (94 prior + 19 new). `cargo test --workspace`:
`vpn_core` 58/58 (52 prior + 6 new), and every other crate is unaffected.
`python demo.py` runs end to end.

### Deliberately not done (Week 8 — see that section below)

- Layer 3 is not wired into `AnomalyCombiner`, so the THREAT
  `produced_by` text is unchanged. **Done Week 8.**
- There is no Rust export of the forest. It will go through either `skl2onnx`
  or a hand-written tree-traversal port. The joblib pickle is Python-only and
  tied to scikit-learn 1.5.0 (recorded in the calibration file). **Done Week 8:
  hand-written tree-traversal, tree-structure-level export.**
- There is no Rust port of Layers 1–2 + the combiner, and no
  `contracts/anomaly_vectors.json`. **Done Week 8.**

### Status against the Week 7 mandate

- **Decision gate:** the Rust reference port is shipped and verified against
  the vectors. The vectors now cover the Week 4 bug, and the replay was shown
  to catch it.
- **Layer 3:** the feature vector is defined, benign training data exists
  (simulated, and stated as such), and the Isolation Forest is trained,
  calibrated and tested.

## Week 8 (Member 3 track, 2026-11-27) — real-grounded Layer 3, full Rust anomaly port

The delivery plan's Week 8 task is: wire `LAYER3_CPU_GATE` into the combiner,
port Layers 1–2 + the combiner to Rust, export the Isolation Forest for Rust
consumption, and build `contracts/anomaly_vectors.json`. Asked for deeper
rather than bigger: real-capture tooling for Layer 3 (today still 100%
synthetic, see below), an adversarial mimicry/evasion test against the
forest's known axis-aligned-split weakness, tree-structure-level Rust parity
(not just output-level), property-based tests, a `cargo-fuzz` harness, and a
local CPU-gate profile — on top of the plan's own checklist, not instead of
it.

### Combiner wiring — `client/rl_agent/anomaly_detector.py`

`AnomalyCombiner` now owns an `IsolationForestLayer` and calls it when
`cpu_load < LAYER3_CPU_GATE` (0.40 — stricter than Layer 2's 0.70, since it
is the most expensive layer), folding its score into the existing `max()`
over every ready layer exactly as Layer 2's signatures already do. A
not-ready Layer 3 tick (sparse, below `LAYER3_MIN_PACKETS`) contributes
nothing rather than a masking 0.0, so a real Layer 1/2 hit still surfaces.
`contracts/state_vector.json`'s THREAT field description is updated to say so.

**Tests** — new `tests/test_phase8.py` (5 tests, a fake Layer 3 injected via
the constructor so the wiring is checked independent of the forest's own
quality): the gate boundary in both directions, Layer 3's score
participating in the max, and the not-ready-doesn't-mask rule.

### Real-capture tooling — exercised, model is now real-grounded

`client/rl_agent/capture_scenario.sh` (idle/browsing/beaconing/
bulk_upload/port_scan, sudo-gated `tcpdump`/`nmap`, loopback-only for
anything attack-shaped) and `client/rl_agent/capture_real_traffic.py`
(offline `scapy` parsing into `tick_features()`'s shape, direction
classified by source IP on a real NIC or by TCP port role/flags on
loopback) have now been run for real: 238 real ticks (idle 59, browsing 62,
beaconing 91, bulk_upload 25, port_scan 1 — a full 65535-port loopback SYN
scan has zero network latency and finishes inside one 5s tick; a real scan
over an actual network would span many). Written to
`models/real_capture_corpus.json` (committed; raw pcaps gitignored).
`train_isolation_forest.py` mixes the real benign rows into training with a
**temporal** train/test split (never random-shuffled: adjacent real ticks
are autocorrelated); `score_median`/`score_threshold` are retrained against
this mix, so `real_capture_used: true` in the calibration file now reflects
reality, not a placeholder.

**Two bugs found and fixed while running the capture for real, not
hypothetically:** `bulk_upload`'s original `cat file | nc -w timeout` push
was verified to silently truncate a 200MB transfer to a few KB — BSD `nc`'s
stream-closing semantics on macOS aren't reliable for a one-shot bulk send.
Replaced with a `curl PUT` against a one-shot Python `http.server` (explicit
`Content-Length` framing, verified byte-identical). The first fix then hit a
second, different failure: pushed at full loopback speed, `sudo tcpdump`
dropped 68% of packets (kernel buffer overrun, not a logic bug) — fixed by
pacing the push at a fixed 5MB/s with `curl --limit-rate`, which also better
matches what a real sustained exfiltration looks like than an instant burst
would. Final capture: 57,640/57,774 packets (99.77%) — see
`capture_scenario.sh`'s `bulk_upload` case and its inline comments for both.

### Retraining — ROC/PR and bootstrap CIs replace point estimates, now on real traffic

`train_isolation_forest.py`'s `calibrate_and_measure` now reports, for every
benign profile and attack mix, a bootstrapped 95% CI (2,000 resamples)
alongside the point rate, plus an overall ROC/PR curve and AUC for the
benign-vs-attack separation `score_threshold` actually trades off (not
per-mix — one operating point, one curve). Measured against the final
model, trained on the 5 synthetic profiles **plus** the real benign rows
above:

| | value |
|---|---|
| ROC AUC | 0.9956 |
| PR AUC | 0.9883 |
| benign FP, synthetic idle (noisiest synthetic) | 1.00%, CI [0.25%, 2.00%] |
| benign FP, other 4 synthetic profiles | 0.00%, CI [0.00%, 0.00%] |
| benign FP, **real idle** | **16.67%**, CI [0.00%, 38.89%] |
| benign FP, **real browsing** | **21.05%**, CI [5.26%, 42.11%] |
| exfiltration (synthetic) detected | 100.00%, CI [100.00%, 100.00%] |
| port scan (synthetic) detected | 99.25%, CI [98.25%, 100.00%] |
| beaconing (synthetic) detected | 47.25%, CI [42.25%, 52.25%] |
| real bulk-upload detected | 100.00%, CI [100.00%, 100.00%] |
| real port scan detected | 100.00% (n=1, not statistically meaningful) |
| real beaconing detected | 28.57%, CI [19.78%, 38.46%] |

**The headline finding, stated plainly rather than buried:** real ambient
traffic is measurably noisier than the synthetic generator ever modeled —
real idle/browsing false-positive at 16.67%/21.05%, an order of magnitude
above any synthetic profile's worst case (1.00%). Mixing it into training
also shifted `score_threshold` up enough to cost beaconing detection
specifically: synthetic beaconing fell from Week 7's 87.50% to 47.25% (real
beaconing itself detected at only 28.57%). This is the real-world gap the
synthetic-only model was always going to have and couldn't show — it is not
a regression, it is what grounding in real data was for. `tests/test_phase7.py`'s
beaconing floor is lowered accordingly (0.70 → 0.40, with the reasoning
recorded inline) rather than silently left to fail or quietly loosened
without explanation.

### Mimicry/evasion test — `tests/test_phase8_mimicry.py`

Precise framing, because it matters for how these numbers should be read: a
low detection rate below demonstrates that a *stateless, per-tick-only*
detector can be defeated this way. It is not a general claim that Isolation
Forests are weak — a session-level detector would plausibly catch both.

- **Slow-and-low exfiltration**: a session built entirely from the top 1%
  most upload-heavy *benign-generator* ticks (not hand-invented numbers —
  real samples the model was trained to consider normal), replayed 300
  times. Real exfiltration is caught at 100%; this construction is flagged
  **8.0%** of the time (post real-grounding; was 28.0% against the
  synthetic-only Week 7 model). Each assertion compares against the real
  attack's own measured rate (read from the calibration file, margin 30
  points) so the test stays meaningful across a retrain rather than pinned
  to one model's numbers.
- **Thin-spread port scan, sub-minimum variant**: pacing probes below
  `LAYER3_MIN_PACKETS` per tick means every tick is simply never scored —
  **0%** flagged, not because the forest misjudges it but because Layer 3
  never runs.
- **Thin-spread port scan, idle-shaped variant** (exactly 3 packets/tick,
  sized and timed like the idle profile): **18.7%** flagged (was 44.3%
  pre-real-grounding) vs. real port scan's 99.2% — the forest cannot tell a
  few small idle-ish packets from a few small scan probes from shape alone
  within one tick.

### `vpn_core::anomaly` (`core/src/anomaly/mod.rs`, `core/src/anomaly/forest.rs`)

Line-for-line port of Layers 1–2 + the combiner, mirroring `rl::gate.rs`'s
exact pattern (plain numeric inputs, `pub const`s, inline
`#[cfg(test)] mod tests` with `include_str!` of `contracts/anomaly_vectors.json`,
a constants drift-guard). Layer 3 is a genuinely new pattern for this
codebase: `forest.rs` embeds two Python-exported JSON artifacts at compile
time (`isolation_forest_trees.json`, `isolation_forest_calibration.json`)
and walks the exact trees sklearn trained — `feature`/`threshold`/
`children_left`/`children_right` plus a **precomputed per-leaf path-length
correction** (`client/rl_agent/export_forest_trees.py` computes sklearn's
own `c(n)` harmonic-number formula once in Python) — rather than
reimplementing sklearn's path-length math in Rust. `serde_json` is promoted
from a dev- to a normal dependency of `core` to parse these at startup:
`core`'s first production dependency beyond crypto/rl primitives.

**New `contracts/anomaly_vectors.json`** (`build_anomaly_vectors()` in
`export_contracts.py`, mirroring `build_decision_gate_vectors()`): 11 named
scenarios replayed tick-by-tick through a real `AnomalyCombiner` — Layer 1
alone, each Layer 2 signature individually, Layer 3 on real benign- and
scan-shaped ticks, both gate boundaries, and the not-ready-masking case —
with every `expect` field captured as actual run output, never hand-typed.
**New `contracts/isolation_forest_test_vectors.json`**: representative
feature vectors (every benign profile, every attack mix, zero and extreme
edge cases) with sklearn's own `score_samples()` output, for numeric parity
at both the raw-score and calibrated-threat-score level.

**Tests (12 new, 70 total in `vpn_core`):** the two vector replays above,
a constants drift-guard, a not-ready-contributes-nothing check, and
property tests (`proptest`, new dev-dependency): the tree walk always
terminates at a leaf with a finite positive score for any finite input
including boundary/extreme values and ±infinity; gating a layer off never
increases the combiner's score; `threat_score` always stays in [0,1];
`tick_features` is always finite whenever it returns `Some`.

### `core/fuzz/` — cargo-fuzz, scoped to its own nightly toolchain

New subcrate, own `rust-toolchain.toml` (`channel = "nightly"`) and own
empty `[workspace]` table, excluded from the root workspace so its
nightly-only tooling never touches the stable build the rest of the project
uses. Two targets: `anomaly_forest` (arbitrary `[f64; 6]` vectors, including
NaN/Inf from raw bytes, straight into the tree-walker) and
`anomaly_combiner` (arbitrary numeric inputs plus independently-sized
packet arrays into the full combiner). Both ran clean for 20s locally with
no crashes (~1.3M execs for the forest target; ~9k execs for the combiner
target — markedly slower because it rebuilds the embedded forest from JSON
every single iteration rather than sharing one across iterations, a
performance limitation worth fixing before a serious fuzzing campaign, not
a correctness one).

### Local CPU-gate profile — `core/examples/anomaly_bench.rs`

1,000,000 ticks (cycling idle/browsing/scan/exfiltration-shaped packets)
through the full combiner + forest: **3.2–3.7 µs/tick** on this Apple
Silicon Mac, profiled with `samply` under the new `[profile.profiling]`.
Against a 5 s real tick that is roughly six orders of magnitude below the
tick budget — a directional sanity check that the <40% CPU gate is in the
right ballpark, explicitly **not** a validation of the deployment droplet's
Linux/x86 1-vCPU architecture, which this was never run on.

### Fixed along the way

`export_contracts.main()` was already calling `export_onnx.main()`, which
failed on a fresh venv (`ModuleNotFoundError: No module named 'onnxscript'`)
— torch's current ONNX exporter path imports it transitively and it was
never in `requirements.txt`. Added. Unrelated to anomaly detection, but it
blocked the whole export pipeline from running at all.

### Tests

`pytest tests/`: 121/121 (`liboqs-python` built from source against a
matching `liboqs` and installed into the venv; `test_phase7.py`'s beaconing
floor lowered 0.70 → 0.40 to match the real-grounded model, reasoning
recorded inline, see above). `cargo test --workspace`: `vpn_core` 70/70
(58 prior + 12 new), `handshake-server` 8/8, `desktop` builds clean (disk
space freed up; no longer excluded). `cargo fuzz run` (both targets, 20s
each): no crashes. `python demo.py`: full end-to-end handshake flow across
every scenario, no errors.

### Deliberately not done

- No session-level or multi-tick detector, which the mimicry section above
  states plainly would plausibly catch both constructions it demonstrates.
- The `anomaly_combiner` fuzz target's per-iteration forest reconstruction
  (performance, not correctness — see above).
- No `cargo-flamegraph` fallback was needed; `samply` worked on the first
  try on this macOS build.
- `real_port_scan`'s 100% detection rate is n=1 (one real scan, which
  finishes inside a single tick over zero-latency loopback) — reported
  above but explicitly flagged as not statistically meaningful, rather than
  presented alongside the bootstrapped rates as if it carried equal weight.

## Week 7 (Member 1 track, 2026-11-27) — RL policy wired into the client

`TEAM_TIMELINE_PROPOSAL.md`'s Week 7 task: integrate Member 3's exported
ONNX policy via the `ort` crate (action count read from the model's output
shape), wire Member 3's Rust decision-gate port into `core::rl`, then `rl/`
output into `crypto/`'s algorithm selection.

### Delivered

- **`core::rl::policy`** — loads `ppo_vpn_agent.onnx` with `ort`
  (2.0.0-rc.15, exact pin; ONNX Runtime downloaded and linked at build
  time, rustls for the download so Linux needs no OpenSSL headers). The
  model is embedded with `include_bytes!`, like Layer 3's trees. Checks at
  load: one input `state`, an output `action_logits`, and the action count
  read from that output's shape. A count other than the gate's 4 is refused
  with an error naming the three files to re-copy, rather than letting
  indices silently change meaning.
- **`core::rl::agent`** — policy → softmax → `DecisionGate` → one `Action`
  per tick: `Stay`, `Renegotiate(level)` or `Rekey(level)`. The level in the
  action is the level the next handshake runs at — this is the
  `rl/` → `crypto/` link. The connect-time algorithm also comes from the
  policy (its top-ranked KEM on the first reading), and the gate is built
  around it, as `INTEGRATION.md` Step 3 asks.
- **`core::protocol::{handshake, rekey}`** — `connect` only returned a
  `TunnelConfig`, so there was no way to send `RekeyRequest`. `handshake`
  now also returns the session id (`Established`), `rekey` sends
  `RekeyRequest` with it and checks the server kept the same id. Connecting
  now uses a timeout, so a dead server can't stall a tick for the OS
  default.
- **`core::session::VpnSession`** — the control loop. `connect` reads the
  device, lets the policy pick the algorithm, handshakes, brings the tunnel
  up. Each `tick(elapsed, threat)`: state vector → agent → carry the action
  out (`ClientHello` for a change of algorithm — the only legal way to step
  down, `PROTOCOL.md` §6 — or `RekeyRequest` for a rekey), hot-swap the PSK
  through `TunnelHandle::rotate_psk`, reset the rekey timer. The session
  owns no clock or thread; the caller drives ticks.
- **`DecisionGate::reset_in_force`** (small addition to Member 3's port) —
  the gate moves `in_force` assuming the handshake will succeed. When it
  doesn't, the session puts it back. It keeps the rekey cooldown on
  purpose, so a failed rekey isn't retried against a dead server every 5 s.

### Decisions worth flagging

- **A renegotiation also resets `TIME_SINCE_REKEY`.** The simulator only
  resets it on the rekey action, but `state_vector.json` defines the field as
  "wall-clock since last key rotation", and a fresh handshake rotates the
  keys. Raised here so Member 3 can object if the policy needs the
  simulator's narrower meaning.
- **A failed action is not an error.** Network or tunnel failures come back
  as `Outcome::Failed` in the tick report and the session continues on its
  old keys; only a policy failure is an `Err`.
- **Pending on the server side**, not this track: Member 2's Week 7 rate
  limit could now reject the client's handshakes. The gate's 3-tick confirm
  and 12-tick rekey cooldown cap the client at well under one handshake per
  15 s, but the two limits haven't been checked against each other yet.

### Verification

`cargo test -p vpn_core` — 90/90 (52 at Week 6, plus Member 3's anomaly
and gate tests, plus 20 new):

- **Policy parity** against all 63 `policy_test_vectors.json` states:
  argmax identical on every vector, and the logits match bit-for-bit
  (worst |diff| = 0, contract tolerance 1e-5). The embedded `.onnx` is
  checked against its SHA-256 in `contracts/manifest.json`.
- **Rekey over a real socket** against `server/handshake-server`'s real
  connection handler: same session id, same tunnel address, new PSK;
  escalation 512 → 1024 accepted; a downgrade rekey refused with
  `ALGO_MISMATCH`; a step down via fresh `ClientHello` accepted for the same
  peer.
- **Control loop end to end** (real policy, real gate, real server handler,
  mock device and tunnel): a steady state touches no network; need rising to
  cellular + stale keys + threat 1.0 renegotiates to ML-KEM-1024 after
  exactly 3 ticks, and the server agrees on the level; the same state with
  the CPU at 90% rekeys instead of moving to the heavier KEM, keeps the
  session id, and the cooldown holds; a dead server leaves the session on
  its old algorithm.

The test states were picked by probing the shipped policy over states the
mock device can produce, each where the policy is decisive (top probability
≈ 1.00), so a retrain is unlikely to flip them without also changing the
policy's behaviour in a way worth noticing.

## Week 8 (Member 1 track, 2026-11-27) — anomaly layers wired in; first vertical slice

`TEAM_TIMELINE_PROPOSAL.md`'s Week 8 task: integrate Member 3's Rust
anomaly port into `core::anomaly` and the state pipeline; implement the
desktop side of the Layer 2 live-input trait; **checkpoint: first full
vertical slice** — real desktop app, real agent decision, real handshake.

### Delivered

- **`core::state::TrafficSignals` + `TrafficSample`** — the platform trait
  the plan's risk table called for: per-tick packet rate/size, up/down
  bytes, new outbound connections / failures / distinct ports, TCP
  retransmit rate, DNS server, and per-packet records for Layer 3. `core`
  defines it; `desktop/` reads it.
- **`StatePipeline::read` + `vector`** — the device is now read once per
  tick and the same readings feed both the anomaly layers and the state
  vector (previously `observe` read it itself, which would have meant two
  pings per tick). `observe` still works as before.
- **`core::session::threat::ThreatMonitor`** — maps one tick's readings onto
  Member 3's `AnomalyCombiner` in the reference units (latency ms,
  packets/s, bytes; 50-tick Layer 1 window). An unreachable probe is fed as
  the ping's 1 s timeout. A failed DNS read reuses the last known server,
  so it can't fire the DNS-change signature.
- **`VpnSession` now owns the anomaly pipeline.** `tick(elapsed)` reads
  device + traffic → anomaly layers → `THREAT` → policy → gate → handshake.
  Each `TickReport` carries the combined reading (which layers the CPU gate
  let run) and the raw traffic sample, so a threat score can be traced to
  its inputs.
- **`desktop/src-tauri/src/traffic.rs`** — real `TrafficSignals`: Windows
  via the IP Helper API (`GetExtendedTcpTable`/`GetTcp6Table`,
  `GetTcpStatisticsEx`, `GetNetworkParams`), Linux via `/proc/net/tcp{,6}`,
  `/proc/net/snmp`, `/etc/resolv.conf`; rates from `sysinfo`. The Linux
  parsers are pure functions tested on every platform.
- **`desktop/src-tauri/src/vpn.rs` + Connect button** — Connect now runs the
  real session on a background thread (handshake, then a tick every 5 s),
  pushing each tick to the UI as a `vpn-event`. The UI has a server form
  (defaults: the droplet and the pinned `server-vk.hex`) and a live panel:
  algorithm in force, threat score, layer status, the policy's
  probabilities, the gate's reason, the action taken.
- **`examples/vertical_slice.rs`** — the same loop, headless, for
  terminals and CI.

### The vertical slice, run

Local `handshake-server` (dry-run PSK install) on this Windows laptop,
client via `cargo run -p desktop --example vertical_slice`, 10 ticks, all
inputs live: the policy picked ML-KEM-512 at connect, a real handshake
completed (tunnel IP assigned, PSK installed server-side), and every tick
ran readings → anomaly → policy → gate → action.

**Not yet done for the checkpoint:** a run against Member 2's droplet, which
needs the team (it registers a peer on the shared server). Command:
`cargo run -p desktop --example vertical_slice` with no arguments uses the
droplet and its pinned key.

### Findings from running it on real data

1. **A bug of mine, found and fixed.** The first run flagged a port scan on
   an idle laptop. The connection tracker counted short connections that had
   already finished (`TIME_WAIT`) as failed attempts. Only `SYN_SENT` counts
   now; regression test added.
2. **For Member 3: on real Wi-Fi, Layer 2 sits at threat 1.0.** With the
   bug fixed, this laptop's network still scores 1.0 nearly every tick, from
   real inputs: system-wide TCP retransmits of 3–39% (the signature's
   threshold is 10%, saturating at 20%) and ping times swinging past the
   300 ms cap. The layers aren't wrong about the inputs; the thresholds were
   set on simulated and loopback traffic. The policy didn't overreact here
   (it stayed on ML-KEM-512 because the keys were fresh), but over a long
   session a permanently saturated `THREAT` will push it towards 1024 and
   rekeys. Worth calibrating against real captures before the Week 9
   evaluation.

### Known limits

- **Connections are sampled, not traced.** A connection that opens and
  closes between two snapshots is invisible, including a probe to a closed
  port. Probes to filtered ports (left in `SYN_SENT`) are seen.
- **Layer 3 gets no packets on desktop yet.** There's no packet capture
  (it needs admin rights or Npcap/libpcap), so Layer 3 reports not-ready and
  adds nothing to the score.
- **The tunnel is still a placeholder** (`vpn::PendingTunnel`): the PSK is
  installed on the server, not on a local WireGuard interface. That's
  Week 9. The client's WireGuard key pair is also generated per session for
  now; Week 9 has to persist it.
- **macOS** gets the `sysinfo` numbers only (out of scope per the plan).

### Verification

- `cargo test -p vpn_core` — 96/96 (6 new: threat mapping, layer gating,
  DNS-read failure, and the session tests now drive `THREAT` through the
  real anomaly layers from traffic inputs).
- `cargo test -p desktop` — 17/17 (11 new: connection tracker incl. the
  `TIME_WAIT` regression, retransmit maths, `/proc` parsers, config
  validation, plus one test that reads this machine's live traffic).
- `cargo test -p handshake-server` — 22 + 8, unchanged.
- `npm run build` in `desktop/` — the TypeScript type-checks and builds.

Build note for Windows: building `desktop` writes a ~1.2 GB static library
(Tauri's mobile crate types with ONNX Runtime linked in). On a nearly full
disk, run `cargo clean` first; `CARGO_INCREMENTAL=0` and `--jobs 2` help on
low-RAM machines.

## Week 7 (Member 2 track, 2026-11-28) — replay protection and handshake rate limiting

Assigned task (`TEAM_TIMELINE_PROPOSAL.md`): replay protection and basic
handshake rate-limiting, because a PQC handshake responder on the internet is
a target. `PROTOCOL.md` §7 had reserved this since Week 1 as a 2-minute
`client_nonce` cache plus a per-source-IP limit returning
`Error(RATE_LIMITED)`, with no wire change needed.

### Why it matters even though the crypto already resists replay

Replaying a recorded `ClientHello` could never produce a working session,
because the replayer doesn't have the client's ephemeral secrets and can't
compute `ClientFinish`. But the server did not know that until message 3. By
then it had already run an ML-KEM encapsulation, an X25519 exchange and,
most expensively, an **ML-DSA-65 signature** for every replayed or flooded
hello. On top of that, the server is blocking thread-per-connection with a
10 s socket timeout, so a client that opens sockets and goes silent holds one
thread each with no upper bound. This week's job is to make both cheap to
refuse.

### `guard.rs` (new)

Three independent checks, all run before any post-quantum work:

| check | where | default | refusal |
|---|---|---|---|
| `NonceCache`: every `client_nonce` remembered for the window | message 1: `ClientHello` *and* the hello inside `RekeyRequest` | 120 s | `Error(0x06 AUTH_FAILED)` |
| `RateLimiter`: token bucket per source IP | accept loop | burst 20, refill 30/min | `Error(0x07 RATE_LIMITED)` |
| concurrent-connection cap (RAII `Permit`, released on drop/panic) | accept loop | 64 | `Error(0x07 RATE_LIMITED)` |

Design choices worth recording:

- **The nonce is recorded on receipt, not on completion.** A hello that was
  never finished still can't be replayed.
- **A replay maps to `0x06`, not a new code.** The v1 code table is frozen
  and has no replay code. A nonce reuse means the handshake can't be shown
  to be fresh, so it falls under `AUTH_FAILED`. A real client with 32 random
  bytes per hello never hits this.
- **The concurrency slot is taken before the rate-limit token**, so a
  connection refused for "server busy" doesn't also burn that IP's budget.
  Unit-tested.
- **A refused check doesn't reset the bucket's refill clock**, so a client
  that keeps retrying still recovers at the configured rate. Unit-tested.
- **Both tables are bounded** (100k nonces, 10k IPs). When a table is full
  of unexpired entries the server fails closed (`0x07`) instead of growing.
  A background thread prunes expired entries every minute.
- **The defaults are generous on purpose.** Several team laptops on campus
  Wi-Fi share one public IP, and the multi-laptop demo must not
  rate-limit itself. Every limit is a CLI flag: `--replay-window-secs`,
  `--rate-burst`, `--rate-per-min`, `--max-conns`.
- **Refusals happen on the accept thread**, with a 200 ms write timeout on
  the error frame, so a peer that never reads can't stall `accept` for
  everyone else.

### Server wiring, without touching `core/`

`server::serve` now takes an `Arc<Guard>` and admits each connection before
spawning its handler, which calls the new `handle_connection_guarded`.
`ServerContext` and `handle_connection(stream, ctx)` are **unchanged**: the
latter is now the guard-less path. `vpn_core`'s Week 6 loopback tests build
a `ServerContext` and call `handle_connection` directly, so adding a field
or changing that signature would have broken Member 1's code for no benefit.
`main.rs` builds the guard from the CLI flags and prints its configuration on
startup.

### Tests

- **12 unit tests in `guard.rs`:**
  - replay inside the window and acceptance again after it;
  - pruning removes only expired entries;
  - a full nonce cache fails closed, then recovers once entries expire;
  - the bucket allows its burst, then refills at the configured rate;
  - a refused attempt doesn't reset the refill clock;
  - zero refill makes the burst a hard total;
  - IP pruning, and a full IP table failing closed only for *new* IPs;
  - concurrency slots are released on drop, and a refused admit doesn't
    leak a slot;
  - a "busy" refusal doesn't spend the IP's token;
  - the protocol code mapping.
- **6 integration tests (`guarded_*` in `tests/handshake.rs`)**, all through
  the real `server::serve` accept loop over loopback TCP:
  - a byte-for-byte replayed `ClientHello` gets `0x06` and never reaches
    `wg set`;
  - a replay of an *abandoned* hello is also refused;
  - a `RekeyRequest` wrapping a spent nonce is refused, while a genuine
    fresh-nonce rekey on the same session still succeeds, with no
    `remove_peer`;
  - the 4th connection from one IP with burst 3 gets `0x07`;
  - a silent connection holding the only slot causes `0x07` for the next,
    and service recovers once it disconnects;
  - 10 normal clients see no difference.
- **Mutation check:** with the nonce check disabled, exactly the three
  replay tests fail. The rate and cap tests are independent of it.
- **Smoke test with the real binaries** (Windows, dry-run, `--rate-burst 2
  --rate-per-min 0`): two `test-client` runs completed handshakes; the third
  printed `server returned Error 0x07: rate limited: too many handshakes
  from this address`, and the server logged `REFUSED 127.0.0.1: …`.

`cargo test --workspace`: `handshake-server` 34 unit + 14 integration (12 + 6
new), `vpn_core` 70/70 (unchanged), `desktop` 6/6.

### Docs

`PROTOCOL.md` §7 is rewritten from "fields defined now, enforcement in Week
7" to the enforced behaviour, with a table of checks, defaults and refusal
codes, plus a change-log row (no wire-format change, no new codes). It also
adds a **note for client implementers**: a connection refused on accept
receives the `Error` frame *before* the client has sent anything. A client
should read what the server sends rather than assume its own `ClientHello`
write goes first, and should treat `0x07` as transient and retry with
backoff. The crate README documents the new module, flags and tests.

### Not verified yet

- **Against the droplet:** it's currently unavailable (the DigitalOcean
  subscription is paused). Nothing here depends on WireGuard, and the guard
  runs identically in dry-run, so the loopback tests and local smoke test
  cover the logic. A redeploy only needs to confirm the `guard :` startup
  line and one `REFUSED` under a forced low `--rate-burst`.
- **Persistence:** the replay cache and buckets are in memory and are lost
  on a restart. That is acceptable here. Replaying a hello after a restart
  still can't complete a handshake. It only costs the server one signature,
  and the rate limiter covers that cost.
- **Per-IP limits can't distinguish clients behind one NAT.** This is the
  reason for the generous defaults above. Per-peer (per-`client_wg_pubkey`)
  limits aren't possible on accept, because the pubkey arrives only in
  message 1, and they would be trivially evaded with fresh keys anyway.

### Status against the Week 7 mandate

Done: replay protection (the `client_nonce` cache, covering rekeys too) and
handshake rate limiting (per-IP token bucket plus a server-wide connection
cap), all refused before any PQC work, with no protocol change, and
`core/` and `contracts/` untouched.

## How to reproduce

```bash
source venv/bin/activate
pip install -r requirements.txt   # liboqs-python must be built separately (see liboqs-python.org)
pytest tests/ -v                  # 121/121
cargo test --workspace            # vpn_core, handshake-server, desktop
cargo run -p handshake-server --bin emit-vectors   # regenerate server/handshake-vectors.json

python demo.py                            # end-to-end walkthrough
python -m client.rl_agent.evaluate        # return vs baselines and the oracle
python -m client.rl_agent.analyze_boundary  # the four before/after metrics
python -m client.rl_agent.export_contracts  # regenerate contracts/ + the .onnx + anomaly_vectors.json

python -m client.rl_agent.verify_policy      # Week 3 robustness checks
python -m client.rl_agent.decision_gate     # sizes the debounce constants
python -m client.rl_agent.train_isolation_forest  # train + calibrate anomaly Layer 3 (ROC/PR, bootstrap CIs)
python -m client.rl_agent.export_forest_trees     # Week 8: tree-structure export for the Rust port

# Week 8: real-capture grounding for Layer 3 -- run each scenario yourself
# (needs interactive sudo for tcpdump/nmap), then parse and retrain:
./client/rl_agent/capture_scenario.sh idle
./client/rl_agent/capture_scenario.sh browsing
./client/rl_agent/capture_scenario.sh beaconing
./client/rl_agent/capture_scenario.sh bulk_upload
./client/rl_agent/capture_scenario.sh port_scan
python -m client.rl_agent.capture_real_traffic     # writes models/real_capture_corpus.json
python -m client.rl_agent.train_isolation_forest   # retrains with real_capture_used=true
python -m client.rl_agent.export_contracts         # re-exports everything downstream

# Week 8: fuzzing + profiling (core/fuzz/ has its own nightly rust-toolchain.toml)
cd core/fuzz && cargo fuzz run anomaly_forest -- -max_total_time=120
cd core/fuzz && cargo fuzz run anomaly_combiner -- -max_total_time=120
cargo build --profile profiling --example anomaly_bench -p vpn_core
samply record ./target/profiling/examples/anomaly_bench

python -m client.rl_agent.train             # single run, Week 2 defaults
python -m client.rl_agent.train --sweep --promote            # the Week 2 experiment
python -m client.rl_agent.train --sweep --ladder week3 --timesteps 300000   # Week 3 ablation
```

Note on `pip install`: installing `onnx`/`onnxruntime` pulls numpy 2.x,
which makes stable_baselines3 emit deprecation warnings when loading a
saved model. `requirements.txt` pins numpy back to 1.26.4 and orders the
entries so a plain install ends up on the pin.
