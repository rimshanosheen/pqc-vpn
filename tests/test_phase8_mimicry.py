"""
Week 8 (Member 3 track): adversarial mimicry/evasion tests against Layer 3
(Isolation Forest).

Kept deliberately separate from `tests/test_phase8.py` (combiner wiring) and
`tests/test_phase7.py` (the forest's own measured behaviour on the synthetic
generator's own attack mixes) -- this file asks a different question: can a
session be built that performs a real attack action while never tripping
the detector, by exploiting the fact that an Isolation Forest splits
axis-aligned and `IsolationForestLayer` scores one tick at a time with no
memory across ticks?

Framing, stated precisely because it matters for how these results should be
read: a measured near-zero detection rate below demonstrates that a
*stateless, per-tick-only* detector can be defeated this way. It is not a
general claim that "Isolation Forests are weak" -- a session-level or
multi-tick detector would plausibly catch both constructions below.

Runs against whatever model `IsolationForestLayer()` currently loads (the
real shipped `.joblib`/calibration), so these numbers automatically reflect
the post-retrain model once `train_isolation_forest.py` has been re-run
against the real-capture corpus -- no change needed here.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from client.rl_agent.anomaly_detector import (  # noqa: E402
    LAYER3_CALIBRATION_PATH, LAYER3_FEATURES, LAYER3_MIN_PACKETS,
    IsolationForestLayer, tick_features,
)
from client.rl_agent import train_isolation_forest as tif  # noqa: E402

EVAL_SEED = 4242  # distinct from EVAL_SEED=1234 in test_phase7.py
N_SESSION_TICKS = 300
# Whatever model is currently installed measured its OWN real-attack
# detection rates at train time (train_isolation_forest.py's
# calibrate_and_measure); comparing each mimicry construction against that
# number, not a hand-guessed constant, is what keeps this file meaningful
# across a retrain -- the absolute evasion rate will drift with the model,
# but "meaningfully below the real attack's own detection rate" should not.
_REAL_ATTACK_DETECTION_RATE = {
    name: m["rate"]
    for name, m in json.loads(LAYER3_CALIBRATION_PATH.read_text())["attack_detection_rate"].items()
}
# How many percentage points below the real attack's detection rate a
# mimicry construction must land to count as "a demonstrated gap" rather
# than noise. 30 points is comfortably above run-to-run sampling variance
# (test_phase7.py's own floors move by low single digits across seeds 0-3)
# while still requiring a real, substantial evasion effect.
_GAP_MARGIN = 0.30


@pytest.fixture(scope="module")
def layer():
    return IsolationForestLayer()


# --------------------------------------------- (a) slow-and-low exfiltration

def _benign_ticks_sorted_by_upload_share(n_per_profile: int, rng: np.random.Generator):
    """Real ticks from the existing benign generator (every feature is, by
    construction, drawn from a distribution IsolationForestLayer was trained
    to consider normal), sorted by `upload_share` descending. Picking from
    the high end gives the attacker the most "exfiltration-like" tick that
    is still, literally, a benign-generator sample -- not a hand-invented
    number that merely happens to fall in a percentile band."""
    rows = []
    for profile in tif.BENIGN_PROFILES.values():
        rows.extend(tif.feature_matrix(profile, n_per_profile, rng).tolist())
    idx = LAYER3_FEATURES.index("upload_share")
    rows.sort(key=lambda r: r[idx], reverse=True)
    return rows


def test_slow_and_low_exfiltration_evades_the_stateless_detector(layer):
    """Build a session entirely out of the top 1% most upload-heavy *benign*
    ticks (genuinely benign-generator samples, so every one of the 6
    features is, by definition, inside the training distribution's marginal
    range), replayed for N_SESSION_TICKS. A real sustained transfer this
    long would move N_SESSION_TICKS * 5s worth of bytes at that upload
    share -- exfiltration-scale -- yet the detector, which only ever scores
    one tick in isolation, should flag close to none of them."""
    rng = np.random.default_rng(EVAL_SEED)
    pool = _benign_ticks_sorted_by_upload_share(2_000, rng)
    top_1pct = pool[: max(1, len(pool) // 100)]

    session = [top_1pct[i % len(top_1pct)] for i in range(N_SESSION_TICKS)]
    results = [layer.update(dict(zip(LAYER3_FEATURES, row))) for row in session]
    detection_rate = float(np.mean([r["anomalous"] for r in results]))

    real_rate = _REAL_ATTACK_DETECTION_RATE["exfiltration"]
    print(f"\nslow-and-low exfiltration: {detection_rate:.1%} of "
          f"{N_SESSION_TICKS} ticks flagged (session built from the top "
          f"1% upload_share benign-generator ticks); real exfiltration is "
          f"caught at {real_rate:.1%}")
    assert detection_rate <= real_rate - _GAP_MARGIN, (
        f"expected the slow-and-low construction to evade detection well "
        f"below the real attack's {real_rate:.1%}, got {detection_rate:.1%} flagged"
    )


# --------------------------------------------------- (b) thin-spread scan

def test_sub_minimum_packet_scan_contributes_nothing():
    """Space scan probes so that no single 5s tick sees LAYER3_MIN_PACKETS
    packets. `tick_features` returns None for every such tick, so Layer 3
    is not-ready and contributes nothing -- the simplest, most airtight
    evasion: it isn't that the forest misjudges the tick, it never scores
    one at all."""
    not_ready = 0
    for _ in range(N_SESSION_TICKS):
        n = LAYER3_MIN_PACKETS - 1  # one below the floor, every tick
        sizes = [58] * n            # tiny, SYN-scan-sized packets
        times = list(range(n))
        outbound = [True] * n
        f = tick_features(sizes, times, outbound)
        assert f is None
        not_ready += 1

    print(f"\nthin-spread scan (<{LAYER3_MIN_PACKETS} packets/tick): "
          f"{not_ready}/{N_SESSION_TICKS} ticks not-ready (never scored)")
    assert not_ready == N_SESSION_TICKS


def test_denser_thin_spread_scan_inside_idle_ranges(layer):
    """A denser variant: LAYER3_MIN_PACKETS exactly (so ticks ARE scored),
    with sizes/timing chosen to resemble the idle profile's own low-rate,
    small-packet shape rather than real nmap SYN/RST packets -- the
    detector has no way to tell "a few small idle-ish packets" from "a few
    small scan probes" from shape alone within a single tick."""
    rng = np.random.default_rng(EVAL_SEED)
    detections = []
    for _ in range(N_SESSION_TICKS):
        n = LAYER3_MIN_PACKETS
        sizes = rng.normal(130, 40, size=n).clip(40, 300)
        gaps = rng.gamma(1.0 / 1.0 ** 2, 1.0 * 1.0, size=n)  # cv ~= 1, idle-like
        times = np.cumsum(gaps)
        outbound = rng.random(n) < 0.5
        f = tick_features(sizes, times, outbound)
        detections.append(layer.update(f)["anomalous"] if f is not None else False)

    detection_rate = float(np.mean(detections))
    real_rate = _REAL_ATTACK_DETECTION_RATE["port_scan"]
    print(f"\ndenser thin-spread scan (idle-shaped, {LAYER3_MIN_PACKETS} "
          f"packets/tick): {detection_rate:.1%} flagged; real port_scan is "
          f"caught at {real_rate:.1%}")
    assert detection_rate <= real_rate - _GAP_MARGIN, (
        f"expected idle-shaped thin-spread probing to evade detection well "
        f"below the real attack's {real_rate:.1%}, got {detection_rate:.1%} flagged"
    )
