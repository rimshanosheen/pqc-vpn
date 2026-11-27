"""
Week 8 (Member 3 track): wires Layer 3 (Isolation Forest) into
`AnomalyCombiner` under its own CPU gate (`LAYER3_CPU_GATE = 0.40`), below
Layer 2's (`LAYER2_CPU_GATE = 0.70`).

Uses a fake Layer 3 (not the real trained forest) so these tests check the
*wiring* — the gate, participation in the combiner's max, and the
not-ready-doesn't-mask-a-real-hit rule — independent of the forest's own
measured quality. `tests/test_phase7.py` covers the forest's own behaviour;
the real-capture retraining regression floors land in a follow-up once
`client/rl_agent/capture_real_traffic.py`'s corpus has been collected and
`train_isolation_forest.py` re-run against it.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from client.rl_agent.anomaly_detector import (  # noqa: E402
    LAYER2_CPU_GATE,
    LAYER3_CPU_GATE,
    AnomalyCombiner,
)

# Baseline kwargs for every Layer 1/2 metric, chosen to be quiet (no layer
# other than the one under test should ever fire) and stable under repeated
# calls (DNS server constant, port/retransmit/upload all well below their
# trigger thresholds).
_QUIET = dict(
    latency=20.0, packet_rate=50.0, packet_size=500.0,
    distinct_ports=2, connection_attempts=5, failed_attempts=0,
    retransmit_rate=0.0, latency_ms=20.0,
    upload_bytes_per_sec=1000.0, download_bytes_per_sec=5000.0,
    dns_server="1.1.1.1",
)
# A tick with enough real packets that tick_features() returns non-None
# (needs >= LAYER3_MIN_PACKETS = 3 packets).
_PACKETS = dict(
    packet_sizes=[500, 520, 480, 510],
    arrival_times=[0.0, 1.0, 2.0, 3.0],
    outbound=[True, False, True, False],
)
_SPARSE = dict(packet_sizes=[60, 60], arrival_times=[0.0, 1.0], outbound=[True, True])


class _FakeLayer3:
    """Returns a fixed result regardless of input, except it reports
    not-ready when `features` is None — mirrors `IsolationForestLayer`'s own
    sparse-tick behavior (`tests/test_phase7.py::test_sparse_tick_is_not_ready`)
    — so a test can pin exactly what "Layer 3 fired" looks like without
    touching the real trained forest."""

    def __init__(self, result: dict):
        self._result = result
        self.calls = 0

    def update(self, features):
        self.calls += 1
        if features is None:
            return {"anomalous": False, "threat_score": 0.0, "ready": False,
                     "detail": {"raw_score": None}}
        return self._result


_FIRES = {"anomalous": True, "threat_score": 0.9, "ready": True, "detail": {"raw_score": 0.9}}
_QUIET_RESULT = {"anomalous": False, "threat_score": 0.1, "ready": True, "detail": {"raw_score": 0.1}}


def _combiner(layer3_result: dict):
    layer3 = _FakeLayer3(layer3_result)
    return AnomalyCombiner(layer3=layer3), layer3


def test_layer3_runs_below_its_gate_and_contributes():
    combiner, layer3 = _combiner(_FIRES)
    r = combiner.update(cpu_load=LAYER3_CPU_GATE - 0.01, **_QUIET, **_PACKETS)
    assert layer3.calls == 1
    assert r["detail"]["layer3_active"] is True
    assert r["threat_score"] == pytest.approx(0.9)
    assert r["anomalous"] is True


def test_layer3_does_not_run_at_or_above_its_gate():
    combiner, layer3 = _combiner(_FIRES)
    r = combiner.update(cpu_load=LAYER3_CPU_GATE, **_QUIET, **_PACKETS)
    assert layer3.calls == 0
    assert r["detail"]["layer3_active"] is False
    assert "layer3" not in r["detail"]["layers"]
    # Layer 3 would have fired had it run, but it never did, and nothing
    # else in _QUIET/_PACKETS should trip any other layer.
    assert r["anomalous"] is False


def test_layer2_runs_but_layer3_does_not_between_the_two_gates():
    """LAYER3_CPU_GATE < LAYER2_CPU_GATE: a cpu_load strictly between them
    should run the five Layer 2 signatures but gate off Layer 3 (the
    stricter of the two gates, since it is the most expensive layer)."""
    combiner, layer3 = _combiner(_FIRES)
    cpu_between = (LAYER3_CPU_GATE + LAYER2_CPU_GATE) / 2
    assert LAYER3_CPU_GATE < cpu_between < LAYER2_CPU_GATE
    r = combiner.update(cpu_load=cpu_between, **_QUIET, **_PACKETS)
    assert layer3.calls == 0
    assert r["detail"]["layer2_active"] is True
    assert r["detail"]["layer3_active"] is False
    assert "port_scan" in r["detail"]["layers"]
    assert "layer3" not in r["detail"]["layers"]


def test_layer3_score_participates_in_the_combiner_max():
    combiner, _ = _combiner(_QUIET_RESULT)
    r = combiner.update(cpu_load=0.0, **_QUIET, **_PACKETS)
    # Every layer here is quiet (<= 0.1); the combiner's max should reflect
    # Layer 3's contribution, not silently drop it.
    assert r["threat_score"] == pytest.approx(0.1)


def test_unready_layer3_tick_does_not_mask_a_real_layer1_hit():
    """A sparse tick (fewer than LAYER3_MIN_PACKETS packets) makes Layer 3
    not-ready; it must contribute nothing, not a masking 0.0, so a real
    Layer 1 hit still surfaces in the combined `anomalous`/`threat_score`."""
    combiner, layer3 = _combiner(_FIRES)  # would fire if it got a real tick
    # Build a non-degenerate (nonzero-variance) Layer 1 baseline: a
    # perfectly constant history gives ZScoreBaseline a std of 0, which its
    # own `_z_score` defines as "no anomaly possible" regardless of the next
    # sample — so the fill ticks vary slightly, like real jitter would.
    for i in range(50):
        quiet = {**_QUIET, "latency": 20.0 + (i % 5)}
        combiner.update(cpu_load=0.0, **quiet, **_SPARSE)
    r = combiner.update(cpu_load=0.0, **{**_QUIET, "latency": 5000.0}, **_SPARSE)
    assert r["detail"]["layers"]["layer3"]["ready"] is False
    assert r["anomalous"] is True
    assert r["threat_score"] == pytest.approx(1.0)  # Layer 1 saturates at >=2x threshold
