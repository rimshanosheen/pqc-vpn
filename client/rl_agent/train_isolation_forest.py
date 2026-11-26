"""
Trains and calibrates anomaly-detection Layer 3 (`IsolationForestLayer`).

There is no captured traffic to train on yet (the live packet source is
Member 1's Week 8 desktop work), so benign traffic is simulated. Each tick
is built packet by packet from one of five profiles (idle, browsing,
streaming, bulk download, video call), then reduced with the same
`tick_features` the live path will use, so the generator exercises the real
feature code rather than skipping it.

Three simulated attack mixes (exfiltration, port scan, beaconing) are used
only to *measure* detection. The forest never sees them: an Isolation
Forest is trained on benign data alone, which is the point of Layer 3.

Every profile below is a reasoned first-pass shape, not a measurement, in
the same spirit as `state_observer.py`'s normalization caps. The numbers are
there to be re-checked once real traffic can be captured, not to be trusted.

Writes:
  client/rl_agent/models/isolation_forest.joblib
  client/rl_agent/models/isolation_forest_calibration.json

Run with: python -m client.rl_agent.train_isolation_forest
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from client.rl_agent.anomaly_detector import (  # noqa: E402
    LAYER3_CALIBRATION_PATH,
    LAYER3_FEATURES,
    LAYER3_MODEL_PATH,
    IsolationForestLayer,
    tick_features,
)

TICK_SECONDS = 5.0
SEED = 0
TRAIN_TICKS_PER_PROFILE = 1_000
HELDOUT_TICKS_PER_PROFILE = 400
ATTACK_TICKS_PER_MIX = 400
N_ESTIMATORS = 100

THRESHOLD_QUANTILE = 0.995

# Week 8: real-capture grounding, on top of the synthetic generator kept as
# a regression floor (see module docstring). `capture_real_traffic.py`
# writes this corpus after `capture_scenario.sh` has been run; its absence
# is not an error, just "no real data yet" -- every function below takes
# `real_corpus=None` and falls back to synthetic-only exactly as before.
REAL_CAPTURE_CORPUS_PATH = LAYER3_MODEL_PATH.parent / "real_capture_corpus.json"
REAL_TRAIN_FRACTION = 0.70
# Never random-shuffle real rows into train/test: adjacent ticks from the
# same capture are autocorrelated (the same TCP connection, the same
# curl loop), so a random split leaks test information into training.
# The first REAL_TRAIN_FRACTION of each scenario, in capture order, trains;
# the chronological remainder is held out.
BOOTSTRAP_RESAMPLES = 2000
# The anomaly threshold is the 99.5th percentile of held-out benign scores,
# so about 1 benign tick in 200 crosses it. At 5 s ticks that is one false
# alarm every ~17 minutes *from this layer alone*, and it lands as a
# threat_score near 0.5, not 1.0. Tighter means missed beacons; looser means
# the RL agent reacts to background noise. Re-derive once real traffic exists.

# Per-profile ranges. A tick draws each value uniformly from its range, so a
# profile covers a spread of intensities rather than one point.
#   rate:  packets/sec
#   cv:    coefficient of variation of inter-arrival gaps (gamma-distributed)
#   p_out: share of packets that are outbound
#   out / in: (mean, sd) packet size in bytes per direction
BENIGN_PROFILES = {
    # Keepalives, background sync, the odd DNS lookup.
    "idle":          dict(rate=(1, 5),      cv=(0.8, 1.3), p_out=(0.4, 0.6),
                          out=(120, 60),   inn=(150, 80)),
    # Page loads: bursts of large inbound packets, small outbound requests/ACKs.
    "browsing":      dict(rate=(20, 200),   cv=(1.5, 3.0), p_out=(0.3, 0.45),
                          out=(90, 60),    inn=(1200, 350)),
    # Near-full-size inbound segments, steady-ish, ACKs going out.
    "streaming":     dict(rate=(150, 600),  cv=(0.6, 1.5), p_out=(0.2, 0.35),
                          out=(66, 10),    inn=(1400, 120)),
    # MTU-sized inbound as fast as the link allows.
    "bulk_download": dict(rate=(500, 2000), cv=(0.4, 1.0), p_out=(0.3, 0.5),
                          out=(66, 8),     inn=(1460, 40)),
    # Symmetric mid-size media frames at a fairly regular cadence.
    "video_call":    dict(rate=(80, 300),   cv=(0.3, 0.7), p_out=(0.45, 0.55),
                          out=(1000, 200), inn=(1000, 200)),
}

ATTACK_MIXES = {
    # Sustained MTU-sized outbound, almost nothing coming back but ACKs.
    "exfiltration":  dict(rate=(300, 1500), cv=(0.3, 1.0), p_out=(0.7, 0.85),
                          out=(1450, 50),  inn=(66, 8)),
    # SYN out, RST back: tiny uniform packets at a fast, machine-regular pace.
    "port_scan":     dict(rate=(200, 2000), cv=(0.1, 0.4), p_out=(0.5, 0.65),
                          out=(58, 4),     inn=(54, 4)),
    # C2 check-ins on a timer: low rate, near-zero jitter, mostly outbound.
    "beaconing":     dict(rate=(1, 3),      cv=(0.0, 0.05), p_out=(0.6, 0.9),
                          out=(250, 40),   inn=(180, 40)),
}


def simulate_tick(profile: dict, rng: np.random.Generator):
    """One tick's packets for `profile`: (sizes, arrival_times, outbound)."""
    rate = rng.uniform(*profile["rate"])
    cv = max(rng.uniform(*profile["cv"]), 0.01)
    p_out = rng.uniform(*profile["p_out"])

    n = max(int(rng.poisson(rate * TICK_SECONDS)), 1)
    # Gamma gaps with shape k have cv = 1/sqrt(k), and mean 1/rate.
    shape = 1.0 / cv ** 2
    gaps = rng.gamma(shape, 1.0 / (rate * shape), size=n)
    times = np.cumsum(gaps)

    outbound = rng.random(n) < p_out
    sizes = np.where(
        outbound,
        rng.normal(*profile["out"], size=n),
        rng.normal(*profile["inn"], size=n),
    )
    sizes = np.clip(sizes, 40, 1500)
    return sizes, times, outbound


def feature_matrix(profile: dict, n_ticks: int, rng: np.random.Generator) -> np.ndarray:
    """`n_ticks` feature rows for `profile`. Ticks below LAYER3_MIN_PACKETS
    are skipped, exactly as the live layer would skip them."""
    rows = []
    while len(rows) < n_ticks:
        f = tick_features(*simulate_tick(profile, rng), tick_seconds=TICK_SECONDS)
        if f is not None:
            rows.append([f[name] for name in LAYER3_FEATURES])
    return np.array(rows)


def benign_matrix(n_per_profile: int, rng: np.random.Generator) -> np.ndarray:
    return np.vstack([feature_matrix(p, n_per_profile, rng) for p in BENIGN_PROFILES.values()])


def load_real_capture_corpus() -> dict | None:
    if not REAL_CAPTURE_CORPUS_PATH.exists():
        return None
    return json.loads(REAL_CAPTURE_CORPUS_PATH.read_text())


def _real_rows_by_scenario(corpus: dict) -> dict[str, np.ndarray]:
    by_scenario: dict[str, list] = {}
    for tick in corpus["ticks"]:
        by_scenario.setdefault(tick["scenario"], []).append(tick)
    out = {}
    for scenario, ticks in by_scenario.items():
        ticks = sorted(ticks, key=lambda t: t["tick_index"])
        out[scenario] = np.array([[t["features"][name] for name in LAYER3_FEATURES] for t in ticks])
    return out


def real_benign_temporal_split(corpus: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """The first REAL_TRAIN_FRACTION of each real BENIGN scenario (in
    capture order) -> one stacked array of train rows; the chronological
    remainder -> held-out, kept per scenario so false positives can still
    be reported per scenario, matching the synthetic profiles' own shape."""
    by_scenario = _real_rows_by_scenario(corpus)
    train_chunks = []
    heldout = {}
    for scenario, meta in corpus["scenarios"].items():
        if meta["category"] != "benign":
            continue
        rows = by_scenario.get(scenario)
        if rows is None or len(rows) == 0:
            continue
        split = max(1, int(len(rows) * REAL_TRAIN_FRACTION))
        train_chunks.append(rows[:split])
        if len(rows) > split:
            heldout[f"real_{scenario}"] = rows[split:]
    train_rows = np.vstack(train_chunks) if train_chunks else np.empty((0, len(LAYER3_FEATURES)))
    return train_rows, heldout


def real_attack_shaped_rows(corpus: dict) -> dict[str, np.ndarray]:
    """Real ticks shaped like an attack mix (see capture_real_traffic.py's
    docstring for why beaconing/bulk_upload/port_scan land here rather than
    in benign training) -- for measurement only, same as the synthetic
    ATTACK_MIXES."""
    by_scenario = _real_rows_by_scenario(corpus)
    out = {}
    for scenario, meta in corpus["scenarios"].items():
        if meta["category"] != "attack_shaped":
            continue
        rows = by_scenario.get(scenario)
        if rows is not None and len(rows):
            out[f"real_{scenario}"] = rows
    return out


def train(seed: int = SEED, n_per_profile: int = TRAIN_TICKS_PER_PROFILE,
          real_corpus: dict | None = None) -> IsolationForest:
    rng = np.random.default_rng(seed)
    x = benign_matrix(n_per_profile, rng)
    if real_corpus is not None:
        real_train, _ = real_benign_temporal_split(real_corpus)
        if len(real_train):
            x = np.vstack([x, real_train])
    return IsolationForest(n_estimators=N_ESTIMATORS, contamination="auto",
                           random_state=seed).fit(x)


def _bootstrap_ci(flagged: np.ndarray, n_boot: int = BOOTSTRAP_RESAMPLES, seed: int = 0) -> list[float]:
    """Nonparametric 95% CI on a flag-rate: resample `flagged` (one bool per
    tick) with replacement `n_boot` times and take the 2.5th/97.5th
    percentile of the resampled rate -- not a single-run point estimate."""
    if len(flagged) == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    n = len(flagged)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_rates = flagged[idx].mean(axis=1)
    return [float(np.quantile(boot_rates, 0.025)), float(np.quantile(boot_rates, 0.975))]


def _downsample(arr: np.ndarray, n: int = 50) -> list[float]:
    """A curve's full point list is too large to carry in the calibration
    JSON every retrain; this keeps a representative, evenly-spaced subset
    (including both endpoints) rather than truncating it."""
    if len(arr) <= n:
        return arr.tolist()
    idx = np.linspace(0, len(arr) - 1, n).astype(int)
    return arr[idx].tolist()


def calibrate_and_measure(model: IsolationForest, seed: int = SEED,
                           real_corpus: dict | None = None) -> dict:
    """Calibrate the thresholds on held-out benign traffic (a different seed
    stream from training), then measure per-profile false positives and
    per-attack detection with those thresholds -- each as a bootstrapped
    rate plus a 95% CI, not a single-run point estimate -- plus an overall
    ROC/PR curve for the benign-vs-attack separation `score_threshold`
    actually trades off."""
    rng = np.random.default_rng(seed + 1)
    heldout = {name: feature_matrix(p, HELDOUT_TICKS_PER_PROFILE, rng)
               for name, p in BENIGN_PROFILES.items()}
    if real_corpus is not None:
        _, real_heldout = real_benign_temporal_split(real_corpus)
        heldout.update(real_heldout)

    all_benign = np.vstack(list(heldout.values()))
    benign_scores = -model.score_samples(all_benign)
    median = float(np.median(benign_scores))
    threshold = float(np.quantile(benign_scores, THRESHOLD_QUANTILE))

    def flag_rate_with_ci(x: np.ndarray) -> dict:
        flagged = (-model.score_samples(x) >= threshold)
        return {"rate": float(flagged.mean()), "ci95": _bootstrap_ci(flagged)}

    attacks = {name: feature_matrix(p, ATTACK_TICKS_PER_MIX, rng)
               for name, p in ATTACK_MIXES.items()}
    if real_corpus is not None:
        attacks.update(real_attack_shaped_rows(real_corpus))
    all_attack = np.vstack(list(attacks.values()))

    y_true = np.concatenate([np.zeros(len(all_benign)), np.ones(len(all_attack))])
    y_score = np.concatenate([benign_scores, -model.score_samples(all_attack)])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    return {
        "score_median": median,
        "score_threshold": threshold,
        "threshold_quantile": THRESHOLD_QUANTILE,
        "benign_false_positive_rate": {n: flag_rate_with_ci(x) for n, x in heldout.items()},
        "attack_detection_rate": {n: flag_rate_with_ci(x) for n, x in attacks.items()},
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_curve": {"fpr": _downsample(fpr), "tpr": _downsample(tpr)},
        "pr_curve": {"precision": _downsample(precision), "recall": _downsample(recall)},
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "real_capture_used": real_corpus is not None,
    }


def main() -> int:
    real_corpus = load_real_capture_corpus()
    model = train(real_corpus=real_corpus)
    measured = calibrate_and_measure(model, real_corpus=real_corpus)

    LAYER3_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, LAYER3_MODEL_PATH, compress=3)
    calibration = {
        "generated_by": "python -m client.rl_agent.train_isolation_forest",
        "features": list(LAYER3_FEATURES),
        "sklearn_version": sklearn.__version__,
        "seed": SEED,
        "n_estimators": N_ESTIMATORS,
        "train_ticks": TRAIN_TICKS_PER_PROFILE * len(BENIGN_PROFILES),
        "note": ("Benign training is simulated traffic (train_isolation_forest.py's "
                 "first-pass profile shapes) plus real captured traffic when "
                 "models/real_capture_corpus.json exists (see real_capture_used "
                 "below) -- run capture_scenario.sh + capture_real_traffic.py "
                 "and retrain to add it."),
        **measured,
    }
    LAYER3_CALIBRATION_PATH.write_text(json.dumps(calibration, indent=2) + "\n")

    # Load back through the real layer, so a save/load mismatch fails here.
    IsolationForestLayer()

    print(f"wrote {LAYER3_MODEL_PATH.name} ({LAYER3_MODEL_PATH.stat().st_size / 1024:.0f} KB) "
          f"and {LAYER3_CALIBRATION_PATH.name}")
    print(f"real_capture_used={measured['real_capture_used']}")
    print(f"score median {measured['score_median']:.4f}, "
          f"threshold {measured['score_threshold']:.4f} (q{THRESHOLD_QUANTILE})")
    print(f"roc_auc={measured['roc_auc']:.4f} pr_auc={measured['pr_auc']:.4f}")
    print(f"benign false-positive rate (held-out, {BOOTSTRAP_RESAMPLES}-resample 95% CI):")
    for name, m in measured["benign_false_positive_rate"].items():
        print(f"  {name:<16} {m['rate']:6.2%}  CI [{m['ci95'][0]:.2%}, {m['ci95'][1]:.2%}]")
    print("attack detection rate:")
    for name, m in measured["attack_detection_rate"].items():
        print(f"  {name:<16} {m['rate']:6.2%}  CI [{m['ci95'][0]:.2%}, {m['ci95'][1]:.2%}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
