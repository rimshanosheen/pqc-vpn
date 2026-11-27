"""
Exports the trained Isolation Forest (`isolation_forest.joblib`) at the
tree-structure level, so the Rust port (`core/src/anomaly/forest.rs`) walks
the exact same trees instead of reimplementing sklearn.

This is the single biggest correctness-risk reducer available for Layer 3:
rather than asking the Rust side to reimplement sklearn's path-length-
correction math (`c(n)`, the average path length of an unsuccessful BST
search -- a harmonic-number formula with its own edge cases at n<=2), this
script precomputes `c(n_node_samples[node])` for every node of every tree,
and the global `c(max_samples_)` normalizer, once, in Python. The Rust side
then does nothing but: walk a tree by comparing a feature to a threshold,
read off the leaf's precomputed correction, sum, average, exponentiate.

Writes:
  client/rl_agent/models/isolation_forest_trees.json   (the trees themselves)
  contracts/isolation_forest_test_vectors.json          (numeric parity vectors)

Run with: python -m client.rl_agent.export_forest_trees
(also invoked from export_contracts.main(), after the forest has been
(re)trained by train_isolation_forest.py)
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from client.rl_agent.anomaly_detector import (  # noqa: E402
    LAYER3_CALIBRATION_PATH, LAYER3_FEATURES, LAYER3_MODEL_PATH, IsolationForestLayer,
)
from client.rl_agent import train_isolation_forest as tif  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
TREES_PATH = LAYER3_MODEL_PATH.with_name("isolation_forest_trees.json")
VECTORS_PATH = REPO_ROOT / "contracts" / "isolation_forest_test_vectors.json"
GENERATOR = "python -m client.rl_agent.export_forest_trees"
EVAL_SEED = 777  # distinct from train_isolation_forest's SEED=0 and test_phase7.py's 1234


def _average_path_length(n_samples) -> np.ndarray:
    """Average path length of an unsuccessful search in a BST holding
    `n_samples` elements (Liu, Ting & Zhou 2008; sklearn's own
    `_average_path_length`, reimplemented here directly off the formula
    rather than importing a private sklearn symbol). 0 for n<=1, 1 for
    n==2, else the harmonic-number form."""
    n = np.asarray(n_samples, dtype=float)
    out = np.zeros_like(n)
    out[n == 2] = 1.0
    mask = n > 2
    out[mask] = 2.0 * (np.log(n[mask] - 1.0) + np.euler_gamma) - 2.0 * (n[mask] - 1.0) / n[mask]
    return out


def export_trees(model) -> dict:
    trees = []
    for estimator in model.estimators_:
        t = estimator.tree_
        leaf_correction = _average_path_length(t.n_node_samples).tolist()
        trees.append({
            "feature": t.feature.tolist(),
            "threshold": t.threshold.tolist(),
            "children_left": t.children_left.tolist(),
            "children_right": t.children_right.tolist(),
            # Precomputed c(n_node_samples[node]) for every node; only the
            # value at the leaf a query lands on is ever used, but storing
            # one array per node (rather than a second leaf-indexed table)
            # keeps every per-tree array the same length.
            "leaf_correction": leaf_correction,
        })

    global_correction = float(_average_path_length(np.array([model.max_samples_]))[0])
    return {
        "schema_version": 1,
        "generated_by": GENERATOR,
        "source_of_truth": "client/rl_agent/models/isolation_forest.joblib "
                            "(train_isolation_forest.py)",
        "purpose": (
            "Tree-structure-level export for a Rust port. For a feature "
            "vector x ordered per `features`: walk each tree from node 0, "
            "comparing x[feature[node]] <= threshold[node] to choose "
            "children_left/children_right, until a node with "
            "children_left[node] == -1 (a leaf); that leaf's depth (edges "
            "from the root) plus its `leaf_correction` is the tree's path "
            "length. Average the path length over all trees, then: "
            "raw_score = 2 ** (-mean_path_length / global_path_length_correction). "
            "This equals `-model.score_samples(x)` (sklearn's own sign "
            "convention is reversed: lower score_samples = more anomalous, "
            "so this codebase negates it throughout, see IsolationForestLayer.raw_score)."
        ),
        "features": list(LAYER3_FEATURES),
        "n_estimators": len(trees),
        "max_samples": int(model.max_samples_),
        "global_path_length_correction": global_correction,
        "trees": trees,
    }


def _sample_vectors(model, n_per_group: int = 8) -> list[dict]:
    rng = np.random.default_rng(EVAL_SEED)
    groups: dict[str, np.ndarray] = {}
    for name, profile in tif.BENIGN_PROFILES.items():
        groups[f"benign_{name}"] = tif.feature_matrix(profile, n_per_group, rng)
    for name, mix in tif.ATTACK_MIXES.items():
        groups[f"attack_{name}"] = tif.feature_matrix(mix, n_per_group, rng)
    # Edge cases: an exact-zero vector and a far-out-of-range vector probe
    # the tree walk's handling of extremes, independent of either generator.
    groups["edge_zero"] = np.zeros((1, len(LAYER3_FEATURES)))
    groups["edge_extreme"] = np.full((1, len(LAYER3_FEATURES)), 1e6)

    vectors = []
    for group, rows in groups.items():
        raw_scores = -model.score_samples(rows)
        for i, (row, raw) in enumerate(zip(rows, raw_scores)):
            vectors.append({
                "name": f"{group}_{i}",
                "features": dict(zip(LAYER3_FEATURES, (round(float(v), 6) for v in row))),
                "expect_raw_score": round(float(raw), 9),
            })
    return vectors


def export_test_vectors(model, calibration: dict) -> dict:
    layer = IsolationForestLayer(model=model, calibration=calibration)
    vectors = _sample_vectors(model)
    for v in vectors:
        r = layer.update(dict(zip(LAYER3_FEATURES, v["features"].values())))
        v["expect_threat_score"] = round(float(r["threat_score"]), 6)
        v["expect_anomalous"] = bool(r["anomalous"])

    return {
        "schema_version": 1,
        "generated_by": GENERATOR,
        "source_of_truth": "client/rl_agent/models/isolation_forest.joblib + "
                            "isolation_forest_calibration.json",
        "purpose": (
            "Numeric parity for a Rust Layer 3 port, both at the raw "
            "tree-walk level (expect_raw_score, compare against "
            "isolation_forest_trees.json's walk) and the calibrated-layer "
            "level (expect_threat_score/expect_anomalous, using "
            "score_median/score_threshold below). Tolerance 1e-5, same "
            "convention as policy_test_vectors.json."
        ),
        "tolerance": 1e-5,
        "score_median": calibration["score_median"],
        "score_threshold": calibration["score_threshold"],
        "vectors": vectors,
    }


def main() -> int:
    model = joblib.load(LAYER3_MODEL_PATH)
    calibration = json.loads(LAYER3_CALIBRATION_PATH.read_text())

    TREES_PATH.write_text(json.dumps(export_trees(model), indent=2) + "\n")
    print(f"wrote {TREES_PATH.relative_to(REPO_ROOT)}")

    VECTORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    VECTORS_PATH.write_text(json.dumps(export_test_vectors(model, calibration), indent=2) + "\n")
    print(f"wrote {VECTORS_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
