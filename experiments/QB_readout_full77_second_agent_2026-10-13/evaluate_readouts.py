# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — matched complete scalar readout evaluation
"""Evaluate both exposed cohorts only after every frozen readout has qualified."""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from fit_readouts import digest, mapped_array


STATE: dict = {}


def initialize(work: str, runs: str, fitted: str) -> None:
    """Open held-out targets only after the complete fit receipt is verified."""
    work, runs, fitted = Path(work), Path(runs), Path(fitted)
    protocol = json.loads((work / "protocol_frozen_v1.json").read_text())
    assert protocol["status"] == "fixed_full77_scalar_readout_study_frozen"
    assert digest(work / "models.json") == protocol["model_source_manifest_sha256"]
    assert digest(work / "cohort.json") == protocol["cohort_sha256"]
    assert digest(Path(__file__)) == protocol["sources_sha256"]["evaluate_readouts.py"]
    assert int(os.environ["OPENBLAS_NUM_THREADS"]) <= protocol["blas_threads_per_worker"]
    receipt = json.loads((fitted / "all77_receipt.json").read_text())
    assert len(receipt["models"]) == 77
    assert receipt["status"] == "all77_fixed_scalar_readouts_fit_qualified_evaluation_required"
    assert receipt["protocol_sha256"] == digest(work / "protocol_frozen_v1.json")
    package = runs / "legacy_tree_fec_first_agent_20261005/deploy/pkg"
    for relative, expected in protocol["legacy_sources_sha256"].items():
        assert digest(package / relative) == expected
    sys.path.insert(0, str(package / "source/fec_adapter"))
    cohort = json.loads((work / "cohort.json").read_text())
    tables = {}
    for key, table_name, cohort_name in (("358", "training", "holdout"),
                                          ("357", "validation", "reserved_confirmation")):
        path = runs / f"tables28_subframe_20261006/table_{table_name}.npz"
        with np.load(path, allow_pickle=False) as archive:
            stems = archive["stems"].astype(str)
            shot = archive["shot_of_row"]
            labels = archive["label_names"].astype(str).tolist()
        allowed = set(cohort[cohort_name])
        counts = np.bincount(shot, minlength=len(stems))
        bounds = np.r_[0, np.cumsum(counts)]
        assert np.array_equal(shot, np.repeat(np.arange(len(stems)), counts))
        chosen = [i for i, stem in enumerate(stems) if stem in allowed]
        assert len(chosen) == int(key) and set(stems[chosen]) == allowed
        rows = np.concatenate([np.arange(bounds[i], bounds[i + 1]) for i in chosen])
        tables[key] = {"features": mapped_array(path, "features"), "rows": rows,
                       "truth": mapped_array(path, "labels")[rows][:, [labels.index("q95"), labels.index("beta_n")]],
                       "stems": stems[chosen], "shot": np.repeat(np.arange(len(chosen)), counts[chosen])}
    STATE.update(work=work, runs=runs, fitted=fitted, tables=tables, protocol=protocol)


def predict_one(task: dict) -> dict:
    """Predict the original and corrected member with identical inputs."""
    from fec_adapter.diii_d_accuracy.network import Network
    source = STATE["runs"] / task["source_relative_to_runs"]
    destination = STATE["fitted"] / task["name"]
    assert digest(source) == task["sha256"]
    receipt = json.loads((destination / "receipt.json").read_text())
    assert receipt["only_qbeta_output_rows_changed"] and receipt["export_reload_bitexact"]
    assert digest(destination / "network.npz") == receipt["network_sha256"]
    original, corrected = Network.load(source), Network.load(destination / "network.npz")
    answers = {}
    for key, table in STATE["tables"].items():
        rows = table["rows"]
        arms = [[], []]
        for start in range(0, len(rows), STATE["protocol"]["chunk"]):
            features = table["features"][rows[start:start + STATE["protocol"]["chunk"]]].astype(float)
            features = np.pad(features, ((0, 0), (0, original.feature_mean.size - features.shape[1])))
            for arm, model in zip(arms, (original, corrected), strict=True):
                _, heads = model.predict(features)
                values = np.column_stack([heads[name] for name in ("q95", "beta_n")])
                assert np.isfinite(values).all()
                arm.append(values)
        answers[key] = np.stack([np.concatenate(arm) for arm in arms])
    return {"name": task["name"], "family": task["family"], "answers": answers}


def summarize(truth: np.ndarray, predictions: np.ndarray, shot: np.ndarray,
              shots: int, seed: int, replicates: int) -> dict:
    """Resample whole shots and recompute each resample's target variance."""
    rng = np.random.default_rng(seed)
    counts = np.stack([np.bincount(rng.integers(shots, size=shots), minlength=shots)
                       for _ in range(replicates)]).astype(float)
    bootstrap = np.zeros(replicates)
    delta = 0.0
    details = {}
    for column, name in enumerate(("q95", "betaN")):
        valid = np.isfinite(truth[:, column])
        y, ids = truth[valid, column].astype(float), shot[valid]
        n = np.bincount(ids, minlength=shots).astype(float)
        total = np.bincount(ids, weights=y, minlength=shots)
        squares = np.bincount(ids, weights=y * y, minlength=shots)
        denominator = squares.sum() - total.sum() ** 2 / n.sum()
        assert denominator > 0
        errors = np.stack([np.bincount(ids, weights=(arm[valid, column].astype(float) - y) ** 2,
                                       minlength=shots) for arm in predictions])
        difference = errors[0] - errors[1]
        change = float(difference.sum() / denominator)
        bootstrap_denominator = counts @ squares - (counts @ total) ** 2 / (counts @ n)
        assert np.all(bootstrap_denominator > 0)
        bootstrap += .075 * (counts @ difference) / bootstrap_denominator
        delta += .075 * change
        details[name] = {"finite_truth_frames": int(valid.sum()),
                         "R2_baseline": float(1 - errors[0].sum() / denominator),
                         "R2_candidate": float(1 - errors[1].sum() / denominator), "delta_R2": change}
    interval = np.quantile(bootstrap, [.005, .995])
    return {"shots": shots, "frames": len(shot), "delta_S": delta, "CI99": interval.tolist(),
            "P_positive": float(np.mean(bootstrap > 0)), "metrics": details,
            "pass": bool(delta >= 1e-5 and interval[0] > 0)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--fitted", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.work / "protocol_frozen_v1.json").read_text())
    assert not (args.work / "evaluation_v1.json").exists()
    for name, expected in protocol["evaluation_tables_sha256"].items():
        assert digest(args.runs / "tables28_subframe_20261006" / name) == expected
    initialize(str(args.work), str(args.runs), str(args.fitted))
    tasks = json.loads((args.work / "models.json").read_text())
    assert len(tasks) == 77
    with ProcessPoolExecutor(max_workers=protocol["workers"], mp_context=get_context("spawn"),
                             initializer=initialize,
                             initargs=(str(args.work), str(args.runs), str(args.fitted))) as pool:
        answers = list(pool.map(predict_one, tasks))
    results, arrays = {}, {}
    for offset, (key, table) in enumerate(STATE["tables"].items()):
        maps = [item["answers"][key] for item in answers if item["family"] != "joint"]
        joint = [item["answers"][key] for item in answers if item["family"] == "joint"]
        assert len(maps) == 69 and len(joint) == 8
        map_mean = np.mean(maps, axis=0).astype(np.float32).astype(float)
        joint_mean = np.mean(joint, axis=0)
        predictions = (.75 * map_mean + .25 * joint_mean).astype(np.float32)
        assert np.isfinite(predictions).all()
        results[key] = summarize(table["truth"], predictions, table["shot"], int(key),
                                 protocol["bootstrap_seed"] + offset, protocol["bootstrap_replicates"])
        arrays.update({f"{key}_predictions": predictions, f"{key}_truth": table["truth"],
                       f"{key}_shot": table["shot"], f"{key}_stems": table["stems"]})
    np.savez_compressed(args.work / "matched_715_scalar_predictions_v1.npz", **arrays)
    result = {"status": "full77_matched_scalar_evaluation_complete", "cohorts_reused": True,
              "all_77_models_in_both_arms": True, "maps_and_geometry_unchanged_by_construction": True,
              "bootstrap_resample_target_means_recomputed": True, "results": results,
              "gate_pass": all(row["pass"] for row in results.values()),
              "protocol_sha256": digest(args.work / "protocol_frozen_v1.json"),
              "source_sha256": digest(Path(__file__)),
              "predictions_sha256": digest(args.work / "matched_715_scalar_predictions_v1.npz")}
    with (args.work / "evaluation_v1.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
