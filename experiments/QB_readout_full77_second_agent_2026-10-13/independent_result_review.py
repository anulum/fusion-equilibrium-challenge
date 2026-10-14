# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — independent complete readout result review
"""Recompute paired score arithmetic using raw arrays and resampled truth means."""
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    here = Path(__file__).resolve().parent
    remote = Path("$FEC_RUNS/ml350/runs/second_agent_qb_readout_full77_20261013")
    protocol = json.loads((here / "protocol_frozen_v1.json").read_text())
    report = json.loads((remote / "work/evaluation_v1.json").read_text())
    terminal = json.loads((remote / "work/terminal_receipt_v1.json").read_text())
    assert terminal["status"] == "complete_frozen_full77_study_review_required"
    assert all(row["returncode"] == 0 for row in terminal["stages"])
    assert report["protocol_sha256"] == digest(here / "protocol_frozen_v1.json")
    for name, expected in protocol["sources_sha256"].items():
        assert digest(here / name) == digest(remote / "work" / name) == expected
    assert report["source_sha256"] == protocol["sources_sha256"]["evaluate_readouts.py"]
    path = remote / "work/matched_715_scalar_predictions_v1.npz"
    assert digest(path) == report["predictions_sha256"]
    cohort = json.loads((here / "cohort.json").read_text())
    audit = {}
    with np.load(path, allow_pickle=False) as arrays:
        for offset, (key, label, expected_frames) in enumerate((("358", "holdout", 78875),
                                                               ("357", "reserved_confirmation", 76251))):
            shots = int(key)
            prediction = arrays[f"{key}_predictions"].astype(float)
            truth = arrays[f"{key}_truth"].astype(float)
            ids, stems = arrays[f"{key}_shot"], arrays[f"{key}_stems"].astype(str)
            assert prediction.shape == (2, expected_frames, 2) and truth.shape == (expected_frames, 2)
            assert len(stems) == shots and set(stems) == set(cohort[label])
            assert np.array_equal(np.unique(ids), np.arange(shots))
            assert np.isfinite(prediction).all() and np.isfinite(truth).all()
            rng = np.random.default_rng(protocol["bootstrap_seed"] + offset)
            multiplicity = np.zeros((protocol["bootstrap_replicates"], shots))
            for index in range(len(multiplicity)):
                np.add.at(multiplicity[index], rng.integers(shots, size=shots), 1)
            delta, draws = 0., np.zeros(len(multiplicity))
            for column, name in enumerate(("q95", "betaN")):
                y = truth[:, column]
                centered = y - y.mean()
                denominator = centered @ centered
                errors = (prediction[:, :, column] - y) ** 2
                change = (errors[0].sum() - errors[1].sum()) / denominator
                assert abs(change - report["results"][key]["metrics"][name]["delta_R2"]) < 1e-12
                n = np.bincount(ids, minlength=shots)
                s = np.bincount(ids, weights=y, minlength=shots)
                ss = np.bincount(ids, weights=y*y, minlength=shots)
                gain = np.bincount(ids, weights=errors[0] - errors[1], minlength=shots)
                variance = multiplicity @ ss - (multiplicity @ s) ** 2 / (multiplicity @ n)
                assert (variance > 0).all()
                delta += .075 * change
                draws += .075 * (multiplicity @ gain) / variance
            interval = np.quantile(draws, [.005, .995])
            declared = report["results"][key]
            assert abs(delta - declared["delta_S"]) < 1e-12
            assert np.max(np.abs(interval - declared["CI99"])) < 1e-12
            assert abs(np.mean(draws > 0) - declared["P_positive"]) < 1e-12
            passed = bool(delta >= 1e-5 and interval[0] > 0)
            assert passed == declared["pass"]
            audit[key] = {"shots": shots, "frames": expected_frames, "delta_S": float(delta),
                          "CI99": interval.tolist(), "gate_pass": passed}
    assert not report["gate_pass"] and not any(part["gate_pass"] for part in audit.values())
    exports = json.loads((here / "all77_export_independent_check_v1.json").read_text())
    assert len(exports["models"]) == 77 and all(row["independent_parameter_guard_PASS"] for row in exports["models"])
    result = {"status": "independent_arithmetic_and_all77_export_review_PASS_scientific_gate_FAIL",
              "cohorts": audit, "same_encoder_and_readout_training_rows": True,
              "cross_fitted_training_activations": False, "both_evaluation_cohorts_reused": True,
              "scope": "fixed ridge.001 scalar output refit on original frozen encoders; no broader architecture impossibility claim",
              "production_change": False, "protocol_sha256": digest(here / "protocol_frozen_v1.json"),
              "source_sha256": digest(Path(__file__)), "input_arrays_sha256": digest(path)}
    with (here / "independent_result_review_v1.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
