# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — independent arithmetic review of equal-family weighting
"""Recompute the complete frozen D52 comparison without changing its gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    """Return the SHA-256 of one input or source."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    """Validate identities and paired whole-shot score changes and bootstrap."""
    here = Path(__file__).resolve().parent
    group = Path("$LAB")
    peer = group / "work/first_agent/d52_family_weight"
    reports = Path("$FEC_RUNS/ml350/runs/d39_finisher_20261011/reports")
    protocol = json.loads((peer / "protocol_d52.json").read_text())
    amendment = json.loads((peer / "protocol_d52_amendment_1.json").read_text())
    expected = {**protocol["sha256"], **amendment["sha256"]}
    for filename, sha in expected.items():
        assert digest(peer / filename) == sha
    assert digest(peer / "protocol_d52.json").startswith(amendment["amends"].split()[-1])
    result = json.loads((reports / "d52_equal_family_358.json").read_text())
    with np.load(reports / "d52_family_means_358.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    with np.load(reports / "d49_qb_frames_358.npz", allow_pickle=False) as archive:
        cached = {name: archive[name] for name in archive.files}
    shot = arrays["shot"]
    assert np.array_equal(shot, cached["shot"])
    assert np.array_equal(arrays["stems"], cached["stems"])
    assert len(shot) == 78875 and len(arrays["stems"]) == 358
    draws = np.random.default_rng(5201).integers(0, 358, size=(10000, 358))
    counts = np.stack([np.bincount(draw, minlength=358) for draw in draws]).astype(float)
    del draws
    screening = np.arange(358) % 3 == 0
    delta = 0.0
    bootstrap = np.zeros(10000)
    parts = np.zeros(2)
    deviations = {}
    means = {"q95": 5.262893061875164, "betaN": 1.2688498003107427}
    for name in ("q95", "betaN"):
        truth = arrays[f"{name}_truth"].astype(float)
        assert np.array_equal(truth, cached[f"{name}_truth"])
        hx, hc, joint = [arrays[f"{name}_{key}"].astype(float) for key in ("hx", "hc", "joint")]
        assert all(np.isfinite(value).all() for value in (truth, hx, hc, joint))
        base = .75 * (21 * hx + 48 * hc) / 69 + .25 * joint
        new = (hx + hc + joint) / 3
        old = .75 * cached[f"{name}_map"] + .25 * cached[f"{name}_joint"]
        deviations[name] = float(np.max(np.abs(base - old)))
        assert deviations[name] < 1e-3
        denominator = np.bincount(shot, weights=(truth - means[name]) ** 2, minlength=358)
        baseline_error = np.bincount(shot, weights=(base - truth) ** 2, minlength=358)
        candidate_error = np.bincount(shot, weights=(new - truth) ** 2, minlength=358)
        difference = baseline_error - candidate_error
        change = float(difference.sum() / denominator.sum())
        assert abs(change - result[name]["dR2"]) < 1e-12
        delta += .075 * change
        bootstrap += .075 * (counts @ difference) / (counts @ denominator)
        for index, pick in enumerate((screening, ~screening)):
            parts[index] += .075 * difference[pick].sum() / denominator[pick].sum()
    interval = np.quantile(bootstrap, [.005, .995])
    assert abs(delta - result["dS"]) < 1e-12
    assert np.max(np.abs(interval - result["dS_ci99"])) < 1e-12
    assert abs(float(np.mean(bootstrap > 0)) - result["P_dS_positive"]) < 1e-12
    for observed, name in zip(parts, ("screening_120", "other_238"), strict=True):
        assert abs(observed - result["dS_by_part"][name]) < 1e-12
    passed = bool(delta >= 1e-5 and interval[0] > 0 and parts.min() > 0)
    assert passed == result["passes"]
    audit = {"status": "independent_arithmetic_PASS_scientific_gate_FAIL" if not passed else "independent_arithmetic_PASS_scientific_gate_PASS",
             "wake_seq": 146389, "shots": 358, "frames": 78875,
             "delta_S": delta, "CI99": interval.tolist(), "gate_pass": passed,
             "maximum_deviation_from_old_cached_baseline": deviations,
             "source_amendment_verified": True,
             "fixed_full_cohort_means_bootstrap_as_declared": True,
             "cohort_reused": True, "production_change_authorized_by_this_review": False,
             "inputs_sha256": {name: digest(reports / name) for name in
                               ("d52_family_means_358.npz", "d52_equal_family_358.json", "d49_qb_frames_358.npz")},
             "protocol_sha256": digest(peer / "protocol_d52.json"),
             "amendment_sha256": digest(peer / "protocol_d52_amendment_1.json"),
             "audit_source_sha256": digest(Path(__file__))}
    with (here / "audit_receipt_v1.json").open("x") as stream:
        json.dump(audit, stream, indent=2)
        stream.write("\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
