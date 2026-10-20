# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — D72 pooled public statistics contract tests
"""Reproduce the actual D72 gate and refuse damaged real per-shot statistics."""

from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from tools.d72_table import Statistics, verify

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "experiments/d72_gs_projection"
DATA = FOLDER / "pooled_statistics_v1.json.gz"
MANIFEST = FOLDER / "pooled_manifest_v1.json"
GATE = FOLDER / "gate_v1.json"


def test_public_cli() -> None:
    """Reproduce pooled components and whole-shot confidence limits via the real CLI."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--parallel-mode",
            "tools/d72_table.py",
            "--data",
            str(DATA),
            "--manifest",
            str(MANIFEST),
            "--gate",
            str(GATE),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )
    actual = json.loads(result.stdout)
    assert actual["status"] == "verified"
    assert actual["maximum_absolute_difference"] < 1e-13
    assert actual["shots"] == 358 and actual["frames"] == actual["used_frames"] == 78875
    assert actual["nonconverged_frames"] == 3435 and actual["shots_with_nonconvergence"] == 185
    assert actual["delta_S"] == pytest.approx(-0.00015708902673605696, abs=1e-14)
    assert actual["CI99"] == pytest.approx([-0.00023884313450385443, -0.00009310851188393185], abs=1e-14)
    stored = json.loads(GATE.read_text())
    for arm in ("base", "gs"):
        for component in ("S", "psi", "qb", "D", "C", "cons_each"):
            assert actual[arm][component] == pytest.approx(stored[arm][component], abs=1e-13)
        assert actual[arm]["lcfs_fails"] == actual[arm]["cons_fails"] == 0
    assert actual["PASS"] is False
    assert actual["maps_recomputed"] is False and actual["physical_improvement_verified"] is False
    assert actual["original_measurements_authenticated"] is False
    assert result.stderr == ""


@pytest.mark.parametrize("fault", ["not_object", "missing", "extra", "seed", "export", "data", "cohort"])
def test_manifest_refusal(tmp_path: Path, fault: str) -> None:
    """Refuse missing identities, substituted method settings and changed shot order."""
    manifest = json.loads(MANIFEST.read_text())
    if fault == "not_object":
        manifest = []
    elif fault == "missing":
        del manifest["schema"]
    elif fault == "extra":
        manifest["unexpected"] = 1
    elif fault == "seed":
        manifest["seed"] -= 1
    elif fault == "export":
        manifest["export_source_sha256"] = "invalid"
    elif fault == "data":
        manifest["data_sha256"] = "0" * 64
    else:
        manifest["cohort"].reverse()
    destination = tmp_path / "manifest.json"
    destination.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verify(DATA, destination, GATE)


def test_frozen_gate_refusal(tmp_path: Path) -> None:
    """Refuse even a numerically unchanged replacement of the frozen gate bytes."""
    gate = tmp_path / "gate.json"
    gate.write_bytes(GATE.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="bytes differ"):
        verify(DATA, MANIFEST, gate)


@pytest.mark.parametrize(
    "fault",
    [
        "missing_field",
        "reorder",
        "used",
        "converged",
        "shape",
        "nonfinite",
        "negative_square",
        "fractional_count",
        "psi_count",
        "scalar_count",
        "extra_arm",
        "negative_residual",
        "distance_shape",
        "infinite_distance",
        "nan_distance",
        "negative_distance",
        "missing_distance",
        "all_missing_distance",
        "fractional_counter",
        "excess_counter",
        "failure_count",
        "scalar_residual",
        "reference_variance",
        "bootstrap_reference",
        "bootstrap_residual",
        "zero_flux_variance",
        "zero_scalar_variance",
        "zero_consistency_variance",
    ],
)
def test_damaged_statistics(tmp_path: Path, fault: str) -> None:
    """Refuse corrupted real records beyond a refreshed transport digest."""
    data = cast(Statistics, json.loads(gzip.decompress(DATA.read_bytes())))
    if fault == "missing_field":
        cast(dict[str, object], cast(object, data)).pop("stems")
    elif fault == "reorder":
        data["stems"].reverse()
    elif fault == "used":
        data["used"][0] -= 1
    elif fault == "converged":
        data["converged"][0] -= 1
    elif fault == "shape":
        data["reference_sum"].pop()
    elif fault == "nonfinite":
        data["reference_sum"][0][0] = float("nan")
    elif fault == "negative_square":
        data["reference_sumsq"][0][0] = -1
    elif fault == "fractional_count":
        data["reference_n"][0][1] -= 0.5
    elif fault == "psi_count":
        data["reference_n"][0][0] -= 1
    elif fault == "scalar_count":
        data["reference_n"][0][1] = 1000000
    elif fault == "extra_arm":
        data["residual"]["unexpected"] = data["residual"]["base"]
    elif fault == "negative_residual":
        data["residual"]["base"][0][0] = -1
    elif fault == "distance_shape":
        data["distance"]["base"].pop()
    elif fault == "infinite_distance":
        data["distance"]["base"][0] = float("inf")
    elif fault == "nan_distance":
        data["distance"]["base"][0] = float("nan")
    elif fault == "negative_distance":
        data["distance"]["base"][0] = -1
    elif fault == "missing_distance":
        data["distance"]["base"][0] = None
    elif fault == "all_missing_distance":
        data["distance"]["base"] = [None] * 358
    elif fault == "fractional_counter":
        data["counters"]["base"][0][0] = 1.5  # type: ignore[call-overload] # Damaged external JSON count.
    elif fault == "excess_counter":
        data["counters"]["base"][0][0] = 1000000
    elif fault == "failure_count":
        data["counters"]["base"][0][15] = 1
    elif fault == "scalar_residual":
        data["residual"]["base"][0][1] += 10
    elif fault == "reference_variance":
        data["reference_sumsq"][0][4] += 1
    elif fault == "bootstrap_reference":
        data["reference_sumsq"][0][4] += 1
        data["reference_sumsq"][1][4] -= 1
    elif fault == "bootstrap_residual":
        value = min(data["residual"]["base"][0][4], data["residual"]["base"][1][4]) / 2
        data["residual"]["base"][0][4] += value
        data["residual"]["base"][1][4] -= value
    else:
        column = {"zero_flux_variance": 0, "zero_scalar_variance": 1, "zero_consistency_variance": 3}[fault]
        for sums, squares in zip(data["reference_sum"], data["reference_sumsq"], strict=True):
            sums[column] = squares[column] = 0
    table = tmp_path / "statistics.json.gz"
    table.write_bytes(gzip.compress(json.dumps(data).encode()))
    manifest = json.loads(MANIFEST.read_text())
    manifest["data_sha256"] = hashlib.sha256(table.read_bytes()).hexdigest()
    destination = tmp_path / "manifest.json"
    destination.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verify(table, destination, GATE)


@pytest.mark.parametrize("fault", ["missing", "truncated_gzip", "invalid_json"])
def test_cli_refusal(tmp_path: Path, fault: str) -> None:
    """Refuse an unavailable input without exposing a path or traceback to the caller."""
    table = tmp_path / "absent.gz"
    manifest_path = MANIFEST
    if fault != "missing":
        contents = DATA.read_bytes()[:-10] if fault == "truncated_gzip" else gzip.compress(b"{")
        table.write_bytes(contents)
        manifest = json.loads(MANIFEST.read_text())
        manifest["data_sha256"] = hashlib.sha256(contents).hexdigest()
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--parallel-mode",
            "tools/d72_table.py",
            "--data",
            str(table),
            "--manifest",
            str(manifest_path),
            "--gate",
            str(GATE),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 2
    assert json.loads(result.stdout) == {
        "status": "refused",
        "message": "D72 inputs are missing, invalid or inconsistent.",
    }
    assert result.stderr == ""
