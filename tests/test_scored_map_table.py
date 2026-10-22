# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — scored-map table contract tests
"""Reproduce the actual complete release and reject changed real statistics."""

from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from tools.scored_map_table import ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE, verify

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments/d72_scored_map_physics"
TABLE, MANIFEST, REPORT = (DATA / n for n in ("per_shot_v1.json.gz", "manifest_v1.json", "report_v1.json"))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_actual_full_release_cli() -> None:
    """Verify all actual 358 shots through the standalone published interface."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--branch",
            "--parallel-mode",
            "-m",
            "tools.scored_map_table",
            "--table",
            str(TABLE),
            "--manifest",
            str(MANIFEST),
            "--manifest-sha256",
            _digest(MANIFEST),
            "--report",
            str(REPORT),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=900,
    )
    verified = json.loads(result.stdout)
    assert verified["status"] == "verified"
    assert verified["shots"] == 358 and verified["frames"] == 78875
    assert verified["maximum_scaled_difference"] <= 1
    assert verified["absolute_tolerance"] == ABSOLUTE_TOLERANCE
    assert verified["relative_tolerance"] == RELATIVE_TOLERANCE
    assert all(
        verified[k] is False
        for k in (
            "frame_quantiles_reproduced",
            "private_maps_remeasured",
            "original_measurements_authenticated",
            "physical_force_balance_certified",
        )
    )
    assert result.stderr == ""


@pytest.mark.parametrize("digest", ["0" * 64, "invalid"])
def test_independent_manifest_anchor_required(digest: str) -> None:
    """A wrong or malformed detached checksum cannot bind the input package."""
    with pytest.raises(ValueError, match="Manifest identity"):
        verify(TABLE, MANIFEST, REPORT, digest)


@pytest.mark.parametrize(
    "fault",
    [
        "frame_type",
        "reference_type",
        "stratum_type",
        "stratum_negative",
        "count_type",
        "count_negative",
        "count_too_large",
        "mean_unavailable",
        "mean_boolean",
        "mean_nan",
        "pair_count_type",
        "pair_count_negative",
        "pair_count_too_large",
        "unsupported_pair_mean",
        "unsupported_pair_change",
        "supported_pair_missing",
        "pair_change_type",
        "manifest_object",
        "method",
        "cohort_order",
        "table_digest",
        "report_digest",
        "table_object",
        "schema",
        "shots_object",
        "missing_shot",
        "duplicate_shot",
        "frame_count",
        "reference_count",
        "strata",
        "stratum_counts",
        "partition",
        "all_frames",
        "metrics",
        "counts_vector",
        "means_vector",
        "pairs",
        "pair_means",
        "missing_key",
        "infinite_change",
        "bootstrap_change",
        "count_change",
        "raw_mean",
        "matched_mean",
        "report_object",
        "report_extra",
        "report_missing",
        "report_list",
        "report_number",
        "report_number_type",
        "report_nan",
        "report_count_type",
        "report_boolean_type",
        "report_certificate",
        "broken_gzip",
        "truncated_gzip",
        "broken_json",
    ],
)
def test_changed_real_release_refused(tmp_path: Path, fault: str) -> None:
    """Transport, identity, structure and arithmetic changes cannot silently pass."""
    manifest = json.loads(MANIFEST.read_text())
    report = json.loads(REPORT.read_text())
    with gzip.open(TABLE, "rt") as stream:
        table = json.load(stream)
    shots = table["shots"]
    entry = shots[0]["statistics"]["all"]["gs_surface"]
    if fault == "frame_type":
        shots[0]["frames"] = float(shots[0]["frames"])
    elif fault == "reference_type":
        shots[0]["reference_frames"] = float(shots[0]["reference_frames"])
    elif fault == "stratum_type":
        shots[0]["stratum_frames"]["unused"] = False
    elif fault == "stratum_negative":
        shots[0]["stratum_frames"]["used_converged"] += 1
        shots[0]["stratum_frames"]["used_nonconverged"] -= 1
    elif fault == "count_type":
        entry["counts"][0] = float(entry["counts"][0])
    elif fault == "count_negative":
        entry["counts"][0] = -1
    elif fault == "count_too_large":
        entry["counts"][0] = shots[0]["frames"] + 1
    elif fault == "mean_unavailable":
        entry["means"][0] = None
    elif fault == "mean_boolean":
        entry["means"][0] = False
    elif fault == "mean_nan":
        entry["means"][0] = float("nan")
    elif fault == "pair_count_type":
        entry["pairs"]["base->gs"]["frames"] = float(entry["pairs"]["base->gs"]["frames"])
    elif fault == "pair_count_negative":
        entry["pairs"]["base->gs"]["frames"] = -1
    elif fault == "pair_count_too_large":
        entry["pairs"]["base->gs"]["frames"] += 1
    elif fault == "unsupported_pair_mean":
        shots[0]["statistics"]["unused"]["gs_surface"]["pairs"]["base->gs"]["means"] = [0.0, 0.0]
    elif fault == "unsupported_pair_change":
        shots[0]["statistics"]["unused"]["gs_surface"]["pairs"]["base->gs"]["change"] = 0.0
    elif fault == "supported_pair_missing":
        entry["pairs"]["base->gs"]["means"] = None
    elif fault == "pair_change_type":
        entry["pairs"]["base->gs"]["change"] = False
    elif fault == "manifest_object":
        manifest = []
    elif fault == "method":
        manifest["seed"] += 1
    elif fault == "cohort_order":
        manifest["cohort"].reverse()
    elif fault == "table_object":
        table = []
    elif fault == "schema":
        table["schema"] = "unknown"
    elif fault == "shots_object":
        table["shots"] = {}
    elif fault == "missing_shot":
        shots.pop()
    elif fault == "duplicate_shot":
        shots[-1] = shots[0]
    elif fault == "frame_count":
        shots[0]["frames"] += 1
    elif fault == "reference_count":
        shots[0]["reference_frames"] -= 1
    elif fault == "strata":
        del shots[0]["statistics"]["unused"]
    elif fault == "stratum_counts":
        del shots[0]["stratum_frames"]["unused"]
    elif fault == "partition":
        shots[0]["stratum_frames"]["used_converged"] -= 1
    elif fault == "all_frames":
        shots[0]["stratum_frames"]["all"] -= 1
    elif fault == "metrics":
        del shots[0]["statistics"]["all"]["gs_surface"]
    elif fault == "counts_vector":
        entry["counts"].pop()
    elif fault == "means_vector":
        entry["means"].pop()
    elif fault == "pairs":
        del entry["pairs"]["base->gs"]
    elif fault == "pair_means":
        entry["pairs"]["base->gs"]["means"].pop()
    elif fault == "missing_key":
        del shots[0]["statistics"]
    elif fault == "infinite_change":
        entry["pairs"]["base->gs"]["change"] = float("inf")
    elif fault == "bootstrap_change":
        entry["pairs"]["base->gs"]["change"] += 1
        shots[179]["statistics"]["all"]["gs_surface"]["pairs"]["base->gs"]["change"] -= 1
    elif fault == "count_change":
        entry["counts"][0] += 1
    elif fault == "raw_mean":
        entry["means"][0] += 1
    elif fault == "matched_mean":
        entry["pairs"]["base->gs"]["means"][0] += 1
    elif fault == "report_object":
        report = []
    elif fault == "report_extra":
        report["unexpected"] = 0
    elif fault == "report_missing":
        del report["strata"]
    elif fault == "report_list":
        report["maps"] = []
    elif fault == "report_number":
        report["strata"]["all"]["measures"]["gs_surface"]["means"][0] += 1
    elif fault == "report_number_type":
        report["strata"]["all"]["measures"]["gs_surface"]["means"][0] = "invalid"
    elif fault == "report_nan":
        report["strata"]["all"]["measures"]["gs_surface"]["means"][0] = float("nan")
    elif fault == "report_count_type":
        report["frames"] = float(report["frames"])
    elif fault == "report_boolean_type":
        report["physical_force_balance_certified"] = 0
    elif fault == "report_certificate":
        report["physical_force_balance_certified"] = True
    destination = tmp_path / "statistics.json.gz"
    destination.write_bytes(gzip.compress(json.dumps(table).encode(), mtime=0))
    if fault == "broken_gzip":
        destination.write_bytes(b"invalid compressed statistics")
    elif fault == "truncated_gzip":
        destination.write_bytes(destination.read_bytes()[:-20])
    elif fault == "broken_json":
        destination.write_bytes(gzip.compress(b"invalid JSON"))
    stored = tmp_path / "report.json"
    stored.write_text(json.dumps(report))
    if isinstance(manifest, dict):
        manifest["table_sha256"] = "0" * 64 if fault == "table_digest" else _digest(destination)
        manifest["report_sha256"] = "0" * 64 if fault == "report_digest" else _digest(stored)
    binding = tmp_path / "manifest.json"
    binding.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verify(destination, binding, stored, _digest(binding))


def test_roundoff_within_declared_tolerance_accepted(tmp_path: Path) -> None:
    """A sub-tolerance decimal perturbation preserves the arithmetic verification."""
    manifest = json.loads(MANIFEST.read_text())
    report = json.loads(REPORT.read_text())
    values = report["strata"]["all"]["measures"]["gs_surface"]["means"]
    values[0] += (ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * abs(values[0])) / 4
    stored = tmp_path / "report.json"
    stored.write_text(json.dumps(report))
    manifest["report_sha256"] = _digest(stored)
    binding = tmp_path / "manifest.json"
    binding.write_text(json.dumps(manifest))
    verified = verify(TABLE, binding, stored, _digest(binding))
    assert 0 < cast(float, verified["maximum_scaled_difference"]) <= 1
    assert cast(float, verified["maximum_absolute_difference"]) > 0


def test_missing_input_cli_has_fixed_refusal(tmp_path: Path) -> None:
    """Unreadable paths produce a fixed sentence and no interpreter traceback."""
    absent = tmp_path / "absent"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--branch",
            "--parallel-mode",
            "-m",
            "tools.scored_map_table",
            "--table",
            str(absent),
            "--manifest",
            str(absent),
            "--manifest-sha256",
            "0" * 64,
            "--report",
            str(absent),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert result.stdout == "Unable to verify the supplied scored-map diagnostic statistics.\n"
    assert result.stderr == ""
