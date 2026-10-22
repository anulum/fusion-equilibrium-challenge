# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — surface-support table contract tests
"""Reproduce the actual complete table and reject damaged copies of its records."""

from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.surface_support_table import verify

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments/d71b_surface_support"
TABLE = DATA / "per_shot_v1.json.gz"
MANIFEST = DATA / "manifest_v1.json"
REPORT = DATA / "report_v1.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_actual_complete_cli() -> None:
    """Verify the actual 358-shot release through the standalone command."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--parallel-mode",
            "-m",
            "tools.surface_support_table",
            "--table",
            str(TABLE),
            "--manifest",
            str(MANIFEST),
            "--manifest-sha256",
            _digest(MANIFEST),
            "--report",
            str(REPORT),
        ],
        text=True,
        cwd=ROOT,
        capture_output=True,
        check=True,
        timeout=120,
    )
    decoded = json.loads(result.stdout)
    assert decoded["status"] == "verified"
    assert decoded["shots"] == 358 and decoded["frames"] == 78875
    assert decoded["maximum_absolute_difference"] <= 1e-14
    assert decoded["frame_quantiles_reproduced"] is False
    assert decoded["private_maps_remeasured"] is False
    assert decoded["original_measurements_authenticated"] is False
    assert result.stderr == ""


def test_manifest_checksum_is_an_independent_anchor() -> None:
    """A valid-length but wrong detached manifest digest cannot bind a result."""
    with pytest.raises(ValueError, match="Manifest identity"):
        verify(TABLE, MANIFEST, REPORT, "0" * 64)
    with pytest.raises(ValueError, match="Manifest identity"):
        verify(TABLE, MANIFEST, REPORT, "not-a-checksum")


@pytest.mark.parametrize(
    "fault",
    [
        "manifest_object",
        "method",
        "cohort_order",
        "table_digest",
        "report_digest",
        "table_object",
        "schema",
        "shots_object",
        "missing_shot",
        "duplicated_shot",
        "frame_count",
        "map_count",
        "surface_count",
        "missing_key",
        "infinite_change",
        "bootstrap_change",
        "level_count",
        "reason_count",
        "condition_support",
        "report_object",
        "report_extra",
        "report_missing",
        "report_list",
        "report_number",
        "report_number_type",
        "report_nan",
        "report_count_type",
        "report_boolean_type",
        "broken_gzip",
        "truncated_gzip",
        "broken_json",
    ],
)
def test_damaged_real_statistics_refused(tmp_path: Path, fault: str) -> None:
    """Changed real records cannot silently alter the cohort or reproduced report."""
    manifest = json.loads(MANIFEST.read_text())
    report = json.loads(REPORT.read_text())
    with gzip.open(TABLE, "rt") as stream:
        table = json.load(stream)
    shots = table["shots"]
    if fault == "manifest_object":
        manifest = []
    elif fault == "method":
        manifest["draws"] += 1
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
    elif fault == "duplicated_shot":
        shots[-1] = shots[0]
    elif fault == "frame_count":
        shots[0]["frames"] += 1
    elif fault == "map_count":
        shots[0]["maps"].pop()
    elif fault == "surface_count":
        shots[0]["maps"][0]["levels"].pop()
    elif fault == "missing_key":
        del shots[0]["maps"]
    elif fault == "infinite_change":
        shots[0]["common"]["change"] = float("inf")
    elif fault == "bootstrap_change":
        shots[0]["common"]["change"] += 1
        shots[1]["common"]["change"] -= 1
    elif fault == "level_count":
        shots[0]["maps"][0]["levels"][0]["used"] += 1
    elif fault == "reason_count":
        shots[0]["maps"][0]["levels"][0]["reasons"]["used"] += 1
    elif fault == "condition_support":
        shots[0]["maps"][0]["levels"][0]["condition_unavailable"] += 1
    elif fault == "report_object":
        report = []
    elif fault == "report_extra":
        report["unexpected"] = 0
    elif fault == "report_missing":
        del report["common"]
    elif fault == "report_list":
        report["common"]["raw_means"] = []
    elif fault == "report_number":
        report["common"]["change"]["mean_change"] += 1
    elif fault == "report_number_type":
        report["common"]["change"]["mean_change"] = "not a number"
    elif fault == "report_nan":
        report["common"]["change"]["mean_change"] = float("nan")
    elif fault == "report_count_type":
        report["frames"] = float(report["frames"])
    elif fault == "report_boolean_type":
        report["private_maps_remeasured"] = 0
    destination = tmp_path / "table.json.gz"
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


def test_cli_missing_inputs_has_fixed_refusal(tmp_path: Path) -> None:
    """Unreadable input paths reach callers as a fixed sentence without a traceback."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--parallel-mode",
            "-m",
            "tools.surface_support_table",
            "--table",
            str(tmp_path / "absent"),
            "--manifest",
            str(tmp_path / "absent"),
            "--manifest-sha256",
            "0" * 64,
            "--report",
            str(tmp_path / "absent"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert result.stdout == "Unable to verify the supplied surface-support statistics.\n"
    assert result.stderr == ""
