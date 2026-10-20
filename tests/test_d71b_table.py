# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — D71b public-table contract tests
"""Verify real D71b data and reject damaged cohort, support and paired statistics."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from tools.d71b_table import verify

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments/d71b_physics_amendment"
TABLE = DATA / "per_shot_v1.csv.gz"
MANIFEST = DATA / "per_shot_manifest_v1.json"
REPORT = DATA / "report_v5.json"


@pytest.fixture(scope="module")
def real_rows() -> list[list[str]]:
    """Read only the actual published full-cohort table."""
    with gzip.open(TABLE, "rt", newline="") as stream:
        return list(csv.reader(stream))


def test_public_cli() -> None:
    """Reproduce every supported comparison through the documented CLI."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--parallel-mode",
            "tools/d71b_table.py",
            "--table",
            str(TABLE),
            "--manifest",
            str(MANIFEST),
            "--report",
            str(REPORT),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=240,
    )
    report = json.loads(result.stdout)
    assert report == {
        "status": "verified",
        "shots": 358,
        "frames": 78875,
        "rows": 61218,
        "maximum_absolute_difference": report["maximum_absolute_difference"],
        "frame_percentiles_verified": False,
        "per_frame_physics_recomputed": False,
        "original_measurements_authenticated": False,
    }
    assert report["maximum_absolute_difference"] < 1e-14


@pytest.mark.parametrize(
    "fault",
    ["non_object", "missing_field", "extra_field", "method", "cohort", "source_digest", "table_digest"],
)
def test_manifest_corruption(tmp_path: Path, fault: str) -> None:
    """Refuse substituted identities, method settings and cohort permutations."""
    manifest: dict[str, object] = json.loads(MANIFEST.read_text())
    if fault == "non_object":
        contents = "[]"
    else:
        if fault == "missing_field":
            del manifest["schema"]
        elif fault == "extra_field":
            manifest["unexpected"] = "not in the export contract"
        elif fault == "method":
            manifest["seed"] = 71019
        elif fault == "cohort":
            cast(list[object], manifest["cohort"]).reverse()
        elif fault == "source_digest":
            manifest["statistics_export_sha256"] = "invalid"
        else:
            manifest["table_sha256"] = "0" * 64
        contents = json.dumps(manifest)
    destination = tmp_path / "manifest.json"
    destination.write_text(contents)
    with pytest.raises(ValueError):
        verify(TABLE, destination, REPORT)


def test_changed_frozen_report(tmp_path: Path) -> None:
    """Refuse report bytes changed independently of the original frozen result."""
    destination = tmp_path / "report.json"
    destination.write_bytes(REPORT.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="bytes differ"):
        verify(TABLE, MANIFEST, destination)


@pytest.mark.parametrize(
    "fault",
    [
        "header",
        "empty",
        "missing_row",
        "extra_row",
        "extra_column",
        "identity",
        "noncanonical_count",
        "negative_count",
        "excess_count",
        "missing_mean",
        "unsupported_mean",
        "empty_support",
        "nan",
        "map_support",
        "pair_support",
        "map_mean",
        "overflow_mean",
        "pair_mean",
        "pair_interval",
        "pair_verdict",
    ],
)
def test_statistics_corruption(tmp_path: Path, real_rows: list[list[str]], fault: str) -> None:
    """Refuse damaged real statistics even when their transport digest is updated."""
    rows = [row.copy() for row in real_rows]
    first = rows[1]
    pair = rows[11]
    assert pair[1:4] == ["pair", "smoothed->margin", "outside_fixed_excess"]
    if fault == "header":
        rows[0][0] = "unknown_column"
    elif fault == "empty":
        rows.clear()
    elif fault == "missing_row":
        rows.pop()
    elif fault == "extra_row":
        rows.append(first.copy())
    elif fault == "extra_column":
        first.append("extra")
    elif fault == "identity":
        first[0] = "d3d_shot_wrong"
    elif fault == "noncanonical_count":
        first[4] = "0" + first[4]
    elif fault == "negative_count":
        first[4] = "-1"
    elif fault == "excess_count":
        first[4] = "999999"
    elif fault == "missing_mean":
        first[5] = ""
    elif fault == "unsupported_mean":
        first[4] = "0"
    elif fault == "empty_support":
        first[4:] = ["0", ""]
    elif fault == "nan":
        first[5] = "nan"
    elif fault == "map_support":
        first[4] = str(int(first[4]) - 1)
    elif fault == "pair_support":
        pair[4] = str(int(pair[4]) - 1)
    elif fault == "map_mean":
        first[5] = "0.1"
    elif fault == "overflow_mean":
        for row in rows[1:]:
            if row[1:4] == ["map", "reference", "outside_fixed_excess"]:
                row[5] = "1e308"
    elif fault == "pair_mean":
        pair[5] = str(float(pair[5]) + 0.01)
    elif fault == "pair_interval":
        second_pair = rows[11 + 9 * 19]
        assert second_pair[1:4] == pair[1:4]
        pair[5] = str(float(pair[5]) + 0.1)
        second_pair[5] = str(float(second_pair[5]) - 0.1)
    else:
        for row in rows[1:]:
            if row[1:4] == ["pair", "smoothed->margin", "outside_fixed_excess"]:
                row[5] = str(-abs(float(row[5])) - 1)
    table = tmp_path / "statistics.csv.gz"
    with gzip.open(table, "wt", newline="") as stream:
        csv.writer(stream).writerows(rows)
    manifest = json.loads(MANIFEST.read_text())
    manifest["table_sha256"] = hashlib.sha256(table.read_bytes()).hexdigest()
    destination = tmp_path / "manifest.json"
    destination.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verify(table, destination, REPORT)


def test_cli_refusal(tmp_path: Path) -> None:
    """Keep missing-input failures caller-safe through the actual process boundary."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--parallel-mode",
            "tools/d71b_table.py",
            "--table",
            str(tmp_path / "missing.csv.gz"),
            "--manifest",
            str(MANIFEST),
            "--report",
            str(REPORT),
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
        "message": "D71b inputs are missing, invalid or inconsistent.",
    }
    assert result.stderr == ""
