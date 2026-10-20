# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — public D71 table contract tests
"""Exercise the public verifier with its actual published data and damaged copies."""

from __future__ import annotations

import csv
import gzip
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from experiments.d71_finisher_physics.per_shot_table import verify

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments/d71_finisher_physics"


def test_dependency_audit_contract() -> None:
    """Keep every pinned environment in the blocking, transitive CI audit matrix."""
    workflow = (ROOT / ".github/workflows/diagnostics.yml").read_text()
    requirements = {path.name for path in ROOT.glob("requirements-*.txt")}
    entries = {
        line.strip().split(": ", 1)[1]
        for line in workflow.splitlines()
        if line.strip().startswith("- requirements:")
    }
    assert entries == requirements
    command = (
        'run: pip-audit --strict --vulnerability-service "${{ matrix.service }}" '
        '--progress-spinner off -r "${{ matrix.requirements }}"'
    )
    assert command in workflow
    for weakened in ("continue-on-error", "|| true", "--ignore-vuln", "--skip-editable"):
        assert weakened not in workflow
    for filename in requirements:
        service = "osv" if filename.startswith("requirements-training-") else "pypi"
        assert f"- requirements: {filename}\n            service: {service}" in workflow
    assert "if: matrix.service == 'osv'" in workflow
    assert 'test "$(wc -l < "$RUNNER_TEMP/torch-upstream.txt")" -eq 1' in workflow
    assert "https://download-r2.pytorch.org/whl/cpu/torch-" in workflow
    assert "--format cyclonedx-json" in workflow
    assert "set -euo pipefail" in workflow
    assert "Audit and inventory every locked version" in workflow
    assert 'diff -u "$RUNNER_TEMP/expected-versions.txt" "$RUNNER_TEMP/sbom-versions.txt"' in workflow
    assert '"sbom-resolved-${{ matrix.requirements }}.json"' in workflow
    assert 'pip-audit --strict --disable-pip --no-deps --vulnerability-service "$AUDIT_SERVICE"' in workflow
    assert (
        "pip-audit --strict --disable-pip --no-deps --progress-spinner off "
        '-r "$RUNNER_TEMP/torch-upstream.txt"'
    ) in workflow
    assert "  push:\n    branches: [main]\n  pull_request:" in workflow


def test_diagnostic_scope_contract() -> None:
    """Keep maintained source and its real-data tests enrolled in each static gate."""
    workflow = (ROOT / ".github/workflows/diagnostics.yml").read_text()
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    source = "experiments/d71_finisher_physics/per_shot_table.py"
    tests = "tests/test_d71_table.py"
    for command in (
        f"ruff check {source} {tests}",
        f"ruff format --check {source} {tests}",
        f"bandit -q {source}",
        "          mypy\n",
    ):
        assert command in workflow
    support = {"tools/check_doc_links.py", "tools/ci_check_shot.py", "tools/d71b_table.py"}
    support_tests = {
        "tests/test_d71b_table.py",
        "tests/test_check_doc_links.py",
        "tests/test_ci_check_shot.py",
        "tests/test_licensing.py",
    }
    assert set(config["tool"]["mypy"]["files"]) == {source, tests} | support | support_tests
    for path in support | support_tests:
        assert path in workflow
    for path in {source, tests} | support | support_tests:
        for prefix in ("ruff check ", "ruff format --check "):
            assert any(
                line.strip().startswith(prefix) and path in line.split() for line in workflow.splitlines()
            )
    assert config["tool"]["mypy"]["strict"] is True
    assert config["tool"]["coverage"]["run"]["branch"] is True
    assert set(config["tool"]["coverage"]["report"]["include"]) == {source} | support
    assert config["tool"]["coverage"]["report"]["fail_under"] == 100


def test_published_cli() -> None:
    """Reproduce all historical means and intervals through the public CLI."""
    command = [sys.executable, "-m", "coverage", "run", "--parallel-mode"]
    result = subprocess.run(
        [
            *command,
            str(DATA / "per_shot_table.py"),
            "--table",
            str(DATA / "per_shot_v1.csv.gz"),
            "--report",
            str(DATA / "report_v2.json"),
        ],
        text=True,
        capture_output=True,
        check=True,
        timeout=180,
    )
    payload = json.loads(result.stdout)
    assert payload["shots"] == 358
    assert payload["rows"] == 3580
    assert payload["maximum_absolute_difference"] < 1e-14
    assert payload["frame_percentiles_verified"] is False


@pytest.mark.parametrize(
    "fault",
    [
        "shots",
        "frames",
        "maps",
        "measures",
        "seed",
        "draws",
        "map_mean",
        "comparison",
        "undefined_mean",
        "undefined_change",
        "changed_mean",
    ],
)
def test_report_corruption(tmp_path: Path, fault: str) -> None:
    """Reject damaged report metadata, omitted comparisons and false numerical claims."""
    report = json.loads((DATA / "report_v2.json").read_text())
    measure = report["measures"]["current_error_own"]
    if fault in ("shots", "frames", "seed", "draws"):
        report[fault] -= 1
    elif fault == "maps":
        report["maps"].reverse()
    elif fault == "measures":
        del report["measures"]["current_error_own"]
    elif fault == "map_mean":
        del measure["per_map"]["margin"]
    elif fault == "comparison":
        del measure["changes"]["smoothed->margin"]
    elif fault == "undefined_mean":
        measure["per_map"]["margin"]["shot_mean"] = float("nan")
    elif fault == "undefined_change":
        measure["changes"]["smoothed->margin"]["mean_change"] = float("inf")
    else:
        measure["per_map"]["margin"]["shot_mean"] += 0.1
    destination = tmp_path / "report.json"
    destination.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        verify(DATA / "per_shot_v1.csv.gz", destination)


@pytest.mark.parametrize(
    "fault",
    [
        "missing_row",
        "duplicate_row",
        "unexpected_map",
        "reference_frames",
        "inconsistent_frames",
        "negative_support",
        "excess_support",
        "missing_value",
        "nonfinite_value",
        "zero_support_value",
    ],
)
def test_table_corruption(tmp_path: Path, fault: str) -> None:
    """Reject broken shot coverage, support and values in the actual published CSV."""
    with gzip.open(DATA / "per_shot_v1.csv.gz", "rt", newline="") as stream:
        rows = list(csv.DictReader(stream))
    row = rows[0]
    if fault == "missing_row":
        rows.pop()
    elif fault == "duplicate_row":
        rows[-1] = row.copy()
    elif fault == "unexpected_map":
        row["map"] = "unrecognised"
    elif fault == "reference_frames":
        row["frames_with_reference"] = str(int(row["frames_with_reference"]) + 1)
    elif fault == "inconsistent_frames":
        rows[1]["frames_with_reference"] = "1"
    elif fault == "negative_support":
        row["current_error_own_frames"] = "-1"
    elif fault == "excess_support":
        row["current_error_own_frames"] = "999999"
    elif fault == "missing_value":
        row["current_error_own"] = ""
    elif fault == "nonfinite_value":
        row["current_error_own"] = "nan"
    else:
        row["bp_error_ref"] = "0.1"
    destination = tmp_path / "table.csv.gz"
    with gzip.open(destination, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        verify(destination, DATA / "report_v2.json")
