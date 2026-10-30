# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — scored-field replay process tests
"""Run the actual scored-map replay and verify its process and refusal contracts."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.scored_map_bundle import load
from tools.scored_map_replay import REFUSAL, replay

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "experiments/d72_scored_map_replay/illustration_v1"
EXPECTED = EXAMPLE / "expected_statistics.json.gz"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cli(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--branch",
            "--parallel-mode",
            "-m",
            "tools.scored_map_replay",
            *arguments,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )


def _arguments(output: Path) -> list[str]:
    return [
        "--bundle",
        str(EXAMPLE),
        "--manifest-sha256",
        _hash(EXAMPLE / "manifest.json"),
        "--expected-statistics",
        str(EXPECTED),
        "--expected-statistics-sha256",
        _hash(EXPECTED),
        "--out",
        str(output),
    ]


@pytest.mark.parametrize("workers", [1, 2])
def test_actual_replay_cli_and_census_verifier(tmp_path: Path, workers: int) -> None:
    """The complete command writes measured reports and reproducible surface statistics."""
    output = tmp_path / "measured"
    result = _cli(_arguments(output) + ["--workers", str(workers)])
    assert result.returncode == 0 and result.stderr == ""
    receipt = json.loads(result.stdout)
    assert receipt == json.loads((output / "complete.json").read_text())
    assert receipt["shots"] == receipt["frames"] == 3
    assert receipt["scope"] == "illustration" and receipt["original_cohort_report_compared"] is False
    assert receipt["illustration_reference_statistics_compared"] is True
    assert receipt["maximum_scaled_difference"] <= 1
    assert receipt["physical_force_balance_certified"] is False
    for name, checksum in receipt["output_sha256"].items():
        assert checksum == _hash(output / name)
    for name, checksum in receipt["source_sha256"].items():
        assert checksum == _hash(ROOT / "tools" / name)
    surface = output / "surface_per_shot.json.gz"
    verified = subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--branch",
            "--parallel-mode",
            "-m",
            "tools.scored_surface_summary",
            "--table",
            str(surface),
            "--table-sha256",
            _hash(surface),
            "--report",
            str(output / "surface_report.json"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    assert verified.returncode == 0 and verified.stderr == ""
    assert json.loads(verified.stdout)["maximum_absolute_difference"] == 0
    again = _cli(_arguments(output))
    assert again.returncode == 2 and "Traceback" not in again.stderr
    assert json.loads(again.stdout) == {"status": "refused", "message": REFUSAL}


@pytest.mark.parametrize(
    "case",
    [
        "missing-argument",
        "unknown-argument",
        "manifest-pin",
        "statistics-pin",
        "missing-file",
        "workers-zero",
        "workers-excess",
    ],
)
def test_actual_cli_fixed_refusals(tmp_path: Path, case: str) -> None:
    """Malformed real command inputs return exit2 without paths or exception disclosure."""
    arguments = _arguments(tmp_path / "out")
    if case == "missing-argument":
        arguments = []
    elif case == "unknown-argument":
        arguments += ["--private-path", str(tmp_path)]
    elif case == "manifest-pin":
        arguments[3] = "0" * 64
    elif case == "statistics-pin":
        arguments[7] = "0" * 64
    elif case.startswith("workers"):
        arguments += ["--workers", "0" if case == "workers-zero" else "13"]
    else:
        arguments[5] = str(tmp_path / "private-nonexistent")
    result = _cli(arguments)
    assert result.returncode == 2 and result.stderr == ""
    assert json.loads(result.stdout) == {"status": "refused", "message": REFUSAL}
    assert str(tmp_path) not in result.stdout


def test_changed_reference_statistics_reject_a_real_measurement(tmp_path: Path) -> None:
    """An altered expected estimate is detected after the actual map measurement."""
    import gzip

    with gzip.open(EXPECTED, "rt") as stream:
        data = json.load(stream)
    data["shots"][0]["statistics"]["all"]["gs_surface"]["means"][0] += 1
    path = tmp_path / "changed.json.gz"
    with gzip.open(path, "wt") as stream:
        json.dump(data, stream)
    bundle = load(EXAMPLE, _hash(EXAMPLE / "manifest.json"))
    with pytest.raises(ValueError, match="differs"):
        replay(bundle, path, _hash(path), tmp_path / "changed-output")
