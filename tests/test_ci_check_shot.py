# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — submitted-shot verifier consumer tests
"""Compare the submitted prediction and altered copies through the public API and CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools.ci_check_shot import DIGESTS, REFERENCE, compare, main

ROOT = Path(__file__).resolve().parents[1]


def _altered(tmp_path: Path, fault: str, key: str = "shot_0000_psirz") -> Path:
    with np.load(REFERENCE, allow_pickle=False) as original:
        arrays = {name: original[name] for name in original.files}
    selected = arrays[key].copy()
    if fault == "dtype":
        arrays[key] = selected.astype(np.float64)
    elif fault == "shape":
        arrays[key] = selected[:-1]
    elif fault == "missing":
        del arrays[key]
    elif fault == "nan":
        selected.flat[0] = np.nan
        arrays[key] = selected
    else:
        scale = float(np.ptp(selected)) if key.endswith("psirz") else 1.0
        selected.flat[0] += scale * (1e-6 if fault == "within" else 1e-3)
        arrays[key] = selected
    path = tmp_path / "changed.npz"
    np.savez(path, **arrays)
    return path


def test_submitted_prediction() -> None:
    """Accept all original arrays with exact hashes and zero measured difference."""
    report = compare(REFERENCE)
    assert set(report) == set(DIGESTS)
    assert all(item["bit_identical"] and item["max_abs_difference"] == 0 for item in report.values())


@pytest.mark.parametrize("key", list(DIGESTS))
@pytest.mark.parametrize("fault", ["within", "outside"])
def test_numerical_tolerances(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], key: str, fault: str
) -> None:
    """Keep the published flux/scalar limits and distinguish tolerance from identity."""
    path = _altered(tmp_path, fault, key)
    report = compare(path)
    assert report[key]["bit_identical"] is False
    assert report[key]["max_abs_difference"] > 0
    expected = 0 if fault == "within" else 1
    assert main([str(path)]) == expected
    output = capsys.readouterr().out
    assert ("within tolerance" if expected == 0 else "OUTSIDE tolerance") in output
    assert report[key]["limit"] == (1e-5 if key.endswith("psirz") else 1e-4)


@pytest.mark.parametrize("fault", ["dtype", "shape", "missing", "nan"])
def test_structural_rejection(tmp_path: Path, capsys: pytest.CaptureFixture[str], fault: str) -> None:
    """Reject damaged prediction structure without interpreter text reaching the CLI."""
    path = _altered(tmp_path, fault)
    with pytest.raises((ValueError, KeyError)):
        compare(path)
    assert main([str(path)]) == 2
    assert capsys.readouterr().out == "Prediction archive or submitted reference is invalid.\n"


def test_reference_integrity(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Reject reference replacement even when prediction and reference agree with each other."""
    changed = _altered(tmp_path, "within")
    with pytest.raises(ValueError, match="digest"):
        compare(changed, changed)
    assert main([str(changed), "--reference", str(changed)]) == 2
    assert "invalid" in capsys.readouterr().out


@pytest.mark.parametrize("fault", ["dtype", "shape"])
def test_reference_schema(tmp_path: Path, fault: str) -> None:
    """Reject NPZ header changes that leave the reference array's raw bytes unchanged."""
    with np.load(REFERENCE, allow_pickle=False) as original:
        arrays = {name: original[name] for name in original.files}
    key = "shot_0000_psirz"
    if fault == "dtype":
        arrays[key] = arrays[key].view(np.int32)
    else:
        arrays[key] = arrays[key].reshape(240, 65 * 65)
    changed = tmp_path / "reference.npz"
    np.savez(changed, **arrays)
    with pytest.raises(ValueError, match="shape or dtype"):
        compare(REFERENCE, changed)


@pytest.mark.parametrize("fault", ["absent", "truncated", "text", "single_array"])
def test_unreadable_archive(tmp_path: Path, capsys: pytest.CaptureFixture[str], fault: str) -> None:
    """Refuse missing, truncated and non-archive files with a fixed failure message."""
    path = tmp_path / "damaged.npz"
    if fault == "truncated":
        path.write_bytes(REFERENCE.read_bytes()[:50])
    elif fault == "text":
        path.write_text("invalid archive\n")
    elif fault == "single_array":
        with np.load(REFERENCE, allow_pickle=False) as reference, path.open("wb") as stream:
            np.save(stream, reference["shot_0000_psirz"])
    assert main([str(path)]) == 2
    assert capsys.readouterr().out == "Prediction archive or submitted reference is invalid.\n"


def test_real_cli() -> None:
    """Consume the committed reference through the same file-form command used in CI."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--parallel-mode",
            str(ROOT / "tools/ci_check_shot.py"),
            str(REFERENCE),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.stdout.endswith("bit-identical\n")
    assert all(value["bit_identical"] for value in json.loads(result.stdout.rsplit("\n", 2)[0]).values())


def test_optimised_python(tmp_path: Path) -> None:
    """Keep dtype validation effective when Python's assertion statements are disabled."""
    changed = _altered(tmp_path, "dtype")
    result = subprocess.run(
        [sys.executable, "-O", str(ROOT / "tools/ci_check_shot.py"), str(changed)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert result.stdout == "Prediction archive or submitted reference is invalid.\n"
    assert result.stderr == ""
