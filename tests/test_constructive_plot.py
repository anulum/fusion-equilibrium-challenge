# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — recorded coarse figure contracts
"""Recompute original fields and exercise actual figure exports and refusals."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pytest

from tools.constructive_plot import BUNDLE, RECORD, load_comparison

ROOT = Path(__file__).resolve().parents[1]


def test_original_comparison_and_edge_order() -> None:
    """Preserve whole-grid gauge/span, reference-core support and every box-edge node."""
    comparison = load_comparison()
    record = json.loads(RECORD.read_text())
    with np.load(BUNDLE, allow_pickle=False) as data:
        delta = data["forward__psi_Wb_per_rad"] - data["inverse__psi_Wb_per_rad"]
        span = abs(record["coarse_reports"]["inverse"]["flux_span_Wb_per_rad"])
        expected = (delta - delta.mean()) / span
        np.testing.assert_array_equal(comparison.difference, expected)
        np.testing.assert_array_equal(comparison.core, data["inverse__core_mask"] > 0.5)
    report = comparison.report
    assert report["whole_domain_max_ratio"] == pytest.approx(0.009416717819584176, abs=1e-15)
    assert report["inverse_core_max_ratio"] == pytest.approx(0.001368121441242509, abs=1e-15)
    assert report["inverse_core_nodes"] == 740
    assert report["max_location_RZ_m"] == pytest.approx([2.8, -0.05625], abs=1e-14)
    assert report["original_whole_domain_limit"] == 0.005
    assert report["original_gate_passed"] is False and report["new_equilibrium_solve"] is False
    assert comparison.edge_distance.shape == comparison.edge_error.shape == (257,)
    assert np.all(np.diff(comparison.edge_distance) > 0)
    assert comparison.edge_distance[-1] == pytest.approx(12.6, abs=1e-13)
    assert comparison.edge_error[0] == comparison.edge_error[-1]
    np.testing.assert_array_equal(comparison.edge_error[:65], np.abs(expected[:, 0]))
    np.testing.assert_array_equal(comparison.edge_error[65:129], np.abs(expected[-1, 1:]))
    np.testing.assert_array_equal(comparison.edge_error[129:193], np.abs(expected[-2::-1, -1]))
    np.testing.assert_array_equal(comparison.edge_error[193:], np.abs(expected[0, -2::-1]))
    assert comparison.edge_error.max() == np.abs(expected).max()


def test_rendered_cli_is_repeatable(tmp_path: Path) -> None:
    """Render readable PNG/SVG exports twice from the original inputs and compare bytes."""
    destinations = [tmp_path / "first", tmp_path / "second"]
    for destination in destinations:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--branch",
                "--parallel-mode",
                "-m",
                "tools.constructive_plot",
                "--out",
                str(destination),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
            timeout=90,
        )
        assert result.stderr == ""
        assert json.loads(result.stdout) == json.loads((destination / "coarse_difference.json").read_text())
        pixels = mpimg.imread(destination / "coarse_difference.png")
        assert pixels.shape == (972, 2070, 4)
        assert np.unique(pixels.reshape(-1, 4), axis=0).shape[0] > 100
        svg = (destination / "coarse_difference.svg").read_text()
        assert "<svg" in svg and "Anulum Fortis" in svg and "whole-domain gate failed" in svg
        assert all(line == line.rstrip() for line in svg.splitlines())
    for name in ("coarse_difference.png", "coarse_difference.svg", "coarse_difference.json"):
        assert (
            hashlib.sha256((destinations[0] / name).read_bytes()).digest()
            == hashlib.sha256((destinations[1] / name).read_bytes()).digest()
        )


@pytest.mark.parametrize("fault", ["missing_bundle", "changed_bundle", "changed_record", "existing_output"])
def test_real_cli_refusals(tmp_path: Path, fault: str) -> None:
    """Refuse changed originals and an existing destination without traceback or overwrite."""
    output = tmp_path / "output"
    args = ["--out", str(output)]
    if fault == "existing_output":
        output.mkdir()
        (output / "keep.txt").write_text("owner content")
    else:
        path = tmp_path / "private-input"
        if fault == "changed_bundle":
            path.write_bytes(BUNDLE.read_bytes()[:-1])
        elif fault == "changed_record":
            path.write_text("{}")
        args += ["--record" if fault == "changed_record" else "--bundle", str(path)]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--branch",
            "--parallel-mode",
            "-m",
            "tools.constructive_plot",
            *args,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 2 and result.stderr == ""
    assert result.stdout == "Unable to render the recorded coarse comparison.\n"
    if fault == "existing_output":
        assert (output / "keep.txt").read_text() == "owner content"
        assert len(list(output.iterdir())) == 1
    else:
        assert not output.exists()
