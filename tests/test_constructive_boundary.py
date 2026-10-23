# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — independent boundary calibration checks
"""Exercise actual boundary operators against continuous-source quadrature."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools.constructive_boundary import (
    MU0,
    continuous_edge_flux,
    edge_coordinates,
    filament_green,
    prescribed_source,
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    65: (1.0379843954395266e-6, 0.008966860044823596, 0.00016790780250473383),
    129: (2.9057679487247883e-8, 0.006741825290069578, 1.7770224377612354e-6),
    257: (7.139802365265423e-10, 0.005364127783675042, 9.03443051334994e-9),
}


def test_source_total_current_and_compact_support() -> None:
    """Integrate the physical source to 1 MA without a mesh normalization."""
    u, w = np.polynomial.legendre.leggauss(32)
    u, w = (u + 1) / 2, w / 2
    theta, v = np.polynomial.legendre.leggauss(64)
    theta, v = np.pi * (theta + 1), np.pi * v
    r = 1.7 + 0.35 * np.sqrt(u[:, None]) * np.cos(theta)[None, :]
    z = 0.55 * np.sqrt(u[:, None]) * np.sin(theta)[None, :]
    total = np.sum(prescribed_source(r, z) * w[:, None] * v[None, :]) * 0.35 * 0.55 / 2
    assert total == pytest.approx(1e6, rel=2e-14)
    assert prescribed_source(np.asarray([1.7, 1.7]), np.asarray([0.0, 0.6])).tolist() == pytest.approx(
        [5e6 / (np.pi * 0.35 * 0.55), 0.0]
    )


def test_filament_kernel_matches_on_axis_field_limit() -> None:
    """Check Wb/rad normalization against the analytic circular-loop axial field."""
    a, z, radius = 1.7, 0.4, 0.01
    flux = filament_green(np.asarray(a), np.asarray(0.0), np.asarray(radius), np.asarray(z))
    axial_b = MU0 * a**2 / (2 * (a**2 + z**2) ** 1.5)
    assert float(flux) / (radius**2 / 2) == pytest.approx(axial_b, rel=3e-5)


@pytest.mark.parametrize("engine", ["reference", "legacy", "native"])
@pytest.mark.parametrize("size", [65, 129, 257])
@pytest.mark.integration
def test_actual_boundary_cli_against_independent_source(engine: str, size: int, tmp_path: Path) -> None:
    """Run each public operator on each grid and compare its actual edge flux."""
    python = sys.executable
    if engine == "legacy":
        reference = os.environ.get("CONSTRUCTIVE_REFERENCE_PYTHON")
        if reference is None:
            pytest.skip(
                "Requires the pinned legacy interpreter; executed in the constructive integration job."
            )
        python = reference
    output = tmp_path / "boundary.json"
    result = subprocess.run(
        [
            python,
            "-m",
            "coverage",
            "run",
            "--branch",
            "--parallel-mode",
            "--source=tools",
            "-m",
            "tools.constructive_boundary",
            "--engine",
            engine,
            "--size",
            str(size),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(output.read_text())
    assert data["schema"] == "constructive_boundary_v1"
    assert data["engine"] == engine and data["size"] == size
    points, shifted = edge_coordinates(size)
    reference_flux = continuous_edge_flux(points)
    scale = float(np.max(abs(reference_flux)))
    assert scale == pytest.approx(0.3827839291605356, rel=2e-13)
    values = np.asarray(data["psi_edge"])
    assert np.isfinite(values).all() and values.shape == (4 * (size - 1),)
    assert data["current_rectangle_A"] == pytest.approx(1e6, rel=1e-5)
    if engine == "reference":
        np.testing.assert_allclose(values, reference_flux, rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(
            data["psi_at_legacy_outward_offset"], continuous_edge_flux(shifted), rtol=1e-13, atol=1e-15
        )
        assert data["quadrature_probe_max_relative"] < 1e-11
    else:
        error = float(np.max(abs(values - reference_flux)) / scale)
        assert error == pytest.approx(EXPECTED[size][0 if engine == "native" else 1], rel=1e-4, abs=2e-13)
        if engine == "legacy":
            direct = np.asarray(data["psi_edge_public_direct_romberg"])
            direct_error = float(np.max(abs(direct - reference_flux)) / scale)
            assert direct_error == pytest.approx(EXPECTED[size][2], rel=1e-4, abs=2e-13)
