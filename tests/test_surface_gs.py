# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — actual surface support on analytic grid fields
"""Exercise real contour extraction, affine fits and unavailable-support cases."""

from __future__ import annotations

import json

import numpy as np
import pytest
from numpy.typing import NDArray

from tools.surface_gs import SURFACES, measure


@pytest.fixture
def grid() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Use the actual 65 by 65 DIII-D coordinate range of the published diagnostic."""
    radius = np.linspace(0.84, 2.54, 65)
    height = np.linspace(-1.6, 1.6, 65)
    rr, zz = np.meshgrid(radius, height)
    return radius, height, rr, zz


def test_solovev(grid: tuple[NDArray[np.float64], ...]) -> None:
    """Recover all four surfaces of an exact analytic equilibrium with small interpolation error."""
    radius, height, rr, zz = grid
    phi = -((rr**2 - 1.7**2) ** 2) / 8 - 0.9 * zz**2
    gs = -(rr**2) - 1.8
    result = measure(phi, gs, radius, height, phi >= -0.3, float(phi.max()), -0.3)
    assert result["schema"] == "surface-gs-support.v1"
    assert result["full_support"] is True
    assert result["used_levels"] == result["requested_levels"] == list(SURFACES)
    assert result["reason"] == "available"
    assert result["ratio"] is not None and result["ratio"] < 1e-4
    for level in result["levels"]:
        assert level["used"] is True and level["reason"] == "used"
        assert level["contour_points"] == level["finite_points"] > 12
        assert level["rank"] == 2 and level["rank_deficient"] is False
        assert level["condition"] is not None and level["condition"] > 1
        assert len(level["singular_values"]) == 2
        assert level["zero_source"] is False and level["source_energy"] > 0


def test_analytic_deformation(grid: tuple[NDArray[np.float64], ...]) -> None:
    """Detect the GS-form change from an explicit analytic non-equilibrium perturbation."""
    radius, height, rr, zz = grid
    epsilon = 0.4
    phi = -((rr**2 - 1.7**2) ** 2) / 8 - 0.9 * zz**2 + epsilon * (rr - 1.7) ** 2 * zz**2
    gs = -(rr**2) - 1.8 + 2 * epsilon * (1.7 * zz**2 / rr + (rr - 1.7) ** 2)
    result = measure(phi, gs, radius, height, phi >= -0.3, float(phi.max()), -0.3)
    assert result["ratio"] is not None and result["ratio"] > 1e-3
    assert result["full_support"] is True


def test_partial_support(grid: tuple[NDArray[np.float64], ...]) -> None:
    """Keep the legacy ratio while exposing levels lost around a grid-scale central spike."""
    radius, height, rr, zz = grid
    phi = 5 * np.exp(-((rr - 1.7) ** 2 + zz**2) / 0.1)
    phi[np.unravel_index(np.argmax(phi), phi.shape)] += 5
    gz, gr = np.gradient(phi, height, radius)
    gs = np.gradient(gr, radius, axis=1) - gr / radius[None, :] + np.gradient(gz, height, axis=0)
    result = measure(phi, gs, radius, height, phi > 0.1, float(phi.max()), 0.1)
    assert 0 < len(result["used_levels"]) < 4
    assert result["full_support"] is False and result["ratio"] is not None
    assert any(level["reason"] == "no_closed_contour" for level in result["levels"])


@pytest.mark.parametrize("source_kind", ["zero", "missing", "partly_missing"])
def test_source_applicability(grid: tuple[NDArray[np.float64], ...], source_kind: str) -> None:
    """Report zero and unavailable sources separately from contour availability."""
    radius, height, rr, zz = grid
    phi = -((rr**2 - 1.7**2) ** 2) / 8 - 0.9 * zz**2
    gs = -(rr**2) - 1.8
    if source_kind == "zero":
        gs[:] = 0
    elif source_kind == "missing":
        gs[:] = np.nan
    else:
        gs[:, :30] = np.nan
    result = measure(phi, gs, radius, height, phi >= -0.3, float(phi.max()), -0.3)
    if source_kind == "zero":
        assert result["full_support"] is True and result["ratio"] is None
        assert result["reason"] == "zero_source"
        assert all(level["zero_source"] for level in result["levels"])
    elif source_kind == "missing":
        assert result["used_levels"] == [] and result["ratio"] is None
        assert result["reason"] == "no_usable_surface"
        assert all(level["reason"] == "insufficient_finite_samples" for level in result["levels"])
    else:
        assert result["ratio"] is not None
        assert any(level["finite_points"] < level["contour_points"] for level in result["levels"])


@pytest.mark.parametrize("case", ["flat_axis", "nonfinite_axis", "small_mask", "open_contours"])
def test_unavailable_surfaces(grid: tuple[NDArray[np.float64], ...], case: str) -> None:
    """Explain invalid drop, too few plasma pixels and genuinely open contours."""
    radius, height, rr, zz = grid
    phi = -((rr**2 - 1.7**2) ** 2) / 8 - 0.9 * zz**2
    gs = -(rr**2) - 1.8
    mask = phi >= -0.3
    axis, boundary = float(phi.max()), -0.3
    if case == "flat_axis":
        axis = boundary
    elif case == "nonfinite_axis":
        axis = float("nan")
    elif case == "small_mask":
        mask[:] = False
    else:
        phi = rr.copy()
        mask[:] = True
        axis, boundary = float(phi.max()), float(phi.min())
    result = measure(phi, gs, radius, height, mask, axis, boundary)
    assert result["ratio"] is None and result["used_levels"] == [] and result["full_support"] is False
    expected = (
        "invalid_axis_boundary"
        if "axis" in case
        else "insufficient_plasma"
        if case == "small_mask"
        else "no_usable_surface"
    )
    assert result["reason"] == expected


@pytest.mark.parametrize(
    "fault",
    [
        "radius_dim",
        "height_dim",
        "small_grid",
        "nan_radius",
        "nan_height",
        "radius_order",
        "height_order",
        "negative_radius",
        "phi_shape",
        "source_shape",
        "mask_shape",
    ],
)
def test_grid_refusal(grid: tuple[NDArray[np.float64], ...], fault: str) -> None:
    """Refuse inconsistent actual analytic grids before invoking contour extraction."""
    radius, height, rr, zz = grid
    phi = -((rr**2 - 1.7**2) ** 2) / 8 - 0.9 * zz**2
    gs = -(rr**2) - 1.8
    mask = phi >= -0.3
    if fault == "radius_dim":
        radius = rr
    elif fault == "height_dim":
        height = zz
    elif fault == "small_grid":
        radius = radius[:2]
    elif fault == "nan_radius":
        radius[0] = np.nan
    elif fault == "nan_height":
        height[0] = np.nan
    elif fault == "radius_order":
        radius = radius[::-1]
    elif fault == "height_order":
        height = height[::-1]
    elif fault == "negative_radius":
        radius -= 10
    elif fault == "phi_shape":
        phi = phi[:-1]
    elif fault == "source_shape":
        gs = gs[:-1]
    else:
        mask = mask[:-1]
    with pytest.raises(ValueError, match="Inconsistent"):
        measure(phi, gs, radius, height, mask, float(phi.max()), -0.3)


def test_condition_does_not_filter(grid: tuple[NDArray[np.float64], ...]) -> None:
    """Expose numerical rank loss under a large radial offset without changing fit inclusion."""
    radius, height, rr, zz = grid
    phi = -((rr**2 - 1.7**2) ** 2) / 8 - 0.9 * zz**2
    shifted = radius + 1e8
    gradient = -0.5 * rr * (rr**2 - 1.7**2)
    gs = -1.5 * rr**2 + 0.5 * 1.7**2 - gradient / shifted[None, :] - 1.8
    result = measure(phi, gs, shifted, height, phi >= -0.3, float(phi.max()), -0.3)
    assert result["full_support"] is True and result["ratio"] is not None
    for level in result["levels"]:
        assert level["used"] is True and level["rank_deficient"] is True
        assert level["rank"] == 1 and level["condition"] is not None and level["condition"] > 1e16


def test_longest_closed_contour(grid: tuple[NDArray[np.float64], ...]) -> None:
    """Select the larger actual contour when two separated analytic lobes cross a level."""
    radius, height, rr, zz = grid
    large = 0.6 - (rr - 1.65) ** 2 - (zz - 0.5) ** 2
    small = 0.3 - (rr - 1.65) ** 2 - (zz + 0.9) ** 2
    combined = np.maximum(large, small)
    gs = -4 + 2 * (rr - 1.65) / rr
    expected = measure(large, gs, radius, height, large > 0.05, float(large.max()), 0.05)
    actual = measure(combined, gs, radius, height, combined > 0.05, float(combined.max()), 0.05)
    assert expected["full_support"] is True
    assert actual == expected


def test_unrepresentable_condition(grid: tuple[NDArray[np.float64], ...]) -> None:
    """Return null metadata when the affine basis underflows, retaining the original fit."""
    radius, height, rr, zz = grid
    phi = (-((rr**2 - 1.7**2) ** 2) / 8 - 0.9 * zz**2) * 1e-300
    gs = -(rr**2) * 1e20 - 1.8e-300
    result = measure(phi, gs, radius * 1e-160, height, phi >= -0.3e-300, float(phi.max()), -0.3e-300)
    assert result["full_support"] is True and result["ratio"] is not None
    assert all(level["condition"] is None and level["rank_deficient"] for level in result["levels"])
    json.dumps(result, allow_nan=False)
