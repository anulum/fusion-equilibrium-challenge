# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — discrete field and GS diagnostic controls
"""Check known discrete operators, deformation and honest unavailable results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from tools.audit_map_input import MapData, Region, load_map, load_region
from tools.field_diagnostics import MU0, measure


def _load(tmp_path: Path, kind: str = "solovev") -> tuple[MapData, Region]:
    radius, height = np.linspace(0.84, 2.54, 65), np.linspace(-1.6, 1.6, 65)
    rr, zz = np.meshgrid(radius, height)
    phi = -((rr**2 - 1.7**2) ** 2) / 8 - 0.9 * zz**2
    mask = phi >= -0.3
    boundary = -0.3
    options: dict[str, NDArray[np.generic] | float] = {"plasma_mask": mask, "boundary_flux": boundary}
    if kind == "deformed":
        phi += 0.4 * (rr - 1.7) ** 2 * zz**2
    elif kind == "partial":
        phi = 5 * np.exp(-((rr - 1.7) ** 2 + zz**2) / 0.1)
        phi[np.unravel_index(np.argmax(phi), phi.shape)] += 5
        options = {"plasma_mask": phi > 0.1, "boundary_flux": 0.1}
    elif kind == "no_mask":
        del options["plasma_mask"]
    elif kind == "no_boundary":
        del options["boundary_flux"]
    elif kind == "invalid_drop":
        options["boundary_flux"] = 1.0
    elif kind == "overflow_drop":
        options.update(axis_flux=1e308, boundary_flux=-1e308)
    elif kind == "empty":
        options["plasma_mask"] = np.zeros_like(mask)
    elif kind == "tiny":
        small = np.zeros_like(mask)
        small[32, 32] = True
        options["plasma_mask"] = small
    elif kind in {"zero", "constant_source"}:
        radius, height = np.arange(1.0, 10.0), np.arange(-4.0, 5.0)
        rr, zz = np.meshgrid(radius, height)
        phi = rr**2 if kind == "zero" else -(zz**2)
        options = {"plasma_mask": np.ones_like(phi, dtype=bool), "boundary_flux": -100.0}
    elif kind == "rank_loss":
        radius = radius + 1e8
    elif kind == "open":
        phi = rr.copy()
        options = {"plasma_mask": np.ones_like(mask), "boundary_flux": 0.0}
    elif kind == "overflow":
        phi *= 1e200
    path = tmp_path / (kind + ".npz")
    np.savez(path, R=radius, Z=height, psi=phi, allow_pickle=False, **options)
    frame = load_map(path, "m", "Wb/rad", 1)
    return frame, load_region(None, frame)


def _part(result: dict[str, object], key: str) -> dict[str, object]:
    return cast(dict[str, object], result[key])


def test_known_discrete_fields_and_current(tmp_path: Path) -> None:
    """The quartic field has a known h²/4 source offset under these stencils."""
    frame, region = _load(tmp_path)
    rr, zz = np.meshgrid(frame.radius, frame.height)
    h = float(np.diff(frame.radius).mean())
    source = rr[1:-1, 1:-1] ** 2 + 1.8 - h * h / 4
    pr = -0.5 * rr[1:-1, 1:-1] * (rr[1:-1, 1:-1] ** 2 - 1.7**2 + h * h)
    pz = -1.8 * zz[1:-1, 1:-1]
    expected_bp = np.sqrt(np.mean((pr**2 + pz**2) / rr[1:-1, 1:-1] ** 2))
    result = measure(frame, region, frame)
    field = _part(result, "field_on_comparison_region")
    assert field["requested_nodes"] == 4225 and field["used_nodes"] == 3969
    assert field["source_rms_T"] == pytest.approx(np.sqrt(np.mean(source**2)), rel=1e-12)
    assert field["bp_rms_T"] == pytest.approx(expected_bp, rel=1e-12)
    current = _part(result, "plasma_current")
    assert frame.plasma is not None
    expected_current = (
        np.sum((source / (MU0 * rr[1:-1, 1:-1]))[frame.plasma[1:-1, 1:-1]]) * h * np.diff(frame.height).mean()
    )
    assert current["net_A"] == pytest.approx(expected_current, rel=1e-12)
    assert current["absolute_A"] == pytest.approx(expected_current, rel=1e-12)
    measured = measure(frame, region, measured_current_a=-float(expected_current))
    comparison = _part(measured, "measured_current_comparison")
    assert comparison["relative_magnitude_error"] == pytest.approx(0, abs=1e-12)
    polynomial = _part(result, "polynomial_gs")
    assert polynomial["available"] is True and polynomial["rank"] == 8
    assert cast(float, polynomial["ratio"]) < 1e-12
    assert (
        cast(float, polynomial["condition"]) > 1 and len(cast(list[float], polynomial["coefficients"])) == 8
    )
    assert _part(result, "reference_comparison")["bp_relative_l2"] == 0
    assert _part(_part(result, "surface_gs"), "support")["full_support"] is True
    json.dumps(result, allow_nan=False)


def test_deformation_and_fixed_field_comparison(tmp_path: Path) -> None:
    """A real perturbation changes fitted residuals and reference field error."""
    frame, region = _load(tmp_path)
    deformed, other_region = _load(tmp_path, "deformed")
    original, changed = measure(frame, region), measure(deformed, other_region, frame)
    assert region.identity == other_region.identity
    assert cast(float, _part(changed, "polynomial_gs")["ratio"]) > 1e-3
    assert cast(float, _part(_part(changed, "surface_gs"), "support")["ratio"]) > 1e-3
    assert cast(float, _part(changed, "reference_comparison")["bp_relative_l2"]) > 0.01
    assert cast(float, _part(original, "polynomial_gs")["ratio"]) < 1e-12


@pytest.mark.parametrize(
    "kind,reason",
    [
        ("no_mask", "missing_plasma_mask"),
        ("no_boundary", "missing_boundary_flux"),
        ("invalid_drop", "invalid_axis_boundary"),
        ("overflow_drop", "invalid_axis_boundary"),
        ("empty", "insufficient_plasma_nodes"),
        ("tiny", "insufficient_plasma_nodes"),
    ],
)
def test_domain_applicability(tmp_path: Path, kind: str, reason: str) -> None:
    """A missing domain never becomes a zero GS residual or a guessed boundary."""
    frame, region = _load(tmp_path, kind)
    result = measure(frame, region, measured_current_a=1.0)
    assert _part(result, "polynomial_gs")["reason"] == reason
    assert _part(result, "polynomial_gs")["ratio"] is None
    assert _part(result, "field_on_comparison_region")["reason"] == "available"
    if kind == "no_mask":
        assert _part(result, "measured_current_comparison")["reason"] == "missing_plasma_mask"
    elif kind == "empty":
        assert _part(result, "measured_current_comparison")["reason"] == "empty_plasma_current_support"


def test_zero_source_rank_loss_and_constant_source(tmp_path: Path) -> None:
    """Separate exact vacuum, nonzero constant source and a nonidentifiable fit."""
    vacuum, region = _load(tmp_path, "zero")
    result = measure(vacuum, region, vacuum, 0.0)
    assert _part(result, "polynomial_gs")["reason"] == "zero_source"
    assert _part(result, "polynomial_gs")["variance_degenerate"] is True
    assert _part(result, "measured_current_comparison")["reason"] == "zero_measured_current"
    ratio = _part(_part(result, "reference_comparison"), "region_absolute_current_over_reference_plasma")
    assert ratio["reason"] == "zero_reference_current" and ratio["value"] is None
    constant, region = _load(tmp_path, "constant_source")
    result = measure(constant, region)
    assert _part(result, "polynomial_gs")["variance_degenerate"] is True
    assert cast(float, _part(result, "polynomial_gs")["ratio"]) < 1e-12
    rank, region = _load(tmp_path, "rank_loss")
    result = measure(rank, region)
    assert _part(result, "polynomial_gs")["reason"] == "rank_deficient"
    assert _part(result, "polynomial_gs")["available"] is True


@pytest.mark.parametrize("kind", ["partial", "open"])
def test_surface_support(tmp_path: Path, kind: str) -> None:
    """Finite partial support and genuinely open contours have different status."""
    frame, region = _load(tmp_path, kind)
    support = _part(_part(measure(frame, region), "surface_gs"), "support")
    assert support["full_support"] is False
    if kind == "partial":
        assert 0 < len(cast(list[float], support["used_levels"])) < 4
    else:
        assert support["reason"] == "no_usable_surface"


@pytest.mark.parametrize(
    "refkind,expected",
    [
        ("no_mask", "missing_reference_plasma_mask"),
        ("empty", "empty_reference_current_support"),
        ("solovev", "empty_comparison_region"),
    ],
)
def test_reference_current_applicability(tmp_path: Path, refkind: str, expected: str) -> None:
    """A reference current denominator needs its own actual plasma support."""
    frame, region = _load(tmp_path)
    reference, _ = _load(tmp_path, refkind)
    if refkind == "solovev":
        region = Region(np.zeros_like(region.mask), region.policy, region.identity, None)
    result = measure(frame, region, reference)
    ratio = _part(_part(result, "reference_comparison"), "region_absolute_current_over_reference_plasma")
    assert ratio["value"] is None and ratio["reason"] == expected
    if refkind == "solovev":
        assert _part(result, "field_on_comparison_region")["bp_rms_T"] is None


def test_flat_reference_and_changed_grid(tmp_path: Path) -> None:
    """Zero reference field and mismatched grids cannot produce relative errors."""
    frame, region = _load(tmp_path)
    path = tmp_path / "flat.npz"
    np.savez(
        path,
        R=frame.radius,
        Z=frame.height,
        psi=np.zeros_like(frame.phi),
        plasma_mask=np.ones_like(frame.phi, dtype=bool),
        boundary_flux=-1.0,
    )
    flat = load_map(path, "m", "Wb/rad", 1)
    result = measure(frame, region, flat)
    assert _part(result, "reference_comparison")["reason"] == "zero_reference_field_energy"
    polynomial = _part(measure(flat, region), "polynomial_gs")
    assert polynomial["condition"] is None and polynomial["ratio"] is None
    np.savez(path, R=frame.radius + 0.1, Z=frame.height, psi=frame.phi)
    with pytest.raises(ValueError, match="grids must match"):
        measure(frame, region, load_map(path, "m", "Wb/rad", 1))


@pytest.mark.parametrize("current", [True, float("nan"), float("inf")])
def test_invalid_current(tmp_path: Path, current: float) -> None:
    """A current declaration is a finite measurement, never a boolean flag."""
    frame, region = _load(tmp_path)
    with pytest.raises(ValueError, match="finite amperes"):
        measure(frame, region, measured_current_a=current)


def test_invalid_region_and_numerical_range(tmp_path: Path) -> None:
    """Refuse shape mismatches and unrepresentable second-derivative energies."""
    frame, region = _load(tmp_path)
    bad = Region(np.ones((3, 3), dtype=bool), region.policy, region.identity, None)
    with pytest.raises(ValueError, match="Region mask shape"):
        measure(frame, bad)
    large, region = _load(tmp_path, "overflow")
    with pytest.raises(FloatingPointError):
        measure(large, region)
