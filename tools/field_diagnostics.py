# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — external-map field and GS-form diagnostics
"""Measure specified discrete forms without certifying physical equilibrium."""

from __future__ import annotations

import math
from typing import TypedDict, cast

import numpy as np

from tools.audit_map_input import F64, MapData, Mask, Region
from tools.surface_gs import MIN_PIXELS
from tools.surface_gs import measure as surface_measure

MU0 = 4 * math.pi * 1e-7
DEGREE = 3


class Current(TypedDict):
    """Axis-oriented node-area quadrature, with its actual spatial support."""

    requested_nodes: int
    used_nodes: int
    net_A: float | None
    absolute_A: float | None
    density_rms_A_m2: float | None
    reason: str


def _derivatives(frame: MapData) -> tuple[F64, F64, F64]:
    phi, radius, height = frame.phi, frame.radius, frame.height
    dr, dz = float(np.diff(radius).mean()), float(np.diff(height).mean())
    pr, pz, gs = (np.full_like(phi, np.nan) for _ in range(3))
    centre = phi[1:-1, 1:-1]
    pr[1:-1, 1:-1] = (phi[1:-1, 2:] - phi[1:-1, :-2]) / (2 * dr)
    pz[1:-1, 1:-1] = (phi[2:, 1:-1] - phi[:-2, 1:-1]) / (2 * dz)
    gs[1:-1, 1:-1] = (
        (phi[1:-1, 2:] - 2 * centre + phi[1:-1, :-2]) / dr**2
        - pr[1:-1, 1:-1] / radius[None, 1:-1]
        + (phi[2:, 1:-1] - 2 * centre + phi[:-2, 1:-1]) / dz**2
    )
    return pr, pz, gs


def _current(frame: MapData, gs: F64, mask: Mask) -> Current:
    use = mask & np.isfinite(gs)
    result: Current = {
        "requested_nodes": int(mask.sum()),
        "used_nodes": int(use.sum()),
        "net_A": None,
        "absolute_A": None,
        "density_rms_A_m2": None,
        "reason": "empty_source_support",
    }
    if use.any():
        radius = np.broadcast_to(frame.radius[None, :], gs.shape)[use]
        density = -gs[use] / (MU0 * radius)
        area = float(np.diff(frame.radius).mean() * np.diff(frame.height).mean())
        result.update(
            {
                "net_A": float(density.sum() * area),
                "absolute_A": float(np.abs(density).sum() * area),
                "density_rms_A_m2": float(np.sqrt(np.mean(density**2))),
                "reason": "available",
            }
        )
    return result


def _domain_reason(frame: MapData) -> str:
    if frame.plasma is None:
        return "missing_plasma_mask"
    if frame.boundary is None:
        return "missing_boundary_flux"
    if not math.isfinite(frame.axis - frame.boundary) or frame.axis <= frame.boundary:
        return "invalid_axis_boundary"
    return ""


def _polynomial(frame: MapData, gs: F64) -> dict[str, object]:
    reason = _domain_reason(frame)
    use = np.zeros(gs.shape, dtype=np.bool_) if frame.plasma is None else frame.plasma & np.isfinite(gs)
    n = int(use.sum())
    result: dict[str, object] = {
        "available": False,
        "reason": reason or "insufficient_plasma_nodes",
        "requested_nodes": 0 if frame.plasma is None else int(frame.plasma.sum()),
        "used_nodes": n,
        "degree": DEGREE,
        "required_rank": 8,
        "ratio": None,
        "rank": None,
        "condition": None,
        "singular_values": [],
        "coefficients": [],
        "sse_T2": None,
        "tss_T2": None,
        "rms_residual_T": None,
        "rms_source_T": None,
        "variance_degenerate": None,
        "sse_over_tss": None,
    }
    if reason or n < MIN_PIXELS:
        return result
    boundary = cast(float, frame.boundary)
    y = -gs[use]
    psin = (frame.axis - frame.phi[use]) / (frame.axis - boundary)
    radius = np.broadcast_to(frame.radius[None, :], gs.shape)[use]
    powers = np.column_stack([psin**k for k in range(DEGREE + 1)])
    design = np.column_stack((radius[:, None] ** 2 * powers, powers))
    coefficients, _residuals, rank, singular = np.linalg.lstsq(design, y, rcond=None)
    sse = float(np.sum((y - design @ coefficients) ** 2))
    tss, energy = float(np.sum((y - y.mean()) ** 2)), float(np.sum(y**2))
    condition = float(singular[0]) / float(singular[-1]) if singular[-1] > 0 else math.inf
    result.update(
        available=energy > 0,
        reason="zero_source" if energy == 0 else "rank_deficient" if rank < 8 else "available",
        ratio=float(np.sqrt(sse / energy)) if energy > 0 else None,
        rank=int(rank),
        condition=condition if math.isfinite(condition) else None,
        singular_values=singular.tolist(),
        coefficients=coefficients.tolist(),
        sse_T2=sse,
        tss_T2=tss,
        rms_residual_T=float(np.sqrt(sse / n)),
        rms_source_T=float(np.sqrt(energy / n)),
        variance_degenerate=energy == 0 or tss <= 1e-10 * energy,
        sse_over_tss=sse / tss if tss > 1e-10 * energy else None,
    )
    return result


def _surface(frame: MapData, gs: F64) -> dict[str, object]:
    reason = _domain_reason(frame)
    if reason:
        return {"available": False, "reason": reason, "support": None}
    plasma, boundary = cast(Mask, frame.plasma), cast(float, frame.boundary)
    support = surface_measure(frame.phi, gs, frame.radius, frame.height, plasma, frame.axis, boundary)
    return {"available": support["ratio"] is not None, "reason": support["reason"], "support": support}


def _field(frame: MapData, derivatives: tuple[F64, F64, F64], mask: Mask) -> dict[str, object]:
    pr, pz, gs = derivatives
    use = mask & np.isfinite(gs)
    n = int(use.sum())
    radius = np.broadcast_to(frame.radius[None, :], gs.shape)[use]
    return {
        "requested_nodes": int(mask.sum()),
        "used_nodes": n,
        "reason": "available" if n else "empty_source_support",
        "bp_rms_T": float(np.sqrt(np.mean((pr[use] ** 2 + pz[use] ** 2) / radius**2))) if n else None,
        "source_rms_T": float(np.sqrt(np.mean(gs[use] ** 2))) if n else None,
        "current": _current(frame, gs, mask),
    }


def _reference(
    frame: MapData, reference: MapData, region: Region, derivatives: tuple[F64, F64, F64]
) -> dict[str, object]:
    if not np.array_equal(frame.radius, reference.radius) or not np.array_equal(
        frame.height, reference.height
    ):
        raise ValueError("Reference and candidate grids must match exactly")
    pr, pz, gs = derivatives
    rr, rz, rg = _derivatives(reference)
    use = region.mask & np.isfinite(gs) & np.isfinite(rg)
    radius = np.broadcast_to(frame.radius[None, :], gs.shape)[use]
    energy = float(np.sum((rr[use] ** 2 + rz[use] ** 2) / radius**2))
    error = float(np.sum(((pr[use] - rr[use]) ** 2 + (pz[use] - rz[use]) ** 2) / radius**2))
    reason = (
        "empty_comparison_region"
        if not use.any()
        else "zero_reference_field_energy"
        if energy == 0
        else "available"
    )
    current = _current(frame, gs, region.mask)
    ref_current = None if reference.plasma is None else _current(reference, rg, reference.plasma)
    denominator = None if ref_current is None else ref_current["absolute_A"]
    numerator = current["absolute_A"]
    ratio = None if denominator is None or denominator == 0 or numerator is None else numerator / denominator
    return {
        "available": reason == "available",
        "reason": reason,
        "requested_nodes": int(region.mask.sum()),
        "used_nodes": int(use.sum()),
        "bp_relative_l2": float(np.sqrt(error / energy)) if reason == "available" else None,
        "bp_error_energy_T2": error,
        "bp_reference_energy_T2": energy,
        "reference_field": _field(reference, (rr, rz, rg), region.mask),
        "reference_plasma_current": ref_current,
        "region_absolute_current_over_reference_plasma": {
            "value": ratio,
            "reason": "missing_reference_plasma_mask"
            if ref_current is None
            else "empty_reference_current_support"
            if denominator is None
            else "zero_reference_current"
            if denominator == 0
            else "empty_comparison_region"
            if numerator is None
            else "available",
        },
    }


def _measurement(current: Current | None, measured: float | None) -> dict[str, object]:
    estimate = None if current is None else current["net_A"]
    reason = (
        "missing_current_measurement"
        if measured is None
        else "missing_plasma_mask"
        if current is None
        else "empty_plasma_current_support"
        if estimate is None
        else "zero_measured_current"
        if measured == 0
        else "available"
    )
    error = None if measured is None or estimate is None else abs(abs(estimate) - abs(measured))
    return {
        "available": reason == "available",
        "reason": reason,
        "measured_A": measured,
        "absolute_magnitude_error_A": error,
        "relative_magnitude_error": error / abs(measured)
        if reason == "available" and measured is not None and error is not None
        else None,
    }


def measure(
    frame: MapData, region: Region, reference: MapData | None = None, measured_current_a: float | None = None
) -> dict[str, object]:
    """Audit one validated map with fixed spatial and explicit own-map domains.

    Parameters
    ----------
    frame : MapData
        Read-only SI input produced by audit_map_input.load_map.
    region : Region
        Independently identified comparison mask on the same grid.
    reference : MapData or None
        Optional co-registered SI reference; its orientation is declared separately.
    measured_current_a : float or None
        Optional plasma current in amperes; comparisons use magnitudes.

    Returns
    -------
    dict of str to object
        Field/current, cubic GS fit, surface support and optional comparisons.
        Values are discrete diagnostics, not percentages outside equilibrium.

    Raises
    ------
    ValueError
        The region/reference grid or current declaration is inconsistent.
    FloatingPointError
        The calculation exceeds representable numerical range.
    """
    if region.mask.shape != frame.phi.shape:
        raise ValueError("Region mask shape differs")
    if measured_current_a is not None and (
        isinstance(measured_current_a, bool) or not math.isfinite(measured_current_a)
    ):
        raise ValueError("Measured current must be finite amperes")
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        derivatives = _derivatives(frame)
        plasma_current = None if frame.plasma is None else _current(frame, derivatives[2], frame.plasma)
        return {
            "field_on_comparison_region": _field(frame, derivatives, region.mask),
            "plasma_current": plasma_current,
            "measured_current_comparison": _measurement(plasma_current, measured_current_a),
            "polynomial_gs": _polynomial(frame, derivatives[2]),
            "surface_gs": _surface(frame, derivatives[2]),
            "reference_comparison": {"available": False, "reason": "missing_reference"}
            if reference is None
            else _reference(frame, reference, region, derivatives),
        }
