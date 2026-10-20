# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Equilibrium Challenge — D71b: corrected and added physical measures
"""Measures added to D71 on 20 October 2026; D71's physics.py is imported unchanged.

1. Critical points from roots of the interpolated gradient (critical.py), counted signed against the reference.
2. Current density on ONE region per frame, the scorer's eroded envelope outside the union of the reference LCFS
   and the LCFS of all nine stages (dilated by one cell), the same for every map and for the reference, over the
   reference's current density inside its own LCFS (one denominator per frame). An identical field gives an
   identical value whatever its own boundary (version 1 used different regions).
3. The polynomial GS-form residual with its parts: SSE, TSS, pixels, rank, RMS residual and source RMS, and a
   degenerate flag when the source is nearly constant (TSS below 1e-10 of the sum of squares).
4. A flux-surface GS test: on the surfaces psi_N = 0.2, 0.4, 0.6, 0.8 of the map, -Delta* phi is fitted as
   a R^2 + b (one a and b per surface); reported is the residual RMS over the source RMS on those surfaces. This
   tests the Grad-Shafranov structure without assuming a radial profile shape.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from skimage.measure import find_contours

import critical
import physics

F64 = NDArray[np.float64]
SURFACES = (.2, .4, .6, .8)
DEGENERATE = 1e-10


def gs_parts(phi: F64, gs: F64, R: F64, plasma: NDArray[np.bool_], axis_value: float,
             boundary_value: float, degree: int = physics.GS_DEGREE) -> dict[str, float]:
    out = {"gs_sse": np.nan, "gs_tss": np.nan, "gs_pixels": 0., "gs_rank": 0., "gs_rms_residual": np.nan,
           "gs_rms_source": np.nan, "gs_degenerate": np.nan}
    use = plasma & np.isfinite(gs)
    out["gs_pixels"] = float(use.sum())
    if use.sum() < physics.MIN_PIXELS or not axis_value > boundary_value:
        return out
    y = -gs[use]
    psin = (axis_value - phi[use]) / (axis_value - boundary_value)
    RR = np.broadcast_to(R[None, :], phi.shape)[use]
    powers = np.column_stack([psin ** k for k in range(degree + 1)])
    design = np.column_stack((RR[:, None] ** 2 * powers, powers))
    coefficients, _res, rank, _sv = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    sse, tss = float(np.sum(residual ** 2)), float(np.sum((y - y.mean()) ** 2))
    out.update(gs_sse=sse, gs_tss=tss, gs_rank=float(rank), gs_rms_residual=float(np.sqrt(sse / use.sum())),
               gs_rms_source=float(np.sqrt(np.mean(y ** 2))),
               gs_degenerate=float(float(np.sum(y ** 2)) == 0. or tss <= DEGENERATE * float(np.sum(y ** 2))))
    return out


def surface_gs(phi: F64, gs: F64, R: F64, Z: F64, plasma: NDArray[np.bool_], axis_value: float,
               boundary_value: float, bilinear) -> float:
    """RMS of the per-surface affine-in-R^2 fit residual over the RMS source, on SURFACES of the map."""
    if not axis_value > boundary_value or plasma.sum() < physics.MIN_PIXELS:
        return np.nan
    source = np.where(np.isfinite(gs), -gs, np.nan)
    num = den = 0.
    used = 0
    for level in SURFACES:
        value = axis_value - level * (axis_value - boundary_value)
        best = None
        for c in find_contours(np.where(plasma, phi, boundary_value - abs(axis_value - boundary_value)), value):
            if len(c) > 12 and np.allclose(c[0], c[-1]) and (best is None or len(c) > len(best)):
                best = c
        if best is None:
            continue
        r = np.interp(best[:, 1], np.arange(len(R)), R); z = np.interp(best[:, 0], np.arange(len(Z)), Z)
        y = bilinear(source, R, Z, r, z)
        keep = np.isfinite(y)
        if keep.sum() < 12:
            continue
        design = np.column_stack((r[keep] ** 2, np.ones(keep.sum())))
        coefficients, *_ = np.linalg.lstsq(design, y[keep], rcond=None)
        num += float(np.sum((y[keep] - design @ coefficients) ** 2)); den += float(np.sum(y[keep] ** 2))
        used += 1
    return float(np.sqrt(num / den)) if used and den > 0 else np.nan


def density(psi: F64, R: F64, Z: F64, axis_sign: float) -> F64:
    """|Delta* phi| / R on the grid (0 at the edge rows and columns), proportional to |mu0 J_phi|."""
    _, _, gs = physics.derivatives(axis_sign * np.asarray(psi, dtype=np.float64), R, Z)
    return np.where(np.isfinite(gs), np.abs(gs) / R[None, :], 0.)


def measure_b(psi: F64, R: F64, Z: F64, own: F64 | None, reference: F64 | None, axis_sign: float,
              envelope: NDArray[np.bool_], kit: dict[str, object], outside_region: NDArray[np.bool_],
              reference_inside: float) -> dict[str, float]:
    phi = axis_sign * np.asarray(psi, dtype=np.float64)
    _, _, gs = physics.derivatives(phi, R, Z)
    own_mask = physics.inside(own, R, Z, kit["points_in_polygon"])
    eroded = kit["erode"](envelope, physics.EDGE_CELLS)
    current = np.where(np.isfinite(gs), np.abs(gs) / R[None, :], 0.)
    points = critical.critical_points(phi, R, Z, eroded)
    n_max, n_min, n_saddle = critical.counts(points)
    out = {"outside_fixed_over_ref": (float(current[outside_region].sum() / reference_inside)
                                      if reference_inside > 0 else np.nan),
           "outside_pixels": float(outside_region.sum()),
           "cp_max": float(n_max), "cp_min": float(n_min), "cp_saddle": float(n_saddle),
           "gs_surface": np.nan}
    if own is not None:
        _, _, iz, ir, _ = kit["find_o_point"](phi, R, Z, envelope)
        axis_value = float(phi[iz, ir])
        boundary_value = float(np.mean(kit["bilinear"](phi, R, Z, own[:, 0], own[:, 1])))
        out.update(gs_parts(phi, gs, R, own_mask, axis_value, boundary_value))
        out["gs_surface"] = surface_gs(phi, gs, R, Z, own_mask, axis_value, boundary_value, kit["bilinear"])
    else:
        out.update(gs_parts(phi, gs, R, own_mask, np.nan, np.nan))
    return out
