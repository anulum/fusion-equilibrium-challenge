# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Equilibrium Challenge — D71 physical measures of one flux map, identical for reference and prediction
"""Physical measures of a 65 x 65 DIII-D flux map on the organisers' grid.

Every measure is applied with the same operators and masks to the reference map and to each predicted map.
phi = axis_sign * psi, so that the magnetic axis is a maximum of phi (the scorer's orientation). With psi in the
stored convention, Bp = |grad psi| / R and the loop integral of Bp around a flux surface is mu0 times the current
it encloses (the scorer's own li code uses this). No profile is assumed known: the Grad-Shafranov form test fits
p'(psi_N) and FF'(psi_N) per map.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]
GS_DEGREE = 3          # cubic p'(psi_N) and FF'(psi_N): 8 free coefficients per map
EDGE_CELLS = 2         # envelope erosion, as the scorer's O-point search
MIN_PIXELS = 30


def derivatives(phi: F64, R: F64, Z: F64) -> tuple[F64, F64, F64]:
    """Return dphi/dR, dphi/dZ and Delta* phi with second-order central differences.

    Delta* phi = d2phi/dR2 - (1/R) dphi/dR + d2phi/dZ2; edge rows and columns are NaN.
    """
    dR, dZ = float(np.mean(np.diff(R))), float(np.mean(np.diff(Z)))
    pr = np.full_like(phi, np.nan); pz = np.full_like(phi, np.nan); gs = np.full_like(phi, np.nan)
    c = phi[1:-1, 1:-1]
    prr = (phi[1:-1, 2:] - 2 * c + phi[1:-1, :-2]) / dR ** 2
    pzz = (phi[2:, 1:-1] - 2 * c + phi[:-2, 1:-1]) / dZ ** 2
    pr[1:-1, 1:-1] = (phi[1:-1, 2:] - phi[1:-1, :-2]) / (2 * dR)
    pz[1:-1, 1:-1] = (phi[2:, 1:-1] - phi[:-2, 1:-1]) / (2 * dZ)
    gs[1:-1, 1:-1] = prr - pr[1:-1, 1:-1] / R[None, 1:-1] + pzz
    return pr, pz, gs


def laplacian_star_o4(phi: F64, R: F64, Z: F64) -> F64:
    """Delta* phi with fourth-order central differences; two edge rows and columns are NaN (robustness variant)."""
    dR, dZ = float(np.mean(np.diff(R))), float(np.mean(np.diff(Z)))
    out = np.full_like(phi, np.nan)
    c = phi[2:-2, 2:-2]
    prr = (-phi[2:-2, 4:] + 16 * phi[2:-2, 3:-1] - 30 * c + 16 * phi[2:-2, 1:-3] - phi[2:-2, :-4]) / (12 * dR ** 2)
    pzz = (-phi[4:, 2:-2] + 16 * phi[3:-1, 2:-2] - 30 * c + 16 * phi[1:-3, 2:-2] - phi[:-4, 2:-2]) / (12 * dZ ** 2)
    pr = (-phi[2:-2, 4:] + 8 * phi[2:-2, 3:-1] - 8 * phi[2:-2, 1:-3] + phi[2:-2, :-4]) / (12 * dR)
    out[2:-2, 2:-2] = prr - pr / R[None, 2:-2] + pzz
    return out


def inside(polygon: F64 | None, R: F64, Z: F64, points_in_polygon) -> NDArray[np.bool_]:
    """Pixel centres inside a physical (R, Z) polygon, by the scorer's even-odd ray cast."""
    if polygon is None or len(polygon) < 8:
        return np.zeros((len(Z), len(R)), dtype=bool)
    RR, ZZ = np.meshgrid(R, Z)
    return points_in_polygon(RR.ravel(), ZZ.ravel(), polygon).reshape(RR.shape)


def dilate(mask: NDArray[np.bool_], cells: int = 1) -> NDArray[np.bool_]:
    """Binary 8-connected dilation without SciPy."""
    out = mask.copy()
    for _ in range(cells):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]; grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]; grown[:, :-1] |= out[:, 1:]
        grown[1:, 1:] |= out[:-1, :-1]; grown[:-1, :-1] |= out[1:, 1:]
        grown[1:, :-1] |= out[:-1, 1:]; grown[:-1, 1:] |= out[1:, :-1]
        out = grown
    return out


def loop_current(phi: F64, R: F64, Z: F64, polygon: F64 | None, bilinear) -> tuple[float, float]:
    """Return (loop integral of |grad phi|/R, loop integral of outward grad phi . n / R) around a polygon.

    The first is the scorer's mu0 Ip; the second is the signed flux of grad phi / R through the curve, which by the
    divergence theorem equals minus the area integral of (Delta* phi) / R inside it.
    """
    if polygon is None or len(polygon) < 8:
        return np.nan, np.nan
    dz, dr = np.gradient(phi, Z, R)
    C = np.asarray(polygon, float)
    Cn = np.roll(C, -1, axis=0)
    seg = Cn - C
    length = np.sqrt((seg ** 2).sum(axis=1))
    mid = .5 * (C + Cn)
    gr = bilinear(dr, R, Z, mid[:, 0], mid[:, 1]); gz = bilinear(dz, R, Z, mid[:, 0], mid[:, 1])
    orientation = np.sign(np.sum(C[:, 0] * Cn[:, 1] - Cn[:, 0] * C[:, 1]))  # +1 counter-clockwise in (R, Z)
    normal = orientation * np.column_stack((seg[:, 1], -seg[:, 0])) / np.maximum(length, 1e-300)[:, None]
    absolute = float(np.sum(np.sqrt(gr ** 2 + gz ** 2) / mid[:, 0] * length))
    signed = float(np.sum((gr * normal[:, 0] + gz * normal[:, 1]) / mid[:, 0] * length))
    return absolute, signed


def critical_points(phi: F64, envelope: NDArray[np.bool_], local_maxima) -> tuple[int, int, int]:
    """Count O-points (maxima of phi), minima of phi and saddle cells inside the eroded envelope.

    Maxima use the scorer's local_maxima; minima are maxima of -phi. A saddle cell is a 2 x 2 cell in which both
    first differences change sign across the cell and the Hessian determinant at its centre is negative.
    """
    n_max = int(local_maxima(phi, envelope, EDGE_CELLS).sum())
    n_min = int(local_maxima(-phi, envelope, EDGE_CELLS).sum())
    a, b, c, d = phi[:-1, :-1], phi[:-1, 1:], phi[1:, :-1], phi[1:, 1:]
    gr0, gr1 = b - a, d - c            # d/dR along the two rows of the cell
    gz0, gz1 = c - a, d - b            # d/dZ along the two columns
    mixed = gr1 - gr0                  # d2phi/dRdZ, scaled like prr and pzz
    sign_r = np.sign(gr0) != np.sign(gr1)
    sign_z = np.sign(gz0) != np.sign(gz1)
    prr = np.zeros_like(a); pzz = np.zeros_like(a)
    prr[:, 1:] = (phi[:-1, 2:] - phi[:-1, 1:-1]) - (phi[:-1, 1:-1] - phi[:-1, :-2])   # at the cell's first corner
    pzz[1:, :] = (phi[2:, :-1] - phi[1:-1, :-1]) - (phi[1:-1, :-1] - phi[:-2, :-1])
    det = prr * pzz - mixed ** 2
    cells = sign_r & sign_z & (det < 0)
    from_envelope = envelope[:-1, :-1] & envelope[:-1, 1:] & envelope[1:, :-1] & envelope[1:, 1:]
    eroded = from_envelope.copy()
    for _ in range(EDGE_CELLS):
        e = eroded.copy()
        e[1:, :] &= eroded[:-1, :]; e[:-1, :] &= eroded[1:, :]
        e[:, 1:] &= eroded[:, :-1]; e[:, :-1] &= eroded[:, 1:]
        eroded = e
    return n_max, n_min, int((cells & eroded).sum())


def gs_form_residual(phi: F64, gs: F64, R: F64, plasma: NDArray[np.bool_], axis_value: float,
                     boundary_value: float, degree: int = GS_DEGREE) -> float:
    """Fraction of the variance of -Delta* phi inside the plasma not of the form R^2 a(psi_N) + b(psi_N).

    a and b are polynomials of degree GS_DEGREE in psi_N = (phi_axis - phi) / (phi_axis - phi_boundary), fitted
    by least squares for this map. Grad-Shafranov gives -Delta* psi = mu0 R^2 p'(psi) + F F'(psi).
    """
    use = plasma & np.isfinite(gs)
    if use.sum() < MIN_PIXELS or not np.isfinite(axis_value - boundary_value) or axis_value == boundary_value:
        return np.nan
    y = -gs[use]
    psin = (axis_value - phi[use]) / (axis_value - boundary_value)
    RR = np.broadcast_to(R[None, :], phi.shape)[use]
    powers = np.column_stack([psin ** k for k in range(degree + 1)])
    design = np.column_stack((RR[:, None] ** 2 * powers, powers))
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    total = float(np.sum((y - y.mean()) ** 2))
    return float(np.sum(residual ** 2) / total) if total > 0 else np.nan


def measure(psi: F64, R: F64, Z: F64, own: F64 | None, reference: F64 | None, axis_sign: float,
            envelope: NDArray[np.bool_], kit: dict[str, object], reference_field: tuple[F64, F64] | None) -> dict[str, float]:
    """All D71 measures of one map; `own` is its official LCFS, `reference` the reference LCFS."""
    phi = axis_sign * np.asarray(psi, dtype=np.float64)
    pr, pz, gs = derivatives(phi, R, Z)
    own_mask = inside(own, R, Z, kit["points_in_polygon"])
    ref_mask = inside(reference, R, Z, kit["points_in_polygon"])
    dR, dZ = float(np.mean(np.diff(R))), float(np.mean(np.diff(Z)))
    current_density = np.where(np.isfinite(gs), -gs / R[None, :], 0.)          # proportional to mu0 J_phi
    loop_abs, _ = loop_current(phi, R, Z, own, kit["bilinear"])
    _, ref_signed = loop_current(phi, R, Z, reference, kit["bilinear"])
    area_own = float(current_density[own_mask].sum() * dR * dZ) if own_mask.any() else np.nan
    area_ref = float(current_density[ref_mask].sum() * dR * dZ) if ref_mask.any() else np.nan
    eroded = kit["erode"](envelope, EDGE_CELLS)
    outside = eroded & ~dilate(own_mask, 1) if own_mask.any() else np.zeros_like(eroded)
    in_abs = float(np.abs(current_density[own_mask]).sum()) if own_mask.any() else np.nan
    out_fraction = float(np.abs(current_density[outside]).sum() / in_abs) if own_mask.any() and in_abs > 0 else np.nan
    _, _, iz, ir, _ = kit["find_o_point"](phi, R, Z, envelope)
    axis_value = float(phi[iz, ir])
    boundary_value = float(np.mean(kit["bilinear"](phi, R, Z, own[:, 0], own[:, 1]))) if own is not None else np.nan
    gs_fraction = gs_form_residual(phi, gs, R, own_mask, axis_value, boundary_value)
    gs_variants = {"gs_fraction_deg2": gs_form_residual(phi, gs, R, own_mask, axis_value, boundary_value, 2),
                   "gs_fraction_deg5": gs_form_residual(phi, gs, R, own_mask, axis_value, boundary_value, 5),
                   "gs_fraction_o4": gs_form_residual(phi, laplacian_star_o4(phi, R, Z), R, own_mask, axis_value,
                                                      boundary_value)}
    n_max, n_min, n_saddle = critical_points(phi, envelope, kit["local_maxima"])
    result = {"lcfs_ok": float(own is not None), "loop_abs_own": loop_abs, "loop_signed_ref": ref_signed,
              "area_own": area_own, "area_ref": area_ref, "outside_fraction": out_fraction,
              "gs_fraction": gs_fraction, "n_max": n_max, "n_min": n_min, "n_saddle": n_saddle,
              "bp_error_ref": np.nan, "bp_error_envelope": np.nan, **gs_variants}
    if reference_field is not None:
        rr, rz = reference_field
        for name, region in (("bp_error_ref", ref_mask), ("bp_error_envelope", eroded)):
            use = region & np.isfinite(pr) & np.isfinite(rr)
            if use.sum() >= MIN_PIXELS:
                diff = ((pr - rr) ** 2 + (pz - rz) ** 2)[use] / R[None, :].repeat(len(Z), 0)[use] ** 2
                norm = (rr ** 2 + rz ** 2)[use] / R[None, :].repeat(len(Z), 0)[use] ** 2
                result[name] = float(np.sqrt(diff.sum() / norm.sum()))
    return result


def field_of(psi: F64, R: F64, Z: F64, axis_sign: float) -> tuple[F64, F64]:
    """dphi/dR and dphi/dZ of the reference map, for the poloidal-field error of predictions."""
    pr, pz, _ = derivatives(axis_sign * np.asarray(psi, dtype=np.float64), R, Z)
    return pr, pz
