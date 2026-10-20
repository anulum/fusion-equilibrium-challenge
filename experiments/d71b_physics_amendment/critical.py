# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Equilibrium Challenge — D71b: critical points of one scalar interpolant of a flux map
"""Find and classify the critical points of phi on a uniform (R, Z) grid.

phi is represented by ONE scalar interpolant, the interpolating bicubic spline (scipy RectBivariateSpline, s=0).
Its gradient and its Hessian come from the same function, so the Hessian is symmetric by construction. In each
sub-cell of a REFINE-times finer grid (the spline's gradient evaluated there) whose corners show a sign change of
both gradient components (or of one, with the other within NEAR_ZERO of zero at a corner), Newton's method on the spline gradient starts at the sub-cell centre; a root is accepted
when it lies in that sub-cell (a tolerance of 1e-6) and the gradient there is below 1e-8 of the sub-cell's gradient
scale. It is classified by the determinant
and trace of the spline Hessian at the root (|det| below 1e-10 of its scale: "degenerate"). Roots closer than half
a cell are merged. Cells within EDGE_CELLS of the grid's edge are not searched (the spline's end conditions can
make spurious extrema there), and a root whose Hessian is below SIGNIFICANCE of the largest Hessian on the grid is
discarded as numerical noise in a flat region.

Version 2 (20 October 2026): version 1 interpolated the two gradient components
independently and symmetrized their Jacobian, which can change the sign of the determinant.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RectBivariateSpline

F64 = NDArray[np.float64]
MERGE_CELLS = 0.5
NEWTON_STEPS = 30
EDGE_CELLS = 2
REFINE = 4
NEAR_ZERO = 0.1
SIGNIFICANCE = 1e-6


def _grid_checked(R: F64, Z: F64) -> tuple[float, float]:
    dR, dZ = np.diff(R), np.diff(Z)
    if not (np.all(dR > 0) and np.all(dZ > 0) and np.all(R > 0)):
        raise ValueError("R and Z must increase and R must be positive")
    if np.ptp(dR) > 1e-4 * abs(dR.mean()) or np.ptp(dZ) > 1e-4 * abs(dZ.mean()):  # float32-stored grids: ~1e-5
        raise ValueError("the grid must be uniform")
    return float(dR.mean()), float(dZ.mean())


def critical_points(phi: F64, R: F64, Z: F64, region: NDArray[np.bool_]) -> list[tuple[str, float, float]]:
    """Return [(kind, R, Z)] for every critical point found in cells whose four corners are in ``region``.

    kind is "max", "min", "saddle" or "degenerate" for phi.
    """
    dR, dZ = _grid_checked(R, Z)
    spline = RectBivariateSpline(Z, R, np.asarray(phi, dtype=np.float64), kx=3, ky=3, s=0)
    cells = region[:-1, :-1] & region[1:, :-1] & region[:-1, 1:] & region[1:, 1:]
    cells[:EDGE_CELLS, :] = cells[-EDGE_CELLS:, :] = False
    cells[:, :EDGE_CELLS] = cells[:, -EDGE_CELLS:] = False
    hessian_scale = max(float(np.max(np.abs(spline(Z, R, dx=2)))), float(np.max(np.abs(spline(Z, R, dy=2)))),
                        float(np.max(np.abs(spline(Z, R, dx=1, dy=1)))), 1e-300)
    # candidates: sub-cells of a REFINE-times finer grid where both spline gradient components change sign
    Rf = np.linspace(R[0], R[-1], (len(R) - 1) * REFINE + 1); Zf = np.linspace(Z[0], Z[-1], (len(Z) - 1) * REFINE + 1)
    gr, gz = spline(Zf, Rf, dy=1), spline(Zf, Rf, dx=1)

    def corners(g):
        return np.stack((g[:-1, :-1], g[1:, :-1], g[:-1, 1:], g[1:, 1:]))

    def changes(c):
        return (c.min(axis=0) <= 0) & (c.max(axis=0) >= 0)

    def near_zero(c):  # a component touching zero inside the sub-cell (a fold of its zero set) need not change sign
        a = np.abs(c)
        return a.min(axis=0) <= NEAR_ZERO * a.max(axis=0)

    allowed = np.repeat(np.repeat(cells, REFINE, axis=0), REFINE, axis=1)
    cr, cz = corners(gr), corners(gz)
    candidates = allowed & ((changes(cr) & changes(cz)) | (changes(cr) & near_zero(cz)) | (near_zero(cr) & changes(cz)))
    dRf, dZf = dR / REFINE, dZ / REFINE
    found: list[tuple[str, float, float]] = []
    for jz, jr in zip(*np.nonzero(candidates)):
        z0, r0 = float(Zf[jz]), float(Rf[jr])
        scale = max(float(np.max(np.abs(gr[jz:jz + 2, jr:jr + 2]))), float(np.max(np.abs(gz[jz:jz + 2, jr:jr + 2]))),
                    1e-300)
        z, r = z0 + .5 * dZf, r0 + .5 * dRf
        for _ in range(NEWTON_STEPS):
            g = np.array([spline.ev(z, r, dy=1), spline.ev(z, r, dx=1)], dtype=float)        # d/dR, d/dZ
            h = np.array([[spline.ev(z, r, dy=2), spline.ev(z, r, dx=1, dy=1)],
                          [spline.ev(z, r, dx=1, dy=1), spline.ev(z, r, dx=2)]], dtype=float)
            try:
                step = np.linalg.solve(h, g)
            except np.linalg.LinAlgError:
                break
            limit = max(abs(step[0]) / dRf, abs(step[1]) / dZf)
            if limit > .5:
                step *= .5 / limit
            r, z = r - step[0], z - step[1]
            if abs(r - r0) > 2 * dRf or abs(z - z0) > 2 * dZf:
                break
        g = np.array([spline.ev(z, r, dy=1), spline.ev(z, r, dx=1)], dtype=float)
        inside = (-1e-6 <= (r - r0) / dRf <= 1 + 1e-6) and (-1e-6 <= (z - z0) / dZf <= 1 + 1e-6)
        if not inside or float(np.max(np.abs(g))) > 1e-8 * scale:
            continue
        hrr, hrz, hzz = float(spline.ev(z, r, dy=2)), float(spline.ev(z, r, dx=1, dy=1)), float(spline.ev(z, r, dx=2))
        det = hrr * hzz - hrz * hrz
        size = max(hrr * hrr, hzz * hzz, hrz * hrz, 1e-300)
        if np.sqrt(size) < SIGNIFICANCE * hessian_scale:
            continue
        if abs(det) < 1e-10 * size:
            kind = "degenerate"
        elif det < 0:
            kind = "saddle"
        else:
            kind = "max" if hrr + hzz < 0 else "min"
        found.append((kind, float(r), float(z)))
    merged: list[tuple[str, float, float]] = []
    for kind, r, z in found:
        if any(k == kind and abs(r - r2) < MERGE_CELLS * dR and abs(z - z2) < MERGE_CELLS * dZ for k, r2, z2 in merged):
            continue
        merged.append((kind, r, z))
    return merged


def counts(points: list[tuple[str, float, float]]) -> tuple[int, int, int]:
    """Numbers of maxima, minima and saddles (degenerate points are reported separately by the caller)."""
    kinds = [p[0] for p in points]
    return kinds.count("max"), kinds.count("min"), kinds.count("saddle")
