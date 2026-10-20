# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Equilibrium Challenge — D72: replace the inside of a finished map by a Grad-Shafranov solution
"""Project the plasma of a finished map onto a Grad-Shafranov equilibrium with the same boundary values.

phi = axis_sign * psi (axis a maximum). Inside the map's own LCFS the map is replaced by the solution of
    Delta* phi = -(R^2 a(psi_N) + b(psi_N)),
with a and b cubic polynomials fitted to the map's own -Delta* phi inside the LCFS (as D71 measures it), rescaled at
every Picard step so that the area integral of the current density keeps the map's own value. Every pixel outside
the LCFS keeps the finished map's value and is the Dirichlet boundary of the solve. No parameter is fitted to a
reference map; the only inputs are the finished map and its official LCFS.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import splu

F64 = NDArray[np.float64]
DEGREE = 3
ITERATIONS = 40
TOLERANCE = 1e-9      # largest change of phi between Picard steps, relative to the axis-to-boundary flux drop
MIN_PIXELS = 30


def _fit(phi: F64, gs: F64, R: F64, plasma: NDArray[np.bool_], axis_value: float,
         boundary_value: float) -> F64:
    use = plasma & np.isfinite(gs)
    psin = (axis_value - phi[use]) / (axis_value - boundary_value)
    RR = np.broadcast_to(R[None, :], phi.shape)[use]
    powers = np.column_stack([psin ** k for k in range(DEGREE + 1)])
    design = np.column_stack((RR[:, None] ** 2 * powers, powers))
    coefficients, *_ = np.linalg.lstsq(design, -gs[use], rcond=None)
    return coefficients


def _source(coefficients: F64, phi: F64, R: F64, axis_value: float, boundary_value: float) -> F64:
    psin = np.clip((axis_value - phi) / (axis_value - boundary_value), 0., 1.)
    powers = np.stack([psin ** k for k in range(DEGREE + 1)], axis=-1)
    a, b = powers @ coefficients[:DEGREE + 1], powers @ coefficients[DEGREE + 1:]
    return R[None, :] ** 2 * a + b


def projected(psi: F64, R: F64, Z: F64, plasma: NDArray[np.bool_], axis_sign: float, axis_index: tuple[int, int],
              boundary_value_phi: float, gs: F64) -> tuple[F64, dict[str, float]]:
    """Return the projected map and diagnostics; the input map unchanged if the solve is not possible.

    ``plasma`` marks the pixel centres inside the map's own LCFS, ``gs`` is Delta* phi of the map (second-order),
    ``axis_index`` the map's O-point and ``boundary_value_phi`` the mean of phi on its LCFS.
    """
    phi = axis_sign * np.asarray(psi, dtype=np.float64)
    unknown = plasma.copy()
    unknown[0, :] = unknown[-1, :] = unknown[:, 0] = unknown[:, -1] = False
    info = {"pixels": float(unknown.sum()), "iterations": 0., "converged": 0., "used": 0.}
    axis_value = float(phi[axis_index])
    if unknown.sum() < MIN_PIXELS or not axis_value > boundary_value_phi:
        return np.asarray(psi, dtype=np.float64), info
    coefficients = _fit(phi, gs, R, unknown, axis_value, boundary_value_phi)
    dR, dZ = float(np.mean(np.diff(R))), float(np.mean(np.diff(Z)))
    nz, nr = phi.shape
    index = -np.ones(phi.shape, dtype=np.int64)
    zi, ri = np.nonzero(unknown)
    index[zi, ri] = np.arange(len(zi))
    rc = R[ri]
    rows, cols, vals = [], [], []
    rhs_known = np.zeros(len(zi))
    neighbours = ((0, 1, 1 / dR ** 2 - 1 / (2 * dR * rc)), (0, -1, 1 / dR ** 2 + 1 / (2 * dR * rc)),
                  (1, 0, np.full(len(zi), 1 / dZ ** 2)), (-1, 0, np.full(len(zi), 1 / dZ ** 2)))
    rows.append(np.arange(len(zi))); cols.append(np.arange(len(zi)))
    vals.append(np.full(len(zi), -2 / dR ** 2 - 2 / dZ ** 2))
    for dz, dr, weight in neighbours:
        z2, r2 = zi + dz, ri + dr
        inner = index[z2, r2] >= 0
        rows.append(np.arange(len(zi))[inner]); cols.append(index[z2, r2][inner]); vals.append(weight[inner])
        rhs_known[~inner] -= weight[~inner] * phi[z2[~inner], r2[~inner]]
    matrix = sparse.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                               shape=(len(zi), len(zi)))
    solve = splu(matrix).solve
    target = float(np.sum(-gs[unknown] / rc) * dR * dZ)          # the map's own plasma current (area form)
    current = phi.copy()
    drop = axis_value - boundary_value_phi
    for step in range(ITERATIONS):
        axis_now = float(current[unknown].max())
        source = _source(coefficients, current, R, axis_now, boundary_value_phi)[unknown]
        total = float(np.sum(source / rc) * dR * dZ)
        if not np.isfinite(total) or total == 0.:
            return np.asarray(psi, dtype=np.float64), info
        source *= target / total
        solution = solve(-source + rhs_known)
        change = float(np.max(np.abs(solution - current[unknown]))) / drop
        current[unknown] = solution
        info["iterations"] = float(step + 1)
        if change < TOLERANCE:
            info["converged"] = 1.
            break
    if not np.all(np.isfinite(current)):
        return np.asarray(psi, dtype=np.float64), info
    info["used"] = 1.
    return axis_sign * current, info
