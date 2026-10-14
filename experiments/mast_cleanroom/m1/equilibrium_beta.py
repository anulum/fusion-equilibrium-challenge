# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — M1c: normalised beta from the pressure a flux map itself carries
"""βN from the map's own toroidal current, split into its pressure and poloidal-current parts.

Grad–Shafranov: ``Δ*ψ = −μ0 R J_φ`` and ``J_φ = R p′(ψ) + F F′(ψ) / (μ0 R)``. The two parts vary differently
across a flux surface (``R`` against ``1/R``), so a map's current density fixes both. The current is fitted,
inside the last closed surface, with the three terms the Thomson-isoflux solve itself uses —
``(R/R_ref)(1 − x)`` (pressure), ``(R_ref/R)(1 − x)`` and ``(R_ref/R) x (1 − x)`` (poloidal current) — by
non-negative least squares. With the flux oriented so the axis is its maximum and ``x`` the normalised
depth, the pressure term ``a0 (R/R_ref)(1 − x)`` means ``p′ = a0 (1 − x) / R_ref`` and

    p(x) = a0 (ψ_axis − ψ_edge) (1 − x)² / (2 R_ref).

No Thomson sample and no pressure factor enter. The averaging, minor radius and field are those of
``map_scalars.normalised_beta`` so the two estimates differ only in where the pressure comes from.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.optimize import nnls

from fec_adapter.cross_machine import map_scalars

MU0 = 4.0e-7 * np.pi


def current_density(flux: NDArray[np.float64], r_grid: NDArray[np.float64],
                    z_grid: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ``−Δ*ψ / (μ0 R)`` on the grid (A/m², sign as the flux's orientation gives it)."""
    rr = np.meshgrid(r_grid, z_grid)[0]
    d_z, d_r = np.gradient(flux, z_grid, r_grid)
    radial = rr * np.gradient(d_r / rr, r_grid, axis=1)
    vertical = np.gradient(d_z, z_grid, axis=0)
    out: NDArray[np.float64] = -(radial + vertical) / (MU0 * rr)
    return out


def equilibrium_beta(
    flux: NDArray[np.float64], r_grid: NDArray[np.float64], z_grid: NDArray[np.float64],
    axis: tuple[float, float], axis_flux: float, edge_flux: float,
    plasma_current: float, f_vacuum: float, reference_radius: float,
) -> tuple[float, float]:
    """Return ``(βN, pressure-term share of the fitted current)``; NaN where undefined."""
    boundary = map_scalars.surface(flux, r_grid, z_grid, axis_flux + 0.999 * (edge_flux - axis_flux), axis)
    if boundary is None or plasma_current == 0.0 or f_vacuum == 0.0 or edge_flux == axis_flux:
        return float("nan"), float("nan")
    orient = 1.0 if axis_flux > edge_flux else -1.0
    phi = orient * flux
    span = orient * (axis_flux - edge_flux)  # > 0
    depth = (orient * axis_flux - phi) / span
    labels, _count = ndimage.label((depth >= 0.0) & (depth < 1.0))
    row = int(np.argmin(np.abs(z_grid - axis[1])))
    column = int(np.argmin(np.abs(r_grid - axis[0])))
    if labels[row, column] == 0:
        return float("nan"), float("nan")
    region = labels == labels[row, column]
    # the finite-difference Laplacian mixes in the vacuum outside across the boundary's cells
    inside = ndimage.binary_erosion(region, iterations=1)
    if int(inside.sum()) < 20:
        return float("nan"), float("nan")
    rr = np.meshgrid(r_grid, z_grid)[0]
    current = current_density(phi, r_grid, z_grid)[inside]
    r, x = rr[inside], depth[inside]
    basis = np.stack([(r / reference_radius) * (1.0 - x), (reference_radius / r) * (1.0 - x),
                      (reference_radius / r) * x * (1.0 - x)], axis=1)
    amplitudes, _residual = nnls(basis, current)
    parts = basis * amplitudes
    share = float(parts[:, 0].sum() / max(parts.sum(), 1e-30))
    pressure = amplitudes[0] * span * (1.0 - depth[region]) ** 2 / (2.0 * reference_radius)
    mean_pressure = float(np.sum(pressure * rr[region]) / np.sum(rr[region]))
    minor = 0.5 * float(boundary[0].max() - boundary[0].min())
    field = f_vacuum / (0.5 * float(boundary[0].max() + boundary[0].min()))
    beta_percent = 100.0 * 2.0 * MU0 * mean_pressure / field**2
    return float(beta_percent * minor * field / (abs(plasma_current) / 1.0e6)), share
