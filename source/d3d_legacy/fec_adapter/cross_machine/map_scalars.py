# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — q95 and normalised beta from a flux map and the measured profiles

"""Read ``q95`` and ``βN`` off a flux map, on any machine.

``q95`` is the field-line pitch integrated round the surface at 95 % of the
normalised flux, ``q = F / 2π · ∮ dl / (R |∇ψ|)``, with ``F = R·B_φ`` taken as
its vacuum value ``μ0 N I_TF / 2π`` — the plasma's own contribution to the
toroidal field is a few per cent and the inputs do not hold it.

``βN = β_t[%] · a · B0 / Ip[MA]`` needs the volume-averaged pressure. The
Thomson channels give the electron pressure where they sit; the map turns their
positions into normalised flux, which carries the profile to the whole plasma.
The total pressure is the electron pressure times :data:`PRESSURE_FACTOR`
(ions, impurities' dilution and fast ions together), a single number calibrated
on DIII-D training shots, where the label may be used, and applied unchanged
elsewhere.
"""
from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from skimage import measure

MU0: Final = 4.0e-7 * np.pi
ELECTRON_VOLT: Final = 1.602176634e-19

TF_TURNS: Final = {"DIII-D": 144.0, "MAST": 24.0}
"""Turns of the toroidal field circuit the shipped current flows through."""

PRESSURE_FACTOR: Final = 2.25
"""Total over electron pressure, calibrated on DIII-D training shots with their true maps: with
a factor of two the label was 1.123 times the map's value in the median (653 frames of 40 shots,
quartiles 0.94 to 1.30, correlation 0.946). The same run checked the safety factor: label over
map 0.996 in the median (730 frames, correlation 0.990)."""

Q_SHARE: Final = 0.95
PROFILE_BINS: Final = 10
MIN_CHANNELS_INSIDE: Final = 5
CORE_DEPTH: Final = 0.2
"""Normalised flux within which at least one usable Thomson sample must lie."""


def vacuum_f(machine: str, tf_current: float) -> float:
    """Return ``R·B_φ`` of the vacuum toroidal field, in tesla metres."""
    return float(MU0 * TF_TURNS[machine] * abs(tf_current) / (2.0 * np.pi))


def _at(field: NDArray[np.float64], r_grid: NDArray[np.float64], z_grid: NDArray[np.float64],
        r: NDArray[np.float64], z: NDArray[np.float64]) -> NDArray[np.float64]:
    rows = (z - z_grid[0]) / (z_grid[1] - z_grid[0])
    columns = (r - r_grid[0]) / (r_grid[1] - r_grid[0])
    values: NDArray[np.float64] = ndimage.map_coordinates(
        field, [rows, columns], order=1, mode="nearest"
    )
    return values


def surface(
    flux: NDArray[np.float64], r_grid: NDArray[np.float64], z_grid: NDArray[np.float64],
    level: float, axis: tuple[float, float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Return the closed contour of ``flux`` at ``level`` that goes round the axis, ``(R, Z)``."""
    best = None
    for contour in measure.find_contours(flux, level):
        if not np.allclose(contour[0], contour[-1]):
            continue
        r = r_grid[0] + contour[:, 1] * (r_grid[1] - r_grid[0])
        z = z_grid[0] + contour[:, 0] * (z_grid[1] - z_grid[0])
        if not (r.min() < axis[0] < r.max() and z.min() < axis[1] < z.max()):
            continue
        if best is None or r.size > best[0].size:
            best = (r, z)
    return best


def safety_factor(
    flux: NDArray[np.float64], r_grid: NDArray[np.float64], z_grid: NDArray[np.float64],
    axis: tuple[float, float], axis_flux: float, edge_flux: float, f_vacuum: float,
) -> float:
    """Return ``q`` on the surface at :data:`Q_SHARE` of the normalised flux, or NaN without one."""
    found = surface(flux, r_grid, z_grid, axis_flux + Q_SHARE * (edge_flux - axis_flux), axis)
    if found is None:
        return float("nan")
    r, z = found
    slopes = np.gradient(flux, z_grid, r_grid)
    slope_z, slope_r = np.asarray(slopes[0]), np.asarray(slopes[1])
    middle_r, middle_z = 0.5 * (r[1:] + r[:-1]), 0.5 * (z[1:] + z[:-1])
    steepness = np.hypot(_at(slope_r, r_grid, z_grid, middle_r, middle_z),
                         _at(slope_z, r_grid, z_grid, middle_r, middle_z))
    length = np.hypot(np.diff(r), np.diff(z))
    if np.any(steepness <= 0.0):
        return float("nan")
    return float(f_vacuum / (2.0 * np.pi) * np.sum(length / (middle_r * steepness)))


def normalised_beta(
    flux: NDArray[np.float64], r_grid: NDArray[np.float64], z_grid: NDArray[np.float64],
    axis: tuple[float, float], axis_flux: float, edge_flux: float,
    channel_r: NDArray[np.float64], channel_z: NDArray[np.float64],
    temperature_ev: NDArray[np.float64], density: NDArray[np.float64],
    plasma_current: float, f_vacuum: float,
) -> float:
    """Return ``βN`` from the map and one frame's Thomson samples, or NaN where undefined."""
    boundary = surface(flux, r_grid, z_grid, axis_flux + 0.999 * (edge_flux - axis_flux), axis)
    if boundary is None or plasma_current == 0.0 or f_vacuum == 0.0:
        return float("nan")
    span = edge_flux - axis_flux
    depth_at = (_at(flux, r_grid, z_grid, channel_r, channel_z) - axis_flux) / span
    pressure = density * temperature_ev * ELECTRON_VOLT
    usable = (np.isfinite(pressure) & (pressure > 0.0) & (depth_at >= 0.0) & (depth_at <= 1.0)
              & (channel_r > boundary[0].min()) & (channel_r < boundary[0].max()))
    if int(usable.sum()) < MIN_CHANNELS_INSIDE or float(depth_at[usable].min()) > CORE_DEPTH:
        return float("nan")  # without a sample near the axis the average would be the edge's
    edges = np.linspace(0.0, 1.0, PROFILE_BINS + 1)
    which = np.clip(np.digitize(depth_at[usable], edges) - 1, 0, PROFILE_BINS - 1)
    centres, means = [], []
    for index in range(PROFILE_BINS):
        member = which == index
        if member.any():
            centres.append(0.5 * (edges[index] + edges[index + 1]))
            means.append(float(np.mean(pressure[usable][member])))
    rr, _zz = np.meshgrid(r_grid, z_grid)
    depth = (flux - axis_flux) / span
    labels, _count = ndimage.label((depth >= 0.0) & (depth < 1.0))
    row = int(np.argmin(np.abs(z_grid - axis[1])))
    column = int(np.argmin(np.abs(r_grid - axis[0])))
    inside = labels == labels[row, column]
    if labels[row, column] == 0:
        return float("nan")
    profile = np.interp(depth[inside], [*centres, 1.0], [*means, 0.0])
    mean_pressure = PRESSURE_FACTOR * float(np.sum(profile * rr[inside]) / np.sum(rr[inside]))
    minor = 0.5 * float(boundary[0].max() - boundary[0].min())
    field = f_vacuum / (0.5 * float(boundary[0].max() + boundary[0].min()))
    beta_percent = 100.0 * 2.0 * MU0 * mean_pressure / field**2
    return float(beta_percent * minor * field / (abs(plasma_current) / 1.0e6))
