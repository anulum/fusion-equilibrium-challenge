# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — scalars a flux map alone cannot carry

"""Derive the two submitted scalars from a solved equilibrium.

The Challenge asks for the flux map plus exactly the two scalars a flux map
cannot contain: ``q95`` needs the toroidal field function ``F(ψ)``, and
``βN`` needs the pressure. Both are available here because the equilibrium was
solved from free functions rather than regressed, so ``F`` and ``p`` are
integrals of quantities the solve already used.

The toroidal field is set by the machine, not by the plasma. DIII-D's field
circuit has 24 coils of 6 turns; the vacuum field on axis follows from
Ampère's law, and the measured circuit current is in the dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from skimage import measure

from .profiles import MU0

TF_TURNS: Final = 144.0
"""DIII-D toroidal field circuit turns: 24 coils of 6 turns each."""

MAJOR_RADIUS: Final = 1.6955
"""Metres; the radius the DIII-D vacuum toroidal field is quoted on."""

Q_SURFACE: Final = 0.95
"""Normalised flux of the surface ``q95`` is quoted on."""


@dataclass(frozen=True)
class FrameScalars:
    """The two submitted scalars for one frame."""

    q95: float
    beta_n: float


def vacuum_field_on_axis(tf_current_a: float) -> float:
    """Return the vacuum toroidal field at the reference radius, in tesla."""
    return float(MU0 * TF_TURNS * tf_current_a / (2.0 * np.pi * MAJOR_RADIUS))


def toroidal_field_function(
    psi_n_knots: NDArray[np.float64],
    ffprime: NDArray[np.float64],
    delta_psi: float,
    f_boundary: float,
) -> NDArray[np.float64]:
    """Integrate ``FF'`` into ``F(ψ_N)`` in tesla-metres.

    ``d(F²/2)/dψ = FF'`` integrated inwards from the boundary, where ``F`` is
    the vacuum value because the plasma current is enclosed by that surface.
    A negative ``F²`` would mean the integration has driven the toroidal field
    through zero, which no tokamak equilibrium does; it is clipped rather than
    propagated as a complex number, and the clip is visible in the returned
    profile as a flat inner region.
    """
    trapezia = np.diff(psi_n_knots) * 0.5 * (ffprime[1:] + ffprime[:-1])
    integral = np.concatenate([[0.0], np.cumsum(trapezia)])
    f_squared = f_boundary**2 + 2.0 * delta_psi * (integral - integral[-1])
    field: NDArray[np.float64] = np.sqrt(np.maximum(f_squared, 0.0))
    return field


def frame_scalars(
    psi: NDArray[np.float64],
    r_grid: NDArray[np.float64],
    z_grid: NDArray[np.float64],
    psi_axis: float,
    psi_boundary: float,
    psi_n_knots: NDArray[np.float64],
    pressure_knots: NDArray[np.float64],
    ffprime: NDArray[np.float64],
    plasma_current_a: float,
    tf_current_a: float,
) -> FrameScalars:
    """Return ``q95`` and ``βN`` for one solved frame.

    Both are undefined without a closed surface, so a frame whose ``ψ`` has no
    extractable contour at the requested normalised flux returns non-finite
    values rather than a plausible-looking number. The scorer treats a
    non-finite prediction as a miss on that frame, which is the honest
    outcome; a fabricated value would be scored as if it were a prediction.
    """
    delta = psi_boundary - psi_axis
    if not np.isfinite(delta) or delta == 0.0:
        return FrameScalars(float("nan"), float("nan"))

    b_toroidal = vacuum_field_on_axis(tf_current_a)
    f_profile = toroidal_field_function(
        psi_n_knots, ffprime, float(delta), b_toroidal * MAJOR_RADIUS
    )

    contour = _contour(psi, r_grid, z_grid, psi_axis + Q_SURFACE * delta)
    boundary = _contour(psi, r_grid, z_grid, psi_boundary)
    if contour is None or boundary is None:
        return FrameScalars(float("nan"), float("nan"))

    f_at_surface = float(np.interp(Q_SURFACE, psi_n_knots, f_profile))
    q95 = _safety_factor(contour, psi, r_grid, z_grid, f_at_surface)
    beta_n = _normalised_beta(
        psi, r_grid, psi_axis, delta, psi_n_knots, pressure_knots,
        boundary, plasma_current_a, b_toroidal,
    )
    return FrameScalars(q95, beta_n)


def _contour(
    psi: NDArray[np.float64],
    r_grid: NDArray[np.float64],
    z_grid: NDArray[np.float64],
    level: float,
) -> NDArray[np.float64] | None:
    """Return the longest closed contour of ``psi`` at ``level``, in metres.

    Several contours can sit at one flux level on a diverted equilibrium — the
    confined boundary and the private-flux region below the X-point among
    them. The longest closed one is the plasma boundary; an open contour runs
    into the computational wall and is not a flux surface of the confined
    plasma at all.
    """
    paths = measure.find_contours(np.asarray(psi, dtype=np.float64), level)
    best: NDArray[np.float64] | None = None
    best_length = 0.0
    for path in paths:
        if np.hypot(*(path[0] - path[-1])) > 1.0:
            continue
        r = np.interp(path[:, 1], np.arange(r_grid.size, dtype=np.float64), r_grid)
        z = np.interp(path[:, 0], np.arange(z_grid.size, dtype=np.float64), z_grid)
        points = np.column_stack([r, z])
        length = float(np.sum(np.hypot(*np.diff(points, axis=0).T)))
        if length > best_length:
            best, best_length = points, length
    return best


def _safety_factor(
    contour: NDArray[np.float64],
    psi: NDArray[np.float64],
    r_grid: NDArray[np.float64],
    z_grid: NDArray[np.float64],
    f_value: float,
) -> float:
    """Return ``q = (F/2π) ∮ dl / (R² |∇ψ|)`` on a flux surface."""
    # `np.gradient` returns one array per axis; unpacking it directly leaves the
    # element type open, so the pair is named and re-asserted as arrays.
    gradients = np.gradient(np.asarray(psi, dtype=np.float64), z_grid, r_grid)
    grad_z: NDArray[np.float64] = np.asarray(gradients[0], dtype=np.float64)
    grad_r: NDArray[np.float64] = np.asarray(gradients[1], dtype=np.float64)
    r_mid = 0.5 * (contour[1:, 0] + contour[:-1, 0])
    z_mid = 0.5 * (contour[1:, 1] + contour[:-1, 1])
    d_length = np.hypot(*np.diff(contour, axis=0).T)
    gr = _sample(grad_r, r_grid, z_grid, r_mid, z_mid)
    gz = _sample(grad_z, r_grid, z_grid, r_mid, z_mid)
    magnitude = np.hypot(gr, gz)
    usable = magnitude > 0.0
    if not usable.any():
        return float("nan")
    integrand = d_length[usable] / (r_mid[usable] ** 2 * magnitude[usable])
    return float(abs(f_value) * np.sum(integrand) / (2.0 * np.pi))


def _normalised_beta(
    psi: NDArray[np.float64],
    r_grid: NDArray[np.float64],
    psi_axis: float,
    delta: float,
    psi_n_knots: NDArray[np.float64],
    pressure_knots: NDArray[np.float64],
    boundary: NDArray[np.float64],
    plasma_current_a: float,
    b_toroidal: float,
) -> float:
    """Return ``βN`` in the conventional percent-metre-tesla-per-megaampere units."""
    if b_toroidal == 0.0 or plasma_current_a == 0.0:
        return float("nan")
    r_mesh = np.broadcast_to(r_grid[np.newaxis, :], psi.shape)
    psi_n = (np.asarray(psi, dtype=np.float64) - psi_axis) / delta
    inside = (psi_n >= 0.0) & (psi_n <= 1.0)
    if not inside.any():
        return float("nan")
    pressure = np.interp(np.clip(psi_n, 0.0, 1.0), psi_n_knots, pressure_knots)
    weight = 2.0 * np.pi * r_mesh * inside
    mean_pressure = float(np.sum(pressure * weight) / np.sum(weight))

    minor = 0.5 * float(boundary[:, 0].max() - boundary[:, 0].min())
    beta_toroidal = 2.0 * MU0 * mean_pressure / b_toroidal**2
    return float(
        100.0 * beta_toroidal * minor * abs(b_toroidal) / abs(plasma_current_a / 1.0e6)
    )


def _sample(
    field: NDArray[np.float64],
    r_grid: NDArray[np.float64],
    z_grid: NDArray[np.float64],
    r: NDArray[np.float64],
    z: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Bilinearly sample a grid field at scattered positions."""
    ri = np.clip(np.interp(r, r_grid, np.arange(r_grid.size, dtype=np.float64)), 0, r_grid.size - 1)
    zi = np.clip(np.interp(z, z_grid, np.arange(z_grid.size, dtype=np.float64)), 0, z_grid.size - 1)
    r0 = np.floor(ri).astype(int)
    z0 = np.floor(zi).astype(int)
    r1 = np.minimum(r0 + 1, r_grid.size - 1)
    z1 = np.minimum(z0 + 1, z_grid.size - 1)
    fr, fz = ri - r0, zi - z0
    sampled: NDArray[np.float64] = (
        field[z0, r0] * (1 - fr) * (1 - fz)
        + field[z0, r1] * fr * (1 - fz)
        + field[z1, r0] * (1 - fr) * fz
        + field[z1, r1] * fr * fz
    )
    return sampled


LEVEL_SCAN: Final = 96
"""Flux levels tried when searching outwards for the last closed surface."""


def outermost_closed_surface(
    psi: NDArray[np.float64],
    r_grid: NDArray[np.float64],
    z_grid: NDArray[np.float64],
    psi_axis: float,
    envelope: NDArray[np.bool_],
) -> tuple[float, NDArray[np.float64]] | None:
    """Find the last closed flux surface by scanning outwards from the axis.

    Returns the boundary flux and its contour, or ``None`` when no closed
    surface fits inside the envelope at all.

    This does not look for an X-point. A tokamak boundary is the outermost
    surface that closes on itself without leaving the vessel, and that
    definition covers a limited plasma resting on a wall as well as a diverted
    one resting on a saddle — whereas an X-point search answers only for the
    second kind, and answers wrongly rather than not at all when the first kind
    is what it is handed.

    Parameters
    ----------
    psi:
        Solved flux on the reconstruction grid.
    r_grid, z_grid:
        The grid, in metres.
    psi_axis:
        Flux at the magnetic axis; the scan starts just outside it.
    envelope:
        Vessel envelope mask on the same grid. A surface that crosses it has
        left the machine and is not a confined flux surface.

    Notes
    -----
    The scan runs outwards and keeps the last level that closes, rather than
    stopping at the first level that fails. A single level can fail spuriously
    where the contour tracer grazes a grid cell, and stopping there would
    truncate the plasma at an arbitrary radius — which is exactly the failure
    this function was written to replace.
    """
    values = np.asarray(psi, dtype=np.float64)
    edge = values[~np.asarray(envelope, dtype=bool)]
    if edge.size == 0:
        return None
    far = float(edge.max()) if psi_axis < float(np.median(edge)) else float(edge.min())
    found: tuple[float, NDArray[np.float64]] | None = None
    for level in np.linspace(psi_axis, far, LEVEL_SCAN + 1)[1:]:
        contour = _contour(values, r_grid, z_grid, float(level))
        if contour is None:
            continue
        if not _inside_envelope(contour, r_grid, z_grid, envelope):
            continue
        found = (float(level), contour)
    return found


def _inside_envelope(
    contour: NDArray[np.float64],
    r_grid: NDArray[np.float64],
    z_grid: NDArray[np.float64],
    envelope: NDArray[np.bool_],
) -> bool:
    """Return whether every point of a contour lies within the vessel envelope."""
    mask = np.asarray(envelope, dtype=bool)
    r_index = np.rint(
        np.interp(contour[:, 0], r_grid, np.arange(r_grid.size, dtype=np.float64))
    ).astype(int)
    z_index = np.rint(
        np.interp(contour[:, 1], z_grid, np.arange(z_grid.size, dtype=np.float64))
    ).astype(int)
    r_index = np.clip(r_index, 0, r_grid.size - 1)
    z_index = np.clip(z_index, 0, z_grid.size - 1)
    return bool(mask[z_index, r_index].all())
