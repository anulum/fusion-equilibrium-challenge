# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — the magnetic axis of a predicted map, found and moved

"""Find the magnetic axis of predicted flux maps and move it where the network says it is.

The network answers the axis position twice: through the flux map, whose
minimum is the axis, and through two outputs that were fitted to the axis
position directly. The two do not agree, and the direct answer is the better
one. Around the axis the flux is flat, so a map error far too small to show in
the flux term still slides the minimum by centimetres; an output fitted to the
position has no such lever against it. Measured on the twenty reserved shots
against the axis of the true maps:

==========================  =========  =========  ==========
axis taken from             R_axis R²  Z_axis R²  flux R²
==========================  =========  =========  ==========
the predicted map           0.9463     0.8302     0.999685
the network's axis outputs  0.9641     0.8901     —
the map after the move      0.9640     0.8878     0.999686
==========================  =========  =========  ==========

The last row is the organisers' scorer on the archive the build wrote; with it
the Consistency term of that window went from 0.9327 to 0.9435 and the
composite from 0.9831 to 0.9853, every other term unchanged to four places.

The move is a shift of the map inside a smooth bump around the axis, written
as the Taylor series of ``ψ(x - Δ)`` to second order:
``ψ ← ψ - bump · (Δ · ∇ψ - ½ Δᵀ H Δ)``. Near its minimum the map is a
paraboloid, for which that series is exact, so the minimum slides by ``Δ`` and
keeps its depth, and everything outside the bump stays as it was. The bump
follows the axis, so the step is repeated a few times; the width of the bump
and the number of rounds were tried from 0.10 m to 0.30 m and from one to
three, and no choice changed either R² by more than 0.008, which twenty shots
cannot resolve. The values below are not tuned.

Half precision is not the cause of the axis error: the same maps written in
single and in half precision give Z_axis R² of 0.8302 and 0.8293.

The axis is found as the scorer finds it — the deepest interior local minimum
inside the vessel envelope, refined by a parabola through its neighbours — so
that the position moved is the position that will be read.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

BUMP_WIDTH = 0.20
"""Metres; radial e-folding length of the region that moves with the axis."""

BUMP_ELONGATION = 1.6
"""The bump is taller than wide by this factor, as the surfaces around the axis are."""

ROUNDS = 3
"""Repetitions of the step; the bump follows the axis, so one is not quite enough."""

SHIFT_LIMIT = 0.20
"""Metres; the largest move of one round, so a wild output cannot tear the map."""

EDGE_CELLS = 2
"""Cells by which the envelope is shrunk before a minimum counts, as the scorer does."""


def _shrunk(mask: NDArray[np.bool_], cells: int) -> NDArray[np.bool_]:
    """Return ``mask`` eroded by ``cells`` in four-connectivity, grid border excluded."""
    inside = mask.copy()
    inside[0, :] = inside[-1, :] = inside[:, 0] = inside[:, -1] = False
    for _ in range(cells):
        eroded = inside.copy()
        eroded[1:, :] &= inside[:-1, :]
        eroded[:-1, :] &= inside[1:, :]
        eroded[:, 1:] &= inside[:, :-1]
        eroded[:, :-1] &= inside[:, 1:]
        inside = eroded
    return inside


def _offset(before: float, centre: float, after: float) -> float:
    """Return the vertex of the parabola through three samples, in cells, within one cell."""
    curvature = before - 2.0 * centre + after
    if curvature == 0.0:
        return 0.0
    return float(np.clip(0.5 * (before - after) / curvature, -1.0, 1.0))


def axis_positions(
    maps: NDArray[np.floating],
    r_grid: NDArray[np.float64],
    z_grid: NDArray[np.float64],
    envelope: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Return ``(T, 2)`` axis positions ``(R, Z)`` of maps whose axis is a minimum.

    The deepest cell that is no higher than its eight neighbours is taken; where
    a map has no such cell inside the shrunk envelope, the lowest cell there is.
    """
    allowed = _shrunk(envelope, EDGE_CELLS)
    if not allowed.any():
        raise ValueError("the envelope leaves no interior cell")
    positions = np.empty((maps.shape[0], 2))
    for frame, single in enumerate(np.asarray(maps, dtype=np.float64)):
        core = single[1:-1, 1:-1]
        lowest = np.ones_like(core, dtype=bool)
        for dz in (-1, 0, 1):
            for dr in (-1, 0, 1):
                neighbour = single[1 + dz : single.shape[0] - 1 + dz,
                                   1 + dr : single.shape[1] - 1 + dr]
                lowest &= core <= neighbour
        candidates = np.zeros_like(allowed)
        candidates[1:-1, 1:-1] = lowest
        candidates &= allowed
        searched = candidates if candidates.any() else allowed
        iz, ir = np.unravel_index(np.argmin(np.where(searched, single, np.inf)), single.shape)
        positions[frame] = (
            r_grid[ir] + (r_grid[1] - r_grid[0])
            * _offset(single[iz, ir - 1], single[iz, ir], single[iz, ir + 1]),
            z_grid[iz] + (z_grid[1] - z_grid[0])
            * _offset(single[iz - 1, ir], single[iz, ir], single[iz + 1, ir]),
        )
    return positions


def moved_onto(
    maps: NDArray[np.float64],
    target: NDArray[np.float64],
    r_grid: NDArray[np.float64],
    z_grid: NDArray[np.float64],
    envelope: NDArray[np.bool_],
    rounds: int = ROUNDS,
) -> NDArray[np.float64]:
    """Return ``maps`` with each frame's axis moved onto ``target``, shape ``(T, 2)`` as ``(R, Z)``.

    A target that is not finite leaves its frame as it was.
    """
    if target.shape != (maps.shape[0], 2):
        raise ValueError(f"target of shape {target.shape} for {maps.shape[0]} maps")
    moved = np.array(maps, dtype=np.float64)
    radius, height = np.meshgrid(r_grid, z_grid)
    wanted = np.where(np.isfinite(target), target, np.nan)
    for _ in range(rounds):
        now = axis_positions(moved, r_grid, z_grid, envelope)
        shift = np.nan_to_num(np.clip(wanted - now, -SHIFT_LIMIT, SHIFT_LIMIT), nan=0.0)
        slope_z, slope_r = np.gradient(moved, z_grid, r_grid, axis=(1, 2))
        bend_zz, bend_zr = np.gradient(slope_z, z_grid, r_grid, axis=(1, 2))
        bend_rr = np.gradient(slope_r, r_grid, axis=2)
        across = (radius[np.newaxis] - now[:, 0, np.newaxis, np.newaxis]) / BUMP_WIDTH
        along = (height[np.newaxis] - now[:, 1, np.newaxis, np.newaxis]) / (
            BUMP_ELONGATION * BUMP_WIDTH
        )
        bump = np.exp(-(across**2 + along**2))
        by_r = shift[:, 0, np.newaxis, np.newaxis]
        by_z = shift[:, 1, np.newaxis, np.newaxis]
        first = by_r * slope_r + by_z * slope_z
        second = 0.5 * (by_r**2 * bend_rr + by_z**2 * bend_zz) + by_r * by_z * bend_zr
        moved -= bump * (first - second)
    return moved
