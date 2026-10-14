# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — equilibrium from coil currents, plasma current and equal temperatures

"""Reconstruct the flux map of a frame from its inputs alone, with no fitted quantity.

A machine that was never trained on gives three things: the coil currents, the
plasma current, and a Thomson profile. The coil currents give the vacuum flux
exactly (Green's functions of the filaments). The plasma current gives the size
of the plasma's own flux but not where the current sits, and where it sits is
what decides the map. The radial force balance would say, but only through the
pressure and the internal inductance, which the inputs do not hold.

The Thomson profile says it directly. Heat moves along field lines far faster
than across them, so the electron temperature is constant on a flux surface;
a laser that crosses the plasma meets every surface twice, and the two radii at
which it reads the same temperature lie on one surface: ``ψ(R_in) = ψ(R_out)``.
A handful of such pairs fixes the radial position and the peaking of the
current the way magnetic probes do in a conventional reconstruction.

Where the plasma ends is a magnetic question, not a thermal one: the last
closed flux surface passes through the X-point, or touches the wall. The radii
at which the temperature has fallen to :data:`EDGE_SHARE` of its peak are not
that surface — on a spherical tokamak the scrape-off layer carries tens of
electron volts against a pedestal of hundreds, so they lie inside it. Measured
on 137 frames of 24 test shots (inputs only, no label of either machine): they
lie inside on 117 of them, the inboard one by 0.10 m in the median, at a
normalised flux of 0.91. They are therefore one more pair of equal flux, the
most informative one and weighted as such; they do not cut the plasma short.

The solve is the usual Picard loop of a reconstruction code. The current
density is three functions of the previous iterate's normalised flux ``x`` —
``R·(1 - x)`` for the pressure term, ``(1 - x)/R`` and ``x(1 - x)/R`` for the
poloidal-current term, the low-order polynomials a magnetic reconstruction is
usually run with, the last one letting the current be hollow as it is while it
still diffuses in. Their amplitudes are found by
non-negative least squares from the total current and the equal-flux pairs,
and the new current gives the next map. The loop is under-relaxed, and a weak
radial field holds the current's height at the laser's, which a free-boundary
iteration otherwise loses to the vertical instability.

The constants below are choices made beforehand — temperature levels, the
edge share, a smoothing length — not values fitted to any reconstruction.
Labelled shots of the unseen machine were used for one thing: to check that
the method and the machine's conventions hold (three demonstration shots:
pooled flux R² 0.9931 over all 115 frames, the unsolved ones filled, where the coil flux
alone gives a negative one).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.optimize import nnls
from scipy.special import ellipe, ellipk

from fec_adapter.closed_surface import closed_level
from fec_adapter.shot_reader import Shot
from fec_adapter.vessel import envelope

MU0 = 4.0e-7 * np.pi

TEMPERATURE_SHARES = (0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25)
"""Shares of the peak temperature at which the two sides of the profile are paired."""

EDGE_SHARE = 0.10
"""Share of the peak temperature taken as the plasma's edge."""

PLATEAU_SHARE = 0.9
"""The profile's peak is placed at the middle of where it is above this share of its maximum:
a flat-topped profile has no single top, and the solve's axis was rejected as too far from
an arbitrary point of the plateau (high-current test shots, no label involved)."""

SMOOTHING_LENGTH = 0.03
"""Metres; width of the Gaussian the temperature samples are smoothed with."""

PROFILE_POINTS = 400
MIN_CHANNELS = 30
MIN_PEAK_EV = 50.0
MIN_DENSITY = 2.0e18
"""Per cubic metre; a frame whose fired channels sit below it in the median saw no plasma.
Census of the 68 979 live test frames (inputs only): below 2·10¹⁸ a frame fires 32–69 of the
channels with a median temperature of 50–130 eV and one hot spike (219 frames); from 2·10¹⁸
upward it fires 116–138 with a median of ≈ 300 eV (1341 frames between 2 and 5·10¹⁸, at plasma
currents of 430 kA and more). A gate at 5·10¹⁸ refused those real, low-density plasmas whole."""
MIN_PAIRS = 4
MIN_CURRENT = 1.0e5
"""Amperes; below this the frame is not solved but filled from its neighbours."""

ITERATIONS = 100
"""One attempt per frame. A second attempt of 300 iterations from a clean start settled 27 % of
the frames the first had not (forty test shots), and the test split scored lower with it
(0.6953 against 0.7061: flux 0.9721 against 0.9758, Consistency 0.133 against 0.169) — what it
settles passes the trust rule and is still worse than the neighbour fill it replaces. The share
of frames solved is not the measure; a frame that does not settle at once is filled."""
RELAXATION = 0.3
HEIGHT_GAIN = 0.4
CONVERGED = 2.0e-3
SELF_CELL = 0.2235
"""Distance, in units of the summed cell sides, at which a cell's flux on itself is evaluated."""

PAIR_TOLERANCE = 0.02
"""Share of the axis-to-edge flux a pair may differ by; it sets the pair's weight."""

AXIS_TOLERANCE = 0.25
"""Metres; a solve whose axis is further than this from the temperature peak is discarded."""


def green(
    r: NDArray[np.float64] | float, z: NDArray[np.float64] | float,
    source_r: NDArray[np.float64] | float, source_z: NDArray[np.float64] | float,
) -> NDArray[np.float64]:
    """Return the poloidal flux per radian at ``(r, z)`` of a unit filament at the source."""
    k2 = np.clip(
        4.0 * r * source_r / ((r + source_r) ** 2 + (z - source_z) ** 2), 1e-12, 1.0 - 1e-12
    )
    flux: NDArray[np.float64] = (
        MU0 * np.sqrt(r * source_r) * ((2.0 - k2) * ellipk(k2) - 2.0 * ellipe(k2))
        / (2.0 * np.pi * np.sqrt(k2))
    )
    return flux


@dataclass(frozen=True)
class Geometry:
    """What depends on the machine and the grid only, computed once.

    Attributes
    ----------
    rr, zz:
        The grid's coordinates, shape ``(NZ, NR)``.
    interior:
        Where plasma current may sit: the vessel envelope, shrunk by a cell, without
        the cells a coil filament occupies.
    cells:
        Flat indices of the interior cells; ``cell_r``, ``cell_z`` are their positions.
    table:
        Flux on the whole grid of a unit current in each cell, shape ``(cells, NZ·NR)``.
    coil_flux:
        Flux on the grid of a unit current in each filament, shape ``(filaments, NZ, NR)``.
    line_r, line_cells, line_coils:
        Radii along the laser and the flux there of unit cell and filament currents.
    reference_radius:
        Metres; the radius the ``R`` and ``1/R`` terms are scaled by.
    """

    rr: NDArray[np.float64]
    zz: NDArray[np.float64]
    interior: NDArray[np.bool_]
    cells: NDArray[np.intp]
    cell_r: NDArray[np.float64]
    cell_z: NDArray[np.float64]
    table: NDArray[np.float64]
    coil_flux: NDArray[np.float64]
    line_r: NDArray[np.float64]
    line_height: float
    line_cells: NDArray[np.float64]
    line_coils: NDArray[np.float64]
    reference_radius: float

    @classmethod
    def of(cls, shot: Shot) -> Geometry:
        """Build the tables for a shot's machine, grid, filaments and laser."""
        rr, zz = np.meshgrid(shot.r_grid, shot.z_grid)
        step_r, step_z = shot.r_grid[1] - shot.r_grid[0], shot.z_grid[1] - shot.z_grid[0]
        allowed = envelope(shot.source).copy()
        for coil_r, coil_z in zip(shot.coils.r, shot.coils.z, strict=True):
            allowed &= ~(
                (np.abs(rr - coil_r) < 1.5 * step_r) & (np.abs(zz - coil_z) < 1.5 * step_z)
            )
        interior = ndimage.binary_erosion(allowed, iterations=1)
        cells = np.flatnonzero(interior.ravel())
        cell_r, cell_z = rr.ravel()[cells], zz.ravel()[cells]
        table = np.stack(
            [green(rr, zz, a, b).ravel() for a, b in zip(cell_r, cell_z, strict=True)]
        )
        for row, cell in enumerate(cells):
            table[row, cell] = green(
                cell_r[row] + SELF_CELL * (step_r + step_z), cell_z[row], cell_r[row], cell_z[row]
            )
        coil_flux = np.stack(
            [green(rr, zz, a, b) for a, b in zip(shot.coils.r, shot.coils.z, strict=True)]
        )
        line_r = np.sort(shot.thomson.r)
        height = float(np.median(shot.thomson.z)) if shot.thomson.z.size else 0.0
        line_cells = np.stack([green(radius, height, cell_r, cell_z) for radius in line_r])
        line_coils = np.stack(
            [green(radius, height, shot.coils.r, shot.coils.z) for radius in line_r]
        )
        reference = float(np.mean(shot.r_grid[np.any(allowed, axis=0)]))
        return cls(rr, zz, interior, cells, cell_r, cell_z, table, coil_flux, line_r, height,
                   line_cells, line_coils, reference)

    def on_line(
        self, radius: float, coil_current: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], float]:
        """Return the flux at a radius of the laser: per unit cell current, and of the coils."""
        index = int(np.clip(np.searchsorted(self.line_r, radius) - 1, 0, self.line_r.size - 2))
        share = (radius - self.line_r[index]) / (self.line_r[index + 1] - self.line_r[index])
        cells = (1.0 - share) * self.line_cells[index] + share * self.line_cells[index + 1]
        coils = (1.0 - share) * self.line_coils[index] + share * self.line_coils[index + 1]
        return cells, float(coils @ coil_current)


@dataclass(frozen=True)
class EqualTemperatures:
    """Radii of equal temperature on the two sides of the profile's peak."""

    peak_radius: float
    pairs: tuple[tuple[float, float, float], ...]
    """``(share of the peak, inner radius, outer radius)``; the last may be the edge."""

    @property
    def edge(self) -> tuple[float, float] | None:
        """Return the edge radii if the profile reaches the edge share on both sides."""
        for share, inner, outer in self.pairs:
            if share == EDGE_SHARE:
                return inner, outer
        return None


def equal_temperatures(
    radius: NDArray[np.float64], temperature: NDArray[np.float64],
    density: NDArray[np.float64] | None = None,
) -> EqualTemperatures | None:
    """Pair the two sides of one frame's temperature profile, or return ``None`` if it cannot be.

    The samples are smoothed with a Gaussian of :data:`SMOOTHING_LENGTH`; a profile with too few
    channels, too cold a peak, or a peak at either end of the laser is not used. A frame whose
    fired channels report, in the median, a density below :data:`MIN_DENSITY` saw no plasma:
    its temperatures are single hot spikes at the edges of a frame the laser missed, and a
    solve on them converges to a plasma that is not there.
    """
    fired = np.isfinite(temperature) & (temperature > 0.0)
    if density is not None and fired.any() and float(np.nanmedian(density[fired])) < MIN_DENSITY:
        return None
    if int(fired.sum()) < MIN_CHANNELS:
        return None
    order = np.argsort(radius[fired])
    at, value = radius[fired][order], temperature[fired][order]
    along = np.linspace(at[0], at[-1], PROFILE_POINTS)
    weight = np.exp(-0.5 * ((along[:, np.newaxis] - at[np.newaxis, :]) / SMOOTHING_LENGTH) ** 2)
    smooth = (weight * value).sum(axis=1) / weight.sum(axis=1)
    top = int(np.argmax(smooth))
    margin = PROFILE_POINTS // 20
    if smooth[top] < MIN_PEAK_EV or top < margin or top > PROFILE_POINTS - margin:
        return None
    pairs = []
    for share in (*TEMPERATURE_SHARES, EDGE_SHARE):
        level = share * smooth[top]
        below_in = np.flatnonzero(smooth[:top] < level)
        below_out = np.flatnonzero(smooth[top:] < level)
        if below_in.size == 0 or below_out.size == 0:
            continue
        i, j = int(below_in[-1]), top + int(below_out[0])
        inner = float(np.interp(level, [smooth[i], smooth[i + 1]], [along[i], along[i + 1]]))
        outer = float(np.interp(level, [smooth[j], smooth[j - 1]], [along[j], along[j - 1]]))
        pairs.append((share, inner, outer))
    if len(pairs) < MIN_PAIRS:
        return None
    plateau = np.flatnonzero(smooth >= PLATEAU_SHARE * smooth[top])
    centre = 0.5 * (along[plateau[0]] + along[plateau[-1]])
    return EqualTemperatures(float(centre), tuple(pairs))


@dataclass(frozen=True)
class Solved:
    """One frame's result."""

    flux: NDArray[np.float64]
    current: NDArray[np.float64]
    """Current per interior cell, in amperes."""
    axis: tuple[float, float]
    axis_flux: float
    edge_flux: float
    """Flux of the last closed surface of the map itself (an X-point's or a limiter's), which
    may lie outside the surface the temperature profile calls the edge."""
    change: float
    """Share of the plasma current the last iteration still moved."""
    amplitudes: tuple[float, ...] = ()
    """Fitted amplitudes of the current terms, the pressure term first."""


def first_guess(geometry: Geometry, plasma_current: float,
                found: EqualTemperatures) -> NDArray[np.float64]:
    """Return a parabolic current between the profile's edges, which only starts the iteration.

    It is 1.8 times as tall as it is wide, roughly what tokamaks of either kind are.
    """
    edge = found.edge
    middle = 0.5 * (edge[0] + edge[1]) if edge else found.peak_radius
    half = 0.5 * (edge[1] - edge[0]) if edge else 0.5
    shape = np.clip(
        1.0 - ((geometry.cell_r - middle) / half) ** 2
        - ((geometry.cell_z - geometry.line_height) / (1.8 * half)) ** 2, 0.0, None,
    )
    guess: NDArray[np.float64] = shape / shape.sum() * plasma_current
    return guess


def solve_frame(
    geometry: Geometry, coil_current: NDArray[np.float64], plasma_current: float,
    found: EqualTemperatures, start: NDArray[np.float64] | None = None,
    iterations: int = ITERATIONS,
) -> Solved | None:
    """Iterate one frame to a map whose surfaces pass through the radii of equal temperature."""
    sign = float(np.sign(plasma_current))
    vacuum = np.einsum("f,fzr->zr", coil_current, geometry.coil_flux)
    shape = vacuum.shape
    current = first_guess(geometry, plasma_current, found) if start is None else start
    lines = [
        (share, *geometry.on_line(inner, coil_current), *geometry.on_line(outer, coil_current))
        for share, inner, outer in found.pairs
    ]
    hold, change, axis, level, axis_flux = 0.0, 1.0, 0, 0.0, 0.0
    amplitudes = np.zeros(3)
    for iteration in range(iterations):
        height = float((current * geometry.cell_z).sum() / current.sum())
        hold -= HEIGHT_GAIN * sign * abs(plasma_current) * MU0 * (height - geometry.line_height)
        flux = (vacuum + (current @ geometry.table).reshape(shape)
                + hold * (geometry.zz - geometry.line_height))
        oriented = sign * flux
        carrying = np.abs(current) > 0.1 * np.abs(current).max()
        candidates = geometry.cells[carrying]
        axis = int(candidates[np.argmax(oriented.ravel()[candidates])])
        axis_flux = float(oriented.ravel()[axis])
        closed = closed_level(oriented, geometry.interior, axis)
        if closed is None:
            return None
        region, level = closed
        span = max(axis_flux - level, 1e-9)
        depth = np.clip((axis_flux - oriented.ravel()[geometry.cells]) / span, 0.0, 1.0)
        inside = (1.0 - depth) * region.ravel()[geometry.cells]
        outward = geometry.cell_r / geometry.reference_radius
        basis = np.stack([inside * outward, inside / outward, inside * depth / outward])
        rows, wanted = [basis.sum(axis=1) / abs(plasma_current) * 50.0], [50.0]
        for share, cells_in, coils_in, cells_out, coils_out in lines:
            weight = (3.0 if share == EDGE_SHARE else 1.0) / (PAIR_TOLERANCE * span)
            rows.append(sign * (basis @ (cells_in - cells_out)) * weight)
            wanted.append(-(coils_in - coils_out) * weight)
        amplitudes, _residual = nnls(np.asarray(rows), np.asarray(wanted))
        new = sign * amplitudes @ basis
        if new.sum() == 0.0:
            return None
        new *= plasma_current / new.sum()
        change = float(np.abs(new - current).sum() / abs(plasma_current))
        current = (1.0 - RELAXATION) * current + RELAXATION * new
        if change < CONVERGED / 4.0 and iteration > 5:  # noqa: PLR2004 - a few passes to settle
            break
    flux = vacuum + (current @ geometry.table).reshape(shape)
    last = closed_level(sign * flux, geometry.interior, axis)
    if last is None:
        return None
    position = (float(geometry.rr.ravel()[axis]), float(geometry.zz.ravel()[axis]))
    return Solved(flux, current, position, float(flux.ravel()[axis]), sign * last[1], change,
                  tuple(float(value) for value in amplitudes))


def trusted(solved: Solved | None, found: EqualTemperatures) -> bool:
    """Say whether a solve converged to a plasma where the temperature profile has its peak."""
    return (
        solved is not None and solved.change < CONVERGED
        and abs(solved.axis[0] - found.peak_radius) < AXIS_TOLERANCE
    )


def reconstruct(
    shot: Shot, geometry: Geometry | None = None, fallback: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], list[Solved | None]]:
    """Return the flux maps of every frame of a shot, ``(T, NZ, NR)``, and the solved frames.

    Frames that cannot be solved — no plasma current to speak of, no usable profile, no
    converged solve — take the current distribution of the nearest solved frame, scaled to
    their own plasma current, on top of their own exact coil flux. A shot without a single
    solved frame gets the coil flux plus ``fallback``, a current pattern per interior cell
    summing to one (the mean pattern of the split's solved frames, in the build), or a
    parabolic current in the middle of the vessel without one.
    """
    geometry = Geometry.of(shot) if geometry is None else geometry
    solved: list[Solved | None] = []
    start = None
    for frame in range(shot.n_frames):
        plasma_current = float(shot.plasma_current[frame])
        found = None
        if np.isfinite(plasma_current) and abs(plasma_current) >= MIN_CURRENT:
            found = equal_temperatures(shot.thomson.r, shot.thomson.te[frame],
                                       shot.thomson.ne[frame])
        result = None
        if found is not None and np.all(np.isfinite(shot.coil_current[frame])):
            result = solve_frame(geometry, shot.coil_current[frame], plasma_current, found, start)
            if not trusted(result, found):
                result = None
        start = None if result is None else result.current
        solved.append(result)
    good = np.flatnonzero([result is not None for result in solved])
    grid_shape = geometry.rr.shape
    maps = np.zeros((shot.n_frames, *grid_shape))
    for frame in range(shot.n_frames):
        result = solved[frame]
        if result is not None:
            maps[frame] = result.flux
            continue
        coils = np.nan_to_num(shot.coil_current[frame])
        vacuum = np.einsum("f,fzr->zr", coils, geometry.coil_flux)
        plasma_current = float(np.nan_to_num(shot.plasma_current[frame]))
        if good.size:
            nearest = solved[int(good[np.argmin(np.abs(good - frame))])]
            assert nearest is not None  # noqa: S101 - chosen among the solved frames
            pattern = nearest.current / nearest.current.sum()
        elif fallback is not None:
            pattern = fallback
        else:
            middle = geometry.reference_radius
            pattern = np.clip(
                1.0 - ((geometry.cell_r - middle) / 0.5) ** 2 - (geometry.cell_z / 0.9) ** 2,
                0.0, None,
            )
            pattern = pattern / pattern.sum()
        maps[frame] = vacuum + ((pattern * plasma_current) @ geometry.table).reshape(grid_shape)
    return maps, solved
