# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — per-frame predictive equilibrium solve

"""Solve one reconstruction frame from coils, kinetic profiles and the current.

The solver in :mod:`scpn_fusion.core.jax_free_boundary_predictive` takes the
free functions as given. This module supplies them from the measurements, and
that introduces a circularity worth naming: the pressure is measured in real
space, the free functions live in flux space, and the map between them is the
equilibrium being solved for.

It is resolved by iteration rather than by assumption. A first solve uses a
generic profile pair whose only job is to produce a plausible flux map; the
Thomson samples are then mapped through that map into normalised flux, the
measured pressure profile replaces the generic one, and the frame is solved
again. The second solve is the prediction; the first is scaffolding, and its
profiles never reach the output.

Nothing in this path reads the target equilibrium. The grid, the clock, the
coil currents, the plasma current and the Thomson profiles are the whole
input.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

# Double precision is not optional for this solve: the flux span across the
# plasma is a fraction of a weber on values of order one, and the grid spacing
# enters a uniformity check with a tolerance of 1e-9. In single precision the
# grid reads as non-uniform and the equilibrium lands in a different basin.
#
# The environment variable alone does not guarantee it. JAX reads it when it
# initialises, so a caller that imports `jax` before this module gets single
# precision silently — which is how a version comparison in this session was
# ruined before the cause was found. Setting the variable *and* updating the
# live configuration *and* verifying the result makes the order irrelevant and
# turns any remaining failure into a loud one.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", val=True)

if jnp.zeros(1).dtype != jnp.float64:
    raise RuntimeError(
        "JAX is not in double precision; the equilibrium solve is not valid in "
        "single precision and must not run"
    )

from scpn_fusion.core.jax_free_boundary_gs import vacuum_field_si
from scpn_fusion.core.jax_free_boundary_predictive import (
    build_response_matrix,
    solve_predictive_equilibrium_newton,
)
from scpn_fusion.core.jax_o_point import smooth_axis_flux
from scpn_fusion.core.jax_plasma_support import soft_axis_connected_support

from . import derived, profiles, vessel
from .shot_reader import Shot

BOOTSTRAP_PRESSURE_FRACTION: Final = 0.4
"""Share of the plasma current carried by the pressure term in the first solve.

Only the ratio of the two generic terms matters there, because the solver
normalises the total current to the measured one. Four tenths is the usual
order for a diverted DIII-D discharge and the second solve replaces it.
"""

DEFAULT_ITERATIONS: Final = 40
"""Newton iterations allowed in each of the solver's two stages."""


@dataclass(frozen=True)
class Geometry:
    """Grid-dependent quantities that are computed once and reused.

    The von Hagenow response matrix depends only on the grid, so building it
    per frame would repeat an identical calculation thousands of times.
    """

    r_grid: jnp.ndarray
    z_grid: jnp.ndarray
    response: jnp.ndarray
    wall_idx: jnp.ndarray
    source_idx: jnp.ndarray


@dataclass(frozen=True)
class FrameSolution:
    """One frame's solved equilibrium and the quantities derived alongside it.

    Attributes
    ----------
    psi:
        Poloidal flux on the reconstruction grid, shape ``(NZ, NR)``, with the
        solenoid's flux superposed.
    psi_solver:
        The flux as the solver produced it, without the solenoid. This is what
        the next frame warm-starts from, because it is the field the solver's
        own topology checks were written against.
    psi_axis, psi_boundary:
        Flux at the magnetic axis and at the last closed surface, in webers.
    pprime, ffprime:
        The free functions the solution was produced with, on the knots.
    pressure:
        The measured pressure profile on the knots, in pascals. Carried out of
        the solve because the normalised beta needs the pressure itself, not
        only its derivative.
    converged:
        Whether the solver's own fail-closed status reported convergence. A
        frame can be usable without it: a finite field that stalled short of
        tolerance is still the best estimate available for that frame, and
        discarding it would hand the scorer a hole rather than an approximation.
        The flag is reported so the convergence rate can be stated instead of
        assumed.
    usable:
        Whether the flux map is finite and therefore worth scoring.
    """

    psi: NDArray[np.float64]
    psi_solver: NDArray[np.float64]
    psi_axis: float
    psi_boundary: float
    pprime: NDArray[np.float64]
    ffprime: NDArray[np.float64]
    pressure: NDArray[np.float64]
    converged: bool
    usable: bool


def geometry(shot: Shot) -> Geometry:
    """Build the reusable grid quantities for one shot's reconstruction grid."""
    r_grid = jnp.asarray(shot.r_grid)
    z_grid = jnp.asarray(shot.z_grid)
    response, wall_idx, source_idx = build_response_matrix(r_grid, z_grid)
    return Geometry(r_grid, z_grid, response, wall_idx, source_idx)


def solve_frame(
    shot: Shot,
    frame: int,
    geom: Geometry,
    model: profiles.ProfileModel,
    psi_n_knots: NDArray[np.float64],
    psi_init: NDArray[np.float64] | None = None,
    iterations: int = DEFAULT_ITERATIONS,
) -> FrameSolution:
    """Solve one frame and return its flux map and free functions.

    Parameters
    ----------
    shot:
        The shot whose frame is being solved.
    frame:
        Index into the reconstruction time base.
    geom:
        Grid quantities from :func:`geometry`.
    model:
        The parameters the measurements do not determine.
    psi_n_knots:
        Normalised-flux knots the free functions are sampled on.
    psi_init:
        Optional warm start — normally the previous frame's solution, since
        consecutive reconstruction times are milliseconds apart and the
        equilibrium moves little between them.
    iterations:
        Maximum Anderson iterations per solve.
    """
    # The solenoid is superposed after the solve rather than solved with.
    # Outside a long solenoid the flux is nearly a constant, and the
    # Grad-Shafranov equation is indifferent to a constant: the equilibrium is
    # the same with or without it. The solver's seed selection is not — handed
    # a field shifted by several tenths of a weber it rejects every candidate
    # axis ("no finite central-axis field", measured). Leaving the solenoid out
    # of the solve and adding its exact vacuum flux afterwards gives the same
    # physics to first order and keeps the solver in the regime it works in.
    solenoid = shot.coils.is_solenoid
    coil_i = jnp.asarray(np.where(solenoid, 0.0, shot.coil_current[frame]))
    ip_target = float(shot.plasma_current[frame])
    coil_r = jnp.asarray(shot.coils.r)
    coil_z = jnp.asarray(shot.coils.z)
    knots = jnp.asarray(psi_n_knots)

    bootstrap_pprime, bootstrap_ffprime = bootstrap_profiles(shot, psi_n_knots, ip_target, model)
    psi, _bootstrap_converged = _solve(
        geom, coil_i, coil_r, coil_z, bootstrap_pprime, bootstrap_ffprime, knots, ip_target,
        psi_init, iterations,
    )
    if not bool(jnp.all(jnp.isfinite(psi))):
        return FrameSolution(
            psi=np.zeros((shot.z_grid.size, shot.r_grid.size)),
            psi_solver=np.zeros((shot.z_grid.size, shot.r_grid.size)),
            psi_axis=float("nan"),
            psi_boundary=float("nan"),
            pprime=np.asarray(bootstrap_pprime),
            ffprime=np.asarray(bootstrap_ffprime),
            pressure=np.zeros_like(psi_n_knots),
            converged=False,
            usable=False,
        )

    pprime, ffprime, pressure = measured_profiles(
        shot, frame, psi, geom, model, psi_n_knots, ip_target
    )
    psi, converged = _solve(
        geom, coil_i, coil_r, coil_z, jnp.asarray(pprime), jnp.asarray(ffprime), knots,
        ip_target, psi, iterations,
    )
    usable = bool(jnp.all(jnp.isfinite(psi)))
    psi_solver = np.asarray(psi, dtype=np.float64)
    psi = psi + vacuum_field_si(
        geom.r_grid,
        geom.z_grid,
        jnp.asarray(shot.coils.r[solenoid]),
        jnp.asarray(shot.coils.z[solenoid]),
        jnp.asarray(shot.coil_current[frame][solenoid]),
    )
    axis = float(smooth_axis_flux(psi)) if usable else float("nan")
    boundary = (
        _boundary_flux(shot, np.asarray(psi, dtype=np.float64), axis)
        if usable
        else float("nan")
    )
    return FrameSolution(
        psi=np.asarray(psi, dtype=np.float64),
        psi_solver=psi_solver,
        psi_axis=axis,
        psi_boundary=boundary,
        pprime=pprime,
        ffprime=ffprime,
        pressure=pressure,
        converged=converged,
        usable=usable,
    )


def measured_profiles(
    shot: Shot,
    frame: int,
    psi: jnp.ndarray,
    geom: Geometry,
    model: profiles.ProfileModel,
    psi_n_knots: NDArray[np.float64],
    ip_target: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Derive ``p'``, ``FF'`` and the pressure for one frame from the measurements.

    The Thomson samples are mapped into normalised flux through ``psi``, the
    pressure profile is reduced onto the knots, and its flux derivative is the
    measured ``p'``. The ``FF'`` shape comes from the model and its amplitude
    from the measured plasma current, so the two free functions between them
    reproduce that current without the solver having to rescale them.
    """
    psi_axis = smooth_axis_flux(psi)
    psi_boundary = _boundary_flux(shot, np.asarray(psi, dtype=np.float64), float(psi_axis))
    delta = float(psi_boundary) - float(psi_axis)

    sample_psi_n = _sample_normalised_flux(
        shot, psi, psi_axis, jnp.asarray(psi_boundary)
    )
    pressure = profiles.total_pressure(shot.thomson.te[frame], shot.thomson.ne[frame], model)
    pressure_knots = profiles.pressure_on_knots(sample_psi_n, pressure, psi_n_knots, model)
    pprime = profiles.pprime_from_pressure(pressure_knots, psi_n_knots, delta)

    shape = profiles.ffprime_shape(psi_n_knots, model)
    pprime_current, shape_current = _current_integrals(
        psi, geom, psi_axis, jnp.asarray(psi_boundary), psi_n_knots, pprime, shape
    )
    amplitude = profiles.ffprime_amplitude(ip_target, pprime_current, shape_current)
    return pprime, shape * amplitude, pressure_knots


def bootstrap_profiles(
    shot: Shot, psi_n_knots: NDArray[np.float64], ip_target: float, model: profiles.ProfileModel
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the generic profile pair used to produce a first flux map.

    The absolute scale is irrelevant because the solver normalises the total
    current; only the split between the two terms matters, and that is set by
    :data:`BOOTSTRAP_PRESSURE_FRACTION`.
    """
    radius = float(np.mean(shot.r_grid))
    pressure_shape = np.clip(1.0 - psi_n_knots, 0.0, None)
    ff_shape = profiles.ffprime_shape(psi_n_knots, model)
    pressure_term = BOOTSTRAP_PRESSURE_FRACTION * ip_target / max(radius, 1e-6)
    ff_term = (1.0 - BOOTSTRAP_PRESSURE_FRACTION) * ip_target * profiles.MU0 * radius
    return jnp.asarray(pressure_shape * pressure_term), jnp.asarray(ff_shape * ff_term)


def _solve(
    geom: Geometry,
    coil_i: jnp.ndarray,
    coil_r: jnp.ndarray,
    coil_z: jnp.ndarray,
    pprime: jnp.ndarray,
    ffprime: jnp.ndarray,
    knots: jnp.ndarray,
    ip_target: float,
    psi_init: NDArray[np.float64] | jnp.ndarray | None,
    iterations: int,
) -> tuple[jnp.ndarray, bool]:
    """Solve one frame through the two-stage Newton path.

    The Anderson fixed-point solver is not used here. On these shots it leaves
    the physical branch: measured on a flat-top frame, forty of its iterations
    turn an analytic starting guess worth R² = +0.59 against the reconstruction
    into a field worth -0.81, and the resulting flux extremum sits on the
    computational wall rather than at an interior magnetic axis. That is the
    saddle-point behaviour the solver module itself warns about.

    The Newton path converges on the same frame to a scaled residual of
    ~1e-10 and places the axis inside the chamber, because it carries a
    selector that rejects an axis assignment outside it. Its returned status is
    fail-closed, so the convergence flag is reported rather than assumed.
    """
    def attempt(start: jnp.ndarray | None) -> Any:
        return solve_predictive_equilibrium_newton(
            coil_i,
            pprime,
            ffprime,
            geom.r_grid,
            geom.z_grid,
            coil_r,
            coil_z,
            knots,
            ip_target,
            geom.response,
            geom.wall_idx,
            geom.source_idx,
            psi_init=start,
            seed_picard_iterations=iterations,
            fixed_support_newton_iterations=iterations,
            dynamic_support_newton_iterations=iterations,
        )

    start = None if psi_init is None else jnp.asarray(psi_init)
    try:
        result = attempt(start)
    except ValueError:
        # The solver refuses a warm start whose axis is not in the chamber, and
        # it is right to: continuing from a field that is not an equilibrium of
        # this machine would carry the previous frame's failure forward through
        # the whole shot. Falling back to a cold start loses the continuity but
        # keeps the frame independent, which is the lesser loss.
        if start is None:
            raise
        result = attempt(None)
    return result.equilibrium, bool(result.finite and result.dynamic_support_converged)


def _sample_normalised_flux(
    shot: Shot,
    psi: jnp.ndarray,
    psi_axis: jnp.ndarray,
    psi_boundary: jnp.ndarray,
) -> NDArray[np.float64]:
    """Return the normalised flux at each Thomson measurement position."""
    values = _bilinear(psi, shot.r_grid, shot.z_grid, shot.thomson.r, shot.thomson.z)
    delta = float(psi_boundary - psi_axis)
    if not np.isfinite(delta) or delta == 0.0:
        return np.full(values.shape, np.nan)
    return (values - float(psi_axis)) / delta


def _current_integrals(
    psi: jnp.ndarray,
    geom: Geometry,
    psi_axis: jnp.ndarray,
    psi_boundary: jnp.ndarray,
    psi_n_knots: NDArray[np.float64],
    pprime: NDArray[np.float64],
    ff_shape: NDArray[np.float64],
) -> tuple[float, float]:
    """Integrate each current term over the plasma support.

    The support is the solver's own axis-connected smooth weight, so the
    amplitude solved for here is the one the solver will actually integrate —
    using a different mask would make the two disagree and reintroduce the
    rescaling this construction exists to avoid.
    """
    denominator = psi_boundary - psi_axis
    raw = (psi - psi_axis) / jnp.where(jnp.abs(denominator) < profiles.TINY, 1.0, denominator)
    support = soft_axis_connected_support(psi, raw, 0.03)
    psi_n = jnp.clip(raw, 0.0, 1.0)
    knots = jnp.asarray(psi_n_knots)
    r_safe = jnp.maximum(geom.r_grid[jnp.newaxis, :], 1e-6)
    d_area = float((grid_spacing(geom.r_grid)) * (grid_spacing(geom.z_grid)))
    pressure_term = jnp.sum(r_safe * jnp.interp(psi_n, knots, jnp.asarray(pprime)) * support)
    shape_term = jnp.sum(
        jnp.interp(psi_n, knots, jnp.asarray(ff_shape)) / (profiles.MU0 * r_safe) * support
    )
    return float(pressure_term) * d_area, float(shape_term) * d_area


def grid_spacing(grid: jnp.ndarray) -> float:
    """Return the uniform spacing of a grid axis in metres."""
    return float(grid[1] - grid[0])


def _bilinear(
    field: jnp.ndarray,
    r_grid: NDArray[np.float64],
    z_grid: NDArray[np.float64],
    r: NDArray[np.float64],
    z: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Interpolate a ``(NZ, NR)`` field at scattered positions.

    Positions outside the grid are clamped to its edge rather than dropped:
    a Thomson channel just outside the computational boundary is still a real
    measurement of a cold edge, and the clamp keeps it in the sample cloud at
    a defensible flux value.
    """
    values = np.asarray(field, dtype=np.float64)
    r_cells = np.arange(r_grid.size, dtype=np.float64)
    z_cells = np.arange(z_grid.size, dtype=np.float64)
    r_index = np.clip(np.interp(r, r_grid, r_cells), 0, r_grid.size - 1)
    z_index = np.clip(np.interp(z, z_grid, z_cells), 0, z_grid.size - 1)
    r0 = np.floor(r_index).astype(int)
    z0 = np.floor(z_index).astype(int)
    r1 = np.minimum(r0 + 1, r_grid.size - 1)
    z1 = np.minimum(z0 + 1, z_grid.size - 1)
    fr = r_index - r0
    fz = z_index - z0
    sampled: NDArray[np.float64] = (
        values[z0, r0] * (1 - fr) * (1 - fz)
        + values[z0, r1] * fr * (1 - fz)
        + values[z1, r0] * (1 - fr) * fz
        + values[z1, r1] * fr * fz
    )
    return sampled


def _boundary_flux(shot: Shot, psi: NDArray[np.float64], psi_axis: float) -> float:
    """Return the flux at the last closed surface, or ``nan`` when none closes.

    The solver's X-point finder is not used for this. On these shots it returns
    a level that encloses a plasma about a centimetre across on an equilibrium
    whose axis is otherwise in the right place — which then collapses the
    normalised beta and the flux mapping together. Scanning outwards for the
    last surface that closes inside the vessel answers the same question
    without assuming the boundary is a saddle rather than a wall contact.
    """
    found = derived.outermost_closed_surface(
        psi, shot.r_grid, shot.z_grid, psi_axis, vessel.envelope(shot.source)
    )
    return float("nan") if found is None else found[0]
