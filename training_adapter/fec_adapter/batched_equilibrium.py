# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — a whole shot solved as batches on every device

"""Solve every frame of a shot in batches, and keep the frames on the confined branch.

The per-frame Newton path in :mod:`fec_adapter.equilibrium` costs tens of
seconds a frame. The batched compiled solve costs milliseconds, but its
fixed-point iteration has a second, collapsed attractor and its convergence
flag does not tell the two apart (see :mod:`fec_adapter.branch_check`). This
module runs the fast solve and steers it:

1. **Cold pass.** Every frame from the coil field, generic profiles, its own
   plasma current.
2. **Branch check.** The confined cross-section of every field, from the field
   alone.
3. **Warm rounds.** A frame that passed keeps its field as the start; a frame
   that failed starts from the nearest frame in time that passed — consecutive
   frames are milliseconds apart. The Thomson samples are mapped through the
   start field, the measured pressure replaces the generic one, and the batch
   is solved again with no current ramp. A new field replaces the old one
   unless the old one passed and the new one does not.
4. **Fallback.** A frame that still fails takes the plasma's own flux from its
   nearest passing neighbour, scaled by the ratio of the plasma currents, plus
   its own exact coil flux. The coil flux is the part known exactly, so the
   approximation is confined to the part that moves least between frames.

The solenoid is kept out of every solve and its exact flux is superposed at the
end, for the reason recorded in :func:`fec_adapter.equilibrium.solve_frame`.
No reconstruction column is read anywhere in this path.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Final

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from scpn_fusion.core.jax_free_boundary_gs import vacuum_field_si
from scpn_fusion.core.jax_o_point import smooth_axis_flux

from . import equilibrium, profiles, vessel
from .branch_check import MINIMUM_CONFINED_AREA, BranchReport, VesselGrid, check_branch
from .device_batches import FrameBatch, Machine, SolverSettings, solve_across_devices
from .shot_reader import Shot


class FrameStatus(Enum):
    """How a frame's field was arrived at."""

    CONFINED_COLD = "confined after the cold pass"
    CONFINED_WARM = "confined after a warm round"
    FALLBACK = "neighbour's plasma flux with this frame's coil flux"
    UNRESOLVED = "no confined frame in the shot to lean on"


@dataclass(frozen=True)
class EllipticalSupport:
    """A machine-defined plasma support for the cold pass: a soft-edged ellipse.

    The weight is ``(1 - tanh((rho - 1) / edge_width)) / 2`` with
    ``rho² = ((R - r_centre) / minor_radius)² + ((Z - z_centre) / (elongation · minor_radius))²``.

    Attributes
    ----------
    r_centre, z_centre:
        Centre of the ellipse in metres.
    minor_radius:
        Horizontal half-width in metres.
    elongation:
        Ratio of the vertical to the horizontal half-width.
    edge_width:
        Width of the soft edge in units of ``rho``.
    """

    r_centre: float
    z_centre: float
    minor_radius: float
    elongation: float
    edge_width: float = 0.1

    def weights(
        self, r_grid: NDArray[np.float64], z_grid: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return the support on a grid, shape ``(NZ, NR)``, between zero and one."""
        r_mesh, z_mesh = np.meshgrid(r_grid, z_grid)
        rho = np.hypot(
            (r_mesh - self.r_centre) / self.minor_radius,
            (z_mesh - self.z_centre) / (self.elongation * self.minor_radius),
        )
        support: NDArray[np.float64] = 0.5 * (1.0 - np.tanh((rho - 1.0) / self.edge_width))
        return support


MACHINE_COLD_SUPPORT: Final = {
    "DIII-D": EllipticalSupport(
        r_centre=1.675, z_centre=-0.054, minor_radius=0.379, elongation=1.80
    ),
    "MAST": EllipticalSupport(
        r_centre=0.869, z_centre=0.0, minor_radius=0.352, elongation=1.6
    ),
}
"""Cold-pass support of each machine that has one.

DIII-D: centre and elongation are the medians of the reconstructed boundary over 1860
frames of 40 calibration-bucket shots (centre 1.675 m, -0.054 m; minor radius 0.583 m;
elongation 1.80). The support covers 0.65 of that minor radius: scanned on eight
calibration-bucket shots, the pooled flux R² was 0.8823 with no fixed support and 0.8997,
0.9070, 0.8941, 0.8887 at 0.5, 0.65, 0.8 and 0.95 of the minor radius, and frames needing
the fallback fell from 302 to 49 of 1931. Nothing here was chosen on the validation bucket.
MAST: from its inputs alone, because no reconstruction of MAST may choose anything (the
competition's rule of 2026-09-23). The midplane Thomson reads above a tenth of its peak between
R 0.327 m and 1.418 m in the median (2323 frames of 120 test shots), so the centre is 0.869 m on the
midplane and the minor radius 0.542 m; the support covers 0.65 of it, DIII-D's measured share. The
elongation was chosen by the count of frames the pipeline ends on the confined branch, again
no label read: on 12 test shots (791 frames) 125 confined with no support, and 771, 767 and 762
at 1.6, 1.8 and 2.0.
A machine without an entry gets no fixed support until it has been measured.
"""


@dataclass(frozen=True)
class PipelineSettings:
    """Settings of the whole-shot solve.

    Attributes
    ----------
    cold, warm:
        Iteration settings of the cold pass and of the warm rounds. The warm
        rounds must not ramp the current.
    max_rounds:
        Cap on the warm rounds. A round that changes no frame's status ends
        the loop earlier.
    minimum_area:
        Confined cross-section, in square metres, a field must reach.
    cold_support:
        Plasma support held fixed during the cold pass. ``None`` takes the
        machine's entry in :data:`MACHINE_COLD_SUPPORT` when
        ``machine_cold_support`` is set, and otherwise lets the solver find the
        support from each iterate. The warm rounds always find their own: the
        fixed support only has to deliver a start on the right branch, as the
        first stage of the Newton path does.
    machine_cold_support:
        Whether a missing ``cold_support`` falls back to the machine's own.
    """

    cold: SolverSettings = field(default_factory=lambda: SolverSettings(n_iter=180, ip_ramp=30))
    warm: SolverSettings = field(default_factory=lambda: SolverSettings(n_iter=300, ip_ramp=1))
    max_rounds: int = 3
    minimum_area: float = MINIMUM_CONFINED_AREA
    cold_support: EllipticalSupport | None = None
    machine_cold_support: bool = True

    def cold_support_for(self, machine: str) -> EllipticalSupport | None:
        """Return the support the cold pass holds fixed on ``machine``, if any."""
        if self.cold_support is not None or not self.machine_cold_support:
            return self.cold_support
        return MACHINE_COLD_SUPPORT.get(machine)


@dataclass(frozen=True)
class ShotSolution:
    """Every frame of one shot.

    Attributes
    ----------
    psi:
        Flux with the solenoid superposed, shape ``(T, NZ, NR)``, in the
        solver's own sign convention.
    psi_axis, psi_boundary:
        Flux at the magnetic axis and on the last closed surface inside the
        vessel, shape ``(T,)``; ``nan`` where there is none.
    enclosed_area:
        Confined cross-section of each frame's final field, in square metres.
    pprime, ffprime, pressure:
        The free functions and the measured pressure each frame ended with, on
        the knots, shape ``(T, N_knots)``.
    status:
        How each frame's field was arrived at.
    rounds:
        Warm rounds actually run.
    """

    psi: NDArray[np.float64]
    psi_axis: NDArray[np.float64]
    psi_boundary: NDArray[np.float64]
    enclosed_area: NDArray[np.float64]
    pprime: NDArray[np.float64]
    ffprime: NDArray[np.float64]
    pressure: NDArray[np.float64]
    status: tuple[FrameStatus, ...]
    rounds: int

    def count(self, status: FrameStatus) -> int:
        """Return how many frames ended with ``status``."""
        return sum(1 for entry in self.status if entry is status)


def nearest_member(members: NDArray[np.intp], size: int) -> NDArray[np.intp]:
    """Return, for each index below ``size``, the nearest index in ``members``.

    An index that is itself a member maps to itself; a tie goes to the earlier
    frame, the one whose field already existed when the later one was measured.
    """
    if members.size == 0:
        raise ValueError("no member to be nearest to")
    ordered = np.sort(members)
    distance = np.abs(np.arange(size)[:, np.newaxis] - ordered[np.newaxis, :])
    nearest: NDArray[np.intp] = ordered[np.argmin(distance, axis=1)]
    return nearest


def solve_shot(
    shot: Shot,
    model: profiles.ProfileModel,
    settings: PipelineSettings | None = None,
    devices: Sequence[jax.Device] | None = None,
) -> ShotSolution:
    """Solve every frame of ``shot`` and report how each field was arrived at.

    Parameters
    ----------
    shot:
        The shot to solve; every frame goes through in one batch.
    model:
        The parameters the measurements do not determine.
    settings:
        Pipeline settings; the defaults are the measured ones.
    devices:
        Devices to spread each batch over; every visible device when omitted.
    """
    settings = settings or PipelineSettings()
    cold_support = settings.cold_support_for(shot.source)
    context = _context(shot, devices)
    knots = context.machine.psi_n_knots
    unit_pprime, unit_ffprime = equilibrium.bootstrap_profiles(shot, knots, 1.0, model)
    current = context.plasma_current[:, np.newaxis]
    pprime = np.asarray(unit_pprime)[np.newaxis, :] * current
    ffprime = np.asarray(unit_ffprime)[np.newaxis, :] * current
    psi, _ = solve_across_devices(
        context.machine,
        FrameBatch(context.coil_current, pprime, ffprime, context.plasma_current),
        settings.cold,
        devices,
        None if cold_support is None else cold_support.weights(shot.r_grid, shot.z_grid),
    )
    reports = _branch_reports(psi + context.solenoid_flux, context.grid, settings.minimum_area)
    state = _Frames(psi, pprime, ffprime, np.zeros_like(pprime), reports, [
        FrameStatus.CONFINED_COLD if report.confined else FrameStatus.UNRESOLVED
        for report in reports
    ])
    rounds = _warm_rounds(shot, model, context, state, settings)
    _fallback(context, state, settings)

    full = state.psi + context.solenoid_flux
    return ShotSolution(
        psi=full,
        psi_axis=_axis_fluxes(full),
        psi_boundary=np.asarray([report.boundary_flux for report in state.reports]),
        enclosed_area=np.asarray([report.enclosed_area for report in state.reports]),
        pprime=state.pprime,
        ffprime=state.ffprime,
        pressure=state.pressure,
        status=tuple(state.status),
        rounds=rounds,
    )


@dataclass(frozen=True)
class _Context:
    """What stays fixed while a shot is being solved."""

    geom: equilibrium.Geometry
    machine: Machine
    grid: VesselGrid
    coil_current: NDArray[np.float64]
    plasma_current: NDArray[np.float64]
    solenoid_flux: NDArray[np.float64]
    devices: Sequence[jax.Device] | None


@dataclass
class _Frames:
    """The per-frame fields and records the rounds update in place.

    ``psi`` is the solver's field, without the solenoid.
    """

    psi: NDArray[np.float64]
    pprime: NDArray[np.float64]
    ffprime: NDArray[np.float64]
    pressure: NDArray[np.float64]
    reports: list[BranchReport]
    status: list[FrameStatus]

    @property
    def confined(self) -> NDArray[np.bool_]:
        """Which frames currently hold a field on the confined branch."""
        return np.asarray([report.confined for report in self.reports], dtype=bool)

    def replace(
        self,
        rows: NDArray[np.bool_],
        psi: NDArray[np.float64],
        free_functions: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        reports: list[BranchReport],
    ) -> None:
        """Overwrite the chosen frames with new fields, profiles and reports."""
        self.psi = np.where(rows[:, np.newaxis, np.newaxis], psi, self.psi)
        self.pprime = np.where(rows[:, np.newaxis], free_functions[0], self.pprime)
        self.ffprime = np.where(rows[:, np.newaxis], free_functions[1], self.ffprime)
        self.pressure = np.where(rows[:, np.newaxis], free_functions[2], self.pressure)
        self.reports = [
            new if chosen else old
            for new, old, chosen in zip(reports, self.reports, rows, strict=True)
        ]


def _context(shot: Shot, devices: Sequence[jax.Device] | None) -> _Context:
    """Build the quantities that do not change from round to round."""
    geom = equilibrium.geometry(shot)
    solenoid = shot.coils.is_solenoid
    return _Context(
        geom=geom,
        machine=Machine(geom, shot.coils.r, shot.coils.z, profiles.knots()),
        grid=VesselGrid(shot.r_grid, shot.z_grid, vessel.envelope(shot.source)),
        coil_current=np.where(solenoid[np.newaxis, :], 0.0, shot.coil_current),
        plasma_current=np.asarray(shot.plasma_current, dtype=np.float64),
        solenoid_flux=_vacuum_flux(
            geom, shot.coils.r[solenoid], shot.coils.z[solenoid], shot.coil_current[:, solenoid]
        ),
        devices=devices,
    )


def _warm_rounds(
    shot: Shot,
    model: profiles.ProfileModel,
    context: _Context,
    state: _Frames,
    settings: PipelineSettings,
) -> int:
    """Run warm rounds until one gains no frame or the cap is reached; return the count."""
    rounds = 0
    while rounds < settings.max_rounds and state.confined.any():
        rounds += 1
        confined = state.confined
        source = nearest_member(np.flatnonzero(confined), shot.n_frames)
        start = state.psi[source]
        free_functions = _measured_profiles(
            shot, start, context.geom, model, context.machine.psi_n_knots
        )
        batch = FrameBatch(
            context.coil_current, free_functions[0], free_functions[1],
            context.plasma_current, start,
        )
        psi, _ = solve_across_devices(context.machine, batch, settings.warm, context.devices)
        reports = _branch_reports(
            psi + context.solenoid_flux, context.grid, settings.minimum_area
        )
        passed = np.asarray([report.confined for report in reports], dtype=bool)
        # A frame that already held a confined field gives it up only for another one.
        state.replace(passed | ~confined, psi, free_functions, reports)
        # A frame that keeps its earlier field still gains the pressure measured in it:
        # its samples were mapped through that very field, and without this a frame
        # confined by the cold pass alone would carry no pressure at all.
        kept = confined & ~passed
        state.pressure = np.where(kept[:, np.newaxis], free_functions[2], state.pressure)
        gained = passed & ~confined
        for frame in np.flatnonzero(gained):
            state.status[frame] = FrameStatus.CONFINED_WARM
        if not gained.any():
            break
    return rounds


def _fallback(context: _Context, state: _Frames, settings: PipelineSettings) -> None:
    """Give every frame still failing its neighbour's plasma flux and its own coil flux."""
    confined = state.confined
    if confined.all() or not confined.any():
        return
    failing = ~confined
    source = nearest_member(np.flatnonzero(confined), failing.size)
    coil_flux = _vacuum_flux(
        context.geom, context.machine.coil_r, context.machine.coil_z, context.coil_current
    )
    ratio = context.plasma_current / context.plasma_current[source]
    borrowed = (state.psi - coil_flux)[source] * ratio[:, np.newaxis, np.newaxis] + coil_flux
    free_functions = (
        state.pprime[source] * ratio[:, np.newaxis],
        state.ffprime[source] * ratio[:, np.newaxis],
        state.pressure[source],
    )
    reports = _branch_reports(
        borrowed + context.solenoid_flux, context.grid, settings.minimum_area
    )
    state.replace(failing, borrowed, free_functions, reports)
    for frame in np.flatnonzero(failing):
        state.status[frame] = FrameStatus.FALLBACK


def _vacuum_flux(
    geom: equilibrium.Geometry,
    coil_r: NDArray[np.float64],
    coil_z: NDArray[np.float64],
    currents: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the exact flux of a filament set for every frame, shape ``(T, NZ, NR)``."""
    r, z = jnp.asarray(coil_r), jnp.asarray(coil_z)
    per_frame = jax.vmap(lambda row: vacuum_field_si(geom.r_grid, geom.z_grid, r, z, row))
    return np.asarray(per_frame(jnp.asarray(currents)), dtype=np.float64)


_AXIS_FLUX_OF_EACH = jax.jit(jax.vmap(smooth_axis_flux))


def _axis_fluxes(fields: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the flux at the magnetic axis of every field, ``nan`` where one is not finite."""
    finite = np.all(np.isfinite(fields), axis=(1, 2))
    safe = np.where(finite[:, np.newaxis, np.newaxis], fields, 0.0)
    values = np.asarray(_AXIS_FLUX_OF_EACH(jnp.asarray(safe)), dtype=np.float64)
    return np.where(finite, values, np.nan)


def _branch_reports(
    fields: NDArray[np.float64], grid: VesselGrid, minimum_area: float
) -> list[BranchReport]:
    """Check every field of a batch."""
    axes = _axis_fluxes(fields)
    return [
        check_branch(psi, float(axis), grid, minimum_area)
        for psi, axis in zip(fields, axes, strict=True)
    ]


def _measured_profiles(
    shot: Shot,
    start_fields: NDArray[np.float64],
    geom: equilibrium.Geometry,
    model: profiles.ProfileModel,
    knots: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Map each frame's Thomson samples through its start field.

    A frame whose mapping fails — no closed surface in its start field, or no
    usable Thomson sample — borrows the profiles of the nearest frame whose
    mapping worked, scaled by the ratio of the plasma currents so that the
    current they reproduce is still its own.
    """
    shape = (shot.n_frames, knots.size)
    pprime, ffprime, pressure = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    mapped = np.zeros(shot.n_frames, dtype=bool)
    for frame in range(shot.n_frames):
        current = float(shot.plasma_current[frame])
        frame_pprime, frame_ffprime, frame_pressure = equilibrium.measured_profiles(
            shot, frame, jnp.asarray(start_fields[frame]), geom, model, knots, current
        )
        mapped[frame] = bool(
            np.all(np.isfinite(frame_pprime)) and np.all(np.isfinite(frame_ffprime))
            and np.all(np.isfinite(frame_pressure))
        )
        if mapped[frame]:
            pprime[frame], ffprime[frame], pressure[frame] = (
                frame_pprime, frame_ffprime, frame_pressure,
            )
    if not mapped.any():
        raise RuntimeError(f"{shot.name}: the Thomson mapping failed on every frame")
    if not mapped.all():
        source = nearest_member(np.flatnonzero(mapped), shot.n_frames)
        ratio = (shot.plasma_current / shot.plasma_current[source])[:, np.newaxis]
        missing = ~mapped[:, np.newaxis]
        pprime = np.where(missing, pprime[source] * ratio, pprime)
        ffprime = np.where(missing, ffprime[source] * ratio, ffprime)
        pressure = np.where(missing, pressure[source], pressure)
    return pprime, ffprime, pressure
