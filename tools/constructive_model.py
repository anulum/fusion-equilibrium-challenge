# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — fixed-pressure synthetic current profiles
"""Hold physical pressure fixed while normalizing only the FF-prime amplitude.

Flux is Wb/rad, radius is metres, pressure is Pa and current is A. The external
FreeGSNKE dependency supplies limiter-aware core detection and nonlinear solving;
the profile below does not rescale the pressure contribution to enforce current.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from os import environ
from typing import Protocol, cast

import freegs.machine  # type: ignore[import-untyped]  # Release geometry dependency.
import numpy as np
from freegs4e.machine import Wall  # type: ignore[import-untyped]
from freegs4e.shaped_coil import ShapedCoil  # type: ignore[import-untyped]
from freegsnke.equilibrium_update import Equilibrium as NativeEquilibrium  # type: ignore[import-untyped]
from freegsnke.jtor_update import (  # type: ignore[import-untyped]  # Upstream has no typing metadata.
    ConstrainPaxisIp,
)
from freegsnke.machine_update import Machine  # type: ignore[import-untyped]
from numpy.typing import NDArray

from tools.constructive_geometry import limiter_boundary

F64 = NDArray[np.float64]
MU0 = 4e-7 * np.pi


class LegacyCoil(Protocol):
    """Geometry-only boundary to the pinned release's public shaped coils."""

    shape: list[list[float]]
    turns: int
    npoints_per_triangle: int


class StaticMachine(Protocol):
    """Static coil state used by the external forward solver."""

    current_vec: F64


class Equilibrium(Protocol):
    """Fields read from an external solve; no external implementation is vendored."""

    R: F64
    Z: F64
    plasma_psi: F64
    psi_axis: float
    psi_bndry: float
    opt: F64
    xpt: F64
    tokamak: StaticMachine

    def psi(self) -> F64:
        """Return total flux in (R, Z) order."""
        ...

    def q(self, psinorm: F64) -> float:
        """Return the safety factor at a single interior flux surface."""
        ...


def make_equilibrium(size: int, initial: int, currents: F64) -> Equilibrium:
    """Build the declared legacy PF geometry with a valid static limiter.

    Parameters
    ----------
    size : int
        Grid size 2^n+1, at least 17, in each direction.
    initial : int
        Independent sine-product initial guess: 0 gives 0.1 Wb/rad, 1 gives
        0.3 Wb/rad with the sine product squared.
    currents : ndarray
        Exactly 18 frozen FC1..FC18 currents in amperes per turn.
        Original coil turns are carried unchanged; effective ampere-turns
        equal these currents multiplied by the corresponding turn count.

    Returns
    -------
    Equilibrium
        An unsolved FreeGSNKE equilibrium with static-only coil state.

    Raises
    ------
    ValueError
        Grid, initialization, currents or the release limiter are invalid.

    Notes
    -----
    The release wall retraces part of its inner vertical segment. Shapely's
    deterministic ``make_valid`` removes this invalid retracing. The result
    must remain one polygon without holes and preserve area within 1e-12 m².
    No inferred resistance, inductance, passive current or voltage is supplied;
    this constructor is for static forward calculations only.
    """
    if (
        size < 17
        or (size - 1) & (size - 2)
        or initial not in (0, 1)
        or currents.shape != (18,)
        or not np.isfinite(currents).all()
    ):
        raise ValueError("Expected a valid grid, two-start index and 18 finite coil currents")
    if environ.get("EQUILIBRIUM_PATH") is not None:
        raise ValueError("Ambient EQUILIBRIUM_PATH would replace the declared independent initial guess")
    legacy = freegs.machine.DIIID()
    boundary = limiter_boundary(np.column_stack([legacy.wall.R, legacy.wall.Z]))
    limiter = Wall(boundary[:, 0], boundary[:, 1])
    old_coils = cast(list[tuple[str, LegacyCoil]], legacy.coils)
    coils = [
        (
            label,
            ShapedCoil(
                coil.shape,
                current=float(currents[i]),
                turns=coil.turns,
                control=False,
                npoints=coil.npoints_per_triangle,
            ),
        )
        for i, (label, coil) in enumerate(old_coils)
    ]
    machine = Machine(coils, wall=limiter, limiter=limiter)
    machine.current_vec = currents.copy()
    r = np.linspace(0.1, 2.8, size)
    z = np.linspace(-1.8, 1.8, size)
    product = np.sin(np.pi * (r - 0.1) / 2.7)[:, None] * np.sin(np.pi * (z + 1.8) / 3.6)[None, :]
    guess = (0.1 if initial == 0 else 0.3) * product ** (initial + 1)
    return cast(
        Equilibrium,
        NativeEquilibrium(
            tokamak=machine,
            Rmin=0.1,
            Rmax=2.8,
            Zmin=-1.8,
            Zmax=1.8,
            nx=size,
            ny=size,
            psi=guess,
            order=2,
        ),
    )


@dataclass(frozen=True)
class Profile:
    """Physical parameters and the most recently evaluated signed flux span."""

    paxis: float = 20_000.0
    current: float = 1_000_000.0
    fvac: float = 3.4
    power: float = 2.0
    span: float = float("nan")
    amplitude: float = float("nan")

    def __post_init__(self) -> None:
        """Reject invalid physical parameters before upstream initialization."""
        values = np.asarray([self.paxis, self.current, self.fvac, self.power])
        if not np.isfinite(values).all() or min(self.paxis, self.fvac, self.power) <= 0:
            raise ValueError("Pressure, vacuum F and power must be finite and positive")
        if self.current == 0:
            raise ValueError("A nonzero signed plasma current is required")


def current_source(
    profile: Profile,
    radius: F64,
    flux: F64,
    axis: float,
    boundary: float,
    mask: F64,
    cell_area: float,
) -> tuple[F64, Profile]:
    """Evaluate the physical GS source and solve its FF-prime normalization.

    Parameters
    ----------
    profile : Profile
        Fixed pressure, current, vacuum F and FF-prime exponent.
    radius, flux, mask : ndarray
        Aligned two-dimensional radius, total flux and binary core support.
    axis, boundary : float
        Flux at the detected magnetic axis and limiter or separatrix.
    cell_area : float
        Positive rectangular grid-cell area used by the pinned upstream solver.

    Returns
    -------
    current, evaluated : ndarray, Profile
        Toroidal current density and its signed span/FF-prime amplitude.

    Raises
    ------
    ValueError
        Inputs are invalid, the core is empty or the flux span is degenerate.
    """
    if radius.ndim != 2 or flux.shape != radius.shape or mask.shape != radius.shape:
        raise ValueError("Radius, flux and support must be aligned two-dimensional arrays")
    span = boundary - axis
    if (
        not np.isfinite(radius).all()
        or not np.isfinite(flux).all()
        or not np.isfinite(mask).all()
        or np.any(radius <= 0)
        or not np.all((mask == 0) | (mask == 1))
        or not np.isfinite([span, axis, cell_area]).all()
        or abs(span) <= 1e-12
        or cell_area <= 0
    ):
        raise ValueError("Invalid radius, flux span, binary support or cell area")
    s = np.clip((flux - axis) / span, 0.0, 1.0)
    pressure_current = radius * (-3 * profile.paxis / span) * (1 - s) ** 2 * mask
    ff_shape = (1 - s) ** profile.power * mask / radius
    integral = float(np.sum(ff_shape) * cell_area)
    if integral <= 0:
        raise ValueError("The supported FF-prime shape has zero integral")
    amplitude = MU0 * (profile.current - float(np.sum(pressure_current) * cell_area)) / integral
    current = pressure_current + amplitude * ff_shape / MU0
    evaluated = Profile(profile.paxis, profile.current, profile.fvac, profile.power, span, amplitude)
    return current, evaluated


def flux_functions(profile: Profile, s: F64) -> tuple[F64, F64, F64, F64]:
    """Return pressure, p-prime, FF-prime and F on normalized flux surfaces.

    Parameters
    ----------
    profile : Profile
        A profile evaluated by ``current_source`` on an actual supported map.
    s : ndarray
        Finite normalized flux coordinates in the closed interval [0, 1].

    Returns
    -------
    pressure, pprime, ffprime, fpol : ndarray
        Physical flux functions, with the positive vacuum-F sign convention.

    Raises
    ------
    ValueError
        The profile is unevaluated, coordinates are invalid or F squared is negative.
    """
    if (
        not np.isfinite([profile.span, profile.amplitude]).all()
        or abs(profile.span) <= 1e-12
        or not np.isfinite(s).all()
        or np.any((s < 0) | (s > 1))
    ):
        raise ValueError("Evaluated finite profile and normalized flux in [0, 1] required")
    pressure = profile.paxis * (1 - s) ** 3
    pprime = -3 * profile.paxis / profile.span * (1 - s) ** 2
    ffprime = profile.amplitude * (1 - s) ** profile.power
    f_squared = profile.fvac**2 - 2 * profile.amplitude * profile.span / (profile.power + 1) * (1 - s) ** (
        profile.power + 1
    )
    if np.any(f_squared < 0) or not np.isfinite(f_squared).all():
        raise ValueError("The prescribed current profile produces invalid F squared")
    return pressure, pprime, ffprime, np.sqrt(f_squared)


class FixedPressureProfile(ConstrainPaxisIp):  # type: ignore[misc]  # Typed boundary to untyped solver.
    """Use FreeGSNKE's limiter-aware geometry with an independently normalized source."""

    physical: Profile
    limiter_core_mask: F64
    flag_limiter: bool

    def __init__(self, equilibrium: object, profile: Profile) -> None:
        """Initialize upstream geometry without solving an equilibrium."""
        super().__init__(equilibrium, profile.paxis, profile.current, profile.fvac)
        self.physical = profile

    def Jtor_part2(self, R: F64, Z: F64, psi: F64, psi_axis: float, psi_bndry: float, mask: F64) -> F64:
        """Evaluate current after upstream limiter-aware boundary detection.

        Parameters
        ----------
        R, Z, psi, mask : ndarray
            Upstream radius, height, total flux and binary core mask.
        psi_axis, psi_bndry : float
            Signed axis and boundary flux values supplied by upstream geometry.

        Returns
        -------
        ndarray
            Current density integrating to the requested Ip on the upstream grid.
        """
        area = float((R[1, 0] - R[0, 0]) * (Z[0, 1] - Z[0, 0]))
        result, self.physical = current_source(self.physical, R, psi, psi_axis, psi_bndry, mask, area)
        self.jtor = result
        self.psi_bndry = float(psi_bndry)
        return result

    def pressure(self, psinorm: F64) -> F64:
        """Return the fixed physical pressure on requested normalized surfaces."""
        return flux_functions(self.physical, np.asarray(psinorm))[0]

    def pprime(self, psinorm: F64) -> F64:
        """Return dp/dpsi, preserving the signed axis-to-boundary flux span."""
        return flux_functions(self.physical, np.asarray(psinorm))[1]

    def ffprime(self, psinorm: F64) -> F64:
        """Return F dF/dpsi with only its own amplitude adjusted to Ip."""
        return flux_functions(self.physical, np.asarray(psinorm))[2]

    def fpol(self, psinorm: F64) -> F64:
        """Return positive F, refusing profiles with negative F squared."""
        return flux_functions(self.physical, np.asarray(psinorm))[3]

    def copy(self) -> FixedPressureProfile:
        """Copy profile and geometry state without sharing mutable solver arrays."""
        return deepcopy(self)
