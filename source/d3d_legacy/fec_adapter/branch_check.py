# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — is a solved field on the confined-plasma branch

"""Decide from the field alone whether a solve landed on the confined-plasma branch.

The free-boundary fixed-point iteration has more than one attractor. Beside the
equilibrium with a confined plasma there is a vacuum-like one, in which the
current collapses into a sliver and no flux surface of any size closes inside
the vessel. The solver's convergence flag does not tell them apart — on a cold
start it points the wrong way: on 128 flat-top frames of one DIII-D discharge
the frames that met the tolerance had a median flux R² of -1.6 against the
reconstruction and the ones that did not, +0.4.

What does tell them apart needs no reference: the area enclosed by the last
flux surface that closes inside the vessel. On the same frames it correlates
+0.93 with the flux R², and frames above one square metre have a median R² of
+0.67 against -1.54 below. The reconstruction was used to establish that
relation and is never read here; the check takes the field, the grid and the
vessel envelope.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from . import derived

MINIMUM_CONFINED_AREA: Final = 1.0
"""Square metres of poloidal cross-section below which a field is not accepted.

Fixed on DIII-D flat-top frames (shot ``00000c4a7b``, frames 88–215), where the
confined plasma's cross-section is about 1.4–2.0 m² and the collapsed branch
encloses less than 0.9 m². It is a property of that machine and that phase of
the discharge: a smaller machine, or the first frames of a current ramp, need
their own value.
"""


@dataclass(frozen=True)
class VesselGrid:
    """The reconstruction grid of one machine together with its vessel envelope.

    Attributes
    ----------
    r_grid, z_grid:
        The grid, in metres.
    envelope:
        Vessel envelope mask on that grid, shape ``(NZ, NR)``.
    """

    r_grid: NDArray[np.float64]
    z_grid: NDArray[np.float64]
    envelope: NDArray[np.bool_]


@dataclass(frozen=True)
class BranchReport:
    """What the check found in one field.

    Attributes
    ----------
    enclosed_area:
        Area inside the last closed flux surface within the vessel, in square
        metres; zero when no surface closes there.
    boundary_flux:
        Flux on that surface in webers, ``nan`` when there is none.
    confined:
        Whether the enclosed area reaches the threshold the check was given.
    """

    enclosed_area: float
    boundary_flux: float
    confined: bool


def polygon_area(contour: NDArray[np.float64]) -> float:
    """Return the area enclosed by a closed ``(N, 2)`` polygon, by the shoelace formula."""
    r, z = contour[:, 0], contour[:, 1]
    return 0.5 * abs(float(np.dot(r, np.roll(z, 1)) - np.dot(z, np.roll(r, 1))))


def check_branch(
    psi: NDArray[np.float64],
    psi_axis: float,
    vessel_grid: VesselGrid,
    minimum_area: float = MINIMUM_CONFINED_AREA,
) -> BranchReport:
    """Measure the confined cross-section of one field and compare it with a threshold.

    Parameters
    ----------
    psi:
        Flux on the reconstruction grid, shape ``(NZ, NR)``, in webers.
    psi_axis:
        Flux at the magnetic axis of ``psi``.
    vessel_grid:
        The grid ``psi`` is given on and the vessel envelope on it.
    minimum_area:
        Threshold in square metres.

    Notes
    -----
    The area is resolved only as finely as the closed-surface search scans flux
    levels (:data:`fec_adapter.derived.LEVEL_SCAN` of them between the axis and
    the far field). A confined region much smaller than the threshold can fall
    between two levels and be reported as zero; it is below the threshold
    either way, so the decision is unaffected, but the area of a very small
    plasma must not be taken from here as a measurement.

    A field with a non-finite value or a non-finite axis is reported as not
    confined with zero area rather than raised on: a batch contains such frames
    in the ordinary course of a discharge, and the caller's response to them is
    the same as to a collapsed solve.
    """
    if not (np.isfinite(psi_axis) and bool(np.all(np.isfinite(psi)))):
        return BranchReport(0.0, float("nan"), confined=False)
    found = derived.outermost_closed_surface(
        psi, vessel_grid.r_grid, vessel_grid.z_grid, psi_axis, vessel_grid.envelope
    )
    if found is None:
        return BranchReport(0.0, float("nan"), confined=False)
    level, contour = found
    area = polygon_area(contour)
    return BranchReport(area, level, confined=area >= minimum_area)
