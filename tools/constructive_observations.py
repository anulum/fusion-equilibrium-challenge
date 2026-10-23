# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — synthetic kinetic observation comparison
"""Compare fixed Thomson-like channels under a persistent calibration model.

The tolerances are declared synthetic assumptions, not measured DIII-D noise.
Static repetition with a fixed channel bias describes stationary compatibility,
not observability through ignition, transport or a known discharge history.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import (  # type: ignore[import-untyped]  # Upstream has no typing metadata.
    RectBivariateSpline,
    RegularGridInterpolator,
)

F64 = NDArray[np.float64]
ELECTRON_CHARGE = 1.602176634e-19
DENSITY_AXIS = 3e19
PRESSURE_AXIS = 20_000.0
TEMPERATURE_AXIS = PRESSURE_AXIS / (3 * ELECTRON_CHARGE * DENSITY_AXIS)
RELATIVE_ALLOWANCES = (0.0, 0.005, 0.01, 0.02, 0.05)


def channels() -> F64:
    """Return all 49 radial and 49 vertical synthetic channel coordinates in metres."""
    radial = np.column_stack([np.linspace(0.8, 2.5, 49), np.zeros(49)])
    vertical = np.column_stack([np.full(49, 1.7), np.linspace(-1.4, 1.4, 49)])
    return np.vstack([radial, vertical])


def observe(r: F64, z: F64, flux: F64, mask: F64, axis: float, boundary: float) -> F64:
    """Sample fixed density and temperature profiles on every declared channel.

    Parameters
    ----------
    r, z : ndarray
        Strictly increasing uniform one-dimensional coordinates in metres.
    flux, mask : ndarray
        Total flux and binary axis-connected support, in (R, Z) order.
    axis, boundary : float
        Signed axis and boundary flux in Wb/rad.

    Returns
    -------
    ndarray
        98 densities in m^-3 followed by 98 electron temperatures in eV.

    Raises
    ------
    ValueError
        The map, grid or channel support is invalid.
    """
    points = channels()
    if (
        r.ndim != 1
        or z.ndim != 1
        or min(len(r), len(z)) < 4
        or flux.shape != (len(r), len(z))
        or mask.shape != flux.shape
        or not all(np.isfinite(a).all() for a in (r, z, flux, mask))
        or np.any(np.diff(r) <= 0)
        or np.any(np.diff(z) <= 0)
        or not np.all((mask == 0) | (mask == 1))
        or not np.isfinite([axis, boundary]).all()
        or abs(boundary - axis) <= 1e-12
        or points[:, 0].min() < r[0]
        or points[:, 0].max() > r[-1]
        or points[:, 1].min() < z[0]
        or points[:, 1].max() > z[-1]
    ):
        raise ValueError("Invalid map, grid, flux span or channel coverage")
    sampled = np.asarray(RectBivariateSpline(r, z, flux).ev(points[:, 0], points[:, 1]))
    supported = np.asarray(RegularGridInterpolator((r, z), mask, method="nearest")(points)) > 0.5
    s = (sampled - axis) / (boundary - axis)
    if np.any(supported & (s < -1e-10)):
        raise ValueError("A supported channel lies beyond the declared magnetic-axis flux")
    core = supported & (s <= 1)
    distance = np.where(core, 1 - np.clip(s, 0, 1), 0)
    return np.concatenate([DENSITY_AXIS * distance, TEMPERATURE_AXIS * distance**2])


@dataclass(frozen=True)
class CalibrationComparison:
    """All-channel calibration bounds for a pair, with no channel selection."""

    midpoint: F64
    reference_bias: F64
    candidate_bias: F64
    halfwidth: F64
    minimum_relative: float
    minimum_relative_without_floor: float
    primary_compatible: bool
    sensitivity: tuple[bool, ...]


def minimum_allowance(half_difference: F64, midpoint: F64, floor: F64) -> float:
    """Return the smallest common relative halfwidth with explicit zero handling.

    Parameters
    ----------
    half_difference, midpoint, floor : ndarray
        Nonnegative half differences, shared observations and absolute allowances.

    Returns
    -------
    float
        Minimum relative allowance, infinity when a positive excess has zero mean.

    Raises
    ------
    ValueError
        Arrays are empty, misaligned, nonfinite or contain negative magnitudes.
    """
    if (
        half_difference.ndim != 1
        or not half_difference.size
        or midpoint.shape != half_difference.shape
        or floor.shape != half_difference.shape
        or not all(np.isfinite(x).all() for x in (half_difference, midpoint, floor))
        or np.any(half_difference < 0)
        or np.any(floor < 0)
    ):
        raise ValueError("Aligned finite calibration arrays with nonnegative magnitudes required")
    excess = np.maximum(0.0, half_difference - floor)
    denominator = np.abs(midpoint)
    if np.any((denominator == 0) & (excess > 0)):
        return float("inf")
    ratio = np.divide(excess, denominator, out=np.zeros_like(excess), where=denominator != 0)
    return float(np.max(ratio))


def compare_observations(reference: F64, candidate: F64) -> CalibrationComparison:
    """Construct shared observations and persistent channel biases for a pair.

    Parameters
    ----------
    reference, candidate : ndarray
        All 196 nonnegative density/temperature observations in declared order.

    Returns
    -------
    CalibrationComparison
        Midpoint, signed fixed biases, primary halfwidth and sensitivity results.

    Raises
    ------
    ValueError
        Input observations are not aligned finite nonnegative channel vectors.
    """
    if (
        reference.shape != (196,)
        or candidate.shape != reference.shape
        or not np.isfinite(reference).all()
        or not np.isfinite(candidate).all()
        or np.any(reference < 0)
        or np.any(candidate < 0)
    ):
        raise ValueError("All 196 finite nonnegative channel observations are required")
    scale = np.concatenate([np.full(98, DENSITY_AXIS), np.full(98, TEMPERATURE_AXIS)])
    midpoint = (reference + candidate) / 2
    difference = np.abs(reference - candidate) / 2
    floor = 0.001 * scale
    minimum = minimum_allowance(difference, midpoint, floor)
    sensitivity = tuple(bool(np.all(difference <= a * midpoint + floor)) for a in RELATIVE_ALLOWANCES)
    return CalibrationComparison(
        midpoint,
        midpoint - reference,
        midpoint - candidate,
        0.02 * midpoint + floor,
        minimum,
        minimum_allowance(difference, midpoint, np.zeros_like(floor)),
        sensitivity[3],
        sensitivity,
    )
