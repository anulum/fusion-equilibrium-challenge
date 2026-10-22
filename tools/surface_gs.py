# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — surface-GS fit support and conditioning
"""Expose support of the frozen D71b surface statistic without changing its fit.

Surfaces retain marching-squares vertices, including the repeated closing point.
No arclength resampling or rejection by condition number is performed. Rank and
source degeneracy are reported separately from inclusion in the historical ratio.
"""

from __future__ import annotations

from typing import TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from skimage.measure import find_contours

SURFACES = (0.2, 0.4, 0.6, 0.8)
MIN_PIXELS = 30


class LevelSupport(TypedDict):
    """One requested surface's selected-contour support and affine fit metadata."""

    level: float
    contour_points: int
    finite_points: int
    used: bool
    reason: str
    rank: int
    singular_values: list[float]
    condition: float | None
    rank_deficient: bool
    zero_source: bool
    residual_energy: float
    source_energy: float


class SurfaceSupport(TypedDict):
    """Versioned diagnostic with explicit full, partial and unavailable support."""

    schema: str
    requested_levels: list[float]
    used_levels: list[float]
    levels: list[LevelSupport]
    ratio: float | None
    full_support: bool
    reason: str


def _bilinear(
    field: NDArray[np.float64],
    radius: NDArray[np.float64],
    height: NDArray[np.float64],
    rq: NDArray[np.float64],
    zq: NDArray[np.float64],
) -> NDArray[np.float64]:
    fr = np.interp(rq, radius, np.arange(len(radius)))
    fz = np.interp(zq, height, np.arange(len(height)))
    r0 = np.clip(np.floor(fr).astype(int), 0, len(radius) - 2)
    z0 = np.clip(np.floor(fz).astype(int), 0, len(height) - 2)
    tr, tz = fr - r0, fz - z0
    return np.asarray(
        field[z0, r0] * (1 - tr) * (1 - tz)
        + field[z0, r0 + 1] * tr * (1 - tz)
        + field[z0 + 1, r0] * (1 - tr) * tz
        + field[z0 + 1, r0 + 1] * tr * tz,
        dtype=np.float64,
    )


def measure(
    phi: NDArray[np.float64],
    laplacian_star: NDArray[np.float64],
    radius: NDArray[np.float64],
    height: NDArray[np.float64],
    plasma: NDArray[np.bool_],
    axis_value: float,
    boundary_value: float,
) -> SurfaceSupport:
    """Measure surface support using the original D71b contours and fit weights.

    Parameters
    ----------
    phi : ndarray
        Oriented flux on a rectangular grid, with the magnetic axis a maximum.
    laplacian_star : ndarray
        Delta-star of phi on the same grid; unavailable stencil values are NaN.
    radius, height : ndarray
        Strictly increasing physical R and Z coordinates in metres.
    plasma : ndarray of bool
        Pixel mask inside this map's own boundary, as in the original statistic.
    axis_value, boundary_value : float
        This map's axis flux and mean flux on its own boundary, in the same units.

    Returns
    -------
    SurfaceSupport
        Requested and used levels, per-level counts, rank, singular values and
        condition, fit/source energies and reasons. The ratio equals the original
        sqrt(sum residual energy / sum source energy) over included surfaces,
        even for partial or rank-deficient fits. None denotes unavailable ratio.

    Raises
    ------
    ValueError
        Grid coordinates, dimensions or shapes are inconsistent.
    """
    if (
        radius.ndim != 1
        or height.ndim != 1
        or min(len(radius), len(height)) < 3
        or not np.isfinite(radius).all()
        or not np.isfinite(height).all()
        or np.any(np.diff(radius) <= 0)
        or np.any(np.diff(height) <= 0)
        or np.any(radius <= 0)
        or phi.shape != (len(height), len(radius))
        or laplacian_star.shape != phi.shape
        or plasma.shape != phi.shape
    ):
        raise ValueError("Inconsistent surface diagnostic grid")
    valid_axis = np.isfinite(axis_value - boundary_value) and axis_value > boundary_value
    reason = (
        "invalid_axis_boundary"
        if not valid_axis
        else "insufficient_plasma"
        if plasma.sum() < MIN_PIXELS
        else ""
    )
    source = np.where(np.isfinite(laplacian_star), -laplacian_star, np.nan)
    levels: list[LevelSupport] = []
    for level in SURFACES:
        support: LevelSupport = {
            "level": level,
            "contour_points": 0,
            "finite_points": 0,
            "used": False,
            "reason": reason or "no_closed_contour",
            "rank": 0,
            "singular_values": [],
            "condition": None,
            "rank_deficient": False,
            "zero_source": False,
            "residual_energy": 0.0,
            "source_energy": 0.0,
        }
        levels.append(support)
        if reason:
            continue
        value = axis_value - level * (axis_value - boundary_value)
        contours = cast(
            list[NDArray[np.float64]],
            find_contours(  # type: ignore[no-untyped-call] # The upstream contour function is unannotated.
                np.where(plasma, phi, boundary_value - abs(axis_value - boundary_value)),
                value,
            ),
        )
        best = None
        for contour in contours:
            if (
                len(contour) > 12
                and np.allclose(contour[0], contour[-1])
                and (best is None or len(contour) > len(best))
            ):
                best = contour
        if best is None:
            continue
        support["contour_points"] = len(best)
        r = np.interp(best[:, 1], np.arange(len(radius)), radius)
        z = np.interp(best[:, 0], np.arange(len(height)), height)
        y = _bilinear(source, radius, height, r, z)
        keep = np.isfinite(y)
        support["finite_points"] = int(keep.sum())
        if keep.sum() < 12:
            support["reason"] = "insufficient_finite_samples"
            continue
        design = np.column_stack((r[keep] ** 2, np.ones(keep.sum())))
        coefficients, _residuals, rank, singular = np.linalg.lstsq(design, y[keep], rcond=None)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            condition = float(singular[0] / singular[-1])
        support.update(
            {
                "used": True,
                "reason": "used",
                "rank": int(rank),
                "singular_values": singular.tolist(),
                "condition": condition if np.isfinite(condition) else None,
                "rank_deficient": bool(rank < 2),
                "residual_energy": float(np.sum((y[keep] - design @ coefficients) ** 2)),
                "source_energy": float(np.sum(y[keep] ** 2)),
            }
        )
        support["zero_source"] = support["source_energy"] == 0
    used = [s["level"] for s in levels if s["used"]]
    # Python 3.12 sum compensates float additions; retain the frozen sequential accumulation.
    numerator = denominator = 0.0
    for support in levels:
        if support["used"]:
            numerator += support["residual_energy"]
            denominator += support["source_energy"]
    return {
        "schema": "surface-gs-support.v1",
        "requested_levels": list(SURFACES),
        "used_levels": used,
        "levels": levels,
        "ratio": float(np.sqrt(numerator / denominator)) if used and denominator > 0 else None,
        "full_support": len(used) == len(SURFACES),
        "reason": reason
        or ("available" if used and denominator > 0 else "zero_source" if used else "no_usable_surface"),
    }
