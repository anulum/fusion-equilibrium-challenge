# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — scored-field physical remeasurement
"""Remeasure the scored D72 fields with the original contour and region definitions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from tools.scored_map_bundle import F64, Bundle, Case, Mask, ShotData, load_shot
from tools.scored_map_operators import load as load_operators
from tools.scored_map_summary import MAPS, RAW, Shot, extract
from tools.surface_gs import SurfaceSupport
from tools.surface_gs import measure as surface_measure

REASONS = (
    "used",
    "invalid_axis_boundary",
    "insufficient_plasma",
    "no_closed_contour",
    "insufficient_finite_samples",
)
Erode = Callable[[Mask, int], Mask]
Axis = Callable[[F64, F64, F64, Mask], tuple[float, float, int, int, bool]]
Bilinear = Callable[[F64, F64, F64, F64, F64], F64]


@dataclass(frozen=True)
class Measurement:
    """Original frame diagnostics and the additional surface-support census."""

    values: F64
    reference_inside: F64
    support: dict[str, NDArray[np.generic]]
    statistics: Shot
    used: Mask
    converged: Mask


def field_error(pr: F64, pz: F64, rr: F64, rz: F64, region: Mask, radius: F64) -> float:
    """Compute the original reference-normalised field error on a fixed region.

    Parameters
    ----------
    pr, pz, rr, rz : ndarray
        Candidate and reference first derivatives on the same grid.
    region : ndarray of bool
        Explicit comparison support, independent of the candidate values.
    radius : ndarray
        Original radial coordinates in metres.

    Returns
    -------
    float
        Original dimensionless field ratio; NaN for fewer than 30 nodes.
    """
    use = region & np.isfinite(pr) & np.isfinite(rr)
    if use.sum() < 30:
        return float("nan")
    r = np.broadcast_to(radius[None, :], pr.shape)[use]
    diff = ((pr - rr) ** 2 + (pz - rz) ** 2)[use] / r**2
    norm = (rr**2 + rz**2)[use] / r**2
    return float(np.sqrt(diff.sum() / norm.sum()))


def stencil_classes(replaced: Mask) -> dict[str, Mask]:
    """Partition available five-point stencils against the original replaced pixels.

    Parameters
    ----------
    replaced : ndarray of bool
        Original D72 replacement domain on the full grid.

    Returns
    -------
    dict of str to ndarray
        Interior, crossing and exterior node masks; edge nodes have no stencil.
    """
    inner = np.zeros_like(replaced)
    inner[1:-1, 1:-1] = True
    c = replaced[1:-1, 1:-1]
    nb = (replaced[1:-1, 2:], replaced[1:-1, :-2], replaced[2:, 1:-1], replaced[:-2, 1:-1])
    every, anyof = np.zeros_like(replaced), np.zeros_like(replaced)
    every[1:-1, 1:-1] = c & nb[0] & nb[1] & nb[2] & nb[3]
    anyof[1:-1, 1:-1] = c | nb[0] | nb[1] | nb[2] | nb[3]
    return {"interior": every, "crossing": inner & anyof & ~every, "exterior": inner & ~anyof}


def measure(bundle: Bundle, data: ShotData) -> Measurement:
    """Remeasure one validated shot, preserving original frame and contour support.

    Parameters
    ----------
    bundle : Bundle
        Original fixed grid, sign, population and input identity.
    data : ShotData
        Checked float32 fields and original contour/convergence data.

    Returns
    -------
    Measurement
        All 21 raw metrics, current denominator, surface metadata and statistics.

    Raises
    ------
    ValueError, FloatingPointError, AssertionError
        Field/domain invariants, old/new surface parity or numerical range fail.
    """
    if data.case not in bundle.cases:
        raise ValueError("Shot is not a member of this package")
    ops = load_operators()
    physics, corrected, kit = ops.physics, ops.corrected, ops.kit
    erode, axis, bilinear = (
        cast(Erode, kit["erode"]),
        cast(Axis, kit["find_o_point"]),
        cast(Bilinear, kit["bilinear"]),
    )
    radius, height, coarse = bundle.grid.radius, bundle.grid.height, bundle.grid.envelope
    sign, n = float(bundle.grid.axis_sign), data.case.frames
    values = {name: np.full((n, 3), np.nan) for name in RAW}
    reference_inside = np.full(n, np.nan)
    support: dict[str, NDArray[np.generic]] = {
        "used_levels": np.zeros((n, 3, 4), dtype=bool),
        "full_support": np.zeros((n, 3), dtype=bool),
        **{
            key: np.zeros((n, 3, 4), dtype=np.int64)
            for key in ("contour_points", "finite_points", "rank", "reason")
        },
        "condition": np.full((n, 3, 4), np.nan),
        "singular_values": np.full((n, 3, 4, 2), np.nan),
        "zero_source": np.zeros((n, 3, 4), dtype=bool),
        "rank_deficient": np.zeros((n, 3, 4), dtype=bool),
    }
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for k in range(n):
            ref_c = data.contours[data.offsets[k * 3] : data.offsets[k * 3 + 1]]
            maps = {name: np.asarray(data.fields[j][k], dtype=np.float64) for j, name in enumerate(MAPS)}
            owns = {name: data.contour(k, j) for j, name in enumerate(MAPS)}
            union = np.zeros(coarse.shape, dtype=bool)
            for contour in owns.values():
                union |= physics.inside(contour, radius, height, kit["points_in_polygon"])
            region = erode(coarse, physics.EDGE_CELLS) & ~physics.dilate(union, 1)
            ref_mask = physics.inside(ref_c, radius, height, kit["points_in_polygon"])
            ref_inside = float(corrected.density(maps["reference"], radius, height, sign)[ref_mask].sum())
            reference_inside[k] = ref_inside
            rr, rz, _ = physics.derivatives(sign * maps["reference"], radius, height)
            replaced = physics.inside(owns["base"], radius, height, kit["points_in_polygon"]).copy()
            replaced[0, :] = replaced[-1, :] = replaced[:, 0] = replaced[:, -1] = False
            if data.used[k] and not np.array_equal(maps["base"][~replaced], maps["gs"][~replaced]):
                raise ValueError("Projected map changed outside the original replacement set")
            if not data.used[k] and not np.array_equal(maps["base"], maps["gs"]):
                raise ValueError("An unused projection frame changed")
            classes = stencil_classes(replaced)
            eroded = erode(coarse, physics.EDGE_CELLS)
            currents: dict[str, F64] = {}
            for j, name in enumerate(MAPS):
                measured = corrected.measure_b(
                    maps[name],
                    radius,
                    height,
                    owns[name],
                    ref_c,
                    sign,
                    coarse,
                    kit,
                    region,
                    ref_inside,
                )
                for metric in RAW[:13]:
                    values[metric][k, j] = measured[metric]
                pr, pz, gs = physics.derivatives(sign * maps[name], radius, height)
                values["bp_error_ref"][k, j] = field_error(pr, pz, rr, rz, ref_mask, radius)
                values["bp_error_envelope"][k, j] = field_error(pr, pz, rr, rz, eroded, radius)
                currents[name] = np.where(np.isfinite(gs), np.abs(gs) / radius[None, :], 0.0)
                for cname, mask in classes.items():
                    values[f"{cname}_nodes"][k, j] = float(mask.sum())
                    values[f"{cname}_abs_current"][k, j] = float(currents[name][mask].sum())
                own = owns[name]
                plasma = physics.inside(own, radius, height, kit["points_in_polygon"])
                axis_value = boundary_value = float("nan")
                if own is not None:
                    _, _, iz, ir, _ = axis(sign * maps[name], radius, height, coarse)
                    axis_value = float(sign * maps[name][iz, ir])
                    boundary_value = float(
                        np.mean(bilinear(sign * maps[name], radius, height, own[:, 0], own[:, 1]))
                    )
                diagnostic: SurfaceSupport = surface_measure(
                    sign * maps[name],
                    gs,
                    radius,
                    height,
                    plasma,
                    axis_value,
                    boundary_value,
                )
                ratio = diagnostic["ratio"]
                old = values["gs_surface"][k, j]
                np.testing.assert_equal(
                    old,
                    np.nan if ratio is None else ratio,
                    err_msg="Surface support changed the original surface statistic",
                )
                support["full_support"][k, j] = diagnostic["full_support"]
                for level, entry in enumerate(diagnostic["levels"]):
                    support["used_levels"][k, j, level] = entry["used"]
                    for key, value in (
                        ("contour_points", entry["contour_points"]),
                        ("finite_points", entry["finite_points"]),
                        ("rank", entry["rank"]),
                        ("zero_source", entry["zero_source"]),
                        ("rank_deficient", entry["rank_deficient"]),
                    ):
                        support[key][k, j, level] = value
                    support["reason"][k, j, level] = REASONS.index(entry["reason"])
                    if entry["condition"] is not None:
                        support["condition"][k, j, level] = entry["condition"]
                    if entry["singular_values"]:
                        support["singular_values"][k, j, level] = entry["singular_values"]
            np.testing.assert_array_equal(
                currents["base"][classes["exterior"]],
                currents["gs"][classes["exterior"]],
                err_msg="Exterior-stencil currents changed",
            )
    raw = np.stack([values[name] for name in RAW])
    stats = extract(
        data.case.stem, raw, RAW, n, reference_inside, data.used.astype(float), data.converged.astype(float)
    )
    return Measurement(raw, reference_inside, support, stats, data.used.copy(), data.converged.copy())


def measure_case(bundle: Bundle, case: Case) -> Measurement:
    """Load and measure one checked case in the caller's selected process.

    Parameters
    ----------
    bundle : Bundle
        Checked grid, conventions and ordered source population.
    case : Case
        One source case belonging to that bundle.

    Returns
    -------
    Measurement
        Frame diagnostics, support and original projection flags.
    """
    return measure(bundle, load_shot(bundle, case))
