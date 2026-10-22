# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — shot-level surface-support reporting
"""Reproduce equal-shot surface-support means and conditional bootstrap intervals."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

REASONS = (
    "missing_reference",
    "used",
    "invalid_axis_boundary",
    "insufficient_plasma",
    "no_closed_contour",
    "insufficient_finite_samples",
)


class Points(TypedDict):
    """Included contours' sample counts; no distribution quantiles are recoverable."""

    count: int
    total: int
    minimum: int | None
    maximum: int | None


class Level(TypedDict):
    """Support counts and included-contour sizes for one requested level."""

    used: int
    rank_deficient: int
    condition_unavailable: int
    condition_range: list[float] | None
    rank_histogram: list[int]
    zero_source: int
    reasons: dict[str, int]
    vertices: Points
    finite_points: Points


class Map(TypedDict):
    """One map's measured and unavailable frames, preserving their distinct meanings."""

    measured: int
    missing_reference: int
    finite_ratio: int
    used_level_histogram: list[int]
    levels: list[Level]


class Comparison(TypedDict):
    """Three map means and their paired change on exactly the same frames."""

    frames: int
    raw_means: list[float] | None
    excess_means: list[float] | None
    change: float | None


class Historical(TypedDict):
    """Original per-map support, distinguished from original matched-pair support."""

    counts: list[int]
    excess_means: list[float | None]
    paired_frames: int
    paired_change: float | None


class Shot(TypedDict):
    """One shot's support and sufficient statistics in frozen map and level order."""

    stem: str
    frames: int
    reference_valid: int
    maps: list[Map]
    historical: Historical
    common: Comparison
    full: Comparison


class Bootstrap(TypedDict):
    """An equal-shot estimate and availability of every fixed bootstrap draw."""

    mean_change: float | None
    ci99: list[float] | None
    supported_shots: int
    draws: int
    finite_draws: int
    unavailable_draws: int
    sampled_nonempty_histogram: list[int]


def resample(values: NDArray[np.float64], draws: NDArray[np.int64]) -> Bootstrap:
    """Drop unavailable sampled shots and renormalise each fixed whole-shot draw.

    Parameters
    ----------
    values : ndarray
        One paired frame-mean change per original shot; NaN for empty support.
    draws : ndarray of int64
        Fixed whole-cohort indices, one row per draw and one column per shot.
        The complete report uses 20 000 draws of 358 identities, seed 25025.

    Returns
    -------
    Bootstrap
        Equal-shot mean, 99 percent interval over finite draws, and counts of
        sampled nonempty shots. A draw with no supported shot is not redrawn.

    Raises
    ------
    ValueError
        Values or draw dimensions are invalid, or an index is outside the cohort.
    """
    if (
        values.ndim != 1
        or not len(values)
        or np.isinf(values).any()
        or draws.ndim != 2
        or not len(draws)
        or draws.shape[1] != len(values)
        or not np.issubdtype(draws.dtype, np.integer)
        or np.any(draws < 0)
        or np.any(draws >= len(values))
    ):
        raise ValueError("Invalid whole-shot values or draws")
    sampled = values[draws]
    available = np.isfinite(sampled)
    counts = available.sum(axis=1)
    total = np.where(available, sampled, 0.0).sum(axis=1)
    boot = np.divide(total, counts, out=np.full(len(draws), np.nan), where=counts > 0)
    finite = np.isfinite(boot)
    supported = np.isfinite(values)
    return {
        "mean_change": float(values[supported].mean()) if supported.any() else None,
        "ci99": [float(x) for x in np.quantile(boot[finite], [0.005, 0.995])] if finite.any() else None,
        "supported_shots": int(supported.sum()),
        "draws": len(draws),
        "finite_draws": int(finite.sum()),
        "unavailable_draws": int((~finite).sum()),
        "sampled_nonempty_histogram": [int(x) for x in np.bincount(counts, minlength=len(values) + 1)],
    }


def _points(entries: Sequence[Points]) -> Points:
    minimum = [value for item in entries if (value := item["minimum"]) is not None]
    maximum = [value for item in entries if (value := item["maximum"]) is not None]
    return {
        "count": sum(item["count"] for item in entries),
        "total": sum(item["total"] for item in entries),
        "minimum": min(minimum) if minimum else None,
        "maximum": max(maximum) if maximum else None,
    }


def _map(entries: Sequence[Map]) -> Map:
    levels: list[Level] = []
    for q in range(4):
        selected = [item["levels"][q] for item in entries]
        conditions = [value for item in selected if (value := item["condition_range"]) is not None]
        levels.append(
            {
                "used": sum(item["used"] for item in selected),
                "rank_deficient": sum(item["rank_deficient"] for item in selected),
                "condition_unavailable": sum(item["condition_unavailable"] for item in selected),
                "condition_range": [min(x[0] for x in conditions), max(x[1] for x in conditions)]
                if conditions
                else None,
                "rank_histogram": [
                    sum(item["rank_histogram"][rank] for item in selected) for rank in range(3)
                ],
                "zero_source": sum(item["zero_source"] for item in selected),
                "reasons": {reason: sum(item["reasons"][reason] for item in selected) for reason in REASONS},
                "vertices": _points([item["vertices"] for item in selected]),
                "finite_points": _points([item["finite_points"] for item in selected]),
            }
        )
    return {
        "measured": sum(item["measured"] for item in entries),
        "missing_reference": sum(item["missing_reference"] for item in entries),
        "finite_ratio": sum(item["finite_ratio"] for item in entries),
        "used_level_histogram": [sum(item["used_level_histogram"][k] for item in entries) for k in range(5)],
        "levels": levels,
    }


def _group(
    shots: Sequence[Shot], name: Literal["common", "full"], draws: NDArray[np.int64]
) -> dict[str, object]:
    selected = [shot[name] for shot in shots]
    changes = np.array([np.nan if item["change"] is None else item["change"] for item in selected])
    raw = [value for item in selected if (value := item["raw_means"]) is not None]
    excess = [value for item in selected if (value := item["excess_means"]) is not None]
    return {
        "frames": sum(item["frames"] for item in selected),
        "shots": len(raw),
        "raw_means": [float(x) for x in np.mean(raw, axis=0)] if raw else None,
        "excess_means": [float(x) for x in np.mean(excess, axis=0)] if excess else None,
        "change": resample(changes, draws),
    }


def summarise(shots: Sequence[Shot], cohort_stems: Sequence[str]) -> dict[str, object]:
    """Report sufficient statistics using fixed shared whole-cohort bootstrap draws.

    Parameters
    ----------
    shots : Sequence of Shot
        Ordered shot aggregates from the validated cohort. A qualification cohort
        remains explicitly distinct from the complete 358-shot publication.
    cohort_stems : Sequence of str
        Expected ordered identities from the independently bound cohort.

    Returns
    -------
    dict of str to object
        Counts, historical point estimates and common/full-support comparisons.
        Bootstrap seed is 25025, with 20 000 equal-shot draws of original identities.

    Raises
    ------
    ValueError
        No shots are supplied, identities are duplicated, or their order differs.
    """
    if not shots:
        raise ValueError("No shots for support report")
    if [shot["stem"] for shot in shots] != list(cohort_stems) or len(set(cohort_stems)) != len(cohort_stems):
        raise ValueError("Shot order differs from the bound cohort")
    draws = np.random.default_rng(25025).integers(0, len(shots), size=(20000, len(shots)))
    historical: dict[str, object] = {}
    for j, name in enumerate(("reference", "smoothed", "margin")):
        values = [value for shot in shots if (value := shot["historical"]["excess_means"][j]) is not None]
        historical[name] = {
            "frames": sum(shot["historical"]["counts"][j] for shot in shots),
            "shots": len(values),
            "shot_mean": float(np.mean(values)) if values else None,
        }
    changes = [value for shot in shots if (value := shot["historical"]["paired_change"]) is not None]
    historical["smoothed->margin"] = {
        "frames": sum(shot["historical"]["paired_frames"] for shot in shots),
        "shots": len(changes),
        "mean_change": float(np.mean(changes)) if changes else None,
    }
    return {
        "shots": len(shots),
        "frames": sum(shot["frames"] for shot in shots),
        "reference_valid": sum(shot["reference_valid"] for shot in shots),
        "seed": 25025,
        "draws": 20000,
        "maps": [_map([shot["maps"][j] for shot in shots]) for j in range(10)],
        "historical": historical,
        "common": _group(shots, "common", draws),
        "full": _group(shots, "full", draws),
        "reference_valid_frames_removed_by_full_support": sum(
            shot["common"]["frames"] - shot["full"]["frames"] for shot in shots
        ),
        "frame_quantiles_reproduced": False,
        "private_maps_remeasured": False,
    }
