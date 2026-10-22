# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — scored-map diagnostic report arithmetic
"""Reproduce matched-support diagnostic estimates and whole-shot intervals."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]


MAPS = ("reference", "base", "gs")


PAIRS = ((1, 2), (0, 1), (0, 2))


STRATA = ("all", "used_converged", "used_nonconverged", "unused")


RAW = (
    "outside_fixed_over_ref",
    "cp_max",
    "cp_min",
    "cp_saddle",
    "gs_surface",
    "gs_sse",
    "gs_tss",
    "gs_pixels",
    "gs_rank",
    "gs_rms_residual",
    "gs_rms_source",
    "gs_degenerate",
    "outside_pixels",
    "bp_error_ref",
    "bp_error_envelope",
    "interior_nodes",
    "interior_abs_current",
    "crossing_nodes",
    "crossing_abs_current",
    "exterior_nodes",
    "exterior_abs_current",
)


METRICS = (
    *RAW,
    "gs_rms_norm",
    *(
        f"{name}_over_refinside"
        for name in (
            "interior_abs_current",
            "crossing_abs_current",
            "exterior_abs_current",
        )
    ),
)


class Pair(TypedDict):
    """Two map means and their difference on exactly the same finite frames."""

    frames: int
    means: list[float] | None
    change: float | None


class Entry(TypedDict):
    """Independent per-map levels and explicitly separate matched comparisons."""

    counts: list[int]
    means: list[float | None]
    pairs: dict[str, Pair]


class Shot(TypedDict):
    """Sufficient statistics retaining every original shot and explanatory stratum."""

    stem: str
    frames: int
    reference_frames: int
    stratum_frames: dict[str, int]
    statistics: dict[str, dict[str, Entry]]


def bootstrap(values: F64, weights: NDArray[np.int64]) -> dict[str, object]:
    """Renormalise fixed multinomial whole-shot weights over available shot means.

    Parameters
    ----------
    values : ndarray
        Per-shot matched changes; NaN denotes no common support in that shot.
    weights : ndarray of int64
        Nonnegative whole-shot counts with one column per original identity.

    Returns
    -------
    dict of str to object
        Equal-shot estimate, conditional 99 percent interval and effective draws.

    Raises
    ------
    ValueError
        Weights or changes do not describe the same nonempty original cohort.
    """
    if values.ndim != 1:
        raise ValueError("Invalid original-cohort multinomial weights")
    return _Resampler(weights, len(values))(values)


class _Resampler:
    """Reuse original-cohort weights and counts without changing any reductions."""

    def __init__(self, weights: NDArray[np.int64], size: int) -> None:
        """Validate the shared multinomial matrix once and retain its float conversion."""
        if (
            not size
            or weights.ndim != 2
            or not len(weights)
            or weights.shape[1] != size
            or not np.issubdtype(weights.dtype, np.integer)
            or np.any(weights < 0)
            or np.any(weights.sum(axis=1) != size)
        ):
            raise ValueError("Invalid original-cohort multinomial weights")
        self.weights = weights
        self.float_weights = weights.astype(np.float64)
        self.counts: dict[bytes, NDArray[np.int64]] = {}

    def __call__(self, values: F64) -> dict[str, object]:
        """Keep each pair's support, weighted reductions and quantiles separate."""
        if np.isinf(values).any():
            raise ValueError("Invalid original-cohort multinomial weights")
        finite = np.isfinite(values)
        key = finite.tobytes()
        if key not in self.counts:
            self.counts[key] = self.weights @ finite.astype(np.int64)
        counts = self.counts[key]
        total = self.float_weights @ np.where(finite, values, 0.0)
        boot = np.divide(total, counts, out=np.full(len(self.weights), np.nan), where=counts > 0)
        available = np.isfinite(boot)
        return {
            "mean_change": float(np.mean(values[finite])) if finite.any() else None,
            "ci99": [float(x) for x in np.quantile(boot[available], [0.005, 0.995])]
            if available.any()
            else None,
            "shots": int(finite.sum()),
            "finite_draws": int(available.sum()),
            "unavailable_draws": int((~available).sum()),
            "sampled_nonempty_histogram": [int(x) for x in np.bincount(counts, minlength=len(values) + 1)],
        }


def summarise(shots: Sequence[Shot], expected_stems: Sequence[str]) -> dict[str, object]:
    """Report all prespecified pairs and strata using one shared set of weights.

    Parameters
    ----------
    shots : Sequence of Shot
        Ordered source-verified shot aggregates.
    expected_stems : Sequence of str
        Independently bound cohort order; qualification remains a separate scope.

    Returns
    -------
    dict of str to object
        Raw/normalised levels, applicability, matched changes and 99 percent intervals.

    Raises
    ------
    ValueError
        Shot identities are absent, repeated or out of order.
    """
    n = len(shots)
    if not n or [shot["stem"] for shot in shots] != list(expected_stems) or len(set(expected_stems)) != n:
        raise ValueError("Original shot order differs")
    weights = np.random.default_rng(23023).multinomial(n, np.full(n, 1 / n), size=20000)
    resampler = _Resampler(weights, n)
    strata: dict[str, object] = {}
    for stratum in STRATA:
        measures: dict[str, object] = {}
        for name in METRICS:
            selected = [shot["statistics"][stratum][name] for shot in shots]
            means: list[float | None] = []
            for j in range(3):
                present = [value for item in selected if (value := item["means"][j]) is not None]
                means.append(float(np.mean(present)) if present else None)
            pairs: dict[str, object] = {}
            for a, b in PAIRS:
                label = f"{MAPS[a]}->{MAPS[b]}"
                chosen = [item["pairs"][label] for item in selected]
                changes = np.array([np.nan if item["change"] is None else item["change"] for item in chosen])
                pair_means = [pair_value for item in chosen if (pair_value := item["means"]) is not None]
                pairs[label] = {
                    "frames": sum(item["frames"] for item in chosen),
                    "matched_means": [float(x) for x in np.mean(pair_means, axis=0)] if pair_means else None,
                    **resampler(changes),
                }
            measures[name] = {
                "counts": [sum(item["counts"][j] for item in selected) for j in range(3)],
                "means": means,
                "pairs": pairs,
            }
        strata[stratum] = {
            "frames": sum(shot["stratum_frames"][stratum] for shot in shots),
            "measures": measures,
        }
    return {
        "shots": n,
        "frames": sum(shot["frames"] for shot in shots),
        "reference_frames": sum(shot["reference_frames"] for shot in shots),
        "maps": list(MAPS),
        "seed": 23023,
        "draws": 20000,
        "bootstrap": "fixed original-cohort multinomial weights; drop unavailable and renormalise",
        "strata": strata,
        "physical_force_balance_certified": False,
    }
