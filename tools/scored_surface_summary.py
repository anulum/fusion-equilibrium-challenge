# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — scored-map surface support statistics
"""Report float32 scored-map support without redefining the historical GS ratio."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from tools.scored_map_summary import MAPS, PAIRS, STRATA, Entry, bootstrap, frame_statistics
from tools.scored_map_table import compare
from tools.surface_gs import SURFACES

REASONS = (
    "used",
    "invalid_axis_boundary",
    "insufficient_plasma",
    "no_closed_contour",
    "insufficient_finite_samples",
)
F64 = NDArray[np.float64]
Mask = NDArray[np.bool_]


class Range(TypedDict):
    """Finite sample support and extrema; no pooled quantiles are inferred."""

    count: int
    minimum: float | None
    maximum: float | None


class Level(TypedDict):
    """One level's counts, ranks, conditions and sampling diagnostics."""

    used: int
    ranks: list[int]
    reasons: list[int]
    zero_source: int
    rank_deficient: int
    contour_points: Range
    finite_points: Range
    condition: Range


class Map(TypedDict):
    """All requested levels and full-support counts for one scored map."""

    full: int
    used_histogram: list[int]
    levels: list[Level]


class Stratum(TypedDict):
    """Original frame group with pair-specific common and full-surface support."""

    frames: int
    common: Entry
    full: Entry


class Shot(TypedDict):
    """Sufficient statistics of one original shot's exact scored-map measurements."""

    stem: str
    frames: int
    maps: list[Map]
    strata: dict[str, Stratum]


def _range(values: NDArray[np.generic]) -> Range:
    finite = values[np.isfinite(values)]
    return {
        "count": len(finite),
        "minimum": float(finite.min()) if len(finite) else None,
        "maximum": float(finite.max()) if len(finite) else None,
    }


def extract(
    stem: str, ratios: F64, support: dict[str, NDArray[np.generic]], used: Mask, converged: Mask
) -> Shot:
    """Reduce measured surface metadata while preserving paired frame support.

    Parameters
    ----------
    stem : str
        Original shot identity.
    ratios : ndarray
        Original surface-GS ratios, shape (frames, reference/base/gs).
    support : dict of str to ndarray
        Exact output of the scored-map measurement kernel, without resampling.
    used, converged : ndarray of bool
        Original D72 replacement and pre-conversion convergence flags.

    Returns
    -------
    Shot
        Counts, fit extrema and paired common/full-support shot means.

    Raises
    ------
    ValueError
        Dimensions, level identities, ranks or recorded support disagree.
    """
    if any(
        support[key].dtype != np.dtype(bool)
        for key in ("used_levels", "full_support", "zero_source", "rank_deficient")
    ) or any(
        support[key].dtype != np.dtype(np.int64)
        for key in ("rank", "reason", "contour_points", "finite_points")
    ):
        raise ValueError("Scored-surface metadata dtypes differ")
    n = len(ratios)
    rank = cast(NDArray[np.int64], support["rank"])
    reason = cast(NDArray[np.int64], support["reason"])
    level_used = support["used_levels"].astype(bool)
    full = support["full_support"].astype(bool)
    if (
        not n
        or ratios.shape != (n, 3)
        or used.shape != (n,)
        or converged.shape != (n,)
        or level_used.shape != (n, 3, 4)
        or full.shape != (n, 3)
        or not np.array_equal(full, level_used.all(axis=2))
        or np.any(converged & ~used)
        or any(
            support[key].shape != (n, 3, 4)
            for key in (
                "rank",
                "reason",
                "condition",
                "contour_points",
                "finite_points",
                "zero_source",
                "rank_deficient",
            )
        )
        or np.any((rank < 0) | (rank > 2))
        or np.any((reason < 0) | (reason >= len(REASONS)))
        or not np.array_equal(level_used, support["reason"] == 0)
    ):
        raise ValueError("Scored-surface dimensions or support differ")
    maps: list[Map] = []
    for j in range(3):
        levels: list[Level] = []
        for q in range(4):
            keep = level_used[:, j, q]
            levels.append(
                {
                    "used": int(keep.sum()),
                    "ranks": np.bincount(support["rank"][:, j, q].astype(np.int64), minlength=3).tolist(),
                    "reasons": np.bincount(support["reason"][:, j, q].astype(np.int64), minlength=5).tolist(),
                    "zero_source": int(support["zero_source"][:, j, q].sum()),
                    "rank_deficient": int(support["rank_deficient"][:, j, q].sum()),
                    "contour_points": _range(support["contour_points"][keep, j, q]),
                    "finite_points": _range(support["finite_points"][keep, j, q]),
                    "condition": _range(support["condition"][keep, j, q]),
                }
            )
        maps.append(
            {
                "full": int(full[:, j].sum()),
                "used_histogram": np.bincount(level_used[:, j].sum(axis=1), minlength=5).tolist(),
                "levels": levels,
            }
        )
    groups = {
        "all": np.ones(n, dtype=bool),
        "used_converged": used & converged,
        "used_nonconverged": used & ~converged,
        "unused": ~used,
    }
    restricted = np.where(full, ratios, np.nan)
    return {
        "stem": stem,
        "frames": n,
        "maps": maps,
        "strata": {
            name: {
                "frames": int(mask.sum()),
                "common": frame_statistics(ratios, mask),
                "full": frame_statistics(restricted, mask),
            }
            for name, mask in groups.items()
        },
    }


def _combine_ranges(entries: Sequence[Range]) -> Range:
    lo = [x["minimum"] for x in entries if x["minimum"] is not None]
    hi = [x["maximum"] for x in entries if x["maximum"] is not None]
    return {
        "count": sum(x["count"] for x in entries),
        "minimum": min(lo) if lo else None,
        "maximum": max(hi) if hi else None,
    }


def _count(value: int, limit: int) -> None:
    if type(value) is not int or not 0 <= value <= limit:
        raise ValueError("Invalid scored-surface count")


def _validate(shot: Shot) -> None:
    n = shot["frames"]
    _count(n, 4096)
    if not n or len(shot["maps"]) != 3 or set(shot["strata"]) != set(STRATA):
        raise ValueError("Scored-surface population dimensions differ")
    if (
        shot["strata"]["all"]["frames"] != n
        or sum(shot["strata"][name]["frames"] for name in STRATA[1:]) != n
    ):
        raise ValueError("Scored-surface strata do not partition the original frames")
    for item in shot["maps"]:
        histogram = item["used_histogram"]
        if len(histogram) != 5 or len(item["levels"]) != 4:
            raise ValueError("Scored-surface level dimensions differ")
        for number in [item["full"], *histogram]:
            _count(number, n)
        if sum(histogram) != n or item["full"] != histogram[4]:
            raise ValueError("Scored-surface full-support count differs")
        if sum(k * count for k, count in enumerate(histogram)) != sum(
            level["used"] for level in item["levels"]
        ):
            raise ValueError("Scored-surface included-level mass differs")
        for level in item["levels"]:
            if len(level["ranks"]) != 3 or len(level["reasons"]) != 5:
                raise ValueError("Scored-surface fit dimensions differ")
            for number in [
                level["used"],
                level["zero_source"],
                level["rank_deficient"],
                *level["ranks"],
                *level["reasons"],
            ]:
                _count(number, n)
            if (
                sum(level["ranks"]) != n
                or sum(level["reasons"]) != n
                or level["reasons"][0] != level["used"]
                or level["ranks"][0] != n - level["used"]
                or level["rank_deficient"] != level["ranks"][1]
                or level["zero_source"] > level["used"]
            ):
                raise ValueError("Scored-surface fit support disagrees")
            for key in ("contour_points", "finite_points", "condition"):
                extent = level[key]
                _count(extent["count"], level["used"])
                lo, hi = extent["minimum"], extent["maximum"]
                if extent["count"] == 0:
                    if lo is not None or hi is not None:
                        raise ValueError("Unavailable fit has numeric extrema")
                elif (
                    lo is None
                    or hi is None
                    or not math.isfinite(lo)
                    or not math.isfinite(hi)
                    or lo < 0
                    or hi < lo
                ):
                    raise ValueError("Scored-surface fit extrema differ")
            if any(level[key]["count"] != level["used"] for key in ("contour_points", "finite_points")):
                raise ValueError("Scored-surface sample counts differ")
    for stratum in shot["strata"].values():
        size = stratum["frames"]
        _count(size, n)
        for mode in ("common", "full"):
            entry = stratum[mode]
            if (
                len(entry["counts"]) != 3
                or len(entry["means"]) != 3
                or set(entry["pairs"]) != {f"{MAPS[a]}->{MAPS[b]}" for a, b in PAIRS}
            ):
                raise ValueError("Scored-surface pair dimensions differ")
            for count, value in zip(entry["counts"], entry["means"], strict=True):
                _count(count, size)
                if (count == 0) != (value is None) or value is not None and not math.isfinite(value):
                    raise ValueError("Scored-surface map availability differs")
            for a, b in PAIRS:
                pair = entry["pairs"][f"{MAPS[a]}->{MAPS[b]}"]
                _count(pair["frames"], min(entry["counts"][a], entry["counts"][b]))
                means, change = pair["means"], pair["change"]
                if pair["frames"] == 0:
                    if means is not None or change is not None:
                        raise ValueError("Unavailable pair has numeric estimates")
                elif (
                    means is None
                    or len(means) != 2
                    or change is None
                    or not all(math.isfinite(value) for value in [*means, change])
                ):
                    raise ValueError("Scored-surface pair availability differs")
        if any(a > b for a, b in zip(stratum["full"]["counts"], stratum["common"]["counts"], strict=True)):
            raise ValueError("Full-support counts exceed primary support")
        if any(
            stratum["full"]["pairs"][label]["frames"] > stratum["common"]["pairs"][label]["frames"]
            for label in stratum["common"]["pairs"]
        ):
            raise ValueError("Full-support pairs exceed primary support")


def summarise(shots: Sequence[Shot], stems: Sequence[str]) -> dict[str, object]:
    """Reconstruct census and sensitivity using fixed whole-shot bootstrap weights.

    Parameters
    ----------
    shots : Sequence of Shot
        Measured support aggregates, retaining all original frames.
    stems : Sequence of str
        Independently bound ordered shot identities.

    Returns
    -------
    dict of str to object
        All-map support and common/full-support sensitivity, seed 23023,
        20 000 draws. Full-support results supplement the all-valid primary result.

    Raises
    ------
    ValueError
        Original identities are missing, duplicated or reordered.
    """
    n = len(shots)
    if not n or [s["stem"] for s in shots] != list(stems) or len(set(stems)) != n:
        raise ValueError("Scored-surface shot order differs")
    for shot in shots:
        _validate(shot)
    weights = np.random.default_rng(23023).multinomial(n, np.full(n, 1 / n), size=20000)
    maps: list[Map] = []
    for j in range(3):
        selected = [s["maps"][j] for s in shots]
        levels: list[Level] = []
        for q in range(4):
            chosen = [m["levels"][q] for m in selected]
            levels.append(
                {
                    "used": sum(x["used"] for x in chosen),
                    "ranks": [sum(x["ranks"][k] for x in chosen) for k in range(3)],
                    "reasons": [sum(x["reasons"][k] for x in chosen) for k in range(5)],
                    "zero_source": sum(x["zero_source"] for x in chosen),
                    "rank_deficient": sum(x["rank_deficient"] for x in chosen),
                    "contour_points": _combine_ranges([x["contour_points"] for x in chosen]),
                    "finite_points": _combine_ranges([x["finite_points"] for x in chosen]),
                    "condition": _combine_ranges([x["condition"] for x in chosen]),
                }
            )
        maps.append(
            {
                "full": sum(m["full"] for m in selected),
                "used_histogram": [sum(m["used_histogram"][k] for m in selected) for k in range(5)],
                "levels": levels,
            }
        )
    strata: dict[str, object] = {}
    for name in STRATA:
        entries = [shot["strata"][name] for shot in shots]
        comparisons: dict[str, object] = {}
        for a, b in PAIRS:
            label = f"{MAPS[a]}->{MAPS[b]}"
            modes: dict[str, object] = {}
            for mode in ("common", "full"):
                pairs = [entry[mode]["pairs"][label] for entry in entries]
                changes = np.array([np.nan if p["change"] is None else p["change"] for p in pairs])
                means = [p["means"] for p in pairs if p["means"] is not None]
                modes[mode] = {
                    "frames": sum(p["frames"] for p in pairs),
                    "matched_means": np.mean(means, axis=0).tolist() if means else None,
                    **bootstrap(changes, weights),
                }
            comparisons[label] = modes
        strata[name] = {"frames": sum(e["frames"] for e in entries), "pairs": comparisons}
    return {
        "schema": "scored-surface-support.v1",
        "shots": n,
        "frames": sum(s["frames"] for s in shots),
        "map_order": list(MAPS),
        "requested_levels": list(SURFACES),
        "reasons": list(REASONS),
        "seed": 23023,
        "draws": 20000,
        "maps": maps,
        "strata": strata,
        "full_support_replaces_primary": False,
        "physical_force_balance_certified": False,
    }


def main() -> int:
    """Recompute a hash-pinned support table without accessing private field caches.

    Returns
    -------
    int
        Zero for matching reports, two for malformed or inconsistent inputs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--table-sha256", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        with args.table.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        if checksum != args.table_sha256:
            raise ValueError("Scored-surface table identity differs")
        with gzip.open(args.table, "rb") as stream:
            raw = stream.read(64 * 2**20 + 1)
        if len(raw) > 64 * 2**20:
            raise ValueError("Scored-surface table exceeds its limit")
        data = json.loads(raw)
        if data["schema"] != "scored-surface-shot-statistics.v1":
            raise ValueError("Scored-surface table schema differs")
        shots = cast(list[Shot], data["shots"])
        report = summarise(shots, data["cohort"])
        difference = compare(report, json.loads(args.report.read_text()))
    except Exception:
        print("Unable to verify the supplied scored-surface support statistics.")
        return 2
    print(
        json.dumps(
            {
                "status": "verified",
                "shots": len(shots),
                "frames": report["frames"],
                "maximum_absolute_difference": difference[0],
                "maximum_scaled_difference": difference[1],
                "original_maps_remeasured": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
