# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Equilibrium Challenge — verify the published D71 shot means and paired intervals
"""Recompute D71 summaries from the public table; no private model or frame cache is needed.

This verifies historical arithmetic, including withdrawn topology diagnostics. It does not validate those
diagnostics physically or recover frame percentiles from shot means.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import TypedDict, cast

import numpy as np

MAPS = ("reference", "raw_64coords", "kept_45coords", "smoothed", "axis", "contour", "shape", "x_points",
        "reshape", "margin")
MEASURES = ("current_error_own", "current_error_ref", "outside_excess", "gs_excess", "bp_error_ref",
            "bp_error_envelope", "extra_o_points", "saddle_difference", "lcfs_failed", "gs_excess_deg2",
            "gs_excess_deg5", "gs_excess_o4")


class Change(TypedDict):
    """Stored paired-change fields used by the verification."""

    mean_change: float
    ci99: list[float]


class Level(TypedDict):
    """Stored per-map mean; frame quantiles are not verified here."""

    shot_mean: float


class Measure(TypedDict):
    """The report's per-map means and stage comparisons."""

    per_map: dict[str, Level]
    changes: dict[str, Change]


class Report(TypedDict):
    """D71 report fields needed to reproduce its shot-level arithmetic."""

    shots: int
    frames: int
    maps: list[str]
    seed: int
    draws: int
    measures: dict[str, Measure]


def verify(table: Path, report_path: Path) -> float:
    """Verify shot coverage, means and paired bootstrap intervals.

    Parameters
    ----------
    table : Path
        Published gzip-compressed CSV of per-shot means and support counts.
    report_path : Path
        Historical D71 JSON report containing the comparison values and seed.

    Returns
    -------
    float
        Largest absolute difference across the checked means and interval endpoints.

    Raises
    ------
    ValueError
        Coverage is incomplete, values or counts are invalid, or summaries differ by more than 1e-14.
    """
    report = cast(Report, json.loads(report_path.read_text()))
    if report["shots"] != 358 or report["frames"] != 78875:
        raise ValueError("Expected D71's complete 358-shot, 78875-frame report")
    maps = report["maps"]
    if tuple(maps) != MAPS or set(report["measures"]) != set(MEASURES):
        raise ValueError("Maps or measures differ from the frozen D71 report")
    if report["seed"] != 71019 or report["draws"] != 20000:
        raise ValueError("Bootstrap settings differ from the frozen D71 report")
    with gzip.open(table, "rt", newline="") as stream:
        rows = list(csv.DictReader(stream))
    indexed = {(row["stem"], row["map"]): row for row in rows}
    shots = sorted({row["stem"] for row in rows})
    if len(shots) != 358 or len(rows) != 3580 or len(indexed) != len(rows):
        raise ValueError("Expected 3580 unique shot/map rows")
    if set(indexed) != {(shot, name) for shot in shots for name in maps}:
        raise ValueError("Table is missing a map or contains an unexpected map")
    reference_frames = {shot: int(indexed[shot, "reference"]["frames_with_reference"]) for shot in shots}
    if sum(reference_frames.values()) != report["frames"]:
        raise ValueError("Reference frame coverage differs from the report")
    rng = np.random.default_rng(report["seed"])
    draws = rng.integers(0, len(shots), size=(report["draws"], len(shots)))
    worst = 0.0
    for key, measure in report["measures"].items():
        expected_steps = {"smoothed->margin", *[f"{a}->{b}" for a, b in zip(maps[1:-1], maps[2:])]}
        if set(measure["per_map"]) != set(maps[1:]) or set(measure["changes"]) != expected_steps:
            raise ValueError(f"Incomplete report comparisons: {key}")
        means = np.empty((len(shots), len(maps)), dtype=np.float64)
        for i, shot in enumerate(shots):
            for j, name in enumerate(maps):
                row = indexed[shot, name]
                count = int(row[key + "_frames"])
                if int(row["frames_with_reference"]) != reference_frames[shot] or not 0 <= count <= reference_frames[shot]:
                    raise ValueError(f"Invalid frame support: {shot}, {name}, {key}")
                value = float(row[key]) if row[key] else np.nan
                if (count > 0) != bool(row[key]) or (row[key] and not np.isfinite(value)):
                    raise ValueError(f"Mean and support disagree: {shot}, {name}, {key}")
                means[i, j] = value
        for name, level in measure["per_map"].items():
            difference = abs(float(np.nanmean(means[:, maps.index(name)])) - level["shot_mean"])
            if not np.isfinite(difference):
                raise ValueError(f"Undefined per-map comparison: {key}, {name}")
            worst = max(worst, difference)
        for step, recorded in measure["changes"].items():
            before, after = step.split("->")
            difference_by_shot = means[:, maps.index(after)] - means[:, maps.index(before)]
            bootstrap = np.nanmean(difference_by_shot[draws], axis=1)
            interval = np.nanquantile(bootstrap, [0.005, 0.995])
            differences = [abs(float(np.nanmean(difference_by_shot)) - recorded["mean_change"]),
                           *np.abs(interval - recorded["ci99"])]
            if not np.all(np.isfinite(differences)):
                raise ValueError(f"Undefined paired comparison: {key}, {step}")
            worst = max(worst, *differences)
    if worst > 1e-14:
        raise ValueError(f"Summaries differ from the report: {worst}")
    return float(worst)


def main() -> None:
    """Check a public CSV/report pair through the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    error = verify(args.table, args.report)
    print(json.dumps({"shots": 358, "rows": 3580, "maximum_absolute_difference": error,
                      "frame_percentiles_verified": False}))


if __name__ == "__main__":
    main()
