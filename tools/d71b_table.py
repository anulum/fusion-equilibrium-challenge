# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — verify D71b shot means and matched-pair intervals
"""Recompute supported D71b report fields from public sufficient statistics.

The table retains pair-specific means over common finite frames. This verifies
arithmetic and the published cohort binding; it does not remeasure private maps,
authenticate the original measurements, or recover frame quantiles.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import re
import zlib
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
from numpy.typing import NDArray

MAPS = (
    "reference",
    "raw_64coords",
    "kept_45coords",
    "smoothed",
    "axis",
    "contour",
    "shape",
    "x_points",
    "reshape",
    "margin",
)
PAIRS = ("smoothed->margin", *(f"{a}->{b}" for a, b in zip(MAPS[1:-1], MAPS[2:], strict=True)))
MEASURES = (
    "outside_fixed_excess",
    "gs_surface_excess",
    "gs_rms_norm_excess",
    "max_extra",
    "max_missing",
    "saddle_extra",
    "saddle_missing",
    "min_extra",
    "gs_degenerate",
)
FIELDS = ["stem", "kind", "label", "measure", "count", "mean"]
REPORT_SHA256 = "368410df403a3e204eed90c77370681ed33fe779f4c9a3b776cd112c5c92b2b0"
COHORT_SHA256 = "5f719b4324cb1cde8308bf2285f4bad73ae9b365815a2b705b927139bef41f0b"


class Shot(TypedDict):
    """One member of the hash-bound original ordered cohort."""

    stem: str
    frames: int
    reference_frames: int
    measures_sha256: str


class Level(TypedDict):
    """Report fields recoverable from per-map shot means and counts."""

    shot_mean: float
    frames_valid: int


class Change(TypedDict):
    """Report fields recoverable from common-frame shot differences."""

    mean_change: float
    ci99: list[float]
    verdict: str
    shots: int
    matched_frames: int


class Measure(TypedDict):
    """The frozen report's levels and matched-pair comparisons."""

    per_map: dict[str, Level]
    changes: dict[str, Change]


class Report(TypedDict):
    """The hash-bound report fields used in this public verification."""

    measures: dict[str, Measure]


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _cohort(manifest_path: Path, table: Path, report: Path) -> list[Shot]:
    decoded: object = json.loads(manifest_path.read_text())
    if not isinstance(decoded, dict):
        raise ValueError("Expected a statistics manifest object")
    manifest = cast(dict[str, object], decoded)
    fixed: dict[str, object] = {
        "schema": "d71b-shot-statistics.v1",
        "shots": 358,
        "frames": 78875,
        "maps": list(MAPS),
        "pairs": list(PAIRS),
        "measures": list(MEASURES),
        "seed": 71020,
        "draws": 20000,
        "activation_sha256": "5436f808fbf4eea8735a384cb564dc437abbca0b2d82918b1ce698f9fca5f7a8",
        "run_receipt_sha256": "579a02a8da967fdf0073e6ad5925b1e9aea16e751661af97667d1218295fb78e",
        "report_sha256": REPORT_SHA256,
        "reporter_sha256": "f1922260bcfee6e6b2cc2626722fec9582520676642675b1cc7e32f0dec04743",
        "cohort_sha256": COHORT_SHA256,
    }
    if set(manifest) != set(fixed) | {"cohort", "table_sha256", "statistics_export_sha256"}:
        raise ValueError("Unexpected or missing manifest fields")
    if any(manifest[key] != expected for key, expected in fixed.items()):
        raise ValueError("Manifest differs from the frozen D71b cohort or method")
    if not re.fullmatch(r"[0-9a-f]{64}", str(manifest["statistics_export_sha256"])):
        raise ValueError("Invalid extraction-source identity")
    if _sha256(table) != manifest["table_sha256"] or _sha256(report) != REPORT_SHA256:
        raise ValueError("Table or frozen report bytes differ")
    encoded = json.dumps(manifest["cohort"], sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(encoded).hexdigest() != COHORT_SHA256:
        raise ValueError("Shot order, support or original measurement identities differ")
    # The pinned digest identifies this exact JSON value, including its schema.
    return cast(list[Shot], manifest["cohort"])


def _statistics(table: Path, cohort: list[Shot]) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    labels = (*MAPS, *PAIRS)
    means = np.full((len(MEASURES), len(cohort), len(labels)), np.nan, dtype=np.float64)
    counts = np.zeros(means.shape, dtype=np.int64)
    expected = itertools.product(range(len(cohort)), range(len(MEASURES)), range(len(labels)))
    with gzip.open(table, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        if next(reader, []) != FIELDS:
            raise ValueError("Unexpected statistics columns")
        for (shot, measure, label), row in zip(expected, reader, strict=True):
            kind = "map" if label < len(MAPS) else "pair"
            if len(row) != len(FIELDS) or row[:4] != [
                cohort[shot]["stem"],
                kind,
                labels[label],
                MEASURES[measure],
            ]:
                raise ValueError("Statistics rows are missing, duplicated or out of order")
            count = int(row[4])
            if str(count) != row[4] or not 0 <= count <= cohort[shot]["reference_frames"]:
                raise ValueError("Invalid common-frame support")
            if (count > 0) != bool(row[5]):
                raise ValueError("Mean and frame support disagree")
            value = float(row[5]) if count else np.nan
            if count and not np.isfinite(value):
                raise ValueError("Non-finite supported mean")
            if label == 0 and MEASURES[measure].endswith("_excess") and count and value != 0:
                raise ValueError("Reference excess must be zero for each shot")
            means[measure, shot, label] = value
            counts[measure, shot, label] = count
    return means, counts


def _difference(actual: float, expected: float) -> float:
    difference = abs(actual - expected)
    if not np.isfinite(difference) or difference > 1e-14:
        raise ValueError("Recomputed D71b report differs")
    return difference


def verify(table: Path, manifest: Path, report: Path) -> dict[str, object]:
    """Verify D71b levels, support and paired whole-shot bootstrap intervals.

    Parameters
    ----------
    table : Path
        Gzip CSV of per-map means and pair-specific common-frame differences.
    manifest : Path
        Versioned table digest, ordered cohort and frozen source identities.
    report : Path
        Original, hash-bound D71b v5 report.

    Returns
    -------
    dict[str, object]
        Verified coverage and maximum absolute arithmetic discrepancy, with
        explicit exclusions for private remeasurement and frame quantiles.

    Raises
    ------
    ValueError
        Inputs fail the cohort, support, table or arithmetic contract.
    OSError
        A file is unavailable or its compressed bytes cannot be read.
    """
    cohort = _cohort(manifest, table, report)
    means, counts = _statistics(table, cohort)
    stored = cast(Report, json.loads(report.read_text()))
    draws = np.random.default_rng(71020).integers(0, len(cohort), size=(20000, len(cohort)))
    worst = 0.0
    for i, key in enumerate(MEASURES):
        measure = stored["measures"][key]
        for j, name in enumerate(MAPS):
            level = measure["per_map"][name]
            if int(counts[i, :, j].sum()) != level["frames_valid"]:
                raise ValueError("Per-map finite-frame coverage differs")
            with np.errstate(over="ignore", invalid="ignore"):
                actual = float(np.nanmean(means[i, :, j]))
            worst = max(worst, _difference(actual, level["shot_mean"]))
        for j, name in enumerate(PAIRS, start=len(MAPS)):
            change = measure["changes"][name]
            values = means[i, :, j]
            support = (int(counts[i, :, j].sum()), int(np.isfinite(values).sum()))
            if support != (change["matched_frames"], change["shots"]):
                raise ValueError("Matched-pair support differs")
            with np.errstate(over="ignore", invalid="ignore"):
                actual = float(np.nanmean(values))
                interval = np.nanquantile(np.nanmean(values[draws], axis=1), [0.005, 0.995])
            verdict = "worse" if interval[0] > 0 else "better" if interval[1] < 0 else "inconclusive"
            if verdict != change["verdict"]:
                raise ValueError("Paired-interval interpretation differs")
            worst = max(worst, _difference(actual, change["mean_change"]))
            for bound, recorded in zip(interval, change["ci99"], strict=True):
                worst = max(worst, _difference(float(bound), recorded))
    return {
        "status": "verified",
        "shots": len(cohort),
        "frames": sum(c["reference_frames"] for c in cohort),
        "rows": int(means.size),
        "maximum_absolute_difference": worst,
        "frame_percentiles_verified": False,
        "per_frame_physics_recomputed": False,
        "original_measurements_authenticated": False,
    }


def main() -> int:
    """Run the public table verifier with a fixed refusal message.

    Returns
    -------
    int
        Zero on successful verification; two on an invalid or unreadable input.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(args.table, args.manifest, args.report)
    except (ValueError, OSError, csv.Error, EOFError, zlib.error):
        print(
            json.dumps({"status": "refused", "message": "D71b inputs are missing, invalid or inconsistent."})
        )
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
