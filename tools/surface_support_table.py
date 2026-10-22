# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — surface-support table verification
"""Recompute counts, equal-shot means and intervals from bound shot aggregates.

The caller supplies the independently published manifest digest. Verification
checks transport, cohort and arithmetic; it does not authenticate measurements,
remeasure private maps or reconstruct frame quantiles.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import zlib
from pathlib import Path
from typing import TypedDict, cast

from tools.surface_support_summary import Shot, summarise

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
COHORT_SHA256 = "5f719b4324cb1cde8308bf2285f4bad73ae9b365815a2b705b927139bef41f0b"
PRODUCER_SHA256 = "7a8632bebcada7f0619f33e3c4a9ebfc7bae95701303f0642ce4cdda67bf83a8"
ACTIVATION_SHA256 = "93a6c48e14cf23385b52c5b4517d32a1c8ddf405c6bb057a62164dd2abef9c72"


class CohortShot(TypedDict):
    """An original cohort member and its independently bound reference support."""

    stem: str
    frames: int
    reference_frames: int
    measures_sha256: str


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Expected a statistics object")
    return cast(dict[str, object], value)


def _compare(actual: object, expected: object) -> float:
    if isinstance(actual, dict):
        left, right = _object(actual), _object(expected)
        if left.keys() != right.keys():
            raise ValueError("Report fields differ")
        return max((_compare(value, right[key]) for key, value in left.items()), default=0.0)
    if isinstance(actual, list):
        if not isinstance(expected, list) or len(actual) != len(expected):
            raise ValueError("Report dimensions differ")
        return max((_compare(a, b) for a, b in zip(actual, expected, strict=True)), default=0.0)
    if isinstance(actual, float):
        if type(expected) not in (int, float):
            raise ValueError("Report numerical availability differs")
        difference = abs(actual - cast(float, expected))
        if not math.isfinite(difference) or difference > 1e-14:
            raise ValueError("Recomputed report differs")
        return difference
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError("Report count or availability differs")
    return 0.0


def _manifest(path: Path, checksum: str, table: Path, report: Path) -> list[CohortShot]:
    if not re.fullmatch(r"[0-9a-f]{64}", checksum) or _digest(path) != checksum:
        raise ValueError("Manifest identity differs")
    manifest = _object(json.loads(path.read_text()))
    fixed: dict[str, object] = {
        "schema": "surface-support-shot-manifest.v1",
        "shots": 358,
        "frames": 78875,
        "maps": list(MAPS),
        "levels": [0.2, 0.4, 0.6, 0.8],
        "comparison_maps": ["reference", "smoothed", "margin"],
        "seed": 25025,
        "draws": 20000,
        "cohort_sha256": COHORT_SHA256,
        "census_producer_sha256": PRODUCER_SHA256,
        "census_activation_sha256": ACTIVATION_SHA256,
        "baseline_table_sha256": "1c89463fe6dd5dd455a841613551880aff8c4ad33470987d19ccf189635b0da8",
        "baseline_report_sha256": "368410df403a3e204eed90c77370681ed33fe779f4c9a3b776cd112c5c92b2b0",
    }
    if any(manifest.get(key) != value for key, value in fixed.items()):
        raise ValueError("Cohort or measurement method differs")
    encoded = json.dumps(manifest["cohort"], sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(encoded).hexdigest() != COHORT_SHA256:
        raise ValueError("Original shot order or support differs")
    if _digest(table) != manifest["table_sha256"] or _digest(report) != manifest["report_sha256"]:
        raise ValueError("Table or report bytes differ")
    return cast(list[CohortShot], manifest["cohort"])


def _shots(table: Path, cohort: list[CohortShot]) -> list[Shot]:
    with gzip.open(table, "rt", encoding="utf-8") as stream:
        decoded = _object(json.load(stream))
    if set(decoded) != {"schema", "shots"} or decoded["schema"] != "surface-support-shot-statistics.v1":
        raise ValueError("Statistics schema differs")
    if not isinstance(decoded["shots"], list):
        raise ValueError("Expected ordered shot statistics")
    shots = cast(list[Shot], decoded["shots"])
    if [shot["stem"] for shot in shots] != [case["stem"] for case in cohort]:
        raise ValueError("Shot identities are missing, duplicated or reordered")
    for shot, case in zip(shots, cohort, strict=True):
        if shot["frames"] != case["frames"] or shot["reference_valid"] != case["reference_frames"]:
            raise ValueError("Original frame support differs")
        if len(shot["maps"]) != 10 or any(len(m["levels"]) != 4 for m in shot["maps"]):
            raise ValueError("Map or surface dimensions differ")
    return shots


def verify(table: Path, manifest: Path, report: Path, manifest_sha256: str) -> dict[str, object]:
    """Reproduce the complete published surface-support report.

    Parameters
    ----------
    table : Path
        Compressed JSON of ordered shot aggregates.
    manifest : Path
        Cohort, method, source and transport identities.
    report : Path
        Report to reproduce, bound by the manifest.
    manifest_sha256 : str
        Independently published checksum of the manifest, supplied by the caller.

    Returns
    -------
    dict of str to object
        Verified cohort size and maximum numerical difference, with the scope of
        reproduction stated explicitly.

    Raises
    ------
    ValueError
        An identity, cohort, shape or recomputed field differs.
    OSError
        A supplied input cannot be read.
    """
    try:
        cohort = _manifest(manifest, manifest_sha256, table, report)
        shots = _shots(table, cohort)
        recomputed = summarise(shots, [case["stem"] for case in cohort])
        expected = _object(json.loads(report.read_text()))
        worst = _compare(recomputed, expected)
    except (
        KeyError,
        TypeError,
        IndexError,
        AttributeError,
        gzip.BadGzipFile,
        EOFError,
        zlib.error,
        OverflowError,
        RecursionError,
    ):
        raise ValueError("Invalid support statistics structure") from None
    return {
        "status": "verified",
        "shots": len(shots),
        "frames": sum(shot["frames"] for shot in shots),
        "maximum_absolute_difference": worst,
        "frame_quantiles_reproduced": False,
        "private_maps_remeasured": False,
        "original_measurements_authenticated": False,
    }


def main() -> int:
    """Run the verifier and return fixed refusal text for invalid inputs.

    Returns
    -------
    int
        Zero on success; two when an input is unreadable or fails verification.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(args.table, args.manifest, args.report, args.manifest_sha256)
    except (OSError, ValueError):
        print("Unable to verify the supplied surface-support statistics.")
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
