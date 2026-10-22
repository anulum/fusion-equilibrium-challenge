# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — scored-map physical table verification
"""Recompute matched diagnostic means and intervals from bound shot aggregates.

The command verifies the supplied package and arithmetic. It does not remeasure
private maps, authenticate original measurements or certify physical equilibrium.
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

from tools.scored_map_summary import MAPS, METRICS, PAIRS, STRATA, Shot, summarise

COHORT_SHA256 = "5f719b4324cb1cde8308bf2285f4bad73ae9b365815a2b705b927139bef41f0b"
PRODUCER_SHA256 = "aae8f6c128795ca3b67c8e4dc5b92eba51b6387ff25cf812e45bfaa66b3d3584"
ABSOLUTE_TOLERANCE = 1e-13
RELATIVE_TOLERANCE = 1e-12


class CohortShot(TypedDict):
    """An original D71b cohort member and its bound reference support."""

    stem: str
    frames: int
    reference_frames: int
    measures_sha256: str


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Expected a diagnostic statistics object")
    return cast(dict[str, object], value)


def _compare(actual: object, expected: object) -> tuple[float, float]:
    if isinstance(actual, dict):
        left, right = _object(actual), _object(expected)
        if left.keys() != right.keys():
            raise ValueError("Report fields differ")
        differences = [_compare(value, right[key]) for key, value in left.items()]
    elif isinstance(actual, list):
        if not isinstance(expected, list) or len(actual) != len(expected):
            raise ValueError("Report dimensions differ")
        differences = [_compare(a, b) for a, b in zip(actual, expected, strict=True)]
    elif isinstance(actual, float):
        if type(expected) not in (int, float):
            raise ValueError("Report numerical availability differs")
        difference = abs(actual - cast(float, expected))
        tolerance = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * abs(actual)
        if not math.isfinite(difference) or difference > tolerance:
            raise ValueError("Recomputed diagnostic report differs")
        return difference, difference / tolerance
    else:
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError("Report count or availability differs")
        return 0.0, 0.0
    return max((d[0] for d in differences), default=0.0), max((d[1] for d in differences), default=0.0)


def _manifest(path: Path, checksum: str, table: Path, report: Path) -> list[CohortShot]:
    if not re.fullmatch(r"[0-9a-f]{64}", checksum) or _digest(path) != checksum:
        raise ValueError("Manifest identity differs")
    manifest = _object(json.loads(path.read_text()))
    fixed: dict[str, object] = {
        "schema": "scored-map-physics-shot-manifest.v1",
        "shots": 358,
        "frames": 78875,
        "maps": list(MAPS),
        "metrics": list(METRICS),
        "pairs": [f"{MAPS[a]}->{MAPS[b]}" for a, b in PAIRS],
        "strata": list(STRATA),
        "seed": 23023,
        "draws": 20000,
        "cohort_sha256": COHORT_SHA256,
        "measurement_producer_sha256": PRODUCER_SHA256,
        "d72_activation_sha256": "5b156162e3add75896f89e2f44e9b3f98b2151fdd2f61f7e8fe33024b5c8eb0c",
        "d71b_activation_sha256": "5436f808fbf4eea8735a384cb564dc437abbca0b2d82918b1ce698f9fca5f7a8",
        "cohort_manifest_sha256": "7afd91aea533d4b79e7db595b657fe4c8c5f0c0cef712090912c76c11fdcdd54",
    }
    if any(manifest.get(key) != value for key, value in fixed.items()):
        raise ValueError("Cohort or measurement method differs")
    encoded = json.dumps(manifest["cohort"], sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(encoded).hexdigest() != COHORT_SHA256:
        raise ValueError("Original shot order or support differs")
    if _digest(table) != manifest["table_sha256"] or _digest(report) != manifest["report_sha256"]:
        raise ValueError("Table or report bytes differ")
    return cast(list[CohortShot], manifest["cohort"])


def _finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(cast(float, value))


def _shots(table: Path, cohort: list[CohortShot]) -> list[Shot]:
    with gzip.open(table, "rt", encoding="utf-8") as stream:
        decoded = _object(json.load(stream))
    if set(decoded) != {"schema", "shots"} or decoded["schema"] != "r23-matched-shot-statistics.v1":
        raise ValueError("Statistics schema differs")
    if not isinstance(decoded["shots"], list):
        raise ValueError("Expected ordered shot statistics")
    shots = cast(list[Shot], decoded["shots"])
    if [s["stem"] for s in shots] != [c["stem"] for c in cohort]:
        raise ValueError("Shot identities are missing, duplicated or reordered")
    pair_names = {f"{MAPS[a]}->{MAPS[b]}" for a, b in PAIRS}
    for shot, case in zip(shots, cohort, strict=True):
        if (
            type(shot["frames"]) is not int
            or type(shot["reference_frames"]) is not int
            or shot["frames"] != case["frames"]
            or shot["reference_frames"] != case["reference_frames"]
        ):
            raise ValueError("Original frame support differs")
        if set(shot["statistics"]) != set(STRATA) or set(shot["stratum_frames"]) != set(STRATA):
            raise ValueError("Diagnostic strata differ")
        if (
            shot["stratum_frames"]["all"] != shot["frames"]
            or sum(shot["stratum_frames"][s] for s in STRATA[1:]) != shot["frames"]
        ):
            raise ValueError("Diagnostic strata do not partition the original frames")
        for stratum in STRATA:
            size = shot["stratum_frames"][stratum]
            if type(size) is not int or not 0 <= size <= shot["frames"]:
                raise ValueError("Invalid stratum support")
            entries = shot["statistics"][stratum]
            if set(entries) != set(METRICS):
                raise ValueError("Diagnostic measure order or membership differs")
            for entry in entries.values():
                if len(entry["counts"]) != 3 or len(entry["means"]) != 3 or set(entry["pairs"]) != pair_names:
                    raise ValueError("Diagnostic map or pair dimensions differ")
                if any(type(n) is not int or not 0 <= n <= size for n in entry["counts"]):
                    raise ValueError("Invalid map support")
                for count, value in zip(entry["counts"], entry["means"], strict=True):
                    if (count == 0) != (value is None) or value is not None and not _finite(value):
                        raise ValueError("Map support and numerical availability differ")
                for a, b in PAIRS:
                    pair = entry["pairs"][f"{MAPS[a]}->{MAPS[b]}"]
                    count = pair["frames"]
                    if type(count) is not int or not 0 <= count <= min(
                        entry["counts"][a], entry["counts"][b]
                    ):
                        raise ValueError("Invalid common-frame support")
                    means, change = pair["means"], pair["change"]
                    if count == 0:
                        if means is not None or change is not None:
                            raise ValueError("Unsupported pair has a numerical estimate")
                    elif means is None or len(means) != 2 or not all(_finite(x) for x in [*means, change]):
                        raise ValueError("Matched map dimensions or numerical availability differ")
    return shots


def verify(table: Path, manifest: Path, report: Path, manifest_sha256: str) -> dict[str, object]:
    """Reproduce the complete scored-map diagnostic report from sufficient statistics.

    Parameters
    ----------
    table : Path
        Compressed JSON of ordered per-shot diagnostic statistics.
    manifest : Path
        Independently checksum-bound cohort, method and file identities.
    report : Path
        Published diagnostic report to recompute.
    manifest_sha256 : str
        Published manifest checksum supplied explicitly by the caller.

    Returns
    -------
    dict of str to object
        Verified scope, size, tolerances and absolute/scaled discrepancies.

    Raises
    ------
    ValueError
        Identity, cohort, shape, availability or recomputed arithmetic differs.
    OSError
        A supplied input cannot be read.
    """
    try:
        cohort = _manifest(manifest, manifest_sha256, table, report)
        shots = _shots(table, cohort)
        actual = summarise(shots, [case["stem"] for case in cohort])
        differences = _compare(actual, _object(json.loads(report.read_text())))
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
        raise ValueError("Invalid diagnostic statistics structure") from None
    return {
        "status": "verified",
        "shots": len(shots),
        "frames": sum(s["frames"] for s in shots),
        "maximum_absolute_difference": differences[0],
        "maximum_scaled_difference": differences[1],
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "frame_quantiles_reproduced": False,
        "private_maps_remeasured": False,
        "original_measurements_authenticated": False,
        "physical_force_balance_certified": False,
    }


def main() -> int:
    """Run the verifier with fixed caller-safe refusal text.

    Returns
    -------
    int
        Zero on success; two for unreadable or inconsistent inputs.
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
        print("Unable to verify the supplied scored-map diagnostic statistics.")
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
