# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — submitted-shot numerical regression check
"""Compare a regenerated DIII-D shot with the immutable Development prediction.

The reference is d3d_shot_0047b7837b from submission 986068. Its array hashes
are checked before comparison. The accepted differences remain 1e-5 of the
reference flux range and 1e-4 for either scalar; bit identity is reported
separately. These tolerances do not establish full-fold reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict
from zipfile import BadZipFile

import numpy as np

REFERENCE = Path(__file__).resolve().parent / "reference" / "d3d_shot_0047b7837b_submitted_986068.npz"
DIGESTS = {
    "shot_0000_psirz": "908d3d5e756c2e4c11487bd138b84b94a9cc3ed9490c5fc2dcdf985056761ee6",
    "shot_0000_q95": "84ae16249f1277cd7c3676adb7b8fc586bc41e56fb70bf803f18011e222cd1d7",
    "shot_0000_betaN": "e41095668132e6c97c2a673769a57bfad0627dfdd821c678f9ddeb52c184bb1b",
}
SHAPES = {"shot_0000_psirz": (240, 65, 65), "shot_0000_q95": (240,), "shot_0000_betaN": (240,)}


class Comparison(TypedDict):
    """Observed identity and maximum error for one submitted array."""

    bit_identical: bool
    max_abs_difference: float
    max_difference_over_scale: float
    limit: float


def compare(archive: Path, reference: Path = REFERENCE) -> dict[str, Comparison]:
    """Check reference integrity and compare each map or scalar array.

    Parameters
    ----------
    archive : Path
        Regenerated archive containing the first DIII-D shot.
    reference : Path
        Original reference archive, whose three array digests are immutable.

    Returns
    -------
    dict[str, Comparison]
        Identity and scaled maximum difference for each submitted array.

    Raises
    ------
    ValueError
        A reference digest, prediction shape, dtype or finite-value check fails.
    OSError
        An archive cannot be read.
    KeyError
        A required submitted array is absent.
    """
    report: dict[str, Comparison] = {}
    with np.load(archive, allow_pickle=False) as new, np.load(reference, allow_pickle=False) as ref:
        for key, digest in DIGESTS.items():
            a, b = new[key], ref[key]
            if b.dtype != np.float32 or b.shape != SHAPES[key]:
                raise ValueError("Reference shape or dtype does not match the submitted prediction.")
            if hashlib.sha256(np.ascontiguousarray(b).tobytes()).hexdigest() != digest:
                raise ValueError("Reference array digest does not match the submitted prediction.")
            if a.dtype != np.float32 or a.shape != b.shape or not np.isfinite(a).all():
                raise ValueError("Prediction must have the reference shape and finite float32 values.")
            difference = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
            scale = float(np.ptp(b)) if key.endswith("psirz") else 1.0
            report[key] = {
                "bit_identical": bool(np.array_equal(a, b)),
                "max_abs_difference": difference,
                "max_difference_over_scale": difference / scale,
                "limit": 1e-5 if key.endswith("psirz") else 1e-4,
            }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Report the numerical comparison and return a failing exit code on damage.

    Parameters
    ----------
    argv : Sequence[str] or None
        Command arguments, or the process arguments when omitted.

    Returns
    -------
    int
        Zero for an accepted comparison, one for excessive error, two for an
        unreadable or structurally invalid archive.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    args = parser.parse_args(argv)
    try:
        report = compare(args.archive, args.reference)
    except (OSError, ValueError, TypeError, KeyError, EOFError, BadZipFile):
        print("Prediction archive or submitted reference is invalid.")
        return 2
    ok = all(item["max_difference_over_scale"] <= item["limit"] for item in report.values())
    print(json.dumps(report, indent=1))
    if all(item["bit_identical"] for item in report.values()):
        print("bit-identical")
    elif ok:
        print("within tolerance (not bit-identical on this CPU)")
    else:
        print("OUTSIDE tolerance")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
