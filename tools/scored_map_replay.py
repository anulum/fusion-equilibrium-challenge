# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — scored-field replay command
"""Remeasure original scored maps through a source-bound serial or process interface."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from importlib.metadata import version
from itertools import repeat
from multiprocessing import get_context
from pathlib import Path
from typing import Never

import numpy as np

from tools.scored_map_bundle import Bundle
from tools.scored_map_bundle import load as load_bundle
from tools.scored_map_measure import measure_case
from tools.scored_map_operators import load as load_operators
from tools.scored_map_summary import RAW, Shot, summarise
from tools.scored_map_table import compare, reference
from tools.scored_surface_summary import Shot as SurfaceShot
from tools.scored_surface_summary import extract as surface_extract
from tools.scored_surface_summary import summarise as surface_summarise
from tools.surface_gs import SURFACES

ROOT = Path(__file__).resolve().parents[1]
REFUSAL = "Scored-map replay refused: check package identities, fields, geometry and output directory."


class Parser(argparse.ArgumentParser):
    """Keep parser refusals independent of caller paths and exception text."""

    def error(self, message: str) -> Never:
        """Refuse malformed arguments without printing their contents."""
        raise ValueError("Invalid replay arguments")


def replay(
    bundle: Bundle, expected_path: Path, expected_sha256: str, output: Path, workers: int = 1
) -> dict[str, object]:
    """Measure maps and compare actual statistics to independently pinned references.

    Parameters
    ----------
    bundle : Bundle
        Validated original scored-field package.
    expected_path : Path
        Original public per-shot statistics, or explicit three-frame illustration.
    expected_sha256 : str
        Independently supplied compressed-statistics identity.
    output : Path
        New output directory; existing paths are refused.
    workers : int, default 1
        One serial worker or at most twelve isolated measurement processes.

    Returns
    -------
    dict of str to object
        Measured scope, numerical comparisons and actual measuring-source identities.

    Raises
    ------
    OSError, ValueError
        Inputs, historical parity, surface support or output are inconsistent.
    """
    if not 1 <= workers <= 12:
        raise ValueError("Measurement worker count must be between one and twelve")
    stems = [case.stem for case in bundle.cases]
    expected, expected_report = reference(expected_path, expected_sha256, bundle.scope, stems)
    output.mkdir(parents=True, exist_ok=False)
    shots: list[Shot] = []
    surface_shots: list[SurfaceShot] = []
    discrepancies: list[tuple[float, float]] = []
    manager = (
        ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))
        if workers > 1
        else nullcontext()
    )
    with manager as executor:
        results = (
            map(measure_case, repeat(bundle), bundle.cases)
            if executor is None
            else executor.map(measure_case, repeat(bundle), bundle.cases, chunksize=1)
        )
        for case, old, result in zip(bundle.cases, expected, results, strict=True):
            discrepancies.append(compare(result.statistics, old))
            np.savez_compressed(
                output / (case.stem + ".npz"),
                values=result.values,
                reference_inside_current=result.reference_inside,
                allow_pickle=False,
                **result.support,
            )
            shots.append(result.statistics)
            surface_shots.append(
                surface_extract(
                    case.stem,
                    result.values[RAW.index("gs_surface")],
                    result.support,
                    result.used,
                    result.converged,
                )
            )
    encoded = json.dumps(
        {"schema": "r23-matched-shot-statistics.v1", "shots": shots}, allow_nan=False
    ).encode()
    with (output / "per_shot.json.gz").open("xb") as stream:
        with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as zipped:
            zipped.write(encoded)
    report = summarise(shots, stems)
    discrepancies.append(compare(report, expected_report))
    (output / "report.json").write_text(json.dumps(report, allow_nan=False, indent=2) + "\n")
    surface_report = surface_summarise(surface_shots, stems)
    (output / "surface_report.json").write_text(json.dumps(surface_report, allow_nan=False, indent=2) + "\n")
    surface_encoded = json.dumps(
        {"schema": "scored-surface-shot-statistics.v1", "cohort": stems, "shots": surface_shots},
        allow_nan=False,
    ).encode()
    with (output / "surface_per_shot.json.gz").open("xb") as stream:
        with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as zipped:
            zipped.write(surface_encoded)
    sources = (
        "scored_map_measure.py",
        "scored_map_replay.py",
        "scored_map_bundle.py",
        "scored_map_operators.py",
        "surface_gs.py",
        "scored_surface_summary.py",
        "scored_map_summary.py",
        "scored_map_table.py",
    )
    artifacts = [case.stem + ".npz" for case in bundle.cases] + [
        "per_shot.json.gz",
        "report.json",
        "surface_per_shot.json.gz",
        "surface_report.json",
    ]
    output_sha256: dict[str, str] = {}
    for name in artifacts:
        with (output / name).open("rb") as stream:
            output_sha256[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    receipt: dict[str, object] = {
        "schema": "scored-map-replay.v1",
        "scope": bundle.scope,
        "shots": len(shots),
        "frames": sum(case.frames for case in bundle.cases),
        "workers": workers,
        "output_sha256": output_sha256,
        "input_manifest_sha256": bundle.checksum,
        "expected_statistics_sha256": expected_sha256,
        "maximum_absolute_difference": max(d[0] for d in discrepancies),
        "maximum_scaled_difference": max(d[1] for d in discrepancies),
        "source_sha256": {
            name: hashlib.sha256((ROOT / "tools" / name).read_bytes()).hexdigest() for name in sources
        },
        "frozen_source_sha256": load_operators().source_sha256,
        "environment": {
            "python": platform.python_version(),
            "packages": {name: version(name) for name in ("numpy", "scipy", "scikit-image")},
        },
        "requested_surface_levels": list(SURFACES),
        "physical_force_balance_certified": False,
        "original_cohort_report_compared": bundle.scope == "full358",
        "original_subset_statistics_compared": bundle.scope == "qualification",
        "illustration_reference_statistics_compared": bundle.scope == "illustration",
    }
    (output / "complete.json").write_text(json.dumps(receipt, allow_nan=False, indent=2) + "\n")
    return receipt


def main() -> int:
    """Measure a caller-pinned package and write only a new output directory.

    Returns
    -------
    int
        Zero on completed measurement, two on refused input or output.
    """
    parser = Parser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--expected-statistics", type=Path, required=True)
    parser.add_argument("--expected-statistics-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    try:
        args = parser.parse_args()
        bundle = load_bundle(args.bundle, args.manifest_sha256)
        receipt = replay(
            bundle, args.expected_statistics, args.expected_statistics_sha256, args.out, args.workers
        )
    except Exception:
        print(json.dumps({"status": "refused", "message": REFUSAL}))
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
