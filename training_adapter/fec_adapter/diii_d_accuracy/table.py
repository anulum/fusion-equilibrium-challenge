# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — the frame table a model is fitted on

"""Build the basis and the frame tables from the extracted shots.

The extracted set is tens of gigabytes of flux maps. A model never needs the
maps themselves while it is being fitted: it needs each frame's features and
the frame's coordinates in the flux basis. This step reads every extracted shot
once more and leaves one table per part of the split — features, coordinates,
the scalar labels, and which shot every row came from — a gigabyte or two in
all, which fits anywhere and loads in seconds.

The basis is fitted on the training part alone and stored with more directions
than a model is likely to use; a model takes the leading ones
(:meth:`FluxBasis.truncated`), so trying another size costs no second pass over
the maps. What the truncation costs is recorded per part: the flux R² that a
perfect model of the kept coordinates would reach.

Rows keep their shot. Frames of one shot are strongly alike, so any split made
later for early stopping or cross-validation has to be by shot, and a pooled
score per shot has to be possible.

Features come in two layouts (:mod:`.layout`). Without ``--history`` they are
made from the extracted inputs on the reconstruction frames — record-relative,
the layout of every network fitted before 22 Sep 2026. With ``--history`` they
are made from each shot's stored discharge record (:mod:`.history`) and read at
the reconstruction frames. The table names its layout, and a network fitted
from it carries that name.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fec_adapter import split
from fec_adapter.diii_d_accuracy.extract import SCALAR_LABELS
from fec_adapter.diii_d_accuracy.features import AlignedInputs, frame_features
from fec_adapter.diii_d_accuracy.flux_basis import FluxBasis, moments_in_parallel
from fec_adapter.diii_d_accuracy.history import load_record
from fec_adapter.diii_d_accuracy.layout import DISCHARGE_LAYOUT, RECORD_LAYOUT, FeatureLayout

BASIS_FILE = "flux_basis.npz"
REPORT_FILE = "table_report.json"
PARTS = ("training", "calibration", "validation")


@dataclass(frozen=True)
class ShotRows:
    """The table rows of one shot, and what the basis leaves out of its maps."""

    stem: str
    features: NDArray[np.float32]
    coordinates: NDArray[np.float32]
    labels: NDArray[np.float32]
    squared_truncation: float
    squared_total: float
    map_sum: float
    cells: int


def shot_rows(path: Path, basis: FluxBasis, history: Path | None = None) -> ShotRows:
    """Return one extracted shot as table rows.

    Parameters
    ----------
    path:
        The shot's extracted file.
    basis:
        The flux basis the maps are encoded in.
    history:
        The directory of stored discharge records; given, the features are the
        discharge layout's, read from the record of the same name.

    Raises
    ------
    ValueError
        If the discharge record does not hold the extracted shot's frames.
    """
    with np.load(path, allow_pickle=False) as stored:
        if history is None:
            features = frame_features(AlignedInputs.from_stored(stored))
        else:
            record = load_record(history / path.name)
            recorded = record.inputs.times_ms[record.frames]
            if not np.array_equal(recorded, np.asarray(stored["times_ms"], dtype=np.float64)):
                raise ValueError(f"{path.stem}: the discharge record holds other frames")
            features = record.features()
        maps = stored["psi"].astype(np.float64)
        labels = np.column_stack([stored[name] for name in SCALAR_LABELS])
    coordinates = basis.encode(maps)
    left_out = maps - basis.decode(coordinates)
    return ShotRows(
        path.stem, features.astype(np.float32), coordinates.astype(np.float32),
        labels.astype(np.float32), float(np.sum(left_out**2)), float(np.sum(maps**2)),
        float(maps.sum()), int(maps.size),
    )


def _rows_of_many(
    task: tuple[list[Path], Path, Path | None],
) -> list[ShotRows | tuple[str, str]]:
    """Build the rows of several shots in a worker; a shot that fails is reported by name."""
    files, basis_file, history = task
    basis = FluxBasis.load(basis_file)
    rows: list[ShotRows | tuple[str, str]] = []
    for path in files:
        try:
            rows.append(shot_rows(path, basis, history))
        except (OSError, ValueError, KeyError) as error:
            rows.append((path.stem, f"{type(error).__name__}: {error}"))
    return rows


def build_part(
    files: Sequence[Path], basis_file: Path, out_file: Path, workers: int,
    history: Path | None = None,
) -> dict[str, object]:
    """Write the table of one part and return its summary.

    ``history`` is the directory of discharge records; given, the table is of the
    discharge layout, otherwise record-relative.
    """
    layout: FeatureLayout = RECORD_LAYOUT if history is None else DISCHARGE_LAYOUT
    share = [list(files[start::workers]) for start in range(min(workers, len(files)))]
    fresh = multiprocessing.get_context("spawn")
    tasks = [(part, basis_file, history) for part in share]
    with ProcessPoolExecutor(max_workers=len(share), mp_context=fresh) as pool:
        gathered = [row for rows in pool.map(_rows_of_many, tasks) for row in rows]
    failures = {row[0]: row[1] for row in gathered if isinstance(row, tuple)}
    shots = sorted((row for row in gathered if isinstance(row, ShotRows)), key=lambda r: r.stem)
    if not shots:
        raise ValueError(f"no shot of this part could be turned into rows: {failures}")
    lengths = np.asarray([shot.features.shape[0] for shot in shots])
    np.savez_compressed(
        out_file,
        allow_pickle=False,
        features=np.concatenate([shot.features for shot in shots]),
        coordinates=np.concatenate([shot.coordinates for shot in shots]),
        labels=np.concatenate([shot.labels for shot in shots]),
        label_names=np.asarray(list(SCALAR_LABELS)),
        shot_of_row=np.repeat(np.arange(len(shots)), lengths),
        stems=np.asarray([shot.stem for shot in shots]),
        feature_layout=np.asarray(layout),
        # Per shot, so that the pooled flux R² of any subset of shots can be formed exactly.
        shot_squares_total=np.asarray([shot.squared_total for shot in shots]),
        shot_squares_left_out=np.asarray([shot.squared_truncation for shot in shots]),
        shot_map_sum=np.asarray([shot.map_sum for shot in shots]),
        shot_cells=np.asarray([shot.cells for shot in shots], dtype=np.int64),
    )
    cells = sum(shot.cells for shot in shots)
    mean = sum(shot.map_sum for shot in shots) / cells
    total = sum(shot.squared_total for shot in shots) - cells * mean**2
    return {
        "shots": len(shots),
        "frames": int(lengths.sum()),
        "features": int(shots[0].features.shape[1]),
        "feature_layout": layout,
        "flux_r2_ceiling_of_the_basis": 1.0 - sum(s.squared_truncation for s in shots) / total,
        # With orthonormal directions the squared error of a model splits exactly into its
        # error in the coordinates plus what the basis leaves out, so these two sums give a
        # model's pooled flux R² from the table alone, without the maps.
        "flux_squares_about_the_mean": total,
        "flux_squares_left_out_by_the_basis": sum(s.squared_truncation for s in shots),
        "failures": failures,
    }


def main() -> int:
    """Fit the basis on the training part and write one table per part."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--directions", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--history", type=Path, default=None,
                        help="stored discharge records: build the discharge layout's features")
    args = parser.parse_args()

    started = time.monotonic()
    files = sorted(
        path for path in args.extracted.glob("*_shot_*.npz") if not path.name.startswith(".")
    )
    by_part = {part: [p for p in files if split.bucket_of(p.stem) == part] for part in PARTS}
    print({part: len(found) for part, found in by_part.items()}, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    basis_file = args.out / BASIS_FILE
    if not basis_file.exists():
        moments = moments_in_parallel(by_part["training"], args.workers)
        with np.load(by_part["training"][0], allow_pickle=False) as first:
            grid_shape = (int(first["psi"].shape[1]), int(first["psi"].shape[2]))
        FluxBasis.from_moments(moments, args.directions, grid_shape).save(basis_file)
        print(f"basis from {moments.frames} training frames, "
              f"{(time.monotonic() - started) / 60:.1f} min", flush=True)
    report: dict[str, object] = {"directions": args.directions}
    for part, found in by_part.items():
        report[part] = build_part(
            found, basis_file, args.out / f"table_{part}.npz", args.workers, args.history
        )
        print(part, report[part], f"{(time.monotonic() - started) / 60:.1f} min", flush=True)
    (args.out / REPORT_FILE).write_text(json.dumps(report, indent=2) + "\n")
    print("FINISHED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
