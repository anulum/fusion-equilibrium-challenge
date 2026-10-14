# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — the shape scalars the scorer reads off a true map, as labels

"""Add to the frame tables the scalars the scorer derives from the true map.

The Consistency term compares seven scalars derived from the submitted map
with the same seven derived from the true one. Five of them — elongation, the
two triangularities, volume and the internal inductance — are read off a
predicted map worse than a network output fitted to them directly predicts
them (validation part, a quarter of the training rows, one network: 0.952 vs
0.969 for κ, 0.946 vs 0.949 for li). The dataset's own ``efit_li`` is not the
scored quantity: it explains the scorer's li with an R² of 0.983 only. So the
labels are made here by running the organisers' derivation
(:mod:`fec_adapter.scoring_kit`) on the true maps, decoded from the tables'
coordinates — the basis keeps them to one part in 10⁸ — and appended to each
part's table as further label columns. A fit then treats them as it treats
``q95`` or ``r_axis``: standardised, weighted, missing where the derivation
failed.

With ``--x-points`` the columns appended are instead the two X-points at the
ends of the scorer's boundary (:mod:`.x_point`): the validation part loses its
boundary distance there, and a network output for them lets the edit put them
in place. With ``--contour`` they are the boundary's radii about the axis in
:data:`.contour_edit.CONTOUR_ANGLES` directions (:mod:`.contour_edit`), NaN where
the scorer finds no boundary.

Nothing here reads a submission; the maps come from the training files' own
``efit_psirz``.
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fec_adapter.diii_d_accuracy.contour_edit import contour_radii
from fec_adapter.diii_d_accuracy.contour_edit import label_names as contour_label_names
from fec_adapter.diii_d_accuracy.flux_basis import FluxBasis
from fec_adapter.diii_d_accuracy.shape_edit import Reader, ReadingBackend
from fec_adapter.diii_d_accuracy.table import BASIS_FILE, PARTS, REPORT_FILE
from fec_adapter.diii_d_accuracy.x_point import LABELS as X_POINT_LABELS
from fec_adapter.diii_d_accuracy.x_point import x_point_positions
from fec_adapter.scoring_kit import CONS_SCALARS

SHAPE_LABELS = {
    "kappa": "kappa", "tri_top": "tri_top", "tri_bot": "tri_bot", "volume": "volume",
    "li_scored": "li",
}
"""Label name in the table → the scorer's name of the derived scalar."""

ROWS_PER_TASK = 2000


def derived_scalars(
    maps: NDArray[np.float64], machine: str, backend: ReadingBackend | None = None
) -> NDArray[np.float64]:
    """Return the scorer's seven derived scalars of every frame, ``(T, 7)``, NaN where undefined.

    ``backend`` chooses who reads the maps (:class:`~.shape_edit.Reader`); both give the
    same numbers.
    """
    reader = Reader(machine, backend)
    out = np.full((maps.shape[0], len(CONS_SCALARS)), np.nan)
    for frame, flux in enumerate(maps):
        reading = reader.read(np.asarray(flux, dtype=np.float64))
        out[frame] = [reading.values[name] for name in CONS_SCALARS]
    return out


def shape_labels(
    maps: NDArray[np.float64], machine: str, backend: ReadingBackend | None = None
) -> NDArray[np.float64]:
    """Return the five shape labels of every frame, in :data:`SHAPE_LABELS` order."""
    derived = derived_scalars(maps, machine, backend)
    columns = [CONS_SCALARS.index(name) for name in SHAPE_LABELS.values()]
    return derived[:, columns]


def x_point_labels(
    maps: NDArray[np.float64], machine: str, backend: ReadingBackend | None = None
) -> NDArray[np.float64]:
    """Return the two boundary X-points of every frame, in :data:`X_POINT_LABELS` order.

    ``NaN`` where the scorer finds no boundary, or where an end of it is not a saddle
    (a limited plasma's end, or a second null too far from the boundary).
    """
    reader = Reader(machine, backend)
    return np.stack([x_point_positions(reader, np.asarray(flux, dtype=np.float64))
                     for flux in maps])


def contour_labels(
    maps: NDArray[np.float64], machine: str, backend: ReadingBackend | None = None
) -> NDArray[np.float64]:
    """Return the boundary radii of every frame, in :func:`.contour_edit.label_names` order."""
    reader = Reader(machine, backend)
    return np.stack([contour_radii(reader, np.asarray(flux, dtype=np.float64))
                     for flux in maps])


def _labels_of_task(task: tuple[Path, NDArray[np.float32], str, str]) -> NDArray[np.float64]:
    basis_file, coordinates, machine, kind = task
    maps = FluxBasis.load(basis_file).decode(coordinates)
    if kind == "x_points":
        return x_point_labels(maps, machine)
    if kind == "contour":
        return contour_labels(maps, machine)
    return shape_labels(maps, machine)


def _names_of(kind: str) -> list[str]:
    if kind == "x_points":
        return list(X_POINT_LABELS)
    if kind == "contour":
        return list(contour_label_names())
    return list(SHAPE_LABELS)


def labelled_table(
    table_file: Path, basis_file: Path, out_file: Path, machine: str, workers: int,
    *, x_points: bool = False, contour: bool = False,
) -> dict[str, object]:
    """Write a table with the shape, X-point or contour labels appended; return a summary."""
    if x_points and contour:
        raise ValueError("append the X-point labels and the contour labels one at a time")
    kind = "x_points" if x_points else "contour" if contour else "shape"
    with np.load(table_file, allow_pickle=False) as stored:
        contents = {name: stored[name] for name in stored.files}
    coordinates = contents["coordinates"]
    bounds = range(0, coordinates.shape[0] + ROWS_PER_TASK, ROWS_PER_TASK)
    tasks = [(basis_file, coordinates[a:b], machine, kind)
             for a, b in itertools.pairwise(bounds)
             if b > a and coordinates[a:b].shape[0]]
    fresh = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=fresh) as pool:
        added = np.concatenate(list(pool.map(_labels_of_task, tasks)))
    names = _names_of(kind)
    old_names = [str(name) for name in contents["label_names"]]
    if set(names) & set(old_names):
        raise ValueError(f"{table_file} already has labels {sorted(set(names) & set(old_names))}")
    contents["labels"] = np.concatenate([contents["labels"], added.astype(np.float32)], axis=1)
    contents["label_names"] = np.asarray([*old_names, *names])
    np.savez_compressed(out_file, allow_pickle=False, **contents)
    finite = np.isfinite(added).mean(axis=0)
    return {
        "frames": int(added.shape[0]),
        "labelled_share": dict(zip(names, np.round(finite, 5).tolist(), strict=True)),
        "median": dict(zip(names, np.round(np.nanmedian(added, axis=0), 4).tolist(),
                           strict=True)),
    }


def main() -> int:
    """Append the shape labels to every part's table and copy the basis and report beside them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--machine", default="DIII-D")
    parser.add_argument("--parts", nargs="+", default=list(PARTS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--x-points", action="store_true",
                        help="append the two boundary X-points instead of the shape labels")
    parser.add_argument("--contour", action="store_true",
                        help="append the boundary's radii about the axis instead")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = json.loads((args.tables / REPORT_FILE).read_text())
    (args.out / BASIS_FILE).write_bytes((args.tables / BASIS_FILE).read_bytes())
    for part in args.parts:
        summary = labelled_table(
            args.tables / f"table_{part}.npz", args.tables / BASIS_FILE,
            args.out / f"table_{part}.npz", args.machine, args.workers,
            x_points=args.x_points, contour=args.contour,
        )
        key = ("x_point_labels" if args.x_points else "contour_labels" if args.contour
               else "shape_labels")
        report[part][key] = summary
        print(json.dumps({"part": part, **summary,
                          "minutes": round((time.monotonic() - started) / 60.0, 1)}), flush=True)
    (args.out / REPORT_FILE).write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
