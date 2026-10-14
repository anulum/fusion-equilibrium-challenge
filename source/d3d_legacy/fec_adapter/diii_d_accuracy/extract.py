# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — every labelled shot brought onto one aligned, compact form

"""Read the labelled DIII-D set once and keep each shot aligned and compact.

The released files carry every diagnostic on its own clock — coil currents at
tens of kilohertz for the whole acquisition window — and the reconstruction in
double precision. A model is fitted on frames: the inputs at the reconstruction
times, and the reconstruction. This step produces exactly that, one file per
shot, so that fitting reads a few tens of gigabytes of aligned frames and not
the raw set again for every experiment.

What is kept is the *aligned inputs*, not features made from them: circuit
currents, plasma and toroidal-field current, the Thomson samples with their
positions, the time base. Which features serve a model is a question asked
many times; reading the raw set is answered once.

The labels kept are the flux map, the two submitted scalars, and three of the
reconstruction's own scalars (internal inductance and the axis position) that
the Consistency term of the score is built from, so that a fit can be judged on
them without deriving them from every map.

Shots of the holdout part are refused, not skipped silently: their labels are
read once, at the end, and the refusal is what keeps that true here
(:func:`fec_adapter.split.refuse_holdout`). The extraction asks the rule before
it opens a file.

The inputs come through the reader, which offers no label; the labels are read
here, beside it, and nowhere else in this entry.

With ``--history`` the step stores each shot's discharge record instead
(:mod:`.history`): the inputs on the reconstruction frames and on the instants
of the discharge before, between and after them, the frames' positions among
those instants, and the discharge window. It holds no label and no flux map, so it is
a small companion of the aligned set, read by the table step to build the
discharge layout's features.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from fec_adapter import split
from fec_adapter.diii_d_accuracy.history import DischargeRecord, save_record
from fec_adapter.input_features import circuit_currents
from fec_adapter.shot_reader import read_row, read_shot

MANIFEST_FILE = "MANIFEST.json"

SCALAR_LABELS = {
    "q95": "efit_q95",
    "beta_n": "efit_beta_n",
    "li": "efit_li",
    "r_axis": "efit_r_axis",
    "z_axis": "efit_z_axis",
}
"""Stored name and source column of every per-frame label kept beside the flux map."""


@dataclass(frozen=True)
class ShotRecord:
    """One line of the manifest: what a stored shot is and which part it belongs to."""

    stem: str
    part: str
    frames: int
    circuits: int
    thomson_channels: int


def extract_shot(path: Path, out_dir: Path) -> ShotRecord:
    """Store one shot's aligned inputs and labels and return its manifest line.

    The file is written under a hidden partial name and renamed when complete,
    so an interrupted run leaves nothing that looks finished.

    Raises
    ------
    PermissionError
        If the shot belongs to the holdout part. Nothing is read in that case.
    ValueError
        If a label series does not have one value per reconstruction time.
    """
    split.refuse_holdout(path.stem)
    shot = read_shot(path)
    row = read_row(path)
    psi = np.asarray(row["efit_psirz"], dtype=np.float32)
    expected = (shot.n_frames, shot.z_grid.size, shot.r_grid.size)
    if psi.shape != expected:
        raise ValueError(f"{path.stem}: flux maps have shape {psi.shape}, expected {expected}")
    labels = {}
    for name, column in SCALAR_LABELS.items():
        values = np.asarray(row[column], dtype=np.float32)
        if values.shape != (shot.n_frames,):
            raise ValueError(f"{path.stem}: {column} has shape {values.shape}")
        labels[name] = values
    circuits = circuit_currents(shot)
    partial = out_dir / f".{path.stem}.partial.npz"
    np.savez_compressed(
        partial,
        allow_pickle=False,
        times_ms=shot.times_ms,
        r_grid=shot.r_grid,
        z_grid=shot.z_grid,
        circuit_names=np.asarray(list(dict.fromkeys(shot.coils.columns))),
        circuits=circuits.astype(np.float32),
        plasma_current=shot.plasma_current.astype(np.float32),
        tf_current=shot.tf_current.astype(np.float32),
        thomson_r=shot.thomson.r.astype(np.float32),
        thomson_z=shot.thomson.z.astype(np.float32),
        thomson_te=shot.thomson.te.astype(np.float32),
        thomson_ne=shot.thomson.ne.astype(np.float32),
        psi=psi,
        **labels,
    )
    partial.rename(out_dir / f"{path.stem}.npz")
    return ShotRecord(
        path.stem, split.bucket_of(path.stem), shot.n_frames,
        int(circuits.shape[1]), int(shot.thomson.r.size),
    )


def extract_history(path: Path, out_dir: Path) -> ShotRecord:
    """Store one shot's discharge record and return its manifest line.

    Raises
    ------
    PermissionError
        If the shot belongs to the holdout part. Nothing is read in that case.
    ValueError
        If the shot's plasma-current record holds no current.
    """
    split.refuse_holdout(path.stem)
    record = DischargeRecord.from_row(read_row(path), path.stem)
    save_record(record, out_dir / f"{path.stem}.npz")
    return ShotRecord(
        path.stem, split.bucket_of(path.stem), int(record.frames.size),
        int(record.inputs.circuits.shape[1]), int(record.inputs.thomson_r.size),
    )


def _outcome(task: tuple[Path, Path, bool]) -> tuple[str, ShotRecord | None, str]:
    """Extract one shot in a worker and report ``(stem, record, error)`` without raising."""
    path, out_dir, history = task
    try:
        stored = extract_history(path, out_dir) if history else extract_shot(path, out_dir)
    except (OSError, ValueError, KeyError) as error:
        return path.stem, None, f"{type(error).__name__}: {error}"
    return path.stem, stored, ""


def pending(files: Iterable[Path], out_dir: Path) -> tuple[list[Path], int]:
    """Return the shots still to extract, and how many holdout shots were left out."""
    todo, held_out = [], 0
    for path in files:
        if split.bucket_of(path.stem) == "holdout":
            held_out += 1
        elif not (out_dir / f"{path.stem}.npz").exists():
            todo.append(path)
    return todo, held_out


def extract_all(
    files: Sequence[Path], out_dir: Path, workers: int, report_every: int = 100,
    *, history: bool = False,
) -> int:
    """Extract every shot that is not stored yet, in parallel, and write the manifest.

    A shot that cannot be read is recorded with its error and does not stop the
    run; the manifest lists the failures so that none is lost in a long log.
    Returns the number of shots newly stored. ``history`` stores discharge
    records (:func:`extract_history`) instead of aligned shots with labels.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    todo, held_out = pending(files, out_dir)
    print(f"{len(files)} files: {held_out} in the holdout part left alone, "
          f"{len(files) - held_out - len(todo)} stored already, {len(todo)} to extract "
          f"with {workers} workers", flush=True)
    started, stored, failures = time.monotonic(), 0, {}
    tasks = [(path, out_dir, history) for path in todo]
    # Workers are started afresh, not forked: a fork of a process that already runs threads
    # can deadlock in the child, and a long unattended run is where that would surface.
    fresh = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=fresh) as pool:
        for done, (stem, record, error) in enumerate(pool.map(_outcome, tasks, chunksize=4), 1):
            if record is None:
                failures[stem] = error
                print(f"FAILED {stem}: {error}", flush=True)
            else:
                stored += 1
            if done % report_every == 0 or done == len(tasks):
                minutes = (time.monotonic() - started) / 60
                print(f"{done}/{len(tasks)} extracted, {len(failures)} failed, "
                      f"{minutes:.1f} min", flush=True)
    write_manifest(out_dir, failures)
    print(f"FINISHED: {stored} new shots, {len(failures)} failed", flush=True)
    return stored


def write_manifest(out_dir: Path, failures: dict[str, str]) -> None:
    """List every stored shot with its part and sizes, and every failure, in one file."""
    records = []
    for path in sorted(out_dir.glob("*_shot_*.npz")):
        if path.name.startswith("."):
            continue
        with np.load(path, allow_pickle=False) as stored:
            # A discharge record lists its reconstruction frames among longer instants.
            frames = stored["frames"] if "frames" in stored.files else stored["times_ms"]
            records.append(asdict(ShotRecord(
                path.stem, split.bucket_of(path.stem), int(frames.size),
                int(stored["circuits"].shape[1]), int(stored["thomson_r"].size),
            )))
    parts: dict[str, int] = {}
    for record in records:
        parts[record["part"]] = parts.get(record["part"], 0) + 1
    summary = {
        "shots": len(records),
        "frames": sum(record["frames"] for record in records),
        "shots_per_part": parts,
        "failures": failures,
        "records": records,
    }
    (out_dir / MANIFEST_FILE).write_text(json.dumps(summary, indent=1) + "\n")


def main() -> int:
    """Extract a directory of labelled shots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="the labelled split")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="first so many files, for a trial")
    parser.add_argument("--history", action="store_true",
                        help="store discharge records (inputs only) instead of aligned shots")
    args = parser.parse_args()
    files = sorted(args.data_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no shot files under {args.data_dir}")
    extract_all(files if args.limit is None else files[: args.limit], args.out, args.workers,
                history=args.history)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
