# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — the archive of a machine reconstructed from its inputs alone

"""Write the prediction archive of a split with the equal-temperature reconstruction.

Every shot is reconstructed by :mod:`.thomson_isoflux`, and ``q95`` and ``βN``
are read off the solved maps by :mod:`.map_scalars`. A frame without a solved
map has no scalar of its own; it takes the value interpolated in time between
the solved frames of its shot, and a shot without any takes the median over the
whole split — a number that comes from the inputs of that split, not a zero,
which the scorer would count as a prediction of zero.

The shots are predicted in parallel in consecutive runs and collected in the
order of the sorted file names, the order the scorer reads the archive in.
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from fec_adapter.archive import PredictionArchive, split_files
from fec_adapter.cross_machine import map_scalars, thomson_isoflux
from fec_adapter.shot_reader import read_shot
from fec_adapter.split import bucket_of, refuse_holdout

MEDIAN_FRAMES = 5

PARTS: Final = ("training", "calibration", "validation")
"""The parts of the fixed split that may be reconstructed; the holdout part is opened only once."""

@dataclass(frozen=True)
class ShotResult:
    """One reconstructed shot.

    Attributes
    ----------
    maps:
        The flux maps, one per frame.
    q95, beta_n:
        The scalars after filling in time and the running median; NaN everywhere when
        no frame gave a value.
    solved:
        How many frames have a solved map.
    pattern_sum:
        The sum of the solved frames' current patterns, each a distribution over the
        interior cells summing to one.
    q95_read, beta_n_read:
        How many solved frames gave a finite value before any filling. A solved map can
        still give none (no closed surface at 95 % of the flux, no Thomson sample inside
        the boundary), and these counts tell that apart from a frame that was not solved.
    """

    maps: NDArray[np.float32]
    q95: NDArray[np.float64]
    beta_n: NDArray[np.float64]
    solved: int
    pattern_sum: NDArray[np.float64]
    q95_read: int
    beta_n_read: int


def predict_shot(
    path: Path, geometry: thomson_isoflux.Geometry | None = None,
    fallback: NDArray[np.float64] | None = None,
) -> ShotResult:
    """Reconstruct one shot file and read the scalars off its solved frames."""
    shot = read_shot(path)
    maps, solved = thomson_isoflux.reconstruct(shot, geometry, fallback)
    q95 = np.full(shot.n_frames, np.nan)
    beta_n = np.full(shot.n_frames, np.nan)
    for frame, result in enumerate(solved):
        if result is None:
            continue
        f_vacuum = map_scalars.vacuum_f(shot.source, float(shot.tf_current[frame]))
        q95[frame] = map_scalars.safety_factor(
            result.flux, shot.r_grid, shot.z_grid, result.axis, result.axis_flux,
            result.edge_flux, f_vacuum,
        )
        beta_n[frame] = map_scalars.normalised_beta(
            result.flux, shot.r_grid, shot.z_grid, result.axis, result.axis_flux,
            result.edge_flux, shot.thomson.r, shot.thomson.z, shot.thomson.te[frame],
            shot.thomson.ne[frame], float(shot.plasma_current[frame]), f_vacuum,
        )
    patterns = [result.current / result.current.sum() for result in solved if result is not None]
    return ShotResult(
        maps=maps.astype(np.float32), q95=steadied(q95), beta_n=steadied(beta_n),
        solved=len(patterns),
        pattern_sum=np.sum(patterns, axis=0) if patterns else np.zeros(0),
        q95_read=int(np.isfinite(q95).sum()), beta_n_read=int(np.isfinite(beta_n).sum()),
    )


def steadied(series: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fill the gaps in time, then take the running median of :data:`MEDIAN_FRAMES` frames.

    A single frame's Thomson samples scatter, and so does what is read off through them; the
    quantities themselves change over many frames.
    """
    filled = filled_in_time(series)
    if not np.isfinite(filled).all():
        return filled
    steady: NDArray[np.float64] = ndimage.median_filter(filled, size=MEDIAN_FRAMES, mode="nearest")
    return steady


def filled_in_time(series: NDArray[np.float64]) -> NDArray[np.float64]:
    """Interpolate over the frames without a value; a series without any stays NaN."""
    known = np.isfinite(series)
    if not known.any():
        return series
    frames = np.arange(series.size)
    filled: NDArray[np.float64] = np.interp(frames, frames[known], series[known])
    return filled


_GEOMETRY: dict[str, thomson_isoflux.Geometry] = {}
"""One geometry per machine and worker process; building it takes longer than solving a shot."""


def _predict_many(files: list[Path]) -> list[ShotResult]:
    results = []
    for path in files:
        source = read_shot(path).source
        if source not in _GEOMETRY:
            _GEOMETRY[source] = thomson_isoflux.Geometry.of(read_shot(path))
        results.append(predict_shot(path, _GEOMETRY[source]))
    return results


def _with_mean_pattern_where_nothing_solved(
    files: list[Path], results: list[ShotResult]
) -> list[ShotResult]:
    """Redo the shots without a solved frame with the split's mean current pattern.

    Their maps were made with a parabolic current in the middle of the vessel; the mean
    pattern of every solved frame of the split is a plasma of this machine's usual shape and
    position, and it, too, comes from the inputs alone.
    """
    summed = [result.pattern_sum for result in results if result.solved > 0]
    if not summed:
        return results
    mean = np.sum(summed, axis=0) / sum(result.solved for result in results)
    redone = list(results)
    for index, result in enumerate(results):
        if result.solved == 0:
            source = read_shot(files[index]).source
            if source not in _GEOMETRY:
                _GEOMETRY[source] = thomson_isoflux.Geometry.of(read_shot(files[index]))
            redone[index] = predict_shot(files[index], _GEOMETRY[source], mean)
    return redone


def selected_files(data_root: Path, split: str, limit: int | None,
                   part: str | None) -> list[Path]:
    """Return the shot files to reconstruct: the first ``limit`` of the split, or one part of it.

    A part (``validation``, ``calibration``, ``training``) is taken from the fixed split in
    sorted order, so a machine with labels can score the reconstruction exactly as the
    accuracy builds are scored; the part that is opened only once is refused.
    """
    if part is None:
        return split_files(data_root, split, limit)
    if part not in PARTS:
        raise PermissionError(f"the part {part!r} is not one that may be read; parts: {PARTS}")
    files = [path for path in split_files(data_root, split, None) if bucket_of(path.stem) == part]
    for path in files:
        refuse_holdout(path.stem)
    return files


def main() -> int:
    """Reconstruct a split and write its archive and a short report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--part", default=None,
                        help="reconstruct this part of the fixed split instead of the first shots")
    args = parser.parse_args()

    started = time.monotonic()
    files = selected_files(args.data_root, args.split, args.limit, args.part)
    bounds = np.linspace(0, len(files), min(args.workers, len(files)) * 4 + 1).astype(int)
    runs = [list(files[a:b]) for a, b in itertools.pairwise(bounds) if b > a]
    fresh = multiprocessing.get_context("spawn")
    results: list[ShotResult] = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=fresh) as pool:
        for done, part in enumerate(pool.map(_predict_many, runs)):
            results.extend(part)
            print(json.dumps({"runs_done": done + 1, "of": len(runs), "shots": len(results),
                              "minutes": round((time.monotonic() - started) / 60.0, 1)}),
                  flush=True)
    results = _with_mean_pattern_where_nothing_solved(files, results)
    every_q = np.concatenate([result.q95 for result in results])
    every_beta = np.concatenate([result.beta_n for result in results])
    fallback = (float(np.nanmedian(every_q)), float(np.nanmedian(every_beta)))
    archive = PredictionArchive()
    for result in results:
        archive.add(result.maps, np.where(np.isfinite(result.q95), result.q95, fallback[0]),
                    np.where(np.isfinite(result.beta_n), result.beta_n, fallback[1]))
    archive.write(args.out / f"{args.split}.npz")
    report = {
        "split": args.split, "shots": archive.shots,
        "frames": int(sum(result.maps.shape[0] for result in results)),
        "solved_frames": int(sum(result.solved for result in results)),
        "q95_read_frames": int(sum(result.q95_read for result in results)),
        "beta_n_read_frames": int(sum(result.beta_n_read for result in results)),
        "shots_without_a_solved_frame": int(sum(result.solved == 0 for result in results)),
        "shots_without_a_read_q95": int(sum(result.q95_read == 0 for result in results)),
        "shots_without_a_read_beta_n": int(sum(result.beta_n_read == 0 for result in results)),
        "scalar_fallback": {"q95": fallback[0], "beta_n": fallback[1]},
        "non_finite_values_written_as_zero": archive.replaced,
        "seconds": round(time.monotonic() - started),
        # Per shot, in archive order: frames, solved frames, and the solved frames that
        # gave a finite q95 and βN before any filling, so the shots the solve or the
        # scalar reading struggles with can be found without their labels.
        "per_shot": [
            [int(result.maps.shape[0]), result.solved, result.q95_read, result.beta_n_read]
            for result in results
        ],
    }
    (args.out / f"isoflux_report_{args.split}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
