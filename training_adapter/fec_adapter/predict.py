# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — prediction driver for the Challenge scorer

"""Predict a set of shots and write them in the form the scorer reads.

The Challenge's local scorer accepts a prediction archive keyed by shot index,
which is how this driver hands its results over. That route keeps the solver
and the scorer in separate processes, and therefore in separate environments:
the scorer's environment is the organisers' own lock file and holds no JAX, and
the solver's holds no scorer, so neither is run against versions it was never
tested with.

This driver is the physics alone and serves every machine. What an entry adds
on top of it lives with that entry (:mod:`fec_adapter.cross_machine`,
:mod:`fec_adapter.diii_d_accuracy`).

Shot selection mirrors the scorer's own: sorted filename order, offset by
``--skip``. The two must agree, because the archive is keyed by position.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from . import batched_equilibrium, derived, equilibrium, profiles
from .archive import PredictionArchive
from .orientation import flux_orientation
from .shot_reader import Shot, read_shot, select_frames
from .temporal import smooth_in_time

DATA_ROOT_VARIABLE = "FEC_DATA_ROOT"
"""Environment variable naming the directory that holds the dataset's splits."""


def shot_paths(root: Path, split: str, count: int, skip: int) -> list[Path]:
    """Return the shot files to predict, in the scorer's own order."""
    files = sorted((root / split).glob("*.parquet"))
    selected = files[skip : skip + count]
    if not selected:
        raise SystemExit(f"no shots selected (skip={skip} of {len(files)} files)")
    return selected


def predict_shot(
    shot: Shot, model: profiles.ProfileModel, iterations: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Solve every frame of one shot and return ``(psi, q95, betaN)``.

    Frames are solved in time order and each one warm-starts from the previous
    solution. Consecutive reconstruction times are a few milliseconds apart,
    so the equilibrium has barely moved, and a warm start both converges
    faster and stays on the same solution branch — a cold start on this
    problem can land on the competing vacuum-like fixed point.
    """
    geom = equilibrium.geometry(shot)
    knots = profiles.knots()
    psi = np.empty((shot.n_frames, shot.z_grid.size, shot.r_grid.size), dtype=np.float64)
    q95 = np.full(shot.n_frames, np.nan)
    beta_n = np.full(shot.n_frames, np.nan)

    orientation = flux_orientation(shot.plasma_current)
    warm: NDArray[np.float64] | None = None
    converged = 0
    started = time.monotonic()
    for frame in range(shot.n_frames):
        solution = equilibrium.solve_frame(shot, frame, geom, model, knots, warm, iterations)
        psi[frame] = orientation * solution.psi
        print(
            f"    frame {frame + 1}/{shot.n_frames}  converged={solution.converged}  "
            f"usable={solution.usable}  elapsed={time.monotonic() - started:.0f}s",
            flush=True,
        )
        converged += int(solution.converged)
        if solution.converged:
            warm = solution.psi_solver
        if solution.usable:
            scalars = derived.frame_scalars(
                solution.psi, shot.r_grid, shot.z_grid,
                solution.psi_axis, solution.psi_boundary,
                knots, solution.pressure, solution.ffprime,
                float(shot.plasma_current[frame]), float(shot.tf_current[frame]),
            )
            q95[frame] = scalars.q95
            beta_n[frame] = scalars.beta_n
    print(f"    solver converged on {converged}/{shot.n_frames} frames", flush=True)
    return psi, q95, beta_n


DEFAULT_SMOOTH_FRAMES = 5
"""Half-window of the time smoothing applied to a shot's written flux maps.

Five frames on either side gave the best composite of the three settings tried
on twenty development shots (0.5875 at zero, 0.5931 at two, 0.5946 at five).
"""


def predict_shot_batched(
    shot: Shot,
    model: profiles.ProfileModel,
    smooth_frames: int = DEFAULT_SMOOTH_FRAMES,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Solve a whole shot in batches on every device and return ``(psi, q95, betaN)``.

    The frames go through :func:`fec_adapter.batched_equilibrium.solve_shot`, which
    reports for every frame how its field was arrived at; the rates are printed so
    that a shot leaning mostly on the fallback is seen and not averaged away.
    The written maps are smoothed along time (:mod:`fec_adapter.temporal`).

    The two scalars are derived from each frame's own solution. This is the
    physics alone, the same on every machine; what is learned from one machine's
    reconstructions is added by the entries that are allowed to use it.
    """
    started = time.monotonic()
    settings = batched_equilibrium.PipelineSettings()
    solution = batched_equilibrium.solve_shot(shot, model, settings)
    orientation = flux_orientation(shot.plasma_current)
    knots = profiles.knots()
    psi = smooth_in_time(orientation * solution.psi, smooth_frames)
    q95 = np.full(shot.n_frames, np.nan)
    beta_n = np.full(shot.n_frames, np.nan)
    for frame in range(shot.n_frames):
        axis, boundary = float(solution.psi_axis[frame]), float(solution.psi_boundary[frame])
        if np.isfinite(axis) and np.isfinite(boundary):
            scalars = derived.frame_scalars(
                solution.psi[frame], shot.r_grid, shot.z_grid, axis, boundary,
                knots, solution.pressure[frame], solution.ffprime[frame],
                float(shot.plasma_current[frame]), float(shot.tf_current[frame]),
            )
            q95[frame] = scalars.q95
            beta_n[frame] = scalars.beta_n
    report_statuses(shot, solution, started)
    return psi, q95, beta_n


def report_statuses(shot: Shot, solution: batched_equilibrium.ShotSolution, started: float) -> None:
    """Print how the frames of one shot were arrived at."""
    rates = ", ".join(
        f"{status.name.lower()} {solution.count(status)}"
        for status in batched_equilibrium.FrameStatus
    )
    print(f"    {shot.n_frames} frames, {solution.rounds} warm rounds, "
          f"{time.monotonic() - started:.0f}s: {rates}", flush=True)


def main() -> int:
    """Predict the selected shots and write the scorer's archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=os.environ.get(DATA_ROOT_VARIABLE),
                        help=f"directory of the splits; ${DATA_ROOT_VARIABLE} when omitted")
    parser.add_argument("--split", default="diii_d_train")
    parser.add_argument("--n-shots", type=int, default=3)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=equilibrium.DEFAULT_ITERATIONS)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="cap frames per shot for a quick check (0 = every frame)")
    parser.add_argument("--frame-start", type=int, default=0,
                        help="first frame of the window; the scorer must be given the same "
                             "value, since the archive is keyed by position")
    parser.add_argument("--solver", choices=("batched", "newton"), default="batched",
                        help="batched: whole shot on every device, milliseconds a frame; "
                             "newton: frame by frame, tens of seconds a frame")
    parser.add_argument("--smooth-frames", type=int, default=DEFAULT_SMOOTH_FRAMES,
                        help="half-window of the time smoothing of the written flux maps "
                             "(batched solver; 0 switches it off)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ti-over-te", type=float,
                        default=profiles.ProfileModel.ti_over_te)
    parser.add_argument("--ffprime-alpha", type=float,
                        default=profiles.ProfileModel.ffprime_alpha)
    parser.add_argument("--ffprime-beta", type=float,
                        default=profiles.ProfileModel.ffprime_beta)
    args = parser.parse_args()
    if args.data_root is None:
        parser.error(f"no --data-root given and ${DATA_ROOT_VARIABLE} is not set")

    model = profiles.ProfileModel(
        ti_over_te=args.ti_over_te,
        ffprime_alpha=args.ffprime_alpha,
        ffprime_beta=args.ffprime_beta,
    )
    archive = PredictionArchive()
    report: list[dict[str, object]] = []

    for index, path in enumerate(shot_paths(args.data_root, args.split, args.n_shots, args.skip)):
        started = time.monotonic()
        shot = read_shot(path)
        if args.frame_start or args.max_frames:
            shot = _window(shot, args.frame_start, args.max_frames)
        if args.solver == "batched":
            psi, q95, beta_n = predict_shot_batched(shot, model, args.smooth_frames)
        else:
            psi, q95, beta_n = predict_shot(shot, model, args.iterations)
        archive.add(psi, q95, beta_n)
        elapsed = time.monotonic() - started
        report.append({
            "index": index,
            "shot": shot.name,
            "frames": shot.n_frames,
            "seconds": round(elapsed, 1),
            "q95_finite": int(np.isfinite(q95).sum()),
            "betaN_finite": int(np.isfinite(beta_n).sum()),
        })
        print(f"  {index}: {shot.name}  {shot.n_frames} frames  {elapsed:.1f}s  "
              f"q95 finite {np.isfinite(q95).sum()}/{shot.n_frames}", flush=True)

    archive.write(args.out)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


def _window(shot: Shot, start: int, frames: int) -> Shot:
    """Return the same shot restricted to a window of its frames.

    A window is a smoke test, not a result: it scores only the frames it kept,
    and the scorer must be handed the same window or the two sides disagree.

    The window also has to be chosen before the score is seen, not after. The
    first frames of a shot are the start-up, where the plasma is still forming
    and no confined boundary exists yet; a window there measures breakdown
    rather than reconstruction. That is a reason to state which window a number
    came from, and to quote every window that was run.
    """
    stop = shot.n_frames if not frames else min(start + frames, shot.n_frames)
    return select_frames(shot, slice(start, stop))

if __name__ == "__main__":
    raise SystemExit(main())
