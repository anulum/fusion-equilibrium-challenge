# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — the accuracy entry's archives from a fitted network

"""Predict a split with the fitted network and write the archives the scorer reads.

For every shot, in stream order: the reader aligns the inputs, the features are
built by the same function the fit used, the network gives the flux coordinates
and the two scalars, the basis turns the coordinates back into maps, and the
maps are smoothed along time. No physics solve takes part, so a whole test set
is minutes of host time.

The maps are smoothed along time: consecutive frames are 20 ms apart, the
equilibrium moves slowly against that, and a prediction made frame by frame
carries jitter the truth does not have. How much was measured with the
organisers' scorer on the twenty reserved shots, maps and scalars smoothed alike:

==========  ===========  ========  =========  ============
half-window  composite S  flux R²   scalars    Consistency
==========  ===========  ========  =========  ============
0            0.9820       0.9998    0.9911     0.9267
2            0.9828       0.9997    0.9892     0.9327
5            0.9803       0.9994    0.9834     0.9255
9            0.9759       0.9990    0.9745     0.9117
==========  ===========  ========  =========  ============

The derived quantities of the Consistency term gain from a little smoothing of
the maps; the two submitted scalars only lose by any. So the maps get a
half-window of two frames and the scalars none. Twenty shots separate these
rows by about a thousandth, which is near what twenty shots can resolve; the
ordering is the same in every term, which is why it is acted on.

After the smoothing the magnetic axis of every map is moved onto the axis the
network gives directly, which is the better of its two answers; why, and what
it gains, is in :mod:`fec_adapter.diii_d_accuracy.axis`.

Several networks may be named; they answer with their mean, see
:func:`fec_adapter.diii_d_accuracy.network.predict_together`.

This entry answers for DIII-D only. The scorer wants a file for both machines,
so MAST is written as zeros of the right shape, which score nothing and claim
nothing.

The same function predicts labelled shots for a local score: the split and the
window are arguments, and a window of the labelled split that was kept out of
every fit is scored exactly as the test set will be. ``--part`` takes one part
of the project's fixed split of the labelled shots instead of a window, in
sorted order; the part that is opened only once is refused.
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from fec_adapter.archive import (
    DIII_D_SPLIT,
    PredictionArchive,
    split_files,
)
from fec_adapter.diii_d_accuracy import contour_edit
from fec_adapter.diii_d_accuracy.axis import ROUNDS, axis_positions, moved_onto
from fec_adapter.diii_d_accuracy.flux_basis import FluxBasis
from fec_adapter.diii_d_accuracy.gains import Gains
from fec_adapter.diii_d_accuracy.history import layout_features
from fec_adapter.diii_d_accuracy.network import Network, predict_together
from fec_adapter.diii_d_accuracy.shape_edit import DEFAULT_BACKEND, FADE, SHARE, edited
from fec_adapter.diii_d_accuracy.x_point import LABELS as X_POINT_LABELS
from fec_adapter.diii_d_accuracy.x_point import MOVE_ROUNDS, WINDOW_WIDTH, edited_x_points
from fec_adapter.diii_d_accuracy.x_point import SHARE as X_POINT_SHARE
from fec_adapter.shot_reader import aligned_shot, input_columns
import pyarrow.parquet as pq

def read_row(path: Path):
    """Read only permitted predictor inputs, even on labelled DIII-D rows."""
    columns=input_columns(dict.fromkeys(pq.read_schema(path).names))
    table=pq.read_table(path,columns=columns)
    if table.num_rows != 1:raise ValueError(f"{path.name}: expected one shot row")
    return {name:table[name][0].as_py() for name in columns}
from fec_adapter.split import bucket_of, refuse_holdout
from fec_adapter.temporal import smooth_in_time
from observation_time_smoothing import smooth_observation_times
from fec_adapter.vessel import envelope

AXIS_OUTPUTS = ("r_axis", "z_axis")
"""The network outputs the axis of the maps is moved onto."""

SHAPE_OUTPUTS = {"kappa": "kappa", "tri_top": "tri_top", "tri_bot": "tri_bot",
                 "volume": "volume", "li": "li_scored"}
"""The scorer's name of each shape scalar → the network output the maps are bent towards."""


@dataclass(frozen=True)
class Finish:
    """What is done to a shot's raw prediction before it is written.

    Attributes
    ----------
    smooth_frames, smooth_scalars:
        Half-windows, in frames, of the time smoothing of the maps and of q95 and betaN.
    axis_rounds:
        Rounds of moving the maps' axis onto the network's axis outputs; zero leaves it.
    kept_coordinates:
        How many leading flux coordinates are taken from the networks; the rest are left
        at zero, which is the mean map's share. Zero keeps all. Measured on the validation
        part, the networks' error in the trailing coordinates exceeds those coordinates'
        own variance (at coordinate 48 nine times), so there a zero is the better answer.
    smooth_from:
        The first flux coordinate the time smoothing applies to; the ones before it are
        left as predicted. The leading coordinates carry the discharge's real changes and
        little scatter, the later ones the opposite. Zero smooths all, as before.
    smooth_axis:
        Half-window of the smoothing of the axis outputs the maps are moved onto; a
        negative value takes ``smooth_frames``, as before.
    shape:
        The scorer's names of the shape scalars the maps are bent towards the network's
        outputs of (see :mod:`.shape_edit`); empty leaves the maps' shape as decoded. The
        outputs are smoothed like the axis outputs first.
    x_points:
        Move the boundary's X-points onto the network's X-point outputs after the shape
        edit (see :mod:`.x_point`), the outputs smoothed like the axis outputs first.
    reshape_after_x_points:
        Bend the shape once more after the X-point move. The move alone gave back part of
        what the shape edit had won on κ and the lower triangularity (validation, 23 Sep).
    reshape_targets:
        The shape scalars the second shape edit bends towards, a subset of ``shape``; empty
        means all of ``shape``. On a sample of 16 validation shots the second pass's li step
        looked to cost li and flux; on the whole part, leaving it out scored lower (23 Sep).
    shape_passes:
        How many times the shape edit runs before the X-point move; each pass is the
        edit's full set of rounds. A second pass after the move raised κ beyond a single
        pass (validation, 23 Sep), so the extra rounds alone are measured by this.
    fade:
        Share of the minor radius over which every shape edit's deformation fades out
        beyond the boundary (:data:`.shape_edit.FADE`). A shorter one scored lower on the
        validation part (0.125 and 0.0625, 23 Sep): it spared the flux outside the vessel
        but asked larger levers of the plasma.
    edit_margin:
        Cells beyond the vessel envelope within which the shape edits and the X-point move
        are kept; beyond them the map is put back as it was before the edits. ``None``
        keeps the edits everywhere. The scorer reads every shape scalar inside the envelope,
        while the edits also carry the flux outside it along, where it is scored as flux.
    geometric_share, li_share:
        Share of the way from the map's reading to the network's target that every shape
        edit goes, for the four geometric scalars and for the inductance
        (:data:`.shape_edit.SHARE`). The whole way was measured worse than half of it
        (validation, 24 Sep: geometric 0.5 and inductance 0.6 best).
    reshape_geometric_share, reshape_li_share:
        The shares of the second shape edit, after the X-point move; ``None`` takes the
        first edit's.
    x_point_share:
        Share of the way from the map's X-points to the networks' that the move goes
        (:data:`.x_point.SHARE`). The whole way was measured better than 0.7 and 0.5
        (validation, 24 Sep).
    scalar_shares:
        Per-scalar geometric shares, ``(name, share)`` pairs overriding ``geometric_share``
        (and ``reshape_geometric_share``) for the scalars they name, in both shape edits.
    x_point_window, x_point_rounds:
        Width in metres of the X-point move's window, and its most corrections
        (:data:`.x_point.WINDOW_WIDTH`, :data:`.x_point.MOVE_ROUNDS`). Neither moved the
        validation score (24 Sep: 0.10, 0.20 m and 6 rounds within ±0.00001).
    axis_share_r, axis_share_z:
        Share of the way from the map's axis to the networks' that the axis move goes, in
        ``R`` and in ``Z``; the shape edits gained by going part of the way to noisy heads.
        The whole way was measured best (validation, 24 Sep).
    contour, contour_share, contour_rounds:
        Bend the map last so that its boundary follows the networks' boundary radii
        (:mod:`.contour_edit`), going ``contour_share`` of the way, in at most
        ``contour_rounds`` corrections. Last, it undid the shape edit's elongation and
        triangularities and scored lower (validation, 24 Sep: 0.0004 lower at the whole way).
    contour_first:
        Bend the contour first, before the shape edit, so the shape edit then puts the
        scalars back on their targets.
    """

    smooth_frames: int = 2
    smooth_scalars: int = 0
    axis_rounds: int = ROUNDS
    kept_coordinates: int = 0
    smooth_from: int = 0
    smooth_axis: int = -1
    shape: tuple[str, ...] = ()
    x_points: bool = False
    reshape_after_x_points: bool = False
    shape_passes: int = 1
    reshape_targets: tuple[str, ...] = ()
    fade: float = FADE
    edit_margin: int | None = None
    geometric_share: float = SHARE
    li_share: float = SHARE
    reshape_geometric_share: float | None = None
    reshape_li_share: float | None = None
    x_point_share: float = X_POINT_SHARE
    scalar_shares: tuple[tuple[str, float], ...] = ()
    x_point_window: float = WINDOW_WIDTH
    x_point_rounds: int = MOVE_ROUNDS
    axis_share_r: float = 1.0
    axis_share_z: float = 1.0
    contour: bool = False
    contour_share: float = contour_edit.SHARE
    contour_rounds: int = contour_edit.ROUNDS
    contour_first: bool = False


Prediction = tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]
"""Single precision: a run of shots crosses a process boundary, and the file is single precision."""


def _smoothed(series: NDArray[np.float64], half_window: int) -> NDArray[np.float64]:
    """Return a per-frame series smoothed along time like the maps are."""
    return smooth_in_time(series[:, np.newaxis, np.newaxis], half_window)[:, 0, 0]


def predict_shot(
    path: Path, networks: Sequence[Network], basis: FluxBasis, finish: Finish,
    gains: Gains | None = None,
) -> Prediction:
    """Return ``(psi, q95, betaN)`` of one shot file: the networks' mean, finished as told.

    ``gains``, fitted for these networks together, scale the mean's coordinates first.
    """
    maps, labels = finished_maps(path, networks, basis, finish, gains)
    scalars = [_smoothed(labels[name], finish.smooth_scalars) for name in ("q95", "beta_n")]
    return maps.astype(np.float32), scalars[0].astype(np.float32), scalars[1].astype(np.float32)


def finished_maps(
    path: Path, networks: Sequence[Network], basis: FluxBasis, finish: Finish,
    gains: Gains | None = None,
) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    """Return one shot's finished maps and the networks' mean outputs by name."""
    row = read_row(path)
    shot = aligned_shot(row, path.stem, np.asarray(row["efit_times"], dtype=np.float64))
    features = layout_features(row, shot, {network.feature_layout for network in networks})
    coordinates, labels = predict_together(networks, features)
    if gains is not None:
        coordinates = gains.applied(coordinates)
    if finish.kept_coordinates:
        coordinates[:, finish.kept_coordinates :] = 0.0
    later = coordinates[:, finish.smooth_from :]
    coordinates[:, finish.smooth_from :] = smooth_observation_times(
        later, shot.times_ms, finish.smooth_frames
    )
    maps = basis.decode(coordinates)
    axis_window = finish.smooth_frames if finish.smooth_axis < 0 else finish.smooth_axis
    if finish.axis_rounds:
        missing = [name for name in AXIS_OUTPUTS if name not in labels]
        if missing:
            raise ValueError(f"the network has no output {missing} to move the axis onto")
        target = np.column_stack(
            [_smoothed(labels[name], axis_window) for name in AXIS_OUTPUTS]
        )
        shares = np.array([finish.axis_share_r, finish.axis_share_z])
        if not np.all((shares > 0.0) & (shares <= 1.0)):
            raise ValueError("a share of the way to the axis must lie in (0, 1]")
        if np.any(shares != 1.0):  # the whole way stays bit for bit the target itself
            start = axis_positions(maps, shot.r_grid, shot.z_grid, envelope(shot.source))
            target = np.where(shares == 1.0, target, start + shares * (target - start))
        maps = moved_onto(
            maps, target, shot.r_grid, shot.z_grid, envelope(shot.source), finish.axis_rounds
        )
    maps = _shape_and_x_points(maps, labels, finish, axis_window, shot.source)
    return np.asarray(maps, dtype=np.float64), labels


def _shape_and_x_points(
    maps: NDArray[np.float64], labels: dict[str, NDArray[np.float64]], finish: Finish,
    axis_window: int, source: str,
) -> NDArray[np.float64]:
    """Bend the shape, move the X-points, bend the shape again and the contour, as told."""
    unedited = maps
    if finish.contour and finish.contour_first:
        maps = _contour_bent(maps, labels, finish, axis_window, source)
    if finish.shape:
        missing = [SHAPE_OUTPUTS[name] for name in finish.shape
                   if SHAPE_OUTPUTS[name] not in labels]
        if missing:
            raise ValueError(f"the network has no output {missing} to bend the shape towards")
        targets = {name: _smoothed(labels[SHAPE_OUTPUTS[name]], axis_window)
                   for name in finish.shape}
        if finish.shape_passes < 1:
            raise ValueError("the shape edit needs at least one pass")
        for _ in range(finish.shape_passes):
            maps, _read = edited(maps, targets, source, fade=finish.fade,
                                 geometric_share=finish.geometric_share,
                                 li_share=finish.li_share,
                                 scalar_shares=dict(finish.scalar_shares) or None)
    if finish.x_points:
        missing = [name for name in X_POINT_LABELS if name not in labels]
        if missing:
            raise ValueError(f"the network has no output {missing} to move the X-points onto")
        maps = edited_x_points(
            maps, {name: _smoothed(labels[name], axis_window) for name in X_POINT_LABELS},
            source, share=finish.x_point_share, width=finish.x_point_window,
            rounds=finish.x_point_rounds,
        )
    if finish.reshape_after_x_points:
        if not (finish.shape and finish.x_points):
            raise ValueError("a second shape edit needs both the shape edit and the X-point move")
        second = finish.reshape_targets or finish.shape
        if not set(second) <= set(finish.shape):
            raise ValueError("the second shape edit's targets must be among the first's")
        maps, _read = edited(
            maps, {name: targets[name] for name in second}, source, fade=finish.fade,
            geometric_share=(finish.geometric_share if finish.reshape_geometric_share is None
                             else finish.reshape_geometric_share),
            li_share=(finish.li_share if finish.reshape_li_share is None
                      else finish.reshape_li_share),
            scalar_shares=dict(finish.scalar_shares) or None,
        )
    if finish.contour and not finish.contour_first:
        maps = _contour_bent(maps, labels, finish, axis_window, source)
    return _kept_within_margin(maps, unedited, finish, source)


def _contour_bent(
    maps: NDArray[np.float64], labels: dict[str, NDArray[np.float64]], finish: Finish,
    axis_window: int, source: str,
) -> NDArray[np.float64]:
    """Bend every frame so its boundary follows the networks' smoothed boundary radii."""
    names = contour_edit.label_names()
    missing = [name for name in names if name not in labels]
    if missing:
        raise ValueError(f"the network has no output {missing[:3]}… to bend the contour to")
    return contour_edit.edited(
        maps, {name: _smoothed(labels[name], axis_window) for name in names}, source,
        share=finish.contour_share, rounds=finish.contour_rounds,
    )


def _kept_within_margin(
    maps: NDArray[np.float64], unedited: NDArray[np.float64], finish: Finish, source: str,
) -> NDArray[np.float64]:
    """Put the unedited map back beyond the vessel and its margin, if a margin is set."""
    if finish.edit_margin is None:
        return maps
    if finish.edit_margin < 0:
        raise ValueError("the edit margin is a number of cells, zero or more")
    kept = envelope(source)
    if finish.edit_margin:
        kept = ndimage.binary_dilation(kept, iterations=finish.edit_margin)
    result: NDArray[np.float64] = np.where(kept[np.newaxis], maps, unedited)
    return result


def _predict_many(
    task: tuple[list[Path], list[Path], Path, Finish, Path | None],
) -> list[Prediction]:
    """Predict several shots in a worker, returning them in the order given."""
    files, network_files, basis_file, finish, gains_file = task
    gains = None if gains_file is None else Gains.load(gains_file)
    networks = [Network.load(network_file) for network_file in network_files]
    basis = FluxBasis.load(basis_file).truncated(networks[0].coordinates)
    return [predict_shot(path, networks, basis, finish, gains) for path in files]


def predicted_archive(
    files: Sequence[Path], network_files: Sequence[Path], basis_file: Path, workers: int,
    finish: Finish, gains_file: Path | None = None,
) -> PredictionArchive:
    """Predict ``files`` in parallel and collect them in the order given.

    The files are cut into consecutive runs, one per worker, and the runs are
    added back in their own order, so a shot's position in the archive is its
    position in ``files`` whatever finishes first.
    """
    if not files:
        raise ValueError("no shot files to predict")
    bounds = np.linspace(0, len(files), min(workers, len(files)) + 1).astype(int)
    runs = [list(files[start:stop]) for start, stop in itertools.pairwise(bounds)]
    fresh = multiprocessing.get_context("spawn")
    archive = PredictionArchive()
    with ProcessPoolExecutor(max_workers=len(runs), mp_context=fresh) as pool:
        tasks = [(run, list(network_files), basis_file, finish, gains_file) for run in runs]
        for predictions in pool.map(_predict_many, tasks):
            for maps, q95, beta_n in predictions:
                archive.add(maps, q95, beta_n)
    return archive


def _finish_from(args: argparse.Namespace) -> Finish:
    """Return the finish the command line asks for."""
    return Finish(args.smooth_frames, args.smooth_scalars, args.axis_rounds,
                  args.kept_coordinates, args.smooth_from, args.smooth_axis, tuple(args.shape),
                  args.x_points, args.reshape_after_x_points, args.shape_passes,
                  tuple(args.reshape_targets), args.fade, args.edit_margin,
                  args.geometric_share, args.li_share, args.reshape_geometric_share,
                  args.reshape_li_share, args.x_point_share,
                  tuple((name, float(value)) for name, value in
                        (item.split("=", 1) for item in args.scalar_share)),
                  args.x_point_window, args.x_point_rounds, args.axis_share_r,
                  args.axis_share_z, args.contour, args.contour_share, args.contour_rounds,
                  args.contour_first)


def _parser() -> argparse.ArgumentParser:
    """Return the command line of :func:`main`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--network", type=Path, required=True, nargs="+",
                        help="one fitted network, or several whose mean is predicted")
    parser.add_argument("--basis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default=DIII_D_SPLIT)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--part", default=None,
                        help="predict this part of the fixed split instead of a window")
    parser.add_argument("--smooth-frames", type=int, default=Finish.smooth_frames,
                        help="half-window of the time smoothing of the flux maps")
    parser.add_argument("--smooth-scalars", type=int, default=Finish.smooth_scalars,
                        help="half-window of the time smoothing of q95 and betaN")
    parser.add_argument("--axis-rounds", type=int, default=Finish.axis_rounds,
                        help="rounds of moving the axis onto the network's outputs; 0 leaves it")
    parser.add_argument("--kept-coordinates", type=int, default=Finish.kept_coordinates,
                        help="leading flux coordinates taken from the networks; 0 keeps all")
    parser.add_argument("--smooth-from", type=int, default=Finish.smooth_from,
                        help="first flux coordinate the time smoothing applies to")
    parser.add_argument("--smooth-axis", type=int, default=Finish.smooth_axis,
                        help="half-window for the axis outputs; negative takes --smooth-frames")
    parser.add_argument("--gains", type=Path, default=None,
                        help="gains fitted for these networks together (see the gains module)")
    parser.add_argument("--shape", nargs="*", default=[], choices=sorted(SHAPE_OUTPUTS),
                        help="shape scalars the maps are bent towards the network's outputs of")
    parser.add_argument("--x-points", action="store_true",
                        help="move the boundary X-points onto the network's X-point outputs")
    parser.add_argument("--reshape-after-x-points", action="store_true",
                        help="bend the shape once more after the X-point move")
    parser.add_argument("--reshape-targets", nargs="*", default=[], choices=sorted(SHAPE_OUTPUTS),
                        help="shape scalars of the second shape edit (default: all of --shape)")
    parser.add_argument("--shape-passes", type=int, default=Finish.shape_passes,
                        help="how many times the shape edit runs before the X-point move")
    parser.add_argument("--fade", type=float, default=Finish.fade,
                        help="share of the minor radius over which the shape edit fades out")
    parser.add_argument("--edit-margin", type=int, default=None,
                        help="cells beyond the vessel within which the edits are kept")
    parser.add_argument("--geometric-share", type=float, default=Finish.geometric_share,
                        help="share of the way to the geometric targets each edit goes")
    parser.add_argument("--li-share", type=float, default=Finish.li_share,
                        help="share of the way to the inductance target each edit goes")
    parser.add_argument("--reshape-geometric-share", type=float, default=None,
                        help="geometric share of the second edit (default: --geometric-share)")
    parser.add_argument("--reshape-li-share", type=float, default=None,
                        help="inductance share of the second shape edit (default: --li-share)")
    parser.add_argument("--x-point-share", type=float, default=Finish.x_point_share,
                        help="share of the way to the networks' X-points the move goes")
    parser.add_argument("--scalar-share", nargs="*", default=[], metavar="NAME=SHARE",
                        help="per-scalar geometric shares, e.g. kappa=0.7 tri_top=0.4")
    parser.add_argument("--x-point-window", type=float, default=Finish.x_point_window,
                        help="width in metres of the X-point move's window")
    parser.add_argument("--x-point-rounds", type=int, default=Finish.x_point_rounds,
                        help="most corrections of each X-point move")
    parser.add_argument("--axis-share-r", type=float, default=Finish.axis_share_r,
                        help="share of the way to the networks' axis R the axis move goes")
    parser.add_argument("--axis-share-z", type=float, default=Finish.axis_share_z,
                        help="share of the way to the networks' axis Z the axis move goes")
    parser.add_argument("--contour", action="store_true",
                        help="bend the map last so its boundary follows the networks' radii")
    parser.add_argument("--contour-share", type=float, default=Finish.contour_share,
                        help="share of the way to the networks' boundary the contour bend goes")
    parser.add_argument("--contour-rounds", type=int, default=Finish.contour_rounds,
                        help="most corrections of the contour bend")
    parser.add_argument("--contour-first", action="store_true",
                        help="bend the contour before the shape edit instead of last")
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main() -> int:
    """Write the requested DIII-D archive without touching another machine archive."""
    parser = _parser()
    args = parser.parse_args()
    if (args.out / f"{args.split}.npz").exists():raise FileExistsError("Existing archive retained")

    started = time.monotonic()
    stop = None if args.limit is None else args.skip + args.limit
    files = split_files(args.data_root, args.split, None)
    if args.part is None:
        files = files[args.skip : stop]
    else:
        files = [path for path in files if bucket_of(path.stem) == args.part]
        for path in files:
            refuse_holdout(path.stem)
    finish = _finish_from(args)
    archive = predicted_archive(
        files, args.network, args.basis, args.workers, finish, args.gains
    )
    if archive.replaced:raise RuntimeError("Refuse silent nonfinite-to-zero replacement")
    archive.write(args.out / f"{args.split}.npz")
    summary: dict[str, object] = {
        "split": args.split, "skip": args.skip, "part": args.part,
        "shots": archive.shots, "frames": archive.frames,
        "non_finite_values_written_as_zero": archive.replaced,
        "networks": [str(network) for network in args.network], "smooth_frames": args.smooth_frames,
        "smooth_scalars": args.smooth_scalars, "axis_rounds": args.axis_rounds,
        "kept_coordinates": args.kept_coordinates,
        "smooth_from": args.smooth_from, "smooth_axis": args.smooth_axis, "shape": args.shape,
        "x_points": args.x_points, "reshape_after_x_points": args.reshape_after_x_points,
        "shape_passes": args.shape_passes, "reshape_targets": args.reshape_targets,
        "fade": args.fade, "edit_margin": args.edit_margin,
        "geometric_share": args.geometric_share, "li_share": args.li_share,
        "reshape_geometric_share": args.reshape_geometric_share,
        "reshape_li_share": args.reshape_li_share, "x_point_share": args.x_point_share,
        "scalar_shares": args.scalar_share, "x_point_window": args.x_point_window,
        "x_point_rounds": args.x_point_rounds, "axis_share_r": args.axis_share_r,
        "axis_share_z": args.axis_share_z, "contour": args.contour,
        "contour_share": args.contour_share, "contour_rounds": args.contour_rounds,
        "contour_first": args.contour_first,
        "gains": None if args.gains is None else str(args.gains),
        # Who read the maps the edit steers by; both readers give the same bits.
        "reading": DEFAULT_BACKEND if args.shape or args.x_points else None,
    }
    summary["seconds"] = round(time.monotonic() - started)
    (args.out / "build_report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
