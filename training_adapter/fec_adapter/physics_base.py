# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — the stored result of the physics solve, one file per shot

"""Solve whole shots once and keep what the solve produced, one file per shot.

The solve is the expensive step — tens of seconds of accelerator time a shot —
and everything built on it is cheap, so its result is stored and reused: a new
fit or a new archive reads the files and does not solve again.

One module writes the files and reads them back, so the layout is stated once.
A directory of bases also records the physics settings it was solved with
(:func:`physics_fingerprint`); whoever reads the directory compares that record
with the settings of the running package and refuses a mismatch
(:func:`differing_settings`), because a base solved with another pressure ratio
or another cold support is a different base under the same file names.

The run is resumable and leaves no half-written file under a final name: each
shot is written under a hidden ``.…partial.npz`` name and renamed when complete.
A long run on accelerators can hang without exiting, which no code inside the
hung process can detect; :mod:`fec_adapter.stall_watchdog` runs this module from
outside and restarts it, and resumability is what makes a restart cost one shot.

No reconstruction column is read here: shots come through the reader, which
offers inputs only.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from . import batched_equilibrium, profiles
from .batched_equilibrium import FrameStatus, PipelineSettings, ShotSolution
from .orientation import flux_orientation
from .shot_reader import read_shot

STATUS_ORDER = tuple(FrameStatus)
"""Order of the status codes in a stored base, and of the status flags built from them."""

SETTINGS_FILE = "SETTINGS.json"
SKIPPED_FILE = "SKIPPED.json"
RESERVED_FILE = "RESERVED_FOR_EVALUATION.json"
CURRENT_FILE = "CURRENT"

COMPARED_SETTINGS = ("profile_model", "cold_support", "minimum_area")
"""Entries of the physics fingerprint that every base directory records."""

MINIMUM_FREE_GB = 15.0
"""Free space on the output disk below which the run stops instead of filling it."""


@dataclass(frozen=True)
class PhysicsBase:
    """What the physics solve produced for one shot, in the written orientation.

    Attributes
    ----------
    psi:
        Flux maps, shape ``(T, NZ, NR)``, oriented as they are written.
    status:
        Index into :data:`STATUS_ORDER` per frame, shape ``(T,)``.
    enclosed_area, psi_axis, psi_boundary:
        Per frame, shape ``(T,)``; the two fluxes are in the solver's own sign
        and may be ``nan``.
    pprime, ffprime, pressure:
        On the knots, shape ``(T, N_knots)``.
    """

    psi: NDArray[np.float64]
    status: NDArray[np.int64]
    enclosed_area: NDArray[np.float64]
    psi_axis: NDArray[np.float64]
    psi_boundary: NDArray[np.float64]
    pprime: NDArray[np.float64]
    ffprime: NDArray[np.float64]
    pressure: NDArray[np.float64]

    @classmethod
    def from_solution(cls, solution: ShotSolution, orientation: float) -> PhysicsBase:
        """Take the base straight from a solve."""
        return cls(
            psi=orientation * solution.psi,
            status=np.asarray(
                [STATUS_ORDER.index(entry) for entry in solution.status], dtype=np.int64
            ),
            enclosed_area=solution.enclosed_area,
            psi_axis=solution.psi_axis,
            psi_boundary=solution.psi_boundary,
            pprime=solution.pprime,
            ffprime=solution.ffprime,
            pressure=solution.pressure,
        )

    @classmethod
    def load(cls, path: Path) -> PhysicsBase:
        """Read a stored base."""
        with np.load(path, allow_pickle=False) as stored:
            return cls(
                psi=stored["psi"].astype(np.float64),
                status=stored["status"].astype(np.int64),
                enclosed_area=stored["enclosed_area"].astype(np.float64),
                psi_axis=stored["psi_axis"].astype(np.float64),
                psi_boundary=stored["psi_boundary"].astype(np.float64),
                pprime=stored["pprime"].astype(np.float64),
                ffprime=stored["ffprime"].astype(np.float64),
                pressure=stored["pressure"].astype(np.float64),
            )

    def save(self, path: Path, orientation: float, times_ms: NDArray[np.float64]) -> None:
        """Store the base in single precision, complete or not at all.

        The orientation and the time base are stored with it: neither is needed
        to use the base, both are needed to check it against its shot.
        """
        partial = path.with_name(f".{path.stem}.partial.npz")
        np.savez_compressed(
            partial,
            allow_pickle=False,
            psi=self.psi.astype(np.float32),
            orientation=np.float32(orientation),
            status=self.status.astype(np.uint8),
            enclosed_area=self.enclosed_area.astype(np.float32),
            psi_axis=self.psi_axis.astype(np.float32),
            psi_boundary=self.psi_boundary.astype(np.float32),
            pprime=self.pprime.astype(np.float32),
            ffprime=self.ffprime.astype(np.float32),
            pressure=self.pressure.astype(np.float32),
            times_ms=times_ms.astype(np.float32),
        )
        partial.rename(path)


def physics_fingerprint(
    model: profiles.ProfileModel, settings: PipelineSettings, machine: str
) -> dict[str, object]:
    """Return what decides the physics base: profile parameters, cold support, area threshold."""
    support = settings.cold_support_for(machine)
    return {
        "machine": machine,
        "profile_model": asdict(model),
        "cold_support": None if support is None else asdict(support),
        "minimum_area": settings.minimum_area,
        "max_rounds": settings.max_rounds,
        "cold": asdict(settings.cold),
        "warm": asdict(settings.warm),
    }


def differing_settings(recorded: dict[str, object], physics: dict[str, object]) -> list[str]:
    """Return the settings on which a base directory and the running package disagree."""
    return [
        key for key in COMPARED_SETTINGS
        if json.dumps(recorded.get(key), sort_keys=True) != json.dumps(physics[key], sort_keys=True)
    ]


def finished_files(directory: Path, limit: int | None = None) -> list[Path]:
    """Return the finished per-shot bases of a directory, in sorted order.

    ``limit`` keeps the first so many. The generator works through the shots in
    sorted order, so a limit reproduces the set an earlier fit saw and lets two
    fits differ in one thing only.

    ``Path.glob`` matches hidden names, so the half-written files are left out
    here by name: a fit started while the generator runs must not read one.
    """
    finished = sorted(
        path for path in directory.glob("*_shot_*.npz") if not path.name.startswith(".")
    )
    return finished if limit is None else finished[:limit]


def _listed(path: Path) -> set[str]:
    """Return the shot names a directory's list file holds, or none if there is no file."""
    if not path.exists():
        return set()
    return {str(name) for name in json.loads(path.read_text())}


def generate(
    files: Sequence[Path],
    out_dir: Path,
    model: profiles.ProfileModel,
    settings: PipelineSettings,
    before_each: Callable[[], None] = lambda: None,
    report: Callable[[str], None] = print,
) -> int:
    """Solve every shot that has no stored base yet and return how many were solved.

    Parameters
    ----------
    files:
        Shot files, solved in the order given.
    out_dir:
        Directory of the bases. Shots named in its ``SKIPPED.json`` or
        ``RESERVED_FOR_EVALUATION.json`` are left alone: the first holds shots
        that could not be solved, the second shots that a training base must
        never contain.
    model, settings:
        The physics the base is solved with; recorded in the directory.
    before_each:
        Called before every shot, for checks that belong to the place the run
        happens in, such as a network mount that has to be present.
    report:
        Receives one line per shot and the closing line.

    Raises
    ------
    ValueError
        If the directory already records other physics settings.
    OSError
        If the output disk has less than :data:`MINIMUM_FREE_GB` free.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    left_alone = _listed(out_dir / SKIPPED_FILE) | _listed(out_dir / RESERVED_FILE)
    started, solved = time.monotonic(), 0
    for position, path in enumerate(files, start=1):
        target = out_dir / f"{path.stem}.npz"
        if target.exists() or path.stem in left_alone:
            continue
        (out_dir / CURRENT_FILE).write_text(path.stem + "\n")
        free_gb = shutil.disk_usage(out_dir).free / 1e9
        if free_gb < MINIMUM_FREE_GB:
            raise OSError(f"only {free_gb:.1f} GB free under {out_dir}; stopping")
        before_each()
        shot_started = time.monotonic()
        shot = read_shot(path)
        _record_settings(out_dir, physics_fingerprint(model, settings, shot.source))
        solution = batched_equilibrium.solve_shot(shot, model, settings)
        orientation = flux_orientation(shot.plasma_current)
        PhysicsBase.from_solution(solution, orientation).save(target, orientation, shot.times_ms)
        solved += 1
        report(
            f"{position}/{len(files)} {path.stem} {shot.n_frames} frames "
            f"{time.monotonic() - shot_started:.0f}s | "
            f"fallback {solution.count(FrameStatus.FALLBACK)} "
            f"unresolved {solution.count(FrameStatus.UNRESOLVED)} | free {free_gb:.0f} GB | "
            f"elapsed {(time.monotonic() - started) / 60:.0f} min"
        )
    report(f"FINISHED: {solved} new shots, {len(finished_files(out_dir))} on disk")
    return solved


def _record_settings(out_dir: Path, physics: dict[str, object]) -> None:
    """Write the directory's settings record, or hold an existing one to the same physics."""
    record = out_dir / SETTINGS_FILE
    if record.exists():
        differing = differing_settings(json.loads(record.read_text()), physics)
        if differing:
            raise ValueError(f"{out_dir} holds bases solved with other settings: {differing}")
        return
    codes = {entry.name: code for code, entry in enumerate(STATUS_ORDER)}
    record.write_text(json.dumps({**physics, "status_codes": codes}, indent=2) + "\n")


def mount_check(directory: Path, expected_files: int) -> Callable[[], None]:
    """Return a check that a dataset directory still lists every file it should.

    A network mount that drops mid-run lists nothing or a part; solving on a
    partial listing would store a base that silently lacks shots, so the run
    stops instead.
    """
    def check() -> None:
        try:
            found = sum(1 for _ in directory.iterdir())
        except OSError as error:
            raise OSError(f"{directory} cannot be listed: {error}") from error
        if found != expected_files:
            raise OSError(f"{directory} lists {found} files, expected {expected_files}")

    return check


def main() -> int:
    """Generate the bases of a list of shots, or of every shot of a directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="directory of one split")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stems", type=Path, default=None,
                        help="JSON list of shot names; every shot of the directory when omitted")
    parser.add_argument("--expected-files", type=int, default=None,
                        help="stop if the data directory ever lists another number of files")
    args = parser.parse_args()

    if args.stems is None:
        files = sorted(args.data_dir.glob("*.parquet"))
    else:
        files = [args.data_dir / f"{stem}.parquet" for stem in json.loads(args.stems.read_text())]
    check = (
        mount_check(args.data_dir, args.expected_files)
        if args.expected_files is not None else (lambda: None)
    )
    check()
    generate(files, args.out, profiles.ProfileModel(), PipelineSettings(), check,
             lambda line: print(line, flush=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
