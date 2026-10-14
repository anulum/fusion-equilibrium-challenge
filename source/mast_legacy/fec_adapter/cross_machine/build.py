# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — the cross-machine entry's two archives from stored physics bases

"""Build the cross-machine entry's archives from stored physics bases.

The DIII-D archive is the stored physics base of every test shot plus the
fitted correction, smoothed along time. The solve itself is not repeated here:
the base is generated once on the accelerators, and this step runs on the host
in minutes, so a new bundle costs a rebuild and not a day of solving.

A machine without a model yet is written as zeros of the right shape. The
scorer wants both files; zeros score nothing and claim nothing, and they let
the other machine's result be read off the leaderboard.

Shots are taken in the sorted order of their file names, which is the order
the dataset streams them in. The archive is keyed by position alone
(:mod:`fec_adapter.archive`), so this order is the one thing that must agree
with the scorer; the organisers' validator checks the number of frames of every
position, which a wrong order cannot pass on shots of differing length.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fec_adapter import profiles
from fec_adapter.archive import (
    DIII_D_SPLIT,
    MAST_SPLIT,
    PredictionArchive,
    split_files,
    zero_archive,
)
from fec_adapter.batched_equilibrium import PipelineSettings
from fec_adapter.physics_base import (
    SETTINGS_FILE,
    STATUS_ORDER,
    PhysicsBase,
    differing_settings,
    physics_fingerprint,
)
from fec_adapter.predict import DEFAULT_SMOOTH_FRAMES
from fec_adapter.shot_reader import read_shot
from fec_adapter.temporal import smooth_in_time

from .correction import CorrectionBundle

CONFINED = (0, 1)
"""Positions in :data:`STATUS_ORDER` of the two statuses that mean a confined plasma was found."""


@dataclass
class BuildReport:
    """What went into one archive, to be stored next to it."""

    shots: int = 0
    frames: int = 0
    non_finite_values_written_as_zero: int = 0
    status: dict[str, int] = field(
        default_factory=lambda: {entry.name: 0 for entry in STATUS_ORDER}
    )
    shots_without_confined_frame: list[str] = field(default_factory=list)


def corrected_archive(
    files: list[Path],
    base_dir: Path,
    bundle: CorrectionBundle,
    physics: dict[str, object],
    smooth_frames: int = DEFAULT_SMOOTH_FRAMES,
) -> tuple[PredictionArchive, BuildReport]:
    """Apply the bundle to the stored base of every shot, in the order given.

    Raises
    ------
    ValueError
        If the base directory was solved with other physics settings than
        ``physics``; the bundle would add the wrong difference.
    FileNotFoundError
        If a shot has no stored base. A partial archive is never written,
        because every later shot would sit at the wrong position.
    """
    recorded = json.loads((base_dir / SETTINGS_FILE).read_text())
    differing = differing_settings(recorded, physics)
    if differing:
        raise ValueError(f"{base_dir} was solved with other settings: {differing}")
    archive, report = PredictionArchive(), BuildReport()
    for path in files:
        base_file = base_dir / f"{path.stem}.npz"
        if not base_file.exists():
            raise FileNotFoundError(f"no stored base for {path.stem} (position {archive.shots})")
        shot = read_shot(path)
        base = PhysicsBase.load(base_file)
        psi, q95, beta_n = bundle.apply(shot, base, physics)
        archive.add(smooth_in_time(psi, smooth_frames), q95, beta_n)
        counts = np.bincount(base.status, minlength=len(STATUS_ORDER))
        for entry, count in zip(STATUS_ORDER, counts, strict=True):
            report.status[entry.name] += int(count)
        if not any(counts[position] for position in CONFINED):
            report.shots_without_confined_frame.append(path.stem)
    report.shots, report.frames = archive.shots, archive.frames
    report.non_finite_values_written_as_zero = archive.replaced
    return archive, report


def main() -> int:
    """Build both archives and the report into one directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="directory holding the two public test splits")
    parser.add_argument("--base-dir", type=Path, required=True,
                        help="stored physics bases of the DIII-D test shots")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="first so many shots of each split, for a trial build")
    args = parser.parse_args()

    started = time.monotonic()
    bundle = CorrectionBundle.load(args.bundle)
    physics = physics_fingerprint(profiles.ProfileModel(), PipelineSettings(), "DIII-D")
    diii_d, report = corrected_archive(
        split_files(args.data_root, DIII_D_SPLIT, args.limit), args.base_dir, bundle, physics
    )
    mast = zero_archive(split_files(args.data_root, MAST_SPLIT, args.limit))
    diii_d.write(args.out / f"{DIII_D_SPLIT}.npz")
    mast.write(args.out / f"{MAST_SPLIT}.npz")
    summary = {
        "diii_d": report.__dict__,
        "mast_shots_written_as_zeros": mast.shots,
        "bundle": str(args.bundle),
        "feature_blocks": list(bundle.feature_blocks),
        "seconds": round(time.monotonic() - started),
        "sizes_mb": {
            path.name: round(path.stat().st_size / 1e6, 1)
            for path in sorted(args.out.glob("*.npz"))
        },
    }
    (args.out / "build_report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
