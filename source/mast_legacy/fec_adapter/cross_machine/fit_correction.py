# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — fit the learned correction on a generated physics base

"""Fit the correction bundle on a directory of physics bases and report on shots it never saw.

The split is by shot and by the shot's name, not by its position in a listing:
a shot belongs to the held-out part if a hash of its name says so, so the
held-out set stays the same while the base directory is still growing, and two
fits months apart are compared on the same shots.

The base directory records the physics settings it was generated with. The fit
refuses a base whose settings differ from the ones the running package would
solve with, because the bundle it writes is stamped with the running settings
and would otherwise be stamped wrongly.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fec_adapter import profiles
from fec_adapter.batched_equilibrium import PipelineSettings
from fec_adapter.physics_base import (
    SETTINGS_FILE,
    PhysicsBase,
    differing_settings,
    finished_files,
    physics_fingerprint,
)
from fec_adapter.shot_reader import read_row, read_shot, select_frames

from .correction import (
    CorrectionBundle,
    FitSettings,
    TrainingShot,
    fit_bundle,
)
from .residual_features import DEFAULT_BLOCKS

PERCENT = 100


def is_held_out(stem: str, holdout_percent: int) -> bool:
    """Return whether a shot belongs to the held-out part, from its name alone."""
    if not 0 < holdout_percent < PERCENT:
        raise ValueError(
            f"holdout_percent must lie strictly between 0 and {PERCENT}, got {holdout_percent}"
        )
    digest = hashlib.sha256(f"correction-holdout:{stem}".encode()).hexdigest()
    return int(digest, 16) % PERCENT < holdout_percent


def pooled_r2(truth: NDArray[np.float64], predicted: NDArray[np.float64]) -> float:
    """Return one ratio of sums over every element, as the Challenge scorer pools it."""
    return float(1.0 - np.sum((truth - predicted) ** 2) / np.sum((truth - truth.mean()) ** 2))


def load_training_shot(data_file: Path, base_file: Path, frame_step: int) -> TrainingShot:
    """Read one shot's inputs, base and reconstruction, keeping every ``frame_step``-th frame."""
    row = read_row(data_file)
    shot = read_shot(data_file)
    base = PhysicsBase.load(base_file)
    keep = np.arange(0, shot.n_frames, frame_step)
    shot = select_frames(shot, keep)
    base = PhysicsBase(**{name: getattr(base, name)[keep] for name in fields_of_base()})
    return TrainingShot(
        shot, base,
        np.asarray(row["efit_psirz"], dtype=np.float64)[keep],
        np.asarray(row["efit_q95"], dtype=np.float64)[keep],
        np.asarray(row["efit_beta_n"], dtype=np.float64)[keep],
    )


def main() -> int:
    """Fit on the shots not held out, report on the held-out ones, write the bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--components", type=int, default=16)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--holdout-percent", type=int, default=20)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--with-phase", action="store_true",
                        help="fit with the optional discharge-phase feature block")
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first so many bases in sorted order")
    args = parser.parse_args()

    model, settings = profiles.ProfileModel(), PipelineSettings()
    physics = physics_fingerprint(model, settings, "DIII-D")
    recorded = json.loads((args.base_dir / SETTINGS_FILE).read_text())
    if differing := differing_settings(recorded, physics):
        raise SystemExit(f"the base was generated with other physics settings: {differing}")

    bases = finished_files(args.base_dir, args.limit)
    fit_files = [path for path in bases if not is_held_out(path.stem, args.holdout_percent)]
    held_files = [path for path in bases if is_held_out(path.stem, args.holdout_percent)]
    print(f"{len(fit_files)} shots to fit on, {len(held_files)} held out", flush=True)

    def samples(files: list[Path]) -> list[TrainingShot]:
        return [
            load_training_shot(args.data_dir / f"{path.stem}.parquet", path, args.frame_step)
            for path in files
        ]

    blocks = ("phase",) if args.with_phase else DEFAULT_BLOCKS
    fit_settings = FitSettings(args.components, args.ridge, feature_blocks=blocks)
    bundle = fit_bundle(samples(fit_files), physics, fit_settings)
    bundle.save(args.out)
    report = _held_out_report(bundle, samples(held_files), physics)
    report.update({
        "fit_shots": len(fit_files), "held_out_shots": len(held_files),
        "components": args.components, "ridge": args.ridge, "frame_step": args.frame_step,
        "feature_blocks": list(blocks),
        "held_out": [path.stem for path in held_files],
    })
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "held_out"}, indent=2))
    return 0


def _held_out_report(
    bundle: CorrectionBundle, held: list[TrainingShot], physics: dict[str, object]
) -> dict[str, object]:
    """Pool the flux and the two scalars over the held-out shots, before and after correction."""
    truth, physics_maps, corrected = [], [], []
    scalars: dict[str, tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]] = {
        "q95": ([], []), "beta_n": ([], []),
    }
    for sample in held:
        psi, q95, beta_n = bundle.apply(sample.shot, sample.base, physics)
        truth.append(sample.psi)
        physics_maps.append(sample.base.psi)
        corrected.append(psi)
        pairs = (("q95", sample.q95, q95), ("beta_n", sample.beta_n, beta_n))
        for name, reference, predicted in pairs:
            known = np.isfinite(reference)
            scalars[name][0].append(reference[known])
            scalars[name][1].append(predicted[known])
    all_truth = np.concatenate(truth)
    return {
        "held_out_frames": int(all_truth.shape[0]),
        "r2_psi_physics": round(pooled_r2(all_truth, np.concatenate(physics_maps)), 4),
        "r2_psi_corrected": round(pooled_r2(all_truth, np.concatenate(corrected)), 4),
        **{
            f"r2_{name}": round(pooled_r2(np.concatenate(pair[0]), np.concatenate(pair[1])), 4)
            for name, pair in scalars.items()
        },
    }


def fields_of_base() -> tuple[str, ...]:
    """Return the names of the per-frame arrays of a physics base."""
    return tuple(field.name for field in dataclasses.fields(PhysicsBase))


if __name__ == "__main__":
    raise SystemExit(main())
