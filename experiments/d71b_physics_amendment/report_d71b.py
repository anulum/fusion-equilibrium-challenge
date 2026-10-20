# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Equilibrium Challenge — D71b report: corrected critical points, fixed-region outside current, GS diagnostics
"""Aggregate D71b measures: fail-closed through receipts.validate_run (the activation, the exact D70 cases in order,
every shot's frame count, map order, measure names and shapes and file hash; qualification runs are rejected); per pair of maps, each shot's change is the mean over the frames where BOTH maps have a finite value
(matched support); paired whole-shot bootstrap (20 000 draws, seed 71020, 99 % interval); smoothed -> margin and
consecutive stages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import receipts

SEED, DRAWS = 71020, 20000
PRE, POST = "smoothed", "margin"
SIGNED = ("max_extra", "max_missing", "saddle_extra", "saddle_missing", "min_extra")


def derived(m: dict[str, np.ndarray], ref: int) -> dict[str, np.ndarray]:
    r = lambda key: m[key][:, [ref]]  # noqa: E731
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = m["gs_rms_residual"] / m["gs_rms_source"]
        return {
            "outside_fixed_excess": m["outside_fixed_over_ref"] - r("outside_fixed_over_ref"),
            "gs_surface_excess": m["gs_surface"] - r("gs_surface"),
            "gs_rms_norm_excess": norm - norm[:, [ref]],
            "max_extra": (m["cp_max"] > r("cp_max")).astype(float),
            "max_missing": (m["cp_max"] < r("cp_max")).astype(float),
            "saddle_extra": (m["cp_saddle"] > r("cp_saddle")).astype(float),
            "saddle_missing": (m["cp_saddle"] < r("cp_saddle")).astype(float),
            "min_extra": (m["cp_min"] > r("cp_min")).astype(float),
            "gs_degenerate": m["gs_degenerate"],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True, help="the D70 activation holding the cohort's cases")
    parser.add_argument("--require-shots", type=int, default=358)
    args = parser.parse_args()
    activation = json.loads(args.activation.read_text())
    if receipts.digest(args.cases) != activation["D70_activation_sha256"]:
        raise receipts.ReceiptError("the cases file is not the D70 activation this D71b activation binds")
    cases = json.loads(args.cases.read_text())["cases"]
    if args.require_shots and len(cases) != args.require_shots:
        raise receipts.ReceiptError(f"the cohort has {len(cases)} shots, {args.require_shots} required")
    receipts.validate_run(args.measures, cases, receipts.digest(args.activation))
    folders = [args.measures / str(c["stem"]) for c in cases]
    shot, frames, maps, ref_counts = {}, {}, [], {"cp_max": [], "cp_saddle": [], "cp_min": []}
    per_shot_frames: dict[str, list[np.ndarray]] = {}
    for folder in folders:
        with np.load(folder / "measures.npz", allow_pickle=False) as z:
            m = {key: z[key] for key in z.files}
        maps = [str(x) for x in m["maps"]]
        ref = maps.index("reference")
        valid = np.isfinite(m["cp_max"][:, ref])
        for key in ref_counts:
            ref_counts[key].append(m[key][valid, ref])
        for key, values in derived(m, ref).items():
            v = values[valid]
            frames.setdefault(key, []).append(v)
            per_shot_frames.setdefault(key, []).append(v)
            shot.setdefault(key, []).append(np.nanmean(np.where(np.isfinite(v), v, np.nan), axis=0))
    rng = np.random.default_rng(SEED)
    n = len(folders)
    draws = rng.integers(0, n, size=(DRAWS, n))
    pre, post = maps.index(PRE), maps.index(POST)
    report = {"shots": n, "frames": int(sum(len(v) for v in frames["max_extra"])), "maps": maps, "seed": SEED,
              "draws": DRAWS, "reference_counts_per_frame": {k: {str(c): int((np.concatenate(v) == c).sum())
                                                                 for c in range(4)} for k, v in ref_counts.items()}}
    measures = {}
    for key in shot:
        S, F = np.vstack(shot[key]), np.vstack(frames[key])
        per_map = {name: {"shot_mean": float(np.nanmean(S[:, j])), "frame_median": float(np.nanmedian(F[:, j])),
                          "frame_p95": float(np.nanquantile(F[:, j], .95)), "frames_valid": int(np.isfinite(F[:, j]).sum()),
                          "frames_sum": float(np.nansum(F[:, j]))} for j, name in enumerate(maps)}
        steps = {}
        for a, b in [(pre, post), *[(i, i + 1) for i in range(maps.index("raw_64coords"), len(maps) - 1)]]:
            diff = np.array([np.nanmean(v[:, b][np.isfinite(v[:, a]) & np.isfinite(v[:, b])]
                                        - v[:, a][np.isfinite(v[:, a]) & np.isfinite(v[:, b])])
                             if (np.isfinite(v[:, a]) & np.isfinite(v[:, b])).any() else np.nan
                             for v in per_shot_frames[key]])
            matched = int(sum((np.isfinite(v[:, a]) & np.isfinite(v[:, b])).sum() for v in per_shot_frames[key]))
            boot = np.nanmean(diff[draws], axis=1)
            lo, hi = np.nanquantile(boot, [.005, .995])
            steps[f"{maps[a]}->{maps[b]}"] = {"mean_change": float(np.nanmean(diff)), "ci99": [float(lo), float(hi)],
                                              "verdict": "worse" if lo > 0 else "better" if hi < 0 else "inconclusive",
                                              "shots": int(np.isfinite(diff).sum()), "matched_frames": matched}
        measures[key] = {"per_map": per_map, "changes": steps}
    report["measures"] = measures
    report["rule"] = ("Lower is closer to the reference (gs_degenerate: fewer degenerate frames). 'worse' if the 99 % "
                      "paired whole-shot interval lies above zero, 'better' if below, else inconclusive.")
    args.out.write_text(json.dumps(report, indent=1, allow_nan=True))
    for key in measures:
        print(key, json.dumps(measures[key]["changes"][f"{PRE}->{POST}"]), flush=True)


if __name__ == "__main__":
    main()
