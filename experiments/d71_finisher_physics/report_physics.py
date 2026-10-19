# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Equilibrium Challenge — D71 report: what the finisher does to the physics of the map
"""Aggregate D71 measures: qualification of the operators, per-stage levels, and paired whole-shot changes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MU0 = 4e-7 * np.pi
SEED, DRAWS = 71019, 20000
# Lower is closer to the reference for every derived measure below.
DERIVED = ("current_error_own", "current_error_ref", "outside_excess", "gs_excess", "bp_error_ref",
           "bp_error_envelope", "extra_o_points", "saddle_difference", "lcfs_failed",
           "gs_excess_deg2", "gs_excess_deg5", "gs_excess_o4")
PRE, POST = "smoothed", "margin"      # the finisher's input map and the submitted map


def derived(m: dict[str, np.ndarray], maps: list[str]) -> dict[str, np.ndarray]:
    """Per-frame, per-map distances from the reference, oriented so that 0 is the reference."""
    ref = maps.index("reference")
    r = lambda key: m[key][:, [ref]]  # noqa: E731
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "current_error_own": np.abs(m["loop_abs_own"] / r("loop_abs_own") - 1),
            "current_error_ref": np.abs(m["loop_signed_ref"] / r("loop_signed_ref") - 1),
            "outside_excess": m["outside_fraction"] - r("outside_fraction"),
            "gs_excess": m["gs_fraction"] - r("gs_fraction"),
            "bp_error_ref": m["bp_error_ref"], "bp_error_envelope": m["bp_error_envelope"],
            "extra_o_points": np.abs(m["n_max"] - r("n_max")),
            "saddle_difference": np.abs(m["n_saddle"] - r("n_saddle")),
            "lcfs_failed": np.where(np.isfinite(m["n_max"]), 1 - m["lcfs_ok"], np.nan),
            "gs_excess_deg2": m["gs_fraction_deg2"] - r("gs_fraction_deg2"),
            "gs_excess_deg5": m["gs_fraction_deg5"] - r("gs_fraction_deg5"),
            "gs_excess_o4": m["gs_fraction_o4"] - r("gs_fraction_o4"),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    folders = sorted(p for p in args.measures.iterdir() if (p / "measures.npz").exists())
    shot_means: dict[str, list[np.ndarray]] = {key: [] for key in DERIVED}
    frame_values: dict[str, list[np.ndarray]] = {key: [] for key in DERIVED}
    gauss, surface, convention, raw_levels = [], [], [], {"gs_fraction": [], "outside_fraction": [],
                                                    "gs_fraction_o4": []}
    maps: list[str] = []
    for folder in folders:
        with np.load(folder / "measures.npz", allow_pickle=False) as z:
            m = {key: z[key] for key in z.files}
        maps = [str(x) for x in m["maps"]]
        ref = maps.index("reference")
        valid = np.isfinite(m["n_max"][:, ref])
        gauss.append(m["area_ref"][valid, ref] / -m["loop_signed_ref"][valid, ref])
        surface.append(m["loop_abs_own"][valid, ref] / -m["loop_signed_ref"][valid, ref])
        convention.append(m["loop_abs_own"][valid, ref] / (MU0 * np.maximum(m["ip_A"][valid], 1.)))
        for key in raw_levels:
            raw_levels[key].append(m[key][valid])
        d = derived(m, maps)
        for key in DERIVED:
            values = d[key][valid]
            frame_values[key].append(values)
            shot_means[key].append(np.nanmean(np.where(np.isfinite(values), values, np.nan), axis=0))
    rng = np.random.default_rng(SEED)
    n = len(folders)
    draws = rng.integers(0, n, size=(DRAWS, n))
    pre, post = maps.index(PRE), maps.index(POST)
    report: dict[str, object] = {"shots": n, "frames": int(sum(len(v) for v in frame_values["bp_error_ref"])),
                                 "maps": maps, "seed": SEED, "draws": DRAWS}
    q = lambda x: [float(v) for v in np.nanquantile(np.concatenate(x), [.05, .5, .95])]  # noqa: E731
    report["qualification"] = {"area_over_loop_reference_q05_50_95": q(gauss),
                               "flux_surface_loop_over_signed_reference_q05_50_95": q(surface),
                               "loop_over_mu0_measured_ip_q05_50_95": q(convention)}
    report["levels"] = {key: {name: float(np.nanmedian(np.concatenate(raw_levels[key])[:, j])) for j, name in
                              enumerate(maps)} for key in raw_levels}
    measures = {}
    for key in DERIVED:
        S = np.vstack(shot_means[key])                    # shots x maps
        F = np.vstack(frame_values[key])
        per_map = {name: {"shot_mean": float(np.nanmean(S[:, j])), "frame_median": float(np.nanmedian(F[:, j])),
                          "frame_p95": float(np.nanquantile(F[:, j], .95))} for j, name in enumerate(maps)
                   if name != "reference"}
        steps = {}
        for a, b in [(pre, post), *[(i, i + 1) for i in range(maps.index("raw_64coords"), len(maps) - 1)]]:
            diff = S[:, b] - S[:, a]
            ok = np.isfinite(diff)
            boot = np.nanmean(diff[draws], axis=1)
            lo, hi = np.nanquantile(boot, [.005, .995])
            verdict = "worse" if lo > 0 else "better" if hi < 0 else "inconclusive"
            steps[f"{maps[a]}->{maps[b]}"] = {"mean_change": float(np.nanmean(diff)), "ci99": [float(lo), float(hi)],
                                              "verdict": verdict, "shots": int(ok.sum())}
        measures[key] = {"per_map": per_map, "changes": steps}
    report["measures"] = measures
    report["rule"] = ("Lower is closer to the reference. A change is 'worse' if the 99 % paired whole-shot bootstrap "
                      "interval of the mean shot-level change lies above zero, 'better' if below, else inconclusive.")
    args.out.write_text(json.dumps(report, indent=1, allow_nan=True))
    print(json.dumps(report["qualification"]), flush=True)
    for key in DERIVED:
        c = measures[key]["changes"][f"{PRE}->{POST}"]
        print(key, json.dumps(c), flush=True)


if __name__ == "__main__":
    main()
