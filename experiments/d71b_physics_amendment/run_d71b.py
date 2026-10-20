# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Equilibrium Challenge — D71b: corrected and added physical measures of the reference and the nine native stages
"""D71b: regenerate the nine native stages of D70 unchanged and apply physics_b.measure_b to every map.

The maps come from the D70 sources and activation, unmodified (checksums verified); the final stage must equal the
submitted float32 bits and the field checksums D70 recorded for the same shot. No fitting, no selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import cast

import numpy as np

HERE = Path(__file__).resolve().parent
D70 = Path("$FEC_RUNS/d70_boundary_second_agent_20261015")
sys.path.insert(0, str(D70))

from native_pipeline import TARGETS, NativePipeline, digest  # noqa: E402
from stage_trace import STAGES, observe  # noqa: E402

import physics  # noqa: E402
import physics_b  # noqa: E402
import receipts  # noqa: E402

MAPS = ("reference", *STAGES)
MEASURES = ("outside_fixed_over_ref", "cp_max", "cp_min", "cp_saddle", "gs_surface", "gs_sse", "gs_tss",
            "gs_pixels", "gs_rank", "gs_rms_residual", "gs_rms_source", "gs_degenerate", "outside_pixels")
STATE: dict[str, object] = {}


def initialize(activation_path: Path, output: Path) -> None:
    """Load the exact D70 incumbent and the organisers' scorer functions before fork."""
    activation = json.loads(activation_path.read_text())
    d70 = json.loads((D70 / "activation_v1.json").read_text())
    if digest(D70 / "activation_v1.json") != activation["D70_activation_sha256"]:
        raise ValueError("D70 activation changed")
    for name in ("native_pipeline.py", "stage_trace.py"):
        if digest(D70 / name) != d70["sources"][name]:
            raise ValueError(f"D70 source changed: {name}")
    for name, checksum in activation["D71b_sources"].items():
        if digest(HERE / name) != checksum:
            raise ValueError(f"D71b source changed: {name}")
    manifest = Path(str(d70["D66_manifest"]))
    if digest(manifest) != d70["D66_manifest_sha256"]:
        raise ValueError("Excluded incumbent manifest changed")
    pipeline = NativePipeline(manifest)
    sys.path.insert(0, pipeline.manifest["harness"])
    scorer = __import__("run_parallel_score")
    local = scorer.local_score
    import derive  # noqa: PLC0415 - the organisers' module, importable once the harness is on the path
    import o_point  # noqa: PLC0415
    from common import AXIS_SIGN  # noqa: PLC0415
    R, Z, coarse, fine = scorer._grid()
    STATE.update(pipeline=pipeline, scorer=scorer, local=local, cases=d70["cases"], output=output,
                 R=np.asarray(R, float), Z=np.asarray(Z, float), coarse=coarse, fine=fine,
                 axis_sign=float(AXIS_SIGN[local.MACHINE]),  # submissions in the stored convention: psi_sign +1
                 kit={"points_in_polygon": derive._points_in_polygon, "bilinear": derive._bilinear,
                      "find_o_point": o_point.find_o_point, "local_maxima": o_point.local_maxima,
                      "erode": o_point._erode})


def one(case: dict[str, object]) -> dict[str, object]:
    """Regenerate one shot's nine stages, check them against D70 and measure every map of every frame."""
    started = time.monotonic()
    stem, path = str(case["stem"]), Path(str(case["path"]))
    folder = cast(Path, STATE["output"]) / stem
    for key, checksum in (("path", "raw_sha256"), ("features", "features_sha256"), ("reference", "reference_sha256"),
                          ("D39_cache", "D39_cache_sha256")):  # checked first, also for a cached shot
        if digest(Path(str(case[key]))) != case[checksum]:
            raise ValueError(f"D71b shot dependency changed: {stem}, {key}")
    if (folder / "complete.json").exists():  # a resumed run accepts a cached shot only if every binding holds
        return receipts.validate_shot(folder, case, str(STATE["activation_sha256"]))["summary"]
    pipeline = cast(NativePipeline, STATE["pipeline"])
    row = pipeline.inputs(path)
    with np.load(str(case["features"]), allow_pickle=False) as stored:
        features, times = stored["features"].astype(float), stored["times_ms"].copy()
    coordinates, adjusted, joint, expert = pipeline.heads(features)
    zero = np.zeros((len(features), len(TARGETS)), dtype=float)
    with observe(pipeline.pd, pipeline.basis, coordinates) as snapshots:
        fields = pipeline.finish_maps(path, row, features, coordinates, adjusted, joint, expert, zero)
    if not np.array_equal(snapshots["margin"].astype(np.float32).view(np.uint32), fields["psirz"].view(np.uint32)):
        raise ValueError("Observed final stage differs from the submitted map bits")
    recorded = json.loads((D70 / "full358_v2" / stem / "complete.json").read_text())["final_field_sha256"]
    regenerated = {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in fields.items()}
    if regenerated != recorded:
        raise ValueError(f"Regenerated fields differ from D70's record: {stem}")
    truth = STATE["scorer"]._truth(path)
    reference = pickle.loads(Path(str(case["reference"])).read_bytes())
    contours = reference[0]
    with np.load(str(case["D39_cache"]), allow_pickle=False) as cache:
        ip = np.abs(cache["ip_A"].astype(float))
    R, Z, coarse, fine = STATE["R"], STATE["Z"], STATE["coarse"], STATE["fine"]
    sign, kit, local = float(STATE["axis_sign"]), STATE["kit"], STATE["local"]
    n = len(times)
    out = {measure: np.full((n, len(MAPS)), np.nan) for measure in MEASURES}
    for k in range(n):
        ref_c = contours[k]
        if ref_c is None:
            continue
        ref_psi = np.asarray(truth["psi"][k], dtype=np.float64)
        stage_maps = {name: ref_psi if name == "reference" else np.asarray(snapshots[name][k], dtype=np.float64)
                  for name in MAPS}
        owns = {name: ref_c if name == "reference" else local.extract_lcfs_with_sign(
            stage_maps[name], R, Z, local.MACHINE, coarse, fine, n_iter=local.N_ITER, axis_sign=sign)[0] for name in MAPS}
        union = np.zeros(coarse.shape, dtype=bool)
        for contour in owns.values():
            union |= physics.inside(contour, R, Z, kit["points_in_polygon"])
        region = kit["erode"](coarse, physics.EDGE_CELLS) & ~physics.dilate(union, 1)
        ref_inside = float(physics_b.density(ref_psi, R, Z, sign)[physics.inside(ref_c, R, Z, kit["points_in_polygon"])].sum())
        for j, name in enumerate(MAPS):
            values = physics_b.measure_b(stage_maps[name], R, Z, owns[name], ref_c, sign, coarse, kit, region, ref_inside)
            for measure in MEASURES:
                out[measure][k, j] = values[measure]
    folder.mkdir(exist_ok=True)
    np.savez_compressed(folder / "measures.npz", times_ms=times, ip_A=ip, maps=np.array(MAPS), **out)
    summary = {"stem": stem, "frames": n, "reference_frames": int(sum(c is not None for c in contours)),
               "fields_equal_D70": True, "measures_sha256": digest(folder / "measures.npz"),
               "seconds": time.monotonic() - started}
    with (folder / "complete.json").open("x") as stream:
        json.dump({"summary": summary, "final_field_sha256": regenerated, "maps": list(MAPS),
                   "activation_sha256": STATE["activation_sha256"]}, stream, indent=2)
    print(json.dumps(summary), flush=True)
    return summary


def main() -> None:
    """Run the three frozen qualification shots, or all 358 development shots without selection."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--qualify", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)
    initialize(args.activation, args.out)
    STATE["activation_sha256"] = digest(args.activation)
    if tuple(MAPS) != receipts.MAPS or tuple(MEASURES) != receipts.MEASURES:
        raise ValueError("D71b runner and receipt validator disagree on maps or measures")
    cases = cast(list[dict[str, object]], STATE["cases"])
    if len(cases) != 358 or sum(int(str(c["frames"])) for c in cases) != 78875:
        raise ValueError("D71b requires the complete 358-shot development cohort")
    if args.qualify:
        cases = [cases[index] for index in (0, 179, 357)]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("fork")) as pool:
        done = list(pool.map(one, cases, chunksize=1))
    complete = {"status": "qualification" if args.qualify else "complete", "shots": len(done),
                "frames": sum(d["frames"] for d in done), "activation_sha256": digest(args.activation),
                "members": done, "maps": list(MAPS), "no_fits": True, "no_selection": True}
    (args.out / "complete.json").write_text(json.dumps(complete, indent=2))
    print(json.dumps({k: v for k, v in complete.items() if k != "members"}), flush=True)


if __name__ == "__main__":
    main()
