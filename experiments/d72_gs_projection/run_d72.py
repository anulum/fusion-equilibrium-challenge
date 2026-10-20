# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Equilibrium Challenge — D72: the submitted pipeline against the same maps with a Grad-Shafranov interior
"""Regenerate the D70 native pipeline unchanged on the 358 development shots, project each final map's plasma onto
a Grad-Shafranov equilibrium (gs_project.py), and score both with the organisers' scorer through the D66 wrapper.
The base maps must equal D70's final field checksums and the base official statistics must equal D70's."""
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

from native_pipeline import TARGETS, NativePipeline, digest, load_module  # noqa: E402
from stage_trace import observe  # noqa: E402

import gs_project  # noqa: E402
import physics  # noqa: E402

KEYS = ("ss_res_psi", "ss_res_scal", "ss_res_cons", "cons_frames", "cons_fails", "shot_dlcfs", "lcfs_frames",
        "lcfs_fails", "ref_lcfs_dropped", "missing", "psi_pixels", "psi_nonfinite")
STATE: dict[str, object] = {}


def serial(value: object) -> object:
    if isinstance(value, np.ndarray):
        return [serial(item) for item in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    return value


def initialize(activation_path: Path, output: Path) -> None:
    activation = json.loads(activation_path.read_text())
    d70 = json.loads((D70 / "activation_v1.json").read_text())
    if digest(D70 / "activation_v1.json") != activation["D70_activation_sha256"]:
        raise ValueError("D70 activation changed")
    for name in ("native_pipeline.py", "stage_trace.py"):
        if digest(D70 / name) != d70["sources"][name]:
            raise ValueError(f"D70 source changed: {name}")
    for name, checksum in activation["D72_sources"].items():
        if digest(HERE / name) != checksum:
            raise ValueError(f"D72 source changed: {name}")
    manifest = Path(str(d70["D66_manifest"]))
    if digest(manifest) != d70["D66_manifest_sha256"]:
        raise ValueError("Excluded incumbent manifest changed")
    pipeline = NativePipeline(manifest)
    sys.path.insert(0, pipeline.manifest["harness"])
    scorer = __import__("run_parallel_score")
    wrapper = load_module(Path(pipeline.manifest["score_wrapper"]), "d72_D66_official_wrapper")
    import derive  # noqa: PLC0415
    import o_point  # noqa: PLC0415
    from common import AXIS_SIGN  # noqa: PLC0415
    rows = json.loads(Path(pipeline.manifest["statistics"]).read_text())
    total = {key: sum(np.asarray(row[key]) for row in rows) for key in rows[0]}
    means = (total["psi_sum"] / total["psi_n"], total["scal_sum"] / np.maximum(total["scal_n"], 1),
             total["cons_sum"] / np.maximum(total["cons_n"], 1))
    R, Z, coarse, fine = scorer._grid()
    STATE.update(pipeline=pipeline, scorer=scorer, wrapper=wrapper, means=means, cases=d70["cases"], output=output,
                 R=np.asarray(R, float), Z=np.asarray(Z, float), coarse=coarse, fine=fine,
                 axis_sign=float(AXIS_SIGN[scorer.local_score.MACHINE]),
                 kit={"points_in_polygon": derive._points_in_polygon, "bilinear": derive._bilinear,
                      "find_o_point": o_point.find_o_point, "local_maxima": o_point.local_maxima,
                      "erode": o_point._erode})


def project_shot(maps: np.ndarray) -> tuple[np.ndarray, dict[str, list[float]], np.ndarray]:
    R, Z, coarse, fine = STATE["R"], STATE["Z"], STATE["coarse"], STATE["fine"]
    sign, kit, local = float(STATE["axis_sign"]), STATE["kit"], STATE["scorer"].local_score
    out = np.empty_like(maps, dtype=np.float32)
    info = {"pixels": [], "iterations": [], "converged": [], "used": []}
    gs_share = np.full(len(maps), np.nan)
    for k in range(len(maps)):
        psi = np.asarray(maps[k], dtype=np.float64)
        own, _ = local.extract_lcfs_with_sign(psi, R, Z, local.MACHINE, coarse, fine, n_iter=local.N_ITER,
                                              axis_sign=sign)
        result, frame = psi, {"pixels": 0., "iterations": 0., "converged": 0., "used": 0.}
        if own is not None:
            phi = sign * psi
            plasma = physics.inside(own, R, Z, kit["points_in_polygon"])
            _, _, gs = physics.derivatives(phi, R, Z)
            _, _, iz, ir, _ = kit["find_o_point"](phi, R, Z, coarse)
            boundary = float(np.mean(kit["bilinear"](phi, R, Z, own[:, 0], own[:, 1])))
            result, frame = gs_project.projected(psi, R, Z, plasma, sign, (iz, ir), boundary, gs)
            if frame["used"]:
                p2 = sign * result
                _, _, gs2 = physics.derivatives(p2, R, Z)
                gs_share[k] = physics.gs_form_residual(p2, gs2, R, plasma, float(p2[plasma].max()), boundary)
        out[k] = result.astype(np.float32)
        for key in info:
            info[key].append(float(frame[key]))
    return out, info, gs_share


def one(case: dict[str, object]) -> dict[str, object]:
    started = time.monotonic()
    stem, path = str(case["stem"]), Path(str(case["path"]))
    folder = cast(Path, STATE["output"]) / stem
    if (folder / "complete.json").exists():
        return json.loads((folder / "complete.json").read_text())["summary"]
    for key, checksum in (("path", "raw_sha256"), ("features", "features_sha256"), ("reference", "reference_sha256")):
        if digest(Path(str(case[key]))) != case[checksum]:
            raise ValueError(f"D72 shot dependency changed: {stem}, {key}")
    pipeline = cast(NativePipeline, STATE["pipeline"])
    row = pipeline.inputs(path)
    with np.load(str(case["features"]), allow_pickle=False) as stored:
        features = stored["features"].astype(float)
    coordinates, adjusted, joint, expert = pipeline.heads(features)
    zero = np.zeros((len(features), len(TARGETS)), dtype=float)
    with observe(pipeline.pd, pipeline.basis, coordinates) as snapshots:
        fields = pipeline.finish_maps(path, row, features, coordinates, adjusted, joint, expert, zero)
    if not np.array_equal(snapshots["margin"].astype(np.float32).view(np.uint32), fields["psirz"].view(np.uint32)):
        raise ValueError("Observed final stage differs from the pipeline's map bits")
    d70_record = json.loads((D70 / "full358_v2" / stem / "complete.json").read_text())
    regenerated = {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in fields.items()}
    if regenerated != d70_record["final_field_sha256"]:
        raise ValueError(f"Regenerated fields differ from D70's record: {stem}")
    projected, info, gs_share = project_shot(fields["psirz"])
    truth = STATE["scorer"]._truth(path)
    reference = pickle.loads(Path(str(case["reference"])).read_bytes())
    parts = {}
    for name, maps in (("base", fields["psirz"]), ("gs", projected)):
        prediction = {**fields, "psirz": maps}
        part, _distances, _derived = STATE["wrapper"].score_maps(STATE["scorer"], STATE["means"], truth, reference,
                                                                 prediction)
        parts[name] = serial(part)
    expected = d70_record["parts"]["margin"]
    if any(parts["base"][key] != expected[key] for key in KEYS):
        raise ValueError(f"Base official statistics differ from D70: {stem}")
    folder.mkdir(exist_ok=True)
    np.savez_compressed(folder / "projection.npz", gs_share=gs_share,
                        **{key: np.asarray(value) for key, value in info.items()})
    summary = {"stem": stem, "frames": len(fields["psirz"]), "used": int(sum(info["used"])),
               "converged": int(sum(info["converged"])), "base_equals_D70": True,
               "seconds": time.monotonic() - started}
    with (folder / "complete.json").open("x") as stream:
        json.dump({"summary": summary, "parts": parts,
                   "gs_sha256": hashlib.sha256(projected.tobytes()).hexdigest()}, stream, indent=2)
    print(json.dumps(summary), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--qualify", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)
    initialize(args.activation, args.out)
    cases = cast(list[dict[str, object]], STATE["cases"])
    if len(cases) != 358 or sum(int(str(c["frames"])) for c in cases) != 78875:
        raise ValueError("D72 requires the complete 358-shot development cohort")
    if args.qualify:
        cases = [cases[index] for index in (0, 179, 357)]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("fork")) as pool:
        done = list(pool.map(one, cases, chunksize=1))
    complete = {"status": "qualification" if args.qualify else "complete358", "shots": len(done),
                "frames": sum(d["frames"] for d in done), "activation_sha256": digest(args.activation), "members": done}
    (args.out / "complete.json").write_text(json.dumps(complete, indent=2))
    print(json.dumps({k: v for k, v in complete.items() if k != "members"}), flush=True)


if __name__ == "__main__":
    main()
