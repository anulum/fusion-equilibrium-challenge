# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — deployed model inference on a supplied split
"""Predict a complete supplied split from inputs and pinned weights alone."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


STATE: dict = {}


def digest(path: Path) -> str:
    """Return one asset's SHA-256."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(package: Path, manifest: dict, machine: str) -> None:
    """Verify the applicable frozen source and scientific assets."""
    prefixes = (manifest[f"{machine}_source"], manifest[f"{machine}_entrypoints"])
    for relative, expected in manifest["source_files"].items():
        if relative.startswith(prefixes):
            assert digest(package / relative) == expected
    if machine == "d3d":
        for relative, expected in manifest["runtime_source_files"].items():
            assert digest(package / relative) == expected
        assets = manifest["maps"] + manifest["joint"] + manifest["static"] + manifest["trees"] + [manifest["basis"]]
        assert len(manifest["maps"]) == 69 and len(manifest["joint"]) == len(manifest["static"]) == 8
        for asset in assets:
            assert digest(package / asset["path"]) == asset["sha256"]
        for asset in manifest["trees"]:
            assert digest(package / asset["lookup_path"]) == asset["lookup_sha256"]


def initialize(package_path: str, machine: str, data_root: str, split: str) -> None:
    """Load independent machine modules in a fresh bounded worker."""
    package = Path(package_path)
    manifest = json.loads((package / "deployment.json").read_text())
    verify(package, manifest, machine)
    source = package / manifest[f"{machine}_source"]
    entrypoints = package / manifest[f"{machine}_entrypoints"]
    for folder in (source / "vendor", source, entrypoints):
        sys.path.insert(0, str(folder))
    STATE.update(package=package, manifest=manifest, machine=machine)
    if machine == "mast":
        def forbid_target_mask(event: str, args: tuple) -> None:
            if event == "open" and "mast_envelope.npz" in str(args[0]):
                raise RuntimeError("MAST target envelope is unavailable to inference")
        sys.addaudithook(forbid_target_mask)
        import predict_mast as predictor
        STATE["predictor"] = predictor
        return
    import predict_diii_d as predictor
    import subframe_features
    import xgboost as xgb
    sys.path.insert(0, str(package))
    from tree_replica import predict_like_gpu
    maps = [predictor.Network.load(package / item["path"]) for item in manifest["maps"]]
    joint = [predictor.Network.load(package / item["path"]) for item in manifest["joint"]]
    static = [predictor.Network.load(package / item["path"]) for item in manifest["static"]]
    assert all(network.coordinates == 0 and network.label_names == ("q95", "beta_n") for network in joint)
    x_names = ("x_lower_r", "x_lower_z", "x_upper_r", "x_upper_z")
    assert all(network.coordinates == 0 and network.label_names == x_names for network in static)
    arguments = predictor._parser().parse_args([
        "--data-root", data_root, "--split", split, "--out", "/dev/null",
        "--network", *[str(package / item["path"]) for item in manifest["maps"]],
        "--basis", str(package / manifest["basis"]["path"]), *manifest["finish_flags"]])
    trees = []
    for specification in manifest["trees"]:
        tree = xgb.Booster(params={"nthread": 1, "device": "cpu"})
        tree.load_model(package / specification["path"])
        assert int(tree.attr("best_iteration")) == specification["best_iteration"]
        with np.load(package / specification["lookup_path"], allow_pickle=False) as stored:
            leaves, base = stored["leaf_values"], stored["base_score"]
        assert leaves.dtype == base.dtype == np.float32
        assert leaves.shape[0] == specification["best_iteration"] + 1
        config = json.loads(tree.save_config())["learner"]
        expected_base = np.float32(json.loads(config["learner_model_param"]["base_score"])[0])
        assert base == expected_base
        trees.append((specification, tree, leaves, base))
    native_layout = predictor.layout_features
    native_mean = predictor.predict_together

    def layout(row: dict, shot: object, layouts: set) -> dict:
        result = native_layout(row, shot, layouts)
        base = result["discharge"]
        assert base.shape[1] == 367
        STATE["tree_features"] = np.asarray(base, dtype=np.float32)
        block = subframe_features.features_of(STATE["path"])
        assert block.shape == (base.shape[0], 215)
        result["discharge"] = np.concatenate([base, block.astype(base.dtype)], axis=1)
        return result

    def composed(networks: list, features: dict) -> tuple:
        coordinates, labels = native_mean(networks, features)
        for specification, tree, leaves, base in trees:
            name, share = specification["target"], specification["share"]
            prediction = predict_like_gpu(tree, STATE["tree_features"], leaves, base).astype(float)
            assert prediction.shape == labels[name].shape and np.isfinite(prediction).all()
            labels[name] = labels[name] + share * (prediction - labels[name])
        current = features["discharge"]
        assert current.shape[1] == 582
        answers = [network.predict(np.pad(current, ((0, 0), (0, 32))))[1] for network in static]
        for name in x_names:
            expert = np.mean([answer[name] for answer in answers], axis=0)
            labels[name] = .5 * np.asarray(labels[name], dtype=float) + .5 * expert
        _, heads = native_mean(joint, {"discharge": np.pad(current, ((0, 0), (0, 183)))})
        assert all(np.isfinite(heads[name]).all() for name in ("q95", "beta_n"))
        STATE["joint_heads"] = heads
        return coordinates, labels

    predictor.layout_features = layout
    predictor.predict_together = composed
    STATE.update(predictor=predictor, maps=maps,
                 basis=predictor.FluxBasis.load(package / manifest["basis"]["path"]).truncated(maps[0].coordinates),
                 finish=predictor._finish_from(arguments))


def predict(path: str) -> object:
    """Predict one input-only shot, retaining deployed scalar rounding."""
    STATE["path"] = Path(path)
    if STATE["machine"] == "mast":
        return STATE["predictor"].predict_shot(Path(path))
    maps, q95, beta = STATE["predictor"].predict_shot(
        Path(path), STATE["maps"], STATE["basis"], STATE["finish"], None)
    heads = STATE["joint_heads"]
    shares = STATE["manifest"]["joint_share"]
    q95 = ((1.0-shares["q95"]) * q95.astype(float) + shares["q95"] * heads["q95"]).astype(np.float32)
    beta = ((1.0-shares["beta_n"]) * beta.astype(float) + shares["beta_n"] * heads["beta_n"]).astype(np.float32)
    assert np.isfinite(maps).all() and np.isfinite(q95).all() and np.isfinite(beta).all()
    return maps, q95, beta


def main() -> None:
    """Write every supplied shot in stream order; a limit is qualification only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--machine", choices=("d3d", "mast"), required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not 1 <= args.workers <= 22:
        parser.error("--workers must be between 1 and 22")
    assert os.environ.get("OPENBLAS_NUM_THREADS") == "1" and os.environ.get("OMP_NUM_THREADS") == "1"
    manifest = json.loads((args.package / "deployment.json").read_text())
    verify(args.package, manifest, args.machine)
    assert Path(args.split).name == args.split
    files = sorted((args.data_root / args.split).glob("*.parquet"))
    assert files and not args.out.exists()
    full_count = len(files)
    if args.limit is not None:
        assert args.limit > 0
        files = files[:args.limit]
    expected = [len(pq.read_table(path, columns=["efit_times"], use_threads=False)["efit_times"][0].as_py()) for path in files]
    started = time.time()
    if args.machine == "mast":
        initialize(str(args.package), "mast", str(args.data_root), args.split)
    with ProcessPoolExecutor(max_workers=min(args.workers, len(files)),
                             mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize,
                             initargs=(str(args.package), args.machine, str(args.data_root), args.split)) as pool:
        results = list(pool.map(predict, map(str, files)))
    fallback = None
    if args.machine == "mast":
        results = STATE["predictor"]._with_mean_pattern_where_nothing_solved(files, results)
        fallback = [float(np.nanmedian(np.concatenate([getattr(result, name) for result in results]))) for name in ("q95", "beta_n")]
        assert np.isfinite(fallback).all()
        results = [(result.maps, np.where(np.isfinite(result.q95), result.q95, fallback[0]),
                    np.where(np.isfinite(result.beta_n), result.beta_n, fallback[1])) for result in results]
    entries = {}
    for index, ((maps, q95, beta), frames) in enumerate(zip(results, expected, strict=True)):
        assert maps.shape == (frames, 65, 65) and q95.shape == beta.shape == (frames,)
        for name, array in (("psirz", maps), ("q95", q95), ("betaN", beta)):
            assert np.isfinite(array).all()
            entries[f"shot_{index:04d}_{name}"] = array.astype(np.float32)
    assert len(entries) == len(files) * 3
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **entries)
    with np.load(args.out, allow_pickle=False) as archive:
        assert set(archive.files) == set(entries)
        for name in archive.files:
            np.testing.assert_array_equal(archive[name], entries[name])
    receipt = {"status": "new_split_prediction_complete_numerical_qualification_required",
               "machine": args.machine, "split": args.split, "shots": len(files), "frames": sum(expected),
               "stems": [path.stem for path in files], "limit": args.limit, "whole_split": len(files) == full_count,
               "qualification_only": args.limit is not None, "fallback": fallback,
               "requested_workers": args.workers, "effective_workers": min(args.workers, len(files)),
               "model_manifest_sha256": digest(args.package / "deployment.json"),
               "source_sha256": digest(Path(__file__)), "archive_sha256": digest(args.out),
               "uses_precomputed_test_heads_or_D36_archive": False, "seconds": time.time() - started}
    with args.out.with_suffix(".receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "stems"}))


if __name__ == "__main__":
    main()
