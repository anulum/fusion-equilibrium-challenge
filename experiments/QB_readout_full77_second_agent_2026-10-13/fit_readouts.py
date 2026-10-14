# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — full matched scalar readout fits
"""Stream eligible rows into the original ridge kernel for all declared models."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    """Return an artifact's streaming SHA-256."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def mapped_array(path: Path, name: str) -> np.memmap:
    """Map an uncompressed scientific NPZ member, retaining its declared order."""
    with zipfile.ZipFile(path) as archive:
        member = archive.getinfo(name + ".npy")
        assert member.compress_type == zipfile.ZIP_STORED
    with path.open("rb") as stream:
        stream.seek(member.header_offset)
        header = stream.read(30)
        name_size, extra_size = struct.unpack("<HH", header[26:30])
        stream.seek(name_size + extra_size, 1)
        version = np.lib.format.read_magic(stream)
        reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                  else np.lib.format.read_array_header_2_0)
        shape, fortran, dtype = reader(stream)
        offset = stream.tell()
    assert not dtype.hasobject
    return np.memmap(path, mode="r", dtype=dtype, offset=offset,
                     shape=shape, order="F" if fortran else "C")


def initialize(work: str, runs: str, output: str) -> None:
    """Verify the split before mapping targets; initialise one bounded worker."""
    global STATE
    work_path, runs_path = Path(work), Path(runs)
    protocol = json.loads((work_path / "protocol_frozen_v1.json").read_text())
    assert protocol["status"] == "fixed_full77_scalar_readout_study_frozen"
    assert digest(work_path / "models.json") == protocol["model_source_manifest_sha256"]
    assert digest(work_path / "cohort.json") == protocol["cohort_sha256"]
    assert digest(Path(__file__)) == protocol["sources_sha256"]["fit_readouts.py"]
    assert int(os.environ["OPENBLAS_NUM_THREADS"]) <= protocol["blas_threads_per_worker"]
    package = runs_path / "legacy_tree_fec_first_agent_20261005/deploy/pkg"
    for name, expected in protocol["legacy_sources_sha256"].items():
        assert digest(package / name) == expected
    for folder in (package / "source/fec_adapter/vendor", package / "source/fec_adapter", package / "entrypoints"):
        sys.path.insert(0, str(folder))
    cohort = json.loads((work_path / "cohort.json").read_text())
    training, calibration = set(cohort["training"]), set(cohort["calibration"])
    held = set(cohort["holdout"]) | set(cohort["reserved_confirmation"])
    assert len(training) == 5954 and len(calibration) == 372 and len(held) == 715
    assert not (training & calibration or training & held or calibration & held)
    tables = {}
    for part, allowed in (("training", training), ("calibration", calibration)):
        path = runs_path / f"tables28_subframe_20261006/table_{part}.npz"
        with np.load(path, allow_pickle=False) as archive:
            stems = archive["stems"].astype(str)
            shot = archive["shot_of_row"]
            labels = archive["label_names"].astype(str).tolist()
            assert str(archive["feature_layout"]) == "discharge"
        assert allowed <= set(stems)
        counts = np.bincount(shot, minlength=len(stems))
        bounds = np.r_[0, np.cumsum(counts)]
        assert np.array_equal(shot, np.repeat(np.arange(len(stems)), counts))
        rows = np.concatenate([np.arange(bounds[i], bounds[i + 1])
                               for i, stem in enumerate(stems) if stem in allowed])
        assert set(stems[np.unique(shot[rows])]) == allowed
        features = mapped_array(path, "features")
        assert features.shape == (len(shot), 582) and features.dtype == np.float32
        target = mapped_array(path, "labels")
        assert target.shape[0] == len(shot)
        tables[part] = {"features": features, "labels": target, "rows": rows,
                        "columns": [labels.index(name) for name in ("q95", "beta_n")]}
    STATE = {"work": work_path, "runs": runs_path, "output": Path(output),
             "protocol": protocol, "tables": tables,
             "protocol_sha256": digest(work_path / "protocol_frozen_v1.json")}


STATE: dict = {}


def fit_one(task: dict) -> dict:
    """Refit only named output rows; require exact reload and unchanged map heads."""
    from fec_adapter.diii_d_accuracy.network import Network
    from geometry_readout_streaming import cpu_moments, corrected_from_moments
    name = task["name"]
    source = STATE["runs"] / task["source_relative_to_runs"]
    assert digest(source) == task["sha256"]
    destination = STATE["output"] / name
    receipt_path = destination / "receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        assert receipt["status"] == "full_scalar_readout_fit_and_export_PASS"
        assert receipt["protocol_sha256"] == STATE["protocol_sha256"]
        assert receipt["source_model_sha256"] == task["sha256"]
        assert digest(destination / "network.npz") == receipt["network_sha256"]
        return receipt
    destination.mkdir(exist_ok=False)
    started = time.time()
    model = Network.load(source)
    names, groups = ("q95", "beta_n"), ((0,), (1,))
    total = None
    sample = None
    chunks = 0
    parts = ("training", "calibration") if task["train_on_calibration"] else ("training",)
    for part in parts:
        table = STATE["tables"][part]
        rows = table["rows"]
        for start in range(0, len(rows), STATE["protocol"]["chunk"]):
            selected = rows[start:start + STATE["protocol"]["chunk"]]
            features = table["features"][selected].astype(float)
            assert model.feature_mean.size >= features.shape[1]
            features = np.pad(features, ((0, 0), (0, model.feature_mean.size - features.shape[1])))
            truth = table["labels"][selected][:, table["columns"]].astype(float)
            values = cpu_moments(model, features, truth, names, groups)
            total = values if total is None else tuple(
                tuple(left + right for left, right in zip(a, b, strict=True))
                for a, b in zip(total, values, strict=True))
            if sample is None:
                sample = features[:64].copy()
            chunks += 1
            if chunks % 50 == 0:
                print(json.dumps({"model": name, "chunks": chunks, "seconds": time.time() - started}), flush=True)
    assert total is not None and sample is not None
    corrected, reports = corrected_from_moments(model, total, names, groups, ridge=.001)
    output_rows = [model.coordinates + model.label_names.index(label) for label in names]
    unchanged = np.ones(model.weights[-1].shape[0], dtype=bool)
    unchanged[output_rows] = False
    for old, new in zip(model.weights[:-1], corrected.weights[:-1], strict=True):
        np.testing.assert_array_equal(old, new)
    for old, new in zip(model.biases[:-1], corrected.biases[:-1], strict=True):
        np.testing.assert_array_equal(old, new)
    np.testing.assert_array_equal(model.weights[-1][unchanged], corrected.weights[-1][unchanged])
    np.testing.assert_array_equal(model.biases[-1][unchanged], corrected.biases[-1][unchanged])
    for attribute in ("linear", "feature_mean", "feature_scale", "output_mean", "output_scale"):
        np.testing.assert_array_equal(getattr(model, attribute), getattr(corrected, attribute))
    assert all(np.isfinite(array).all() for array in corrected.weights + corrected.biases)
    old_coordinates, old_heads = model.predict(sample)
    new_coordinates, new_heads = corrected.predict(sample)
    np.testing.assert_array_equal(old_coordinates, new_coordinates)
    for label in set(old_heads) - set(names):
        np.testing.assert_array_equal(old_heads[label], new_heads[label])
    training_errors = {}
    for index, values in enumerate(total):
        count, sum_h, sum_r, hh, hr, rr = values
        row = output_rows[index]
        delta = (corrected.weights[-1][row] - model.weights[-1][row]) * model.output_scale[row]
        bias = float((corrected.biases[-1][row] - model.biases[-1][row]) * model.output_scale[row])
        before = float(rr[0] / count)
        after = float((rr[0] - 2 * delta @ hr[:, 0] - 2 * bias * sum_r[0]
                       + delta @ hh @ delta + 2 * bias * delta @ sum_h + count * bias * bias) / count)
        assert after >= -1e-10 and after <= before + 1e-10
        training_errors[names[index]] = {"finite_rows": int(count), "MSE_before": before, "MSE_after": max(0., after)}
    assert max(row["normal_equation_gradient_max_abs"] for row in reports) < 1e-8
    output = destination / "network.npz"
    assert not output.exists()
    corrected.save(output)
    reloaded = Network.load(output)
    loaded_coordinates, loaded_heads = reloaded.predict(sample)
    np.testing.assert_array_equal(new_coordinates, loaded_coordinates)
    for label in new_heads:
        np.testing.assert_array_equal(new_heads[label], loaded_heads[label])
    receipt = {"status": "full_scalar_readout_fit_and_export_PASS", "name": name,
               "source_model_sha256": task["sha256"], "network_sha256": digest(output),
               "protocol_sha256": STATE["protocol_sha256"], "fit_parts": parts,
               "training_errors_not_admission": training_errors, "reports": reports,
               "only_qbeta_output_rows_changed": True, "all_other_outputs_bitexact": True,
               "export_reload_bitexact": True, "seconds": time.time() - started}
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"model_complete": name, "seconds": receipt["seconds"]}), flush=True)
    return receipt


def main() -> None:
    """Run the complete frozen list; never admit or evaluate partial results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    protocol = json.loads((args.work / "protocol_frozen_v1.json").read_text())
    assert args.workers == protocol["workers"]
    for name, expected in protocol["training_tables_sha256"].items():
        assert digest(args.runs / "tables28_subframe_20261006" / name) == expected
    tasks = json.loads((args.work / "models.json").read_text())
    assert len(tasks) == 77 and sum(task["family"] == "joint" for task in tasks) == 8
    args.out.mkdir(exist_ok=True)
    assert not (args.out / "all77_receipt.json").exists()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn"),
                             initializer=initialize,
                             initargs=(str(args.work), str(args.runs), str(args.out))) as pool:
        receipts = list(pool.map(fit_one, tasks))
    assert len(receipts) == 77 and all(row["only_qbeta_output_rows_changed"] for row in receipts)
    with (args.out / "all77_receipt.json").open("x") as stream:
        json.dump({"status": "all77_fixed_scalar_readouts_fit_qualified_evaluation_required",
                   "protocol_sha256": digest(args.work / "protocol_frozen_v1.json"),
                   "models": receipts, "heldout_scores_read": False}, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
