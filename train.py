# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Equilibrium Challenge — DIII-D training from raw parquet to deployment
"""Rebuild DIII-D assets; --limit exercises every stage with reduced data and epochs."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, replace
from functools import lru_cache, partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "training_adapter"))
sys.path.insert(0, str(HERE / "source/d3d_entrypoints"))
os.environ["FEC_MASK_DIR"] = str(HERE / "source/d3d_legacy/vendor/challenge_masks")

import jax
import jax.numpy as jnp
import numpy as np
import xgboost as xgb
from numpy.typing import NDArray

from fec_adapter.diii_d_accuracy import fit_on_accelerators as fit
from fec_adapter.diii_d_accuracy.contour_edit import radii, label_names as contour_names
from fec_adapter.diii_d_accuracy.fit_network import FitSettings, Table, _Scaling
from fec_adapter.diii_d_accuracy.flux_basis import FluxBasis, Moments
from fec_adapter.diii_d_accuracy.history import DischargeRecord
from fec_adapter.diii_d_accuracy.shape_edit import Reader
from fec_adapter.diii_d_accuracy.shape_labels import SHAPE_LABELS
from fec_adapter.diii_d_accuracy.x_point import ENDS, LABELS as X_NAMES, boundary_saddles
from fec_adapter.shot_reader import read_row
from fec_adapter.split import bucket_of
from geometry_readout_streaming import cpu_moments, corrected_from_moments, validity_groups
from subframe_features import features_of

F32 = NDArray[np.float32]
BASE_NAMES = ("q95", "beta_n", "li", "r_axis", "z_axis")
RAW_NAMES = ("efit_q95", "efit_beta_n", "efit_li", "efit_r_axis", "efit_z_axis")
GEOMETRY_NAMES = (*SHAPE_LABELS, *X_NAMES)
LABEL_NAMES = (*BASE_NAMES, *GEOMETRY_NAMES, *contour_names())


def digest(path: Path) -> str:
    """Return SHA-256 without loading an asset into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record(path: Path, value: object) -> None:
    """Write a new receipt; existing evidence is never overwritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def targets(reader: Reader, maps: NDArray[np.floating]) -> F32:
    """Read shape, saddle and contour targets once per frame using the scorer."""
    out = np.full((len(maps), 73), np.nan, dtype=np.float32)
    for i, raw in enumerate(maps):
        flux = np.asarray(raw, dtype=np.float64)
        reading = reader.read(flux)
        out[i, :5] = [reading.values[name] for name in SHAPE_LABELS.values()]
        saddles = boundary_saddles(reader, flux, reading)
        for k, end in enumerate(ENDS):
            if saddles[end] is not None:
                out[i, 5 + 2*k:7 + 2*k] = saddles[end].position
        if reading.contour is not None:
            out[i, 9:] = radii(reading.contour, reading.axis)
    return out


def prepare(raw_dir: Path, out: Path, limit: int, frames_per_shot: int) -> dict[str, Table]:
    """Extract input-only features and permitted training labels into mapped tables."""
    started = time.monotonic()
    paths = sorted(raw_dir.glob("*.parquet"))
    if not paths or raw_dir.name != "diii_d_train":
        raise ValueError("raw-dir must be the labelled diii_d_train directory")
    groups: dict[str, list[Path]] = {k: [] for k in ("training", "calibration", "validation", "holdout")}
    for path in paths:
        part = bucket_of(path.stem)
        groups[part].append(path)
    if limit:
        groups = {"training": groups["training"][:limit], "calibration": groups["calibration"][:2],
                  "validation": groups["validation"][:2], "holdout": []}
    record(out / "data/split.json", {k: [p.stem for p in v] for k, v in groups.items()})
    cached = out / "data/shots"
    cached.mkdir(parents=True)
    for part, files in groups.items():
        for path in files:
            row = read_row(path)
            if row.get("source") != "DIII-D":
                raise ValueError(f"Wrong machine: {path}")
            maps = np.asarray(row["efit_psirz"], dtype=np.float32)
            features = np.column_stack((DischargeRecord.from_row(row, path.stem).features(), features_of(path)))
            labels = np.column_stack([np.asarray(row[name], dtype=np.float32) for name in RAW_NAMES])
            if features.shape != (len(maps), 582) or labels.shape != (len(maps), 5):
                raise ValueError(f"Feature/target alignment failed: {path}")
            take = np.arange(len(maps))
            if limit:
                take = np.unique(np.linspace(0, len(maps)-1, min(frames_per_shot, len(maps)), dtype=int))
            np.savez(cached / (path.stem + ".npz"), features=features[take].astype(np.float32),
                     psi=maps[take], labels=labels[take], part=np.asarray(part), selected_frames=take)
            print(json.dumps({"extracted": path.stem, "part": part, "frames": len(take)}), flush=True)
    basis_files = [cached / (p.stem + ".npz") for p in groups["training"]]
    if limit:
        flat = np.concatenate([np.load(p)["psi"] for p in basis_files]).reshape(-1, 4225).astype(np.float64)
        if len(flat) < 129:
            raise ValueError("Smoke basis needs at least 129 training frames")
        mean = flat.mean(0)
        centered = flat - mean
        _, singular, directions = np.linalg.svd(centered, full_matrices=False)
        basis = FluxBasis(mean, directions[:128], singular[:128]**2/len(flat),
                          float(np.sum(centered**2)/len(flat)), (65, 65))
        del flat, centered, directions
    else:
        basis = FluxBasis.from_moments(fit_moments(basis_files), 128, (65, 65))
    basis.save(out / "models/flux_basis.npz")
    reader = Reader("DIII-D")
    datasets: dict[str, Table] = {}
    for part, files in groups.items():
        if not files:
            continue
        directory = out / "data" / part
        directory.mkdir()
        counts = [len(np.load(cached / (p.stem + ".npz"))["psi"]) for p in files]
        n = sum(counts)
        arrays = {name: np.lib.format.open_memmap(directory / (name+".npy"), mode="w+", dtype=np.float32,
                   shape=(n, width)) for name, width in (("features", 582), ("coordinates", 128),
                   ("labels", 78), ("geometry", 9))}
        shots = np.lib.format.open_memmap(directory / "shot_of_row.npy", mode="w+", dtype=np.int64, shape=(n,))
        squares, left, total, cells, at = 0.0, 0.0, 0.0, 0, 0
        for i, path in enumerate(files):
            with np.load(cached / (path.stem+".npz"), allow_pickle=False) as z:
                psi, features, labels = z["psi"], z["features"], z["labels"]
            coords = basis.encode(psi).astype(np.float32)
            decoded = basis.decode(coords)
            derived = targets(reader, decoded)
            raw_geometry = targets(reader, psi)[:, :9]
            stop = at + len(psi)
            arrays["features"][at:stop] = features
            arrays["coordinates"][at:stop] = coords
            arrays["labels"][at:stop] = np.column_stack((labels, derived))
            arrays["geometry"][at:stop] = raw_geometry
            shots[at:stop] = i
            squares += float(np.sum(psi.astype(np.float64)**2))
            total += float(psi.sum(dtype=np.float64))
            cells += psi.size
            left += float(np.sum((psi.astype(np.float64)-decoded)**2))
            at = stop
            print(json.dumps({"targets": path.stem, "part": part}), flush=True)
        sums = {"flux_squares_about_the_mean": squares-total**2/cells,
                "flux_squares_left_out_by_the_basis": left}
        record(directory / "metadata.json", {"stems": [p.stem for p in files], "flux_sums": sums,
               "feature_layout": "discharge", "label_names": LABEL_NAMES, "frames": n})
        for array in arrays.values():
            array.flush()
        shots.flush()
        datasets[part] = Table(arrays["features"], arrays["coordinates"], arrays["labels"],
                                LABEL_NAMES, sums, "discharge", shots)
    record(out / "data/preparation.json", {"seconds": time.monotonic()-started,
           "basis_shots": len(basis_files), "basis_method": "exact thin SVD" if limit else "exact covariance eigendecomposition",
           "labelled_source": "DIII-D training only", "smoke_validation_excluded_from_all_fits": bool(limit),
           "full_production_reuses_previously_exposed_validation_and_holdout": not bool(limit)})
    return datasets


def fit_moments(files: list[Path]) -> Moments:
    """Accumulate basis moments with one raw shot in memory."""
    frames = 0
    total = np.zeros(4225)
    outer = np.zeros((4225, 4225))
    for path in files:
        with np.load(path, allow_pickle=False) as z:
            flat = z["psi"].reshape(-1, 4225).astype(np.float64)
        total += flat.sum(0)
        outer += flat.T @ flat
        frames += len(flat)
    return Moments(frames, total, outer)


def merge(parts: list[Table]) -> Table:
    """Combine compatible tables using disk-backed storage for production-size arrays."""
    if len(parts) == 1:
        return parts[0]
    first = parts[0]
    token = hashlib.sha256('|'.join(str(p.features.filename) for p in parts).encode()).hexdigest()[:16]
    directory = Path(first.features.filename).parent.parent / ("joined_" + token)
    directory.mkdir(exist_ok=True)
    arrays = {}
    for name in ("features", "coordinates", "labels", "shot_of_row"):
        values = [getattr(p, name) for p in parts]
        path = directory / (name + '.npy')
        shape = (sum(len(v) for v in values), *values[0].shape[1:])
        if not path.exists():
            joined = np.lib.format.open_memmap(path, mode='w+', dtype=values[0].dtype, shape=shape)
            at, offset = 0, 0
            for value in values:
                for start in range(0, len(value), 4096):
                    stop = min(start+4096, len(value))
                    joined[at+start:at+stop] = value[start:stop] + offset if name == 'shot_of_row' else value[start:stop]
                at += len(value)
                if name == 'shot_of_row':
                    offset += int(value.max())+1
            joined.flush()
        arrays[name] = np.load(path, mmap_mode='r')
    return replace(first, **arrays)


def view(table: Table, names: tuple[str, ...], coordinates: int, padding: int = 0) -> Table:
    """Select a family's labels and append its inactive static embedding."""
    return replace(table, coordinates=table.coordinates[:, :coordinates],
                   labels=table.labels[:, [table.label_names.index(n) for n in names]], label_names=names)


@lru_cache(maxsize=12)
def epoch_kernel(settings_json: str, rows: int, inputs: int, outputs: int):
    """Compile one epoch per architecture and recipe, shared across seeds."""
    settings = FitSettings(**json.loads(settings_json))
    batches = rows // settings.batch
    weights = jnp.asarray(np.r_[np.ones(settings.coordinates), np.full(outputs-settings.coordinates, settings.label_weight)], jnp.float32)
    mix = jnp.eye(outputs, dtype=jnp.float32)

    @jax.jit
    def one_epoch(state, order_key, x, y, initialization):
        def update(carry, selected):
            features = jnp.pad(x[selected], ((0, 0), (0, inputs-x.shape[1])))
            loss, gradient = jax.value_and_grad(fit._loss)(carry[0], features, y[selected], weights,
                                        0.0, jax.random.fold_in(initialization, carry[3]), mix)
            return fit.decayed_adam_step(carry, gradient, settings.epochs*batches, settings), loss
        order = jax.random.permutation(order_key, rows)[:batches*settings.batch]
        state, losses = jax.lax.scan(update, state, order.reshape(batches, settings.batch))
        return state, jnp.mean(losses)
    return one_epoch


class Population:
    """Accumulate finite population means and variances without full-table copies."""

    def __init__(self, columns: int) -> None:
        self.count = np.zeros(columns)
        self.mean = np.zeros(columns)
        self.m2 = np.zeros(columns)

    def add(self, values: NDArray[np.floating]) -> None:
        """Combine one finite-masked block using parallel central moments."""
        raw = np.asarray(values, dtype=np.float64)
        valid = np.isfinite(raw)
        count = valid.sum(0)
        mean = np.where(valid, raw, 0.0).sum(0)/np.maximum(count, 1)
        total = self.count + count
        delta = mean-self.mean
        self.m2 += np.where(valid, raw-mean, 0.0).__pow__(2).sum(0)+delta**2*self.count*count/np.maximum(total, 1)
        self.mean += delta*count/np.maximum(total, 1)
        self.count = total

    def variance(self) -> NDArray[np.float64]:
        """Return finite-column population variances."""
        return self.m2/np.maximum(self.count, 1)


def streaming_scaling(table: Table) -> _Scaling:
    """Fit active normalizers in chunks using only the fitting rows."""
    x = Population(table.features.shape[1])
    y = Population(table.coordinates.shape[1]+table.labels.shape[1])
    for start in range(0, len(table.features), 16384):
        rows = slice(start, start+16384)
        x.add(table.features[rows])
        y.add(np.column_stack((table.coordinates[rows], table.labels[rows])))
    xs, ys = np.sqrt(x.variance()), np.sqrt(y.variance())
    k = table.coordinates.shape[1]
    ys[:k] = np.sqrt(y.variance()[:k].sum())
    return _Scaling(x.mean, np.where(xs > 1e-6, xs, 1.0), y.mean,
                    np.where(ys > 1e-6, ys, 1.0))


@partial(jax.jit, donate_argnums=(0,))
def insert_block(array: jax.Array, block: jax.Array, start: int) -> jax.Array:
    """Upload a block into a donated device table without a second full allocation."""
    return jax.lax.dynamic_update_slice(array, block, (start, 0))


def upload_scaled(table: Table, scaling: _Scaling, device) -> tuple[jax.Array, jax.Array]:
    """Normalize and upload fitting data with bounded host and device temporaries."""
    n = len(table.features)
    x = jax.device_put(np.zeros((1, table.features.shape[1]), np.float32), device)
    y = jax.device_put(np.zeros((1, len(scaling.output_mean)), np.float32), device)
    with jax.default_device(device):
        x = jnp.zeros((n, x.shape[1]), jnp.float32)
        y = jnp.zeros((n, y.shape[1]), jnp.float32)
        for start in range(0, n, 16384):
            rows = slice(start, min(n, start+16384))
            raw_x = np.asarray(table.features[rows], dtype=np.float64)
            raw_y = np.column_stack((table.coordinates[rows], table.labels[rows])).astype(np.float64)
            block_x = np.clip((raw_x-scaling.feature_mean)/scaling.feature_scale, -8., 8.).astype(np.float32)
            block_y = ((raw_y-scaling.output_mean)/scaling.output_scale).astype(np.float32)
            x = insert_block(x, jax.device_put(block_x, device), start)
            y = insert_block(y, jax.device_put(block_y, device), start)
            x.block_until_ready()
            y.block_until_ready()
    return x, y

@jax.jit
def predict64(parameters, features: jax.Array) -> jax.Array:
    """Evaluate the map coordinates for calibration checkpoint selection."""
    return fit.forward(parameters, features, 0.0, None)[:, :64]


def train_network(training: Table, calibration: Table, settings: FitSettings, family: str,
                  destination: Path, device, smoke: bool) -> object:
    """Fit map families with cal-flux selection and scalar families with fixed final epoch."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    scaling = streaming_scaling(training)
    if smoke:
        scaling = replace(scaling, output_mean=np.nan_to_num(scaling.output_mean))
    if family == "static":
        spread = np.nanvar(training.labels.astype(np.float64), axis=0)
        common = float(np.sqrt(np.nanmean(spread)))
        if not np.isfinite(common) or common <= 1e-6:
            raise ValueError("Static X targets have no finite spread")
        scaling = replace(scaling, output_scale=np.full(4, common))
    x, y = upload_scaled(training, scaling, device)
    padding = 183 if family == 'joint' else 32 if family == 'static' else 0
    if padding:
        scaling = replace(scaling, feature_mean=np.r_[scaling.feature_mean, np.zeros(padding)],
                          feature_scale=np.r_[scaling.feature_scale, np.ones(padding)])
    input_width = len(scaling.feature_mean)
    key = jax.random.key(settings.seed)
    if family == "static":
        with jax.default_device(jax.devices("cpu")[0]):
            parameters = fit.initial_parameters(jax.random.key(settings.seed), input_width, y.shape[1], settings)
        parameters = jax.tree.map(lambda a: jax.device_put(a, device), parameters)
        initialization = key
        key = jax.random.fold_in(key, 700)
    else:
        key, initialization = jax.random.split(key)
        parameters = fit.initial_parameters(initialization, input_width, y.shape[1], settings)
    state = (parameters, jax.tree.map(jnp.zeros_like, parameters),
             jax.tree.map(jnp.zeros_like, parameters), jnp.zeros((), jnp.int32))
    signature = json.dumps({**asdict(settings), "seed": 0}, sort_keys=True)
    step = epoch_kernel(signature, len(x), input_width, y.shape[1])
    history = []
    best, criterion, selected_epoch = None, float("inf"), 0
    held = jax.device_put(scaling.inputs(calibration.features).numpy(), device) if settings.coordinates else None

    for epoch in range(settings.epochs):
        key, order = jax.random.split(key)
        state, loss = step(state, order, x, y, initialization)
        value = float(loss.block_until_ready())
        if not np.isfinite(value):
            raise RuntimeError("Nonfinite loss; no numerical retry or seed substitution")
        current = 0.0
        if settings.coordinates:
            predicted = np.asarray(predict64(state[0], held), dtype=np.float64)
            predicted = predicted*scaling.output_scale[:settings.coordinates]+scaling.output_mean[:settings.coordinates]
            squared = np.sort(np.sum((predicted-calibration.coordinates.astype(np.float64))**2, axis=1))
            current = float(squared[:-max(1, len(squared)//fit.TRIMMED_SHARE)].sum())
        if current < criterion or not settings.coordinates:
            best = jax.tree.map(lambda a: np.asarray(a, dtype=np.float64).copy(), state[0])
            criterion, selected_epoch = current, epoch+1
        history.append({"epoch": epoch+1, "loss": value, "calibration_trimmed_SSE": current})
        if (epoch+1) % 10 == 0:
            print(json.dumps({'family': family, 'seed': settings.seed, 'epoch': epoch+1,
                              'loss': value, 'seconds': time.monotonic()-started}), flush=True)
    if best is None:
        raise RuntimeError("No finite checkpoint")
    network = fit._export(best, scaling, settings, training)
    network.save(destination)
    quality = None
    if family == "static" and not smoke:
        losses = np.asarray([r["loss"] for r in history])
        final = float(losses[-10:].mean())
        lowest = float(np.lib.stride_tricks.sliding_window_view(losses, 10).mean(1).min())
        quality = {"final10": final, "lowest10": lowest, "pass": final <= 1.25*lowest}
        if not quality["pass"]:
            raise RuntimeError("Static seed quality failed; no replacement")
    record(destination.with_suffix(".fit.json"), {"family": family, "settings": asdict(settings),
           "frames": len(x), "device": str(device), "device_kind": device.device_kind,
           "seconds": time.monotonic()-started, "history": history, "selected_epoch": selected_epoch,
           "selection": "trimmed calibration flux" if settings.coordinates else "fixed final epoch",
           "network_sha256": digest(destination), "static_quality": quality, "smoke": smoke,
           "static_unused_temporal_encoder_omitted": family == "static"})
    del state, x, y, held, best
    gc.collect()
    return network


def geometry_readout(network, features: F32, truth: F32, destination: Path) -> None:
    """Accumulate finite-target residual moments before solving the frozen ridge readout."""
    groups = validity_groups(truth)
    moments = None
    for start in range(0, len(features), 4096):
        values = cpu_moments(network, features[start:start+4096], truth[start:start+4096], GEOMETRY_NAMES, groups)
        if moments is None:
            moments = values
        else:
            moments = tuple(tuple(a+b for a, b in zip(first, second, strict=True))
                            for first, second in zip(moments, values, strict=True))
    corrected, report = corrected_from_moments(network, moments, GEOMETRY_NAMES, groups, .001)
    corrected.save(destination)
    record(destination.with_suffix(".readout.json"), report)


def train_trees(training: Table, calibration: Table, out: Path, smoke: bool, device: str) -> list[dict]:
    """Fit the four D9 heads on native F367 features using the frozen A/B capacities."""
    result = []
    for name, share in (("li_scored", .26), ("r_axis", .14), ("kappa", .09), ("x_lower_r", .16)):
        started = time.monotonic()
        col = training.label_names.index(name)
        target = np.asarray(training.labels[:, col])
        held = np.asarray(calibration.labels[:, col])
        present = np.isfinite(target)
        valid = np.isfinite(held)
        if present.sum() < 2 or valid.sum() < 2:
            raise ValueError(f"Insufficient finite tree targets: {name}")
        params = {"tree_method": "hist", "device": device, "objective": "reg:squarederror",
                  "max_depth": 8, "eta": .05 if name == "x_lower_r" else .1,
                  "subsample": .8, "colsample_bytree": .5, "min_child_weight": 20,
                  "lambda": 1, "max_bin": 256, "seed": 9, "nthread": 2,
                  "base_score": float(target[present].mean())}
        train = xgb.DMatrix(training.features[:, :367], label=np.nan_to_num(target),
                            weight=present.astype(np.float32), nthread=2)
        cal = xgb.DMatrix(calibration.features[:, :367], label=np.nan_to_num(held),
                          weight=valid.astype(np.float32), nthread=2)
        rounds = 5 if smoke else 3000 if name == "x_lower_r" else 6000
        patience = 2 if smoke else 100 if name == "x_lower_r" else 200
        booster = xgb.train(params, train, rounds, evals=[(cal, "calibration")],
                            early_stopping_rounds=patience, verbose_eval=False)
        path = out / "models/trees" / ("tree_"+name+".ubj")
        path.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(path)
        count = booster.best_iteration+1
        leaves = []
        for raw in booster.get_dump(dump_format="json")[:count]:
            pending = [json.loads(raw)]
            values = {}
            while pending:
                node = pending.pop()
                if "leaf" in node:
                    values[node["nodeid"]] = np.float32(node["leaf"])
                else:
                    pending.extend(node["children"])
            leaves.append(values)
        maximum = max(max(values) for values in leaves)
        matrix = np.full((count, maximum+1), np.nan, dtype=np.float32)
        for i, values in enumerate(leaves):
            for node, value in values.items():
                matrix[i, node] = value
        config = json.loads(booster.save_config())["learner"]["learner_model_param"]
        base = np.float32(json.loads(config["base_score"])[0])
        lookup = path.parent / "tree_lookups" / (name+".npz")
        lookup.parent.mkdir(exist_ok=True)
        np.savez(lookup, leaf_values=matrix, base_score=base)
        item = {"target": name, "share": share, "path": path.relative_to(out).as_posix(),
                "sha256": digest(path), "best_iteration": booster.best_iteration,
                "lookup_path": lookup.relative_to(out).as_posix(), "lookup_sha256": digest(lookup)}
        record(path.with_suffix(".fit.json"), {**item, "parameters": params, "rounds": rounds,
               "patience": patience, "seconds": time.monotonic()-started, "smoke": smoke})
        result.append(item)
    return result


def main() -> None:
    """Run extraction, basis, all network/readout/tree fits and write the new manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="training shots for a smoke run; 0 uses all data")
    parser.add_argument("--frames-per-shot", type=int, default=48, help="used only with --limit")
    parser.add_argument("--epochs", type=int, default=1, help="used only with --limit")
    parser.add_argument("--platform", choices=("cpu", "cuda", "rocm"), default="cpu")
    parser.add_argument("--tree-device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.limit and (args.limit < 3 or args.epochs < 1):
        parser.error("--limit needs at least 3 shots and a positive --epochs")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "models").mkdir()
    started = time.monotonic()
    device = jax.devices('cpu' if args.platform == 'cpu' else 'gpu')[0]
    if args.platform != "cpu" and device.platform != "gpu":
        raise RuntimeError("Requested accelerator is unavailable")
    record(out / "run.json", {"raw_dir": str(args.raw_dir.resolve()), "limit": args.limit,
           "frames_per_shot": args.frames_per_shot if args.limit else None,
           "epochs": args.epochs if args.limit else 60, "platform": args.platform,
           "device": str(device), "device_kind": device.device_kind,
           "recipe_sha256": digest(HERE / "training_recipe.json"),
           "training_source_sha256": digest(Path(__file__)), "deployed_assets_loaded": False})
    data = prepare(args.raw_dir, out, args.limit, args.frames_per_shot)
    smoke = bool(args.limit)
    training = data["training"] if smoke else merge([data["training"], data["validation"], data["holdout"]])
    calibration = data["calibration"]
    all_data = merge([training, calibration])
    geometry = np.concatenate([np.load(out/"data"/part/"geometry.npy", mmap_mode="r") for part in
                (["training"] if smoke else ["training", "validation", "holdout"])])
    cal_geometry = np.load(out/"data/calibration/geometry.npy", mmap_mode="r")
    manifest = json.loads((HERE / "deployment.json").read_text())
    manifest["maps"], manifest["joint"], manifest["static"] = [], [], []
    manifest["basis"] = {"path": "models/flux_basis.npz", "sha256": digest(out/"models/flux_basis.npz")}
    members = json.loads((HERE / "training_recipe.json").read_text())["members"]
    for i, member in enumerate(members):
        settings = FitSettings(**member["fit_settings"])
        if smoke:
            settings = replace(settings, epochs=args.epochs, batch=64, threads=2)
        source = all_data if member["train_on_calibration"] else training
        names = tuple(member["label_names"])
        train = view(source, names, 64)
        cal = view(calibration, names, 64)
        path = out / member["model_path"]
        network = train_network(train, cal, settings, "maps", path, device, smoke)
        truth = np.concatenate((geometry, cal_geometry)) if member["train_on_calibration"] else geometry
        # Keep encoder and readout artifacts separately for provenance.
        path.rename(path.with_name("encoder_network.npz"))
        geometry_readout(network, source.features, truth, path)
        manifest["maps"].append({"path": member["model_path"], "sha256": digest(path)})
        print(json.dumps({"map_readout_complete": i+1, "total": 69}), flush=True)
        del network
        gc.collect()
    tree_training = training if smoke else merge([data["training"], data["validation"]])
    manifest["trees"] = train_trees(tree_training, calibration, out, smoke, args.tree_device)
    for family, seeds, names, padding, batch, weight in (
            ("joint", range(200, 208), ("q95", "beta_n"), 183, 4096, .05),
            ("static", range(500, 508), X_NAMES, 32, 2048, 1.0)):
        train = view(all_data, names, 0, padding)
        cal = view(calibration, names, 0, padding)
        if family == "static":
            train = replace(train, labels=np.concatenate((geometry, cal_geometry))[:, 5:9])
            cal = replace(cal, labels=cal_geometry[:, 5:9])
        for seed in seeds:
            settings = FitSettings(coordinates=0, seed=seed, batch=64 if smoke else batch,
                     epochs=args.epochs if smoke else 60, label_weight=weight, gradient_limit=1, threads=2)
            relative = f"models/{family}8/{'joint_seed' if family == 'joint' else 'static_'}{seed}/network.npz"
            train_network(train, cal, settings, family, out/relative, device, smoke)
            manifest[family].append({"path": relative, "sha256": digest(out/relative)})
            print(json.dumps({"family": family, "seed_complete": seed}), flush=True)
    shutil.copytree(HERE / "source", out / "source")
    for name in ("infer.py", "tree_replica.py", "requirements-inference-cpu.txt", "LICENSE"):
        shutil.copy2(HERE / name, out / name)
    manifest["status"] = "smoke_only_not_a_submission" if smoke else "retrained_DIII_D_requires_accuracy_review"
    manifest["submission"] = None
    manifest["training_scope"] = "DIII-D from scratch; frozen MAST inference assets supplied separately"
    record(out / "deployment.json", manifest)
    record(out / "completed.json", {"status": "all_stages_complete", "map_networks": len(manifest["maps"]),
           "trees": len(manifest["trees"]), "joint": len(manifest["joint"]), "static": len(manifest["static"]),
           "seconds": time.monotonic()-started, "smoke_only": smoke, "manifest_sha256": digest(out/"deployment.json")})
    print(json.dumps({"complete": True, "out": str(out), "seconds": time.monotonic()-started}), flush=True)


if __name__ == "__main__":
    main()
