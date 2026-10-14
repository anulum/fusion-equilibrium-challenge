# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — reproduce original-map geometry readout fitting
"""Refit geometry heads from eligible DIII-D maps and freshly trained base networks."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import jax
import numpy as np
import pyarrow.parquet as pq
from fec_adapter.diii_d_accuracy import contour_edit, x_point
from fec_adapter.diii_d_accuracy.network import Network
from fec_adapter.diii_d_accuracy.shape_edit import Reader
from fec_adapter.diii_d_accuracy.shape_labels import SHAPE_LABELS
from fec_adapter.split import refuse_holdout
from geometry_readout_streaming import (
    corrected_from_moments,
    gpu_moment_function,
    validity_groups,
)
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
NAMES = [*SHAPE_LABELS, *x_point.LABELS, *contour_edit.label_names()]


def derive(path: Path, reader: Reader) -> NDArray[np.float32]:
    """Derive target heads from one eligible DIII-D shot's original float32 maps."""
    refuse_holdout(path.stem)
    table = pq.read_table(path, columns=["source", "efit_psirz"])
    if table.num_rows != 1 or table["source"][0].as_py() != "DIII-D":
        raise ValueError("Only one DIII-D row is admitted")
    maps = np.asarray(table["efit_psirz"][0].as_py(), dtype=np.float32)
    labels = np.full((len(maps), len(NAMES)), np.nan, dtype=np.float32)
    for i, stored in enumerate(maps):
        flux = np.asarray(stored, dtype=np.float64)
        reading = reader.read(flux)
        labels[i, :5] = [reading.values[name] for name in SHAPE_LABELS.values()]
        if reading.contour is None:
            continue
        saddles = x_point.boundary_saddles(reader, flux, reading)
        for j, end in enumerate(x_point.ENDS):
            if saddles[end] is not None:
                labels[i, 5 + 2 * j : 7 + 2 * j] = saddles[end].position
        labels[i, 9:] = contour_edit.radii(reading.contour, reading.axis)
    return labels


def main() -> None:
    """Regenerate eligible targets and apply the fixed ridge to every named model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--base-models", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=4096)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    jax.config.update("jax_enable_x64", True)
    members = json.loads((ROOT / "training_recipe.json").read_text())
    reserved = set(
        json.loads(
            (ROOT / "source/fec_adapter/reserved_for_evaluation.json").read_text()
        )["stems"]
    )
    splits = json.loads((ROOT / "source/fec_adapter/splits.json").read_text())
    reader = Reader("DIII-D")
    tables = {}
    for part in ("training", "calibration"):
        with np.load(args.tables / f"table_{part}.npz", allow_pickle=False) as table:
            stems = table["stems"].astype(str)
            shot_of_row = table["shot_of_row"]
            features = table["features"]
            layout = str(table["feature_layout"])
        if not set(stems) <= set(splits[part]):
            raise ValueError("Split mismatch")
        counts = np.bincount(shot_of_row, minlength=len(stems))
        bounds = np.r_[0, np.cumsum(counts)]
        truth = np.full((len(features), len(NAMES)), np.nan, dtype=np.float32)
        eligible = []
        for index, stem in enumerate(stems):
            if stem in reserved:
                continue
            start, stop = map(int, bounds[index : index + 2])
            refuse_holdout(stem)
            np.testing.assert_array_equal(
                shot_of_row[start:stop], np.full(stop - start, index)
            )
            labels = derive(args.data_root / "diii_d_train" / f"{stem}.parquet", reader)
            if labels.shape != (stop - start, len(NAMES)):
                raise ValueError("Frame mismatch")
            truth[start:stop] = labels
            eligible.append(np.arange(start, stop))
        tables[part] = (features, truth, np.concatenate(eligible), layout)
        print(
            json.dumps(
                {"targets_complete": part, "eligible_rows": len(tables[part][2])}
            ),
            flush=True,
        )
    for member in members:
        old = Network.load(args.base_models / member["name"] / "network.npz")
        names = member["geometry_labels"]
        dest = args.out / f"network_{member['recipe_index']:03d}.npz"
        if not names:
            shutil.copy2(args.base_models / member["name"] / "network.npz", dest)
            continue
        selected = [NAMES.index(n) for n in names]
        parts = (
            ("training", "calibration")
            if member["train_on_calibration"]
            else ("training",)
        )
        all_truth = np.concatenate(
            [tables[p][1][tables[p][2]][:, selected] for p in parts]
        )
        groups = validity_groups(all_truth)
        del all_truth
        kernel = gpu_moment_function(old, names, groups)
        total = None
        for part in parts:
            features, truth, rows, layout = tables[part]
            if layout != old.feature_layout:
                raise ValueError("Feature layout mismatch")
            for offset in range(0, len(rows), args.batch_rows):
                block = rows[offset : offset + args.batch_rows]
                values, bad_predictions, bad_masks = jax.device_get(
                    kernel(
                        features[block, : old.feature_mean.size],
                        truth[block][:, selected],
                    )
                )
                if int(bad_predictions) or int(bad_masks):
                    raise ValueError("Invalid fitting values")
                values = tuple(
                    tuple(np.asarray(v, dtype=np.float64) for v in group)
                    for group in values
                )
                total = (
                    values
                    if total is None
                    else tuple(
                        tuple(a + b for a, b in zip(left, right, strict=True))
                        for left, right in zip(total, values, strict=True)
                    )
                )
        updated, reports = corrected_from_moments(old, total, names, groups, 0.001)
        updated.save(dest)
        with dest.with_suffix(".json").open("x") as stream:
            json.dump(
                {"member": member, "reports": reports, "ridge": 0.001}, stream, indent=2
            )
        print(json.dumps({"model_completed": member["recipe_index"]}), flush=True)
        del kernel
        jax.clear_caches()
    shutil.copy2(args.tables / "flux_basis.npz", args.out / "flux_basis.npz")


if __name__ == "__main__":
    main()
