# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Equilibrium Challenge — native preserved-baseline residual evaluation
"""Apply exported corrections at the qualified native intervention points."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import numpy as np
import pyarrow.parquet as pq  # type: ignore[import-untyped]  # PyArrow supplies no typing marker.
from numpy.typing import NDArray

F64 = NDArray[np.float64]
F32 = NDArray[np.float32]
TARGETS = ("q95", "beta_n", "r_axis", "z_axis", "kappa", "tri_top", "tri_bot", "volume", "li_scored",
           "x_lower_r", "x_lower_z", "x_upper_r", "x_upper_z")
TREE_WEIGHTS = {"li_scored": .26, "r_axis": .14, "kappa": .09, "x_lower_r": .16}


class Network(Protocol):
    """The exported incumbent network's numerical interface."""

    def predict(self, features: F64) -> tuple[F64, dict[str, F64]]:
        """Return native coordinates and named frame heads."""
        ...


def digest(path: Path) -> str:
    """Compute the checksum of a frozen source or model artifact."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_module(path: Path, name: str) -> ModuleType:
    """Load the explicitly pinned deployment source."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativePipeline:
    """Hold the full excluded incumbent without loading unrelated experiments.

    Parameters
    ----------
    manifest_path : pathlib.Path
        Host-specific paths and common checksums, generated before scoring.
    """

    def __init__(self, manifest_path: Path) -> None:
        manifest = json.loads(manifest_path.read_text())
        self.manifest = manifest
        for item in manifest["files"]:
            path = Path(item["path"])
            if digest(path) != item["sha256"]:
                raise ValueError(f"Activated artifact changed: {path}")
        package = Path(manifest["package"])
        for path in (package / "source/fec_adapter/vendor", package / "source/fec_adapter", package / "entrypoints",
                     Path(manifest["integration"]).parent):
            sys.path.insert(0, str(path))
        self.pd = __import__("predict_diii_d")
        self.integration = load_module(Path(manifest["integration"]), "d66_native_x_integration")
        self.maps = [cast(Network, self.pd.Network.load(Path(item["path"]))) for item in manifest["maps"]]
        self.joint = [cast(Network, self.pd.Network.load(Path(item["path"]))) for item in manifest["joint"]]
        self.static = [cast(Network, self.pd.Network.load(Path(item["path"]))) for item in manifest["static"]]
        if (len(self.maps), len(self.joint), len(self.static)) != (69, 8, 8):
            raise ValueError("The preserved incumbent requires all 85 fitted networks")
        self.basis = self.pd.FluxBasis.load(Path(manifest["basis"])).truncated(64)
        args = self.pd._parser().parse_args(["--data-root", "/dev/null", "--network", manifest["maps"][0]["path"],
                                           "--basis", manifest["basis"], "--out", str(manifest_path.parent),
                                           "--split", "diii_d_train", *manifest["finisher_flags"]])
        self.finish = self.pd._finish_from(args)
        self.original = self.pd.read_row, self.pd.layout_features, self.pd.predict_together
        import xgboost as xgb
        self.xgb = xgb
        self.trees = {}
        for name, item in manifest["trees"].items():
            booster = xgb.Booster()
            booster.load_model(item["model"])
            booster.set_param({"device": "cpu", "nthread": 1})
            with np.load(item["lookup"], allow_pickle=False) as stored:
                self.trees[name] = booster, stored["leaf_values"].copy(), stored["base_score"].copy()

    def inputs(self, path: Path) -> dict[str, object]:
        """Read permitted inputs with bounded arrays for magnetic traces.

        Parameters
        ----------
        path : pathlib.Path
            A complete shot parquet.

        Returns
        -------
        dict
            Predictor input columns only.
        """
        columns = self.pd.input_columns(dict.fromkeys(pq.read_schema(path).names))
        table = pq.read_table(path, columns=columns, use_threads=False)
        row: dict[str, object] = {}
        for name in columns:
            value = table[name][0]
            flat = name.startswith("magnetics_") and str(table.schema.field(name).type) in (
                "list<element: double>", "list<element: float>")
            row[name] = value.values.to_numpy(zero_copy_only=False).astype(float) if flat else value.as_py()
        return row

    def heads(self, features: F64) -> tuple[F64, dict[str, F64], dict[str, F64], dict[str, F64]]:
        """Compute CPU heads and trees before any residual intervention.

        Parameters
        ----------
        features : numpy.ndarray
            Cached, qualified native float64 frame inputs.

        Returns
        -------
        tuple
            Coordinates, tree-adjusted heads, joint heads and static X heads.
        """
        coordinates, labels = self.original[2](self.maps, {"discharge": features})
        matrix = self.xgb.DMatrix(np.asarray(features[:, :367], dtype=np.float32, order="C"), nthread=1)
        adjusted = {name: values.copy() for name, values in labels.items()}
        for name, (booster, leaves, base) in self.trees.items():
            indices = booster.predict(matrix, pred_leaf=True, iteration_range=(0, leaves.shape[0])).astype(np.int32)
            leaf_values = leaves[np.arange(leaves.shape[0])[None, :], indices]
            tree = np.cumsum(leaf_values, axis=1, dtype=np.float32)[:, -1] + base
            adjusted[name] += TREE_WEIGHTS[name] * (tree.astype(float) - adjusted[name])
        static_heads = [model.predict(np.pad(features, ((0, 0), (0, 32))))[1] for model in self.static]
        expert = {name: np.mean([heads[name] for heads in static_heads], axis=0) for name in TARGETS[9:]}
        _, joint = self.original[2](self.joint, {"discharge": np.pad(features, ((0, 0), (0, 183)))})
        return cast(F64, coordinates), adjusted, cast(dict[str, F64], joint), expert

    def finish_maps(self, path: Path, row: dict[str, object], features: F64, coordinates: F64,
                    adjusted: dict[str, F64], joint: dict[str, F64], expert: dict[str, F64],
                    delta: F64) -> dict[str, F32]:
        """Finish corrected heads with unchanged map coordinates and options.

        Parameters
        ----------
        path, row, features, coordinates : object
            The complete shot's pinned native prediction inputs.
        adjusted, joint, expert : dict
            All heads of the same excluded incumbent.
        delta : numpy.ndarray
            Physical corrections in the declared thirteen-target order.

        Returns
        -------
        dict
            Submitted float32 maps, q95 and betaN for every frame.
        """
        corrected = {name: values.copy() for name, values in adjusted.items()}
        for index, name in enumerate(TARGETS[2:9], start=2):
            corrected[name] += delta[:, index]
        corrected_expert = {name: expert[name] + 2 * delta[:, index] for index, name in enumerate(TARGETS[9:], start=9)}
        pd = self.pd
        setattr(pd, "read_row", lambda _path: row)
        setattr(pd, "layout_features", lambda *_args: {"discharge": features})
        setattr(pd, "predict_together", lambda *_args: (coordinates.copy(), {name: values.copy() for name, values in corrected.items()}))
        try:
            with self.integration.use_x_specialist(pd, corrected_expert, .5, half_window=0):
                maps, q95, beta = pd.predict_shot(path, self.maps, self.basis, self.finish)
        finally:
            for name, original in zip(("read_row", "layout_features", "predict_together"), self.original, strict=True):
                setattr(pd, name, original)
        fields = {"psirz": maps}
        for name, base, head, share, index in (("q95", q95, "q95", .46, 0), ("betaN", beta, "beta_n", .87, 1)):
            blended = ((1 - share) * base.astype(float) + share * joint[head]).astype(np.float32)
            fields[name] = (blended.astype(float) + delta[:, index]).astype(np.float32)
        if any(value.dtype != np.float32 or not np.isfinite(value).all() for value in fields.values()):
            raise ValueError("A native submitted output is nonfinite or has changed precision")
        return fields


def context_of(fields: dict[str, F32], adjusted: dict[str, F64]) -> F32:
    """Encode the same incumbent context as the paired fitting cache.

    Parameters
    ----------
    fields : dict
        Current-share submitted scalars before the residual correction.
    adjusted : dict
        Tree-adjusted geometry heads before map finishing.

    Returns
    -------
    numpy.ndarray
        Signed-log values and finite flags in the fixed eighteen-column layout.
    """
    values = np.column_stack((fields["q95"], fields["betaN"], *(adjusted[name] for name in TARGETS[2:9])))
    finite = np.isfinite(values)
    values = np.where(finite, values, 0.)
    return np.column_stack((np.sign(values) * np.log1p(np.abs(values)), finite)).astype(np.float32)
