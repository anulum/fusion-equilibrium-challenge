# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — original scored-field package tests
"""Read real D72 fields and refuse changed manifests, headers and source support."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from tools import scored_map_bundle
from tools.scored_map_bundle import load, load_shot, original_cohort
from tools.scored_map_summary import MAPS

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/d72_scored_map_replay"
EXAMPLE = EXPERIMENT / "illustration_v1"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(root: Path, data: object) -> str:
    path = root / "manifest.json"
    path.write_text(json.dumps(data) + "\n")
    return _hash(path)


def _read(root: Path) -> object:
    bundle = load(root, _hash(root / "manifest.json"))
    return load_shot(bundle, bundle.cases[0])


@pytest.fixture
def copied(tmp_path: Path) -> Path:
    """Copy the deterministic original frame-zero package without changing its data."""
    path = tmp_path / "original"
    shutil.copytree(EXAMPLE, path)
    return path


def test_original_selected_arrays_are_read_only_and_unscaled() -> None:
    """All actual map bytes and original parent counts survive the public reader."""
    bundle = load(EXAMPLE, _hash(EXAMPLE / "manifest.json"))
    assert bundle.scope == "illustration" and len(bundle.cases) == 3
    assert [c.source_frames for c in bundle.cases] == [213, 293, 217]
    assert sum(cast(int, c["frames"]) for c in original_cohort()) == 78875
    for case in bundle.cases:
        data = load_shot(bundle, case)
        assert data.source_indices.tolist() == [0] and data.scalars.shape == (1, 2)
        for name, value in zip(MAPS, data.fields, strict=True):
            assert value.dtype == np.dtype("<f4") and value.shape == (1, 65, 65)
            assert not value.flags.writeable
            assert hashlib.sha256(value.tobytes()).hexdigest() == case.arrays_sha256[name]
        assert all(data.contour(0, j) is not None for j in range(3))


def test_actual_whole_shot_manifest_metadata_keeps_parent_identities(copied: Path) -> None:
    """The real 723-frame manifest parses without pretending its fields are bundled here."""
    manifest = json.loads((EXPERIMENT / "qualification_manifest_v1.json").read_text())
    bundle = load(copied, _write(copied, manifest))
    assert bundle.scope == "qualification" and sum(c.frames for c in bundle.cases) == 723
    with pytest.raises(ValueError, match="file differs"):
        load_shot(bundle, bundle.cases[0])
    manifest["shots"][0]["arrays_sha256"]["gs"] = "0" * 64
    with pytest.raises(ValueError, match="original scored-field"):
        load(copied, _write(copied, manifest))


@pytest.mark.parametrize("version", [(1, 0), (2, 0)])
def test_actual_npy_header_versions(copied: Path, version: tuple[int, int]) -> None:
    """Both supported NPY versions retain the actual grid values."""
    manifest = json.loads((copied / "manifest.json").read_text())
    path = copied / "R.npy"
    values = np.load(path, allow_pickle=False)
    with path.open("wb") as stream:
        np.lib.format.write_array(stream, values, version=version, allow_pickle=False)
    manifest["grid"]["R.npy"] = _hash(path)
    _write(copied, manifest)
    _read(copied)


@pytest.mark.parametrize(
    "case",
    [
        "false-full",
        "schema",
        "maps",
        "length",
        "flux",
        "sign-bool",
        "sign",
        "scope",
        "grid-member",
        "shot-list",
        "population",
        "order",
        "source-frames",
        "frames-type",
        "frames-limit",
        "frames-scope",
        "contour-limit",
        "files",
        "identity-member",
        "identity-hash",
        "array-member",
        "array-hash",
        "missing-key",
        "not-object",
    ],
)
def test_changed_manifest_contract_is_refused(copied: Path, case: str) -> None:
    """Transport rehashing cannot legitimise changed scope, conventions or identities."""
    d = json.loads((copied / "manifest.json").read_text())
    row = d["shots"][0]
    if case == "false-full":
        d["scope"] = "full358"
    elif case == "schema":
        d["schema"] = "unknown"
    elif case == "maps":
        d["maps"].reverse()
    elif case == "length":
        d["length_unit"] = "cm"
    elif case == "flux":
        d["flux_convention"] = "guessed"
    elif case == "sign-bool":
        d["axis_sign"] = True
    elif case == "sign":
        d["axis_sign"] = 0
    elif case == "scope":
        d["scope"] = "subset"
    elif case == "grid-member":
        del d["grid"]["Z.npy"]
    elif case == "shot-list":
        d["shots"] = {}
    elif case == "population":
        d["shots"].pop()
    elif case == "order":
        row["stem"] = "different"
    elif case == "source-frames":
        row["source_frames"] = 1
    elif case == "frames-type":
        row["frames"] = True
    elif case == "frames-limit":
        row["frames"] = 0
    elif case == "frames-scope":
        row["frames"] = 2
    elif case == "contour-limit":
        row["contour_points"] = 12289
    elif case == "files":
        del row["files"]["base.npy"]
    elif case == "identity-member":
        row["identities"]["extra"] = "0" * 64
    elif case == "identity-hash":
        row["identities"]["raw_parquet_sha256"] = "unbound"
    elif case == "array-member":
        del row["arrays_sha256"]["reference"]
    elif case == "array-hash":
        row["arrays_sha256"]["base"] = 17
    elif case == "missing-key":
        del d["scope"]
    else:
        d = []
    with pytest.raises(ValueError):
        load(copied, _write(copied, d))


def test_manifest_size_and_independent_pin(copied: Path) -> None:
    """Oversized JSON and an incorrect caller pin fail before numerical allocation."""
    with pytest.raises(ValueError, match="identity"):
        load(copied, "0" * 64)
    path = copied / "manifest.json"
    path.write_bytes(b" " * (4 * 2**20 + 1))
    with pytest.raises(ValueError, match="limit"):
        load(copied, _hash(path))


@pytest.mark.parametrize(
    "case", ["missing", "symlink", "oversize", "hash", "dtype", "shape", "trailing", "version"]
)
def test_changed_numerical_file_is_refused(copied: Path, case: str) -> None:
    """Real malformed NPY files fail even when their new transport hash is declared."""
    d = json.loads((copied / "manifest.json").read_text())
    path = copied / "R.npy"
    array = np.load(path, allow_pickle=False)
    if case == "missing":
        path.unlink()
    elif case == "symlink":
        other = copied / "R-copy.npy"
        path.rename(other)
        path.symlink_to(other)
    elif case == "oversize":
        with path.open("r+b") as stream:
            stream.truncate(512 * 2**20 + 1)
    elif case == "hash":
        path.write_bytes(b"changed")
    elif case == "dtype":
        np.save(path, array.astype(np.float32), allow_pickle=False)
    elif case == "shape":
        np.save(path, array[:-1], allow_pickle=False)
    elif case == "trailing":
        with path.open("ab") as stream:
            stream.write(b"x")
    else:
        with path.open("wb") as stream:
            np.lib.format.write_array(stream, array, version=(3, 0), allow_pickle=False)
    if case not in ("missing", "oversize", "hash"):
        d["grid"]["R.npy"] = _hash(path)
    with pytest.raises(ValueError):
        load(copied, _write(copied, d))


@pytest.mark.parametrize("case", ["nan", "decreasing", "nonuniform", "radius", "envelope"])
def test_changed_grid_is_refused(copied: Path, case: str) -> None:
    """A rehashed actual grid must retain finite uniform positive-radius coordinates."""
    d = json.loads((copied / "manifest.json").read_text())
    name = "envelope.npy" if case == "envelope" else "R.npy"
    path = copied / name
    value = np.load(path, allow_pickle=False)
    if case == "nan":
        value[0] = np.nan
    elif case == "decreasing":
        value = value[::-1]
    elif case == "nonuniform":
        value[5] += 0.001
    elif case == "radius":
        value -= 3
    else:
        value[:] = False
    np.save(path, value, allow_pickle=False)
    d["grid"][name] = _hash(path)
    with pytest.raises(ValueError):
        load(copied, _write(copied, d))


@pytest.mark.parametrize(
    "case",
    [
        "nan",
        "offset-first",
        "offset-end",
        "offset-order",
        "reference-absent",
        "indices",
        "iterations-negative",
        "iterations-limit",
        "pixels-negative",
        "pixels-limit",
        "converged-unused",
        "unused-iterations",
        "array-bytes",
    ],
)
def test_changed_shot_support_is_refused(copied: Path, case: str) -> None:
    """Corrupted original arrays cannot hide behind updated file checksums."""
    d = json.loads((copied / "manifest.json").read_text())
    row = d["shots"][0]
    names = {"nan": "reference", "indices": "source_indices", "array-bytes": "base"}
    name = names.get(
        case,
        "offsets"
        if case.startswith("offset") or case == "reference-absent"
        else "pixels"
        if case.startswith("pixels")
        else "iterations",
    )
    path = copied / row["stem"] / (name + ".npy")
    value = np.load(path, allow_pickle=False)
    if case == "nan":
        value[0, 0, 0] = np.nan
    elif case == "offset-first":
        value[0] = 1
    elif case == "offset-end":
        value[-1] -= 1
    elif case == "offset-order":
        value[1] = -1
    elif case == "reference-absent":
        value[1] = 0
    elif case == "indices":
        value[0] = 1
    elif case == "iterations-negative":
        value[0] = -1
    elif case == "iterations-limit":
        value[0] = 41
    elif case == "pixels-negative":
        value[0] = -1
    elif case == "pixels-limit":
        value[0] = 3970
    elif case == "array-bytes":
        value[0, 20, 20] += 1
    else:
        value[:] = 1
        other = copied / row["stem"] / "used.npy"
        np.save(other, np.array([False]), allow_pickle=False)
        row["files"]["used.npy"] = _hash(other)
        if case == "unused-iterations":
            other = copied / row["stem"] / "converged.npy"
            np.save(other, np.array([False]), allow_pickle=False)
            row["files"]["converged.npy"] = _hash(other)
    np.save(path, value, allow_pickle=False)
    row["files"][name + ".npy"] = _hash(path)
    _write(copied, d)
    with pytest.raises(ValueError):
        _read(copied)


def test_undeclared_shot_refused() -> None:
    """An API caller cannot read a case outside the validated manifest."""
    bundle = load(EXAMPLE, _hash(EXAMPLE / "manifest.json"))
    with pytest.raises(ValueError, match="not in"):
        load_shot(bundle, replace(bundle.cases[0], stem="different"))


def test_changed_original_population_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The independent published original cohort binding cannot be edited silently."""
    path = tmp_path / "experiments/d72_scored_map_physics/manifest_v1.json"
    path.parent.mkdir(parents=True)
    data = json.loads((ROOT / "experiments/d72_scored_map_physics/manifest_v1.json").read_text())
    data["cohort"][0]["frames"] += 1
    path.write_text(json.dumps(data))
    monkeypatch.setattr(scored_map_bundle, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="population differs"):
        original_cohort()
