# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — exact scored-field replay inputs
"""Validate a hash-pinned original-field package without a model or pickle cache."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.lib import format as npformat
from numpy.typing import NDArray

from tools.scored_map_summary import MAPS

F64 = NDArray[np.float64]
F32 = NDArray[np.float32]
I64 = NDArray[np.int64]
Mask = NDArray[np.bool_]
ROOT = Path(__file__).resolve().parents[1]
COHORT_SHA256 = "5f719b4324cb1cde8308bf2285f4bad73ae9b365815a2b705b927139bef41f0b"
MAX_FILE_BYTES = 512 * 2**20
MAX_MANIFEST_BYTES = 4 * 2**20
SHOT_FILES = {
    "reference.npy",
    "base.npy",
    "gs.npy",
    "contours.npy",
    "offsets.npy",
    "times_ms.npy",
    "source_indices.npy",
    "used.npy",
    "converged.npy",
    "iterations.npy",
    "pixels.npy",
    "scalars.npy",
}


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Expected a package object")
    return cast(dict[str, object], value)


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _checksum(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Expected a declared SHA256 identity")
    return value


def _json(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        raw = stream.read(MAX_MANIFEST_BYTES + 1)
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ValueError("Package manifest exceeds its limit")
    return _object(json.loads(raw))


def _integer(value: object, lower: int, upper: int) -> int:
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError("Package count is outside its declared bounds")
    return value


def _array(
    root: Path,
    relative: str,
    checksum: object,
    shape: tuple[int, ...],
    dtype: str,
) -> NDArray[np.generic]:
    path = root / relative
    if (
        path.is_symlink()
        or not path.resolve().is_relative_to(root.resolve())
        or not path.is_file()
        or path.stat().st_size > MAX_FILE_BYTES
        or _digest(path) != _checksum(checksum)
    ):
        raise ValueError("Package numerical file differs")
    with path.open("rb") as stream:
        version = npformat.read_magic(stream)
        if version == (1, 0):
            actual_shape, _fortran, actual_dtype = npformat.read_array_header_1_0(stream)
        elif version == (2, 0):
            actual_shape, _fortran, actual_dtype = npformat.read_array_header_2_0(stream)
        else:
            raise ValueError("Unsupported package NPY version")
        if (
            actual_shape != shape
            or actual_dtype != np.dtype(dtype)
            or path.stat().st_size != stream.tell() + math.prod(shape) * actual_dtype.itemsize
        ):
            raise ValueError("Package array dimensions, dtype or byte length differ")
    return cast(NDArray[np.generic], np.load(path, mmap_mode="r", allow_pickle=False))


def original_cohort() -> list[dict[str, object]]:
    """Return the checksum-bound original ordered 358-shot population.

    Returns
    -------
    list of dict
        Original identifiers, frame counts and reference support.

    Raises
    ------
    OSError, ValueError
        The published population is absent or has changed.
    """
    values = _json(ROOT / "experiments/d72_scored_map_physics/manifest_v1.json")["cohort"]
    raw = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(raw).hexdigest() != COHORT_SHA256:
        raise ValueError("Original scored-map population differs")
    return cast(list[dict[str, object]], values)


@dataclass(frozen=True)
class Grid:
    """Original metre coordinates and the scorer's envelope, without flux rescaling."""

    radius: F64
    height: F64
    envelope: Mask
    axis_sign: int


@dataclass(frozen=True)
class Case:
    """One declared source shot; arrays are loaded separately with bounded memory."""

    stem: str
    frames: int
    source_frames: int
    contour_points: int
    files: dict[str, object]
    identities: dict[str, object]
    arrays_sha256: dict[str, object]


@dataclass(frozen=True)
class Bundle:
    """Bound package metadata; this identity does not authenticate a measurement."""

    root: Path
    checksum: str
    scope: str
    grid: Grid
    cases: list[Case]


@dataclass(frozen=True)
class ShotData:
    """Validated scored fields, original contour geometry and convergence flags."""

    case: Case
    fields: tuple[F32, F32, F32]
    contours: F64
    offsets: I64
    times_ms: F64
    source_indices: I64
    used: Mask
    converged: Mask
    iterations: I64
    pixels: I64
    scalars: F64

    def contour(self, frame: int, map_index: int) -> F64 | None:
        """Return one original contour, or its explicitly absent geometry.

        Parameters
        ----------
        frame, map_index : int
            Validated frame index and reference/base/gs index, respectively.

        Returns
        -------
        ndarray or None
            Original points without resampling; absence is not an invented mask.
        """
        position = 3 * frame + map_index
        start, end = self.offsets[position : position + 2]
        return None if start == end else self.contours[start:end]


def _load(root: Path, manifest_sha256: str) -> Bundle:
    """Validate a replay manifest and its original fixed grid.

    Parameters
    ----------
    root : Path
        Directory with manifest.json and its declared NPY files.
    manifest_sha256 : str
        Independently supplied identity of that manifest.

    Returns
    -------
    Bundle
        Ordered cases, scope and checked grid. Shot fields are read by load_shot.

    Raises
    ------
    OSError, ValueError
        Identities, population, units, grid or metadata are inconsistent.
    """
    manifest = root / "manifest.json"
    if _digest(manifest) != _checksum(manifest_sha256):
        raise ValueError("Replay manifest identity differs")
    data = _json(manifest)
    if (
        data["schema"] != "scored-map-bundle.v1"
        or data["maps"] != list(MAPS)
        or data["length_unit"] != "m"
        or data["flux_convention"] != "released-psirz-unscaled"
        or type(data["axis_sign"]) is not int
        or data["axis_sign"] not in (-1, 1)
        or data["scope"] not in ("full358", "qualification", "illustration")
    ):
        raise ValueError("Replay conventions or scope differ")
    grid_files = _object(data["grid"])
    if set(grid_files) != {"R.npy", "Z.npy", "envelope.npy"}:
        raise ValueError("Replay grid file membership differs")
    radius = cast(F64, _array(root, "R.npy", grid_files["R.npy"], (65,), "<f8"))
    height = cast(F64, _array(root, "Z.npy", grid_files["Z.npy"], (65,), "<f8"))
    envelope = cast(Mask, _array(root, "envelope.npy", grid_files["envelope.npy"], (65, 65), "|b1"))
    for axis in (radius, height):
        steps = np.diff(axis)
        if not np.isfinite(axis).all() or np.any(steps <= 0) or np.ptp(steps) > 1e-4 * steps.mean():
            raise ValueError("Replay grid is not finite, increasing and uniform")
    if np.any(radius <= 0) or not envelope.any():
        raise ValueError("Replay radius or envelope is invalid")
    cohort = original_cohort()
    if data["scope"] != "full358":
        cohort = [cohort[i] for i in (0, 179, 357)]
    if not isinstance(data["shots"], list) or len(data["shots"]) != len(cohort):
        raise ValueError("Replay population is incomplete")
    cases: list[Case] = []
    for entry, source in zip(data["shots"], cohort, strict=True):
        row = _object(entry)
        frames = _integer(row["frames"], 1, 4096)
        if row["stem"] != source["stem"] or row["source_frames"] != source["frames"]:
            raise ValueError("Replay shot order or original support differs")
        if frames != (1 if data["scope"] == "illustration" else source["frames"]):
            raise ValueError("Replay scope and frame support differ")
        files = _object(row["files"])
        if set(files) != SHOT_FILES:
            raise ValueError("Replay shot file membership differs")
        identities = _object(row["identities"])
        if set(identities) != {"raw_parquet_sha256", "d70_base_array_sha256", "d72_gs_array_sha256"}:
            raise ValueError("Replay source identities differ")
        arrays_sha256 = _object(row["arrays_sha256"])
        if set(arrays_sha256) != set(MAPS):
            raise ValueError("Replay field identity membership differs")
        if data["scope"] != "illustration" and (
            arrays_sha256["base"] != identities["d70_base_array_sha256"]
            or arrays_sha256["gs"] != identities["d72_gs_array_sha256"]
        ):
            raise ValueError("Whole-shot and original scored-field identities differ")
        for checksum in [*files.values(), *identities.values(), *arrays_sha256.values()]:
            _checksum(checksum)
        cases.append(
            Case(
                cast(str, row["stem"]),
                frames,
                cast(int, source["frames"]),
                _integer(row["contour_points"], 0, frames * 3 * 4096),
                files,
                identities,
                arrays_sha256,
            )
        )
    return Bundle(
        root,
        manifest_sha256,
        data["scope"],
        Grid(radius, height, envelope, data["axis_sign"]),
        cases,
    )


def load(root: Path, manifest_sha256: str) -> Bundle:
    """Validate a pinned manifest, original population, conventions and grid.

    Parameters
    ----------
    root : Path
        Scored-field package directory, including its manifest.json.
    manifest_sha256 : str
        Independently supplied SHA256 of that manifest.

    Returns
    -------
    Bundle
        Checked metadata and grid; shot arrays remain read-only memory maps.

    Raises
    ------
    OSError, ValueError
        Inputs are unreadable, malformed or inconsistent with the declared scope.
    """
    try:
        return _load(root, manifest_sha256)
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError, RecursionError):
        raise ValueError("Malformed scored-field package") from None


def load_shot(bundle: Bundle, case: Case) -> ShotData:
    """Map one shot's checked fields and original geometry without loading its peers.

    Parameters
    ----------
    bundle : Bundle
        Validated manifest and fixed grid.
    case : Case
        One member of that manifest.

    Returns
    -------
    ShotData
        Read-only original fields, contour points, timestamps and flags.

    Raises
    ------
    OSError, ValueError
        Numerical files, their original field identities or geometry are invalid.
    """
    if case not in bundle.cases:
        raise ValueError("Shot is not in this package")
    n = case.frames
    shapes: dict[str, tuple[tuple[int, ...], str]] = {name + ".npy": ((n, 65, 65), "<f4") for name in MAPS}
    shapes.update(
        {
            "contours.npy": ((case.contour_points, 2), "<f8"),
            "offsets.npy": ((n * 3 + 1,), "<i8"),
            "times_ms.npy": ((n,), "<f8"),
            "source_indices.npy": ((n,), "<i8"),
            "used.npy": ((n,), "|b1"),
            "converged.npy": ((n,), "|b1"),
            "iterations.npy": ((n,), "<i8"),
            "pixels.npy": ((n,), "<i8"),
            "scalars.npy": ((n, 2), "<f8"),
        }
    )
    arrays = {
        name: _array(bundle.root, case.stem + "/" + name, case.files[name], shape, dtype)
        for name, (shape, dtype) in shapes.items()
    }
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Replay numerical arrays contain nonfinite values")
    offsets = cast(I64, arrays["offsets.npy"])
    lengths = np.diff(offsets)
    if (
        offsets[0] != 0
        or offsets[-1] != case.contour_points
        or np.any(lengths < 0)
        or np.any((lengths != 0) & ((lengths < 8) | (lengths > 4096)))
        or np.any(lengths[::3] == 0)
    ):
        raise ValueError("Replay contour offsets or reference support differ")
    indices = cast(I64, arrays["source_indices.npy"])
    times = cast(F64, arrays["times_ms.npy"])
    expected = np.array([0]) if bundle.scope == "illustration" else np.arange(case.source_frames)
    if not np.array_equal(indices, expected) or np.any(np.diff(times) <= 0):
        raise ValueError("Replay frame indices or timestamps differ")
    iterations, pixels = cast(I64, arrays["iterations.npy"]), cast(I64, arrays["pixels.npy"])
    used, converged = cast(Mask, arrays["used.npy"]), cast(Mask, arrays["converged.npy"])
    if (
        np.any(iterations < 0)
        or np.any(iterations > 40)
        or np.any(pixels < 0)
        or np.any(pixels > 3969)
        or np.any(converged & ~used)
        or np.any((~used) & ((iterations != 0) | (pixels != 0)))
    ):
        raise ValueError("Original projection flags are inconsistent")
    fields = tuple(cast(F32, arrays[name + ".npy"]) for name in MAPS)
    for index, name in enumerate(MAPS):
        if hashlib.sha256(fields[index].tobytes()).hexdigest() != case.arrays_sha256[name]:
            raise ValueError("Scored field bytes differ from their declared identity")
    return ShotData(
        case,
        cast(tuple[F32, F32, F32], fields),
        cast(F64, arrays["contours.npy"]),
        offsets,
        times,
        indices,
        used,
        converged,
        iterations,
        pixels,
        cast(F64, arrays["scalars.npy"]),
    )
