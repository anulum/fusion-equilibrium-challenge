# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — bounded external-map inputs and fixed regions
"""Read explicit map conventions without inferring units or a plasma boundary."""

from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast

import numpy as np
from numpy.lib import format as npformat
from numpy.typing import NDArray

F64 = NDArray[np.float64]
Mask = NDArray[np.bool_]
MAX_ARCHIVE = 64 * 2**20
MAX_EXPANDED = 128 * 2**20
MAX_ELEMENTS = 1_048_576
UNIFORM_RTOL = 1e-5
LENGTH_FACTORS = {"m": 1.0, "cm": 0.01, "mm": 0.001}
FLUX_FACTORS = {"Wb/rad": 1.0, "Wb-total": 1 / (2 * math.pi)}
Header = tuple[tuple[int, ...], bool, np.dtype[np.generic]]


@dataclass(frozen=True)
class MapData:
    """Validated read-only SI arrays, declared orientation and own-map metadata."""

    radius: F64
    height: F64
    phi: F64
    plasma: Mask | None
    axis: float
    boundary: float | None
    axis_origin: str
    length_unit: str
    flux_unit: str
    axis_sign: int
    source_sha256: str


@dataclass(frozen=True)
class Region:
    """An independently identified grid mask, unchanged by candidate values."""

    mask: Mask
    policy: str
    identity: str
    source_sha256: str | None


def _arrays(path: Path, required: set[str], optional: set[str]) -> tuple[dict[str, NDArray[np.generic]], str]:
    with path.open("rb") as stream:
        raw = stream.read(MAX_ARCHIVE + 1)
    if len(raw) > MAX_ARCHIVE:
        raise ValueError("Map archive exceeds the byte limit")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = archive.infolist()
            names = [item.filename for item in members]
            keys = {name[:-4] for name in names}
            if (
                len(names) != len(set(names))
                or any(not name.endswith(".npy") for name in names)
                or not required <= keys <= required | optional
                or sum(item.file_size for item in members) > MAX_EXPANDED
            ):
                raise ValueError("Map archive members differ from the input schema")
            for item in members:
                with archive.open(item) as stream:
                    version = npformat.read_magic(stream)
                    readers = {
                        (1, 0): npformat.read_array_header_1_0,
                        (2, 0): npformat.read_array_header_2_0,
                    }
                    if version not in readers:
                        raise ValueError("Unsupported numerical array format")
                    reader = cast(Callable[[BinaryIO], Header], readers[version])
                    shape, _fortran, dtype = reader(cast(BinaryIO, stream))
                mask = item.filename[:-4] in {"plasma_mask", "mask"}
                if (
                    len(shape) > 2
                    or any(n < 0 for n in shape)
                    or math.prod(shape) > MAX_ELEMENTS
                    or dtype.kind not in ("b" if mask else "fiu")
                    or dtype.itemsize > 8
                ):
                    raise ValueError("Unsupported numerical shape or dtype")
        with np.load(io.BytesIO(raw), allow_pickle=False) as stored:
            values = {key: cast(NDArray[np.generic], stored[key]) for key in keys}
    except (zipfile.BadZipFile, EOFError, UnicodeError, NotImplementedError, RuntimeError):
        raise ValueError("Malformed numerical archive") from None
    return values, hashlib.sha256(raw).hexdigest()


def _coordinates(values: dict[str, NDArray[np.generic]], factor: float) -> tuple[F64, F64]:
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        radius, height = (values[key].astype(np.float64) * factor for key in ("R", "Z"))
        for grid in (radius, height):
            if grid.ndim != 1 or len(grid) < 3 or not np.isfinite(grid).all():
                raise ValueError("Expected finite one-dimensional coordinates")
            difference = np.diff(grid)
            if np.any(difference <= 0) or not np.allclose(
                difference, difference.mean(), rtol=UNIFORM_RTOL, atol=0
            ):
                raise ValueError("Coordinates must increase with uniform spacing")
        if np.any(radius <= 0) or len(radius) * len(height) > MAX_ELEMENTS:
            raise ValueError("Invalid cylindrical grid extent")
    return radius, height


def _scalar(values: dict[str, NDArray[np.generic]], key: str, factor: float) -> float | None:
    if key not in values:
        return None
    value = values[key]
    if value.shape != () or not np.isfinite(value).all():
        raise ValueError("Expected a finite scalar flux value")
    result = float(value) * factor
    return result


def load_map(path: Path, length_unit: str, flux_unit: str, axis_sign: int) -> MapData:
    """Load one finite axisymmetric map with explicit coordinate and flux units.

    Parameters
    ----------
    path : Path
        NPZ with R, Z, psi and optional bool plasma_mask and scalar axis_flux,
        boundary_flux. Scalar fluxes use the same input convention as psi.
    length_unit : str
        m, cm or mm; both coordinate arrays share the declaration.
    flux_unit : str
        Wb/rad for flux per radian, or Wb-total for full poloidal flux.
    axis_sign : int
        +1 when the axis is a maximum of input psi, otherwise -1.

    Returns
    -------
    MapData
        Read-only metres and axis-oriented flux per radian, with input identity.

    Raises
    ------
    ValueError
        Units, format, shape, dtype, coordinates or finite values are invalid.
    OSError
        The input cannot be read.
    """
    if length_unit not in LENGTH_FACTORS or flux_unit not in FLUX_FACTORS:
        raise ValueError("Explicit supported units are required")
    if type(axis_sign) is not int or axis_sign not in (-1, 1):
        raise ValueError("Axis orientation must be +1 or -1")
    values, checksum = _arrays(path, {"R", "Z", "psi"}, {"plasma_mask", "axis_flux", "boundary_flux"})
    radius, height = _coordinates(values, LENGTH_FACTORS[length_unit])
    factor = FLUX_FACTORS[flux_unit] * axis_sign
    phi = values["psi"].astype(np.float64) * factor
    if phi.shape != (len(height), len(radius)) or not np.isfinite(phi).all():
        raise ValueError("Expected a finite flux map on the declared grid")
    plasma = values["plasma_mask"].astype(np.bool_) if "plasma_mask" in values else None
    if plasma is not None and plasma.shape != phi.shape:
        raise ValueError("Plasma mask shape differs")
    supplied_axis = _scalar(values, "axis_flux", factor)
    boundary = _scalar(values, "boundary_flux", factor)
    selected = phi[plasma] if plasma is not None and plasma.any() else phi
    axis = float(selected.max()) if supplied_axis is None else supplied_axis
    for array in (radius, height, phi, plasma):
        if array is not None:
            array.setflags(write=False)
    return MapData(
        radius,
        height,
        phi,
        plasma,
        axis,
        boundary,
        "grid-extremum" if supplied_axis is None else "supplied",
        length_unit,
        flux_unit,
        axis_sign,
        checksum,
    )


def load_region(path: Path | None, frame: MapData) -> Region:
    """Bind a fixed comparison mask to an exact SI grid and versioned policy.

    Parameters
    ----------
    path : Path or None
        NPZ with R, Z, bool mask in the map's length unit. None selects all grid
        nodes; unavailable central stencils remain explicitly excluded later.
    frame : MapData
        Validated candidate grid; candidate flux never determines this mask.

    Returns
    -------
    Region
        Read-only mask, policy and hash including SI coordinates and shape.

    Raises
    ------
    ValueError
        The region schema, grid or mask does not match the candidate.
    OSError
        The optional region cannot be read.
    """
    if path is None:
        mask = np.ones(frame.phi.shape, dtype=np.bool_)
        policy, checksum = "grid-interior.v1", None
    else:
        values, checksum = _arrays(path, {"R", "Z", "mask"}, set())
        radius, height = _coordinates(values, LENGTH_FACTORS[frame.length_unit])
        mask = values["mask"].astype(np.bool_)
        if (
            not np.array_equal(radius, frame.radius)
            or not np.array_equal(height, frame.height)
            or mask.shape != frame.phi.shape
        ):
            raise ValueError("Fixed region and map grids differ")
        policy = "fixed-input-region.v1"
    payload = (
        policy.encode()
        + b"\0"
        + json.dumps(list(mask.shape)).encode()
        + b"\0"
        + frame.radius.astype("<f8", copy=False).tobytes()
        + frame.height.astype("<f8", copy=False).tobytes()
        + mask.tobytes()
    )
    mask.setflags(write=False)
    return Region(mask, policy, hashlib.sha256(payload).hexdigest(), checksum)
