# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — external-map input contracts
"""Exercise actual numerical archives, unit conversion and fixed-region identity."""

from __future__ import annotations

import hashlib
import io
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from tools.audit_map_input import MAX_ARCHIVE, MAX_EXPANDED, load_map, load_region


def _values() -> dict[str, NDArray[np.float64]]:
    radius = np.linspace(1, 2, 17)
    height = np.linspace(-1, 1, 19)
    rr, zz = np.meshgrid(radius, height)
    return {"R": radius, "Z": height, "psi": -((rr - 1.5) ** 2) - zz**2}


def test_units_orientation_and_snapshot(tmp_path: Path) -> None:
    """Convert full flux and centimetres, binding exactly the bytes processed."""
    values = _values()
    path = tmp_path / "map.npz"
    np.savez(
        path,
        R=values["R"] * 100,
        Z=values["Z"] * 100,
        psi=-2 * np.pi * values["psi"],
        plasma_mask=values["psi"] > -0.3,
        axis_flux=0.0,
        boundary_flux=2 * np.pi * 0.3,
    )
    frame = load_map(path, "cm", "Wb-total", -1)
    assert np.allclose(frame.phi, values["psi"])
    assert np.allclose(frame.radius, values["R"])
    assert frame.axis == 0 and frame.boundary == pytest.approx(-0.3)
    assert frame.axis_origin == "supplied"
    assert frame.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert frame.plasma is not None
    for array in (frame.phi, frame.radius, frame.height, frame.plasma):
        assert not array.flags.writeable


def test_fixed_regions_and_estimated_axis(tmp_path: Path) -> None:
    """The same grid and mask keep their ID when a candidate's flux changes."""
    values = _values()
    path = tmp_path / "map.npz"
    np.savez(path, allow_pickle=False, **values)
    frame = load_map(path, "m", "Wb/rad", 1)
    region = load_region(None, frame)
    assert region.policy == "grid-interior.v1" and region.source_sha256 is None
    assert frame.axis_origin == "grid-extremum" and frame.axis == 0
    mask = values["psi"] > -0.3
    region_path = tmp_path / "region.npz"
    np.savez(region_path, R=values["R"], Z=values["Z"], mask=mask)
    fixed = load_region(region_path, frame)
    assert fixed.policy == "fixed-input-region.v1" and not fixed.mask.flags.writeable
    assert fixed.identity != region.identity
    for mask_kind in (mask, np.zeros_like(mask)):
        np.savez(path, allow_pickle=False, **(values | {"psi": values["psi"] + 7}), plasma_mask=mask_kind)
        other = load_map(path, "m", "Wb/rad", 1)
        assert other.axis == 7
        assert load_region(None, other).identity == region.identity
        assert load_region(region_path, other).identity == fixed.identity
    np.savez(region_path, R=values["R"], Z=values["Z"], mask=~mask)
    assert load_region(region_path, frame).identity != fixed.identity


@pytest.mark.parametrize(
    "length,flux,sign",
    [
        ("inch", "Wb/rad", 1),
        ("m", "unknown", 1),
        ("m", "Wb/rad", 0),
        ("m", "Wb/rad", True),
        ("m", "Wb/rad", 1.0),
    ],
)
def test_explicit_declarations(tmp_path: Path, length: str, flux: str, sign: int) -> None:
    """No file is opened until unit and axis declarations are valid."""
    with pytest.raises(ValueError):
        load_map(tmp_path / "absent.npz", length, flux, sign)


@pytest.mark.parametrize(
    "fault",
    [
        "Rdim",
        "Zdim",
        "small",
        "nanR",
        "nanZ",
        "reverse",
        "spacing",
        "zeroR",
        "largegrid",
        "shape",
        "nanpsi",
        "maskshape",
        "scalarshape",
        "scalarnan",
        "extra",
        "missing",
        "complex",
        "object",
        "masktype",
        "rank3",
        "widefloat",
    ],
)
def test_invalid_actual_arrays(tmp_path: Path, fault: str) -> None:
    """Reject inconsistent maps and incompatible numerical dtypes before diagnostics."""
    values: dict[str, NDArray[np.generic]] = dict(_values())
    if fault == "Rdim":
        values["R"] = values["psi"]
    elif fault == "Zdim":
        values["Z"] = values["psi"]
    elif fault == "small":
        values["R"] = values["R"][:2]
    elif fault == "nanR":
        values["R"][0] = np.nan
    elif fault == "nanZ":
        values["Z"][0] = np.inf
    elif fault == "reverse":
        values["R"] = values["R"][::-1]
    elif fault == "spacing":
        values["R"][1] += 0.01
    elif fault == "zeroR":
        values["R"] = values["R"].astype(np.float64) - 1
    elif fault == "largegrid":
        values["R"] = np.linspace(1, 2, 1025)
        values["Z"] = np.linspace(-1, 1, 1025)
    elif fault == "shape":
        values["psi"] = values["psi"][:-1]
    elif fault == "nanpsi":
        values["psi"][0, 0] = np.nan
    elif fault == "maskshape":
        values["plasma_mask"] = np.zeros((3, 3), dtype=bool)
    elif fault == "scalarshape":
        values["axis_flux"] = np.array([0.0])
    elif fault == "scalarnan":
        values["boundary_flux"] = np.array(np.nan)
    elif fault == "extra":
        values["secret"] = np.array(1)
    elif fault == "missing":
        del values["R"]
    elif fault == "complex":
        values["psi"] = values["psi"].astype(complex)
    elif fault == "object":
        values["psi"] = np.array([object()], dtype=object)
    elif fault == "masktype":
        values["plasma_mask"] = np.ones(values["psi"].shape)
    elif fault == "rank3":
        values["psi"] = np.zeros((3, 3, 3))
    else:
        values["psi"] = values["psi"].astype(np.longdouble)
    path = tmp_path / "bad.npz"
    np.savez(path, allow_pickle=fault == "object", **values)
    with pytest.raises(ValueError):
        load_map(path, "m", "Wb/rad", 1)


@pytest.mark.parametrize("fault", ["coordinates", "shape"])
def test_region_registration(tmp_path: Path, fault: str) -> None:
    """Do not silently interpolate a region onto a different spatial grid."""
    values = _values()
    path = tmp_path / "map.npz"
    np.savez(path, allow_pickle=False, **values)
    frame = load_map(path, "m", "Wb/rad", 1)
    np.savez(
        tmp_path / "region.npz",
        R=values["R"] + (0.1 if fault == "coordinates" else 0),
        Z=values["Z"],
        mask=np.ones((3, 3) if fault == "shape" else frame.phi.shape, dtype=bool),
    )
    with pytest.raises(ValueError, match="grids differ"):
        load_region(tmp_path / "region.npz", frame)


@pytest.mark.parametrize(
    "fault", ["duplicate", "non_npy", "version", "element_limit", "negative_shape", "v2"]
)
def test_archive_headers(tmp_path: Path, fault: str) -> None:
    """Inspect real NPY headers before allocating declared arrays."""
    path = tmp_path / "header.npz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for key, value in _values().items():
            stream = io.BytesIO()
            if key == "psi" and fault in {"element_limit", "negative_shape", "version"}:
                header = {
                    "descr": "<f8",
                    "fortran_order": False,
                    "shape": (1048577,) if fault == "element_limit" else (-1,),
                }
                np.lib.format.write_array_header_1_0(stream, header)
                raw = stream.getvalue()
                if fault == "version":
                    raw = raw[:6] + b"\x03\x00" + raw[8:]
            elif fault == "v2":
                np.lib.format.write_array_header_2_0(
                    stream, {"descr": value.dtype.str, "fortran_order": False, "shape": value.shape}
                )
                raw = stream.getvalue() + value.tobytes()
            else:
                np.save(stream, value)
                raw = stream.getvalue()
            archive.writestr(key + (".txt" if fault == "non_npy" else ".npy"), raw)
            if key == "psi" and fault == "duplicate":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    archive.writestr(key + ".npy", raw)
    if fault == "v2":
        assert load_map(path, "m", "Wb/rad", 1).phi.shape == (19, 17)
    else:
        with pytest.raises(ValueError):
            load_map(path, "m", "Wb/rad", 1)


def test_bounded_actual_archives(tmp_path: Path) -> None:
    """Both a raw oversized file and a compressible ZIP bomb are bounded."""
    path = tmp_path / "large.npz"
    with path.open("wb") as stream:
        stream.truncate(MAX_ARCHIVE + 1)
    with pytest.raises(ValueError, match="byte limit"):
        load_map(path, "m", "Wb/rad", 1)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("R.npy", "Z.npy", "psi.npy"):
            with archive.open(name, "w") as stream:
                for _ in range(MAX_EXPANDED // 3 // 65536 + 1):
                    stream.write(bytes(65536))
    with pytest.raises(ValueError, match="members"):
        load_map(path, "m", "Wb/rad", 1)
