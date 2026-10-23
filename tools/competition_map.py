# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — competition archive frame selection
"""Extract a declared shot/frame into the existing single-map audit contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Never, cast

import numpy as np
from numpy.lib import format as npformat
from numpy.typing import NDArray

from tools.audit_map_input import FLUX_FACTORS, LENGTH_FACTORS, load_arrays, load_map

type FloatMap = NDArray[np.float32] | NDArray[np.float64]
REFUSAL = "Competition map selection refused: check archive, selection, grid and declarations."


class Parser(argparse.ArgumentParser):
    """Keep malformed selection arguments on a fixed refusal path."""

    def error(self, message: str) -> Never:
        """Refuse invalid syntax without echoing interpreter-generated details."""
        raise ValueError("Invalid selection arguments")


def _frame_index(
    count: int,
    frame: int | None,
    time_ms: float | None,
    times: Path | None,
) -> tuple[int, float | None, str | None]:
    if (frame is None) == (time_ms is None) or (time_ms is not None and not np.isfinite(time_ms)):
        raise ValueError("Choose one finite frame or time")
    axis, checksum = None, None
    if times is not None:
        values, checksum = load_arrays(times, {"times_ms"}, set())
        axis = values["times_ms"].astype(np.float64)
        if axis.shape != (count,) or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
            raise ValueError("Declared times must align and strictly increase")
    if frame is None:
        if axis is None:
            raise ValueError("Time selection requires declared timestamps")
        matches = np.flatnonzero(axis == time_ms)
        if len(matches) != 1:
            raise ValueError("The requested time is not an exact unique timestamp")
        frame = int(matches[0])
    if not 0 <= frame < count:
        raise ValueError("Frame index is outside the shot")
    return frame, None if axis is None else float(axis[frame]), checksum


def extract_frame(
    archive: Path,
    shot: int,
    frame: int | None = None,
    time_ms: float | None = None,
    times: Path | None = None,
    expected_sha256: str | None = None,
) -> tuple[FloatMap, dict[str, object]]:
    """Select one numerical frame without allocating the machine or shot array.

    Parameters
    ----------
    archive : Path
        Competition NPZ containing shot_NNNN_psirz arrays in (time, Z, R) order.
    shot : int
        Nonnegative archive index; this does not infer a discharge identifier.
    frame : int or None
        Zero-based frame, exclusive with time_ms.
    time_ms : float or None
        Exact timestamp from a separately declared axis; no nearest-time fallback.
    times : Path or None
        Bounded NPZ with the sole times_ms vector for this shot.
    expected_sha256 : str or None
        Optional caller-pinned full archive digest.

    Returns
    -------
    psi, selection : ndarray, dict of str to object
        Original finite float32/64 frame and input/selection identities. Other
        arrays and frames are not validated. C/F storage never transposes Z/R.

    Raises
    ------
    OSError, ValueError
        The input, selected header, frame, time or expected identity is invalid.
    """
    if shot < 0:
        raise ValueError("Archive shot index must be nonnegative")
    key = f"shot_{shot:04d}_psirz"
    with archive.open("rb") as source:
        checksum = hashlib.file_digest(source, "sha256").hexdigest()
        size = os.fstat(source.fileno()).st_size
        if expected_sha256 is not None and checksum != expected_sha256:
            raise ValueError("Archive identity differs from the caller's pin")
        source.seek(0)
        with zipfile.ZipFile(source) as stored:
            members = [entry for entry in stored.infolist() if entry.filename == key + ".npy"]
            if len(members) != 1:
                raise ValueError("Selected shot map must occur exactly once")
            with stored.open(members[0]) as stream:
                version = npformat.read_magic(stream)
                if version == (1, 0):
                    shape, fortran, dtype = npformat.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, fortran, dtype = npformat.read_array_header_2_0(stream)
                else:
                    raise ValueError("Unsupported selected numerical array version")
                offset = stream.tell()
                if (
                    len(shape) != 3
                    or shape[0] < 1
                    or shape[1:] != (65, 65)
                    or dtype.kind != "f"
                    or dtype.itemsize not in (4, 8)
                ):
                    raise ValueError("Selected map requires finite real32/64 (T,65,65) fields")
                if members[0].file_size != offset + shape[0] * 65 * 65 * dtype.itemsize:
                    raise ValueError("Selected member length differs from its header")
                index, timestamp, times_checksum = _frame_index(shape[0], frame, time_ms, times)
                if fortran:
                    psi = np.empty((65, 65), dtype=dtype, order="F")
                    pixels = psi.ravel(order="F")
                    for position in range(65 * 65):
                        stream.seek(offset + (position * shape[0] + index) * dtype.itemsize)
                        pixels[position] = np.frombuffer(stream.read(dtype.itemsize), dtype=dtype)[0]
                else:
                    stream.seek(offset + index * 65 * 65 * dtype.itemsize)
                    psi = (
                        np.frombuffer(stream.read(65 * 65 * dtype.itemsize), dtype=dtype)
                        .reshape(65, 65)
                        .copy()
                    )
    if not np.isfinite(psi).all():
        raise ValueError("Selected frame contains nonfinite values")
    return cast(FloatMap, psi), {
        "archive_sha256": checksum,
        "archive_size_bytes": size,
        "array_key": key,
        "shot_index": shot,
        "frame_index": index,
        "declared_shot_frames": shape[0],
        "selected_time_ms": timestamp,
        "times_sha256": times_checksum,
        "source_dtype": dtype.str,
        "source_storage_order": "F" if fortran else "C",
        "frame_sha256_c_order": hashlib.sha256(np.ascontiguousarray(psi).tobytes()).hexdigest(),
        "unselected_arrays_validated": False,
    }


def export_selection(
    psi: FloatMap,
    selection: dict[str, object],
    grid: Path,
    output: Path,
    length_unit: str,
    flux_unit: str,
    axis_sign: int,
) -> dict[str, object]:
    """Validate and export a real selected snapshot with explicit map declarations.

    Parameters
    ----------
    psi, selection : ndarray, dict of str to object
        Frame and receipt from extract_frame.
    grid : Path
        Bounded NPZ with R/Z and optional map-specific plasma_mask, axis_flux and
        boundary_flux, supplied explicitly for the selected frame.
    output : Path
        New directory for map.npz and selection.json; never overwrite an output.
    length_unit, flux_unit : str
        Existing audit unit declarations; stored values retain their convention.
    axis_sign : int
        Explicit +1 or -1 axis orientation.

    Returns
    -------
    dict of str to object
        Selection, grid and exact output identities. This is not a scored result.

    Raises
    ------
    OSError, ValueError
        Geometry, selected values, declarations or output cannot be accepted.
    """
    values, grid_checksum = load_arrays(grid, {"R", "Z"}, {"plasma_mask", "axis_flux", "boundary_flux"})
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".competition-map-", dir=output.parent) as temporary:
        path = Path(temporary) / "map.npz"
        np.savez(path, allow_pickle=False, **{name: values[name] for name in sorted(values)}, psi=psi)
        validated = load_map(path, length_unit, flux_unit, axis_sign)
        report = {
            "schema": "competition-map-selection.v1",
            **selection,
            "grid_sha256": grid_checksum,
            "map_sha256": validated.source_sha256,
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "length_unit": length_unit,
            "flux_unit": flux_unit,
            "axis_sign": axis_sign,
            "competition_score_computed": False,
        }
        raw = path.read_bytes()
    output.mkdir(exist_ok=False)
    with (output / "map.npz").open("xb") as stream:
        stream.write(raw)
    with (output / "selection.json").open("x") as stream:
        stream.write(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return report


def main() -> int:
    """Export one frame or issue a fixed refusal and exit two.

    Returns
    -------
    int
        Zero after a valid selection, two for a refused input or destination.
    """
    parser = Parser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--shot-index", type=int, required=True)
    choose = parser.add_mutually_exclusive_group(required=True)
    choose.add_argument("--frame", type=int)
    choose.add_argument("--time-ms", type=float)
    parser.add_argument("--times", type=Path)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--length-unit", choices=LENGTH_FACTORS, required=True)
    parser.add_argument("--flux-unit", choices=FLUX_FACTORS, required=True)
    parser.add_argument("--axis-sign", type=int, choices=(-1, 1), required=True)
    parser.add_argument("--archive-sha256")
    try:
        args = parser.parse_args()
        psi, selection = extract_frame(
            args.archive, args.shot_index, args.frame, args.time_ms, args.times, args.archive_sha256
        )
        report = export_selection(
            psi, selection, args.grid, args.out, args.length_unit, args.flux_unit, args.axis_sign
        )
    except Exception:
        print(json.dumps({"status": "refused", "message": REFUSAL}))
        return 2
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
