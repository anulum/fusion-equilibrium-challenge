# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — competition-to-audit process contracts
"""Select real submitted fields and exercise declared times, layouts and refusals."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import warnings
import zipfile
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray

from tools.ci_check_shot import REFERENCE
from tools.competition_map import REFUSAL, extract_frame

ROOT = Path(__file__).resolve().parents[1]


def _cli(arguments: list[str], module: str = "tools.competition_map") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "coverage", "run", "--branch", "--parallel-mode", "-m", module, *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=90,
    )


def _arguments(archive: Path, grid: Path, output: Path) -> list[str]:
    return [
        "--archive",
        str(archive),
        "--shot-index",
        "0",
        "--frame",
        "1",
        "--grid",
        str(grid),
        "--out",
        str(output),
        "--length-unit",
        "m",
        "--flux-unit",
        "Wb/rad",
        "--axis-sign",
        "1",
    ]


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path, NDArray[np.float64]]:
    """Write actual challenge-format arrays with distinguishable Z/R and time axes."""
    radius, height = np.linspace(1, 2, 65), np.linspace(-1, 1, 65)
    rr, zz = np.meshgrid(radius, height)
    maps = np.stack([100 * rr + zz**2 + t for t in range(3)])
    archive, grid = tmp_path / "machine.npz", tmp_path / "grid.npz"
    np.savez_compressed(archive, shot_0000_psirz=maps)
    np.savez(grid, R=radius, Z=height)
    return archive, grid, maps


def test_actual_submitted_archive_to_audit(tmp_path: Path) -> None:
    """Compare the audit of a selected original 986068 frame with its direct map."""
    grid, output = tmp_path / "grid.npz", tmp_path / "selected"
    with np.load(ROOT / "tools/reference/d71_figure_frames.npz", allow_pickle=False) as geometry:
        np.savez(grid, R=geometry["R"], Z=geometry["Z"])
    args = _arguments(REFERENCE, grid, output)
    result = _cli(args + ["--archive-sha256", hashlib.sha256(REFERENCE.read_bytes()).hexdigest()])
    assert result.returncode == 0 and result.stderr == ""
    receipt = json.loads(result.stdout)
    assert receipt == json.loads((output / "selection.json").read_text())
    assert receipt["declared_shot_frames"] == 240 and receipt["selected_time_ms"] is None
    assert receipt["times_sha256"] is None and receipt["unselected_arrays_validated"] is False
    with (
        np.load(REFERENCE, allow_pickle=False) as original,
        np.load(output / "map.npz", allow_pickle=False) as selected,
    ):
        psi = original["shot_0000_psirz"][1]
        np.testing.assert_array_equal(selected["psi"], psi)
        direct = tmp_path / "direct.npz"
        np.savez(direct, R=selected["R"], Z=selected["Z"], psi=psi)
    reports = []
    for path in (output / "map.npz", direct):
        audit = _cli(
            ["--map", str(path), "--length-unit", "m", "--flux-unit", "Wb/rad", "--axis-sign", "1"],
            "tools.equilibrium_audit",
        )
        assert audit.returncode == 0 and audit.stderr == ""
        reports.append(json.loads(audit.stdout))
    assert reports[0]["diagnostics"] == reports[1]["diagnostics"]
    assert reports[0]["input"]["snapshot_sha256"] == receipt["map_sha256"]
    assert reports[0]["diagnostics"]["polynomial_gs"]["reason"] == "missing_plasma_mask"
    assert receipt["frame_sha256_c_order"] == hashlib.sha256(psi.tobytes()).hexdigest()
    assert receipt["competition_score_computed"] is False


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("dtype", ["<f4", ">f4", "<f8"])
def test_actual_storage_layouts(
    tmp_path: Path,
    inputs: tuple[Path, Path, NDArray[np.float64]],
    order: Literal["C", "F"],
    dtype: str,
) -> None:
    """Preserve original numerical values and Z/R orientation in actual C/F members."""
    archive, _grid, maps = inputs
    values = np.array(maps, dtype=dtype, order=order)
    np.savez_compressed(archive, shot_0007_psirz=values)
    frame, receipt = extract_frame(archive, 7, frame=1)
    np.testing.assert_array_equal(frame, values[1])
    assert frame.dtype == values.dtype
    assert receipt["source_storage_order"] == order and receipt["shot_index"] == 7
    assert receipt["frame_sha256_c_order"] == hashlib.sha256(values[1].tobytes()).hexdigest()


def test_exact_time_and_frame_label(inputs: tuple[Path, Path, NDArray[np.float64]], tmp_path: Path) -> None:
    """Select an exact supplied time and bind the same axis to an explicit frame."""
    archive, grid, maps = inputs
    times = tmp_path / "times.npz"
    np.savez(times, times_ms=np.array([10.0, 30.0, 50.0]))
    frame, receipt = extract_frame(archive, 0, time_ms=30, times=times)
    np.testing.assert_array_equal(frame, maps[1])
    assert receipt["selected_time_ms"] == 30 and receipt["frame_index"] == 1
    args = _arguments(archive, grid, tmp_path / "selected") + ["--times", str(times)]
    args[args.index("--frame")] = "--time-ms"
    args[args.index("--time-ms") + 1] = "30"
    result = _cli(args)
    assert result.returncode == 0
    assert json.loads(result.stdout)["times_sha256"] == hashlib.sha256(times.read_bytes()).hexdigest()
    _, labelled = extract_frame(archive, 0, frame=2, times=times)
    assert labelled["selected_time_ms"] == 50


@pytest.mark.parametrize("version", [(1, 0), (2, 0)])
def test_actual_npy_headers(inputs: tuple[Path, Path, NDArray[np.float64]], version: tuple[int, int]) -> None:
    """Read both supported NPY header versions without loading another array."""
    archive, _grid, maps = inputs
    stream = io.BytesIO()
    header = {"descr": "<f8", "fortran_order": False, "shape": maps.shape}
    writer = (
        np.lib.format.write_array_header_1_0 if version == (1, 0) else np.lib.format.write_array_header_2_0
    )
    writer(stream, header)
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("shot_0000_psirz.npy", stream.getvalue() + maps.tobytes())
    frame, _receipt = extract_frame(archive, 0, frame=2)
    np.testing.assert_array_equal(frame, maps[2])


@pytest.mark.parametrize(
    "fault",
    [
        "negative_shot",
        "missing_shot",
        "negative_frame",
        "beyond_frame",
        "no_selection",
        "both_selections",
        "time_without_axis",
        "nan_query",
        "missing_time",
        "duplicate_times",
        "descending_times",
        "wrong_time_count",
        "nan_times",
        "times_shape",
        "missing_times_key",
        "nan_selected",
        "shape",
        "empty",
        "integer",
        "float16",
        "complex",
        "object",
        "version",
        "member_length",
        "duplicate_member",
        "wrong_pin",
        "missing_archive",
        "missing_grid",
        "grid_shape",
        "grid_spacing",
        "bad_unit",
        "bad_sign",
        "existing_output",
    ],
)
def test_real_selection_refusals(
    inputs: tuple[Path, Path, NDArray[np.float64]],
    tmp_path: Path,
    fault: str,
) -> None:
    """Reject real malformed archives, selections and declarations at the process boundary."""
    archive, grid, maps = inputs
    output = tmp_path / "selected"
    args = _arguments(archive, grid, output)
    if fault == "negative_shot":
        args[args.index("--shot-index") + 1] = "-1"
    elif fault == "missing_shot":
        args[args.index("--shot-index") + 1] = "9"
    elif fault in {"negative_frame", "beyond_frame"}:
        args[args.index("--frame") + 1] = "-1" if fault == "negative_frame" else "3"
    elif fault == "no_selection":
        position = args.index("--frame")
        del args[position : position + 2]
    elif fault == "both_selections":
        args += ["--time-ms", "30"]
    elif fault.startswith("time") or fault in {
        "nan_query",
        "missing_time",
        "duplicate_times",
        "descending_times",
        "wrong_time_count",
        "nan_times",
        "missing_times_key",
    }:
        args[args.index("--frame")] = "--time-ms"
        args[args.index("--time-ms") + 1] = (
            "nan" if fault == "nan_query" else "31" if fault == "missing_time" else "30"
        )
        if fault != "time_without_axis":
            times = tmp_path / "times.npz"
            values = np.array([10.0, 30.0, 50.0])
            if fault == "duplicate_times":
                values[0] = 30
            elif fault == "descending_times":
                values = values[::-1]
            elif fault == "wrong_time_count":
                values = values[:2]
            elif fault == "nan_times":
                values[0] = np.nan
            elif fault == "times_shape":
                values = values[None, :]
            np.savez(
                times, allow_pickle=False, **{"wrong" if fault == "missing_times_key" else "times_ms": values}
            )
            args += ["--times", str(times)]
    elif fault in {"nan_selected", "shape", "empty", "integer", "float16", "complex", "object"}:
        changed = maps.copy()
        if fault == "nan_selected":
            changed[1, 0, 0] = np.nan
        elif fault == "shape":
            changed = changed[:, :-1]
        elif fault == "empty":
            changed = changed[:0]
        else:
            changed = changed.astype(
                {"integer": int, "float16": np.float16, "complex": complex, "object": object}[fault]
            )
        np.savez(archive, shot_0000_psirz=changed)
    elif fault in {"version", "member_length", "duplicate_member"}:
        with zipfile.ZipFile(archive) as original:
            raw = original.read("shot_0000_psirz.npy")
        if fault == "version":
            raw = raw[:6] + b"\x03\x00" + raw[8:]
        elif fault == "member_length":
            raw = raw[:-1]
        with zipfile.ZipFile(archive, "w") as changed_zip:
            changed_zip.writestr("shot_0000_psirz.npy", raw)
            if fault == "duplicate_member":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    changed_zip.writestr("shot_0000_psirz.npy", raw)
    elif fault == "wrong_pin":
        args += ["--archive-sha256", "0" * 64]
    elif fault == "missing_archive":
        archive.unlink()
    elif fault == "missing_grid":
        grid.unlink()
    elif fault in {"grid_shape", "grid_spacing"}:
        radius = np.linspace(1, 2, 64 if fault == "grid_shape" else 65)
        if fault == "grid_spacing":
            radius[3] += 0.005
        np.savez(grid, R=radius, Z=np.linspace(-1, 1, 65))
    elif fault == "bad_unit":
        args[args.index("--length-unit") + 1] = "inch"
    elif fault == "bad_sign":
        args[args.index("--axis-sign") + 1] = "0"
    else:
        output.mkdir()
        (output / "keep.txt").write_text("owner content")
    result = _cli(args)
    assert result.returncode == 2 and result.stderr == ""
    assert json.loads(result.stdout) == {"status": "refused", "message": REFUSAL}
    if fault == "existing_output":
        assert (output / "keep.txt").read_text() == "owner content"
        assert len(list(output.iterdir())) == 1
    else:
        assert not output.exists()


def test_public_selection_api_requires_one_selector(inputs: tuple[Path, Path, NDArray[np.float64]]) -> None:
    """The public reader refuses ambiguous declarations as well as the CLI parser."""
    archive, _grid, _maps = inputs
    with pytest.raises(ValueError, match="Choose one"):
        extract_frame(archive, 0)
    with pytest.raises(ValueError, match="Choose one"):
        extract_frame(archive, 0, frame=1, time_ms=30)
