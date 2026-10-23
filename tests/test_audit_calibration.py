# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — calibration and illustration counterexamples
"""Check known stencil floors, controlled deformations and real illustration lineage."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from numpy.lib import format as npformat

from tools.audit_calibration import DATA, FRAMES, analytic, illustrations, main
from tools.official_ridge import checksum


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    # These reference copies test known zero field error; they are never a research baseline.
    with np.load(FRAMES, allow_pickle=False) as old:
        arrays = {key + "_psi": old[key + "_reference_psi"] for key in ("p50", "p90", "p99", "max")}
        examples = {
            key: {"stem": str(old[key + "_stem"]), "time_ms": float(old[key + "_time_ms"])}
            for key in ("p50", "p90", "p99", "max")
        }
    base = tmp_path / "known_reference_copy.npz"
    np.savez_compressed(base, **arrays)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "examples": examples,
                "training_shots": ["d3d_shot_0000000001"],
                "excluded_shots": [item["stem"] for item in examples.values()],
                "baseline_frames_sha256": checksum(base),
                "reference_frames_sha256": checksum(FRAMES),
            }
        )
    )
    return manifest, base, FRAMES


def _part(record: dict[str, object], key: str) -> dict[str, object]:
    return cast(dict[str, object], record[key])


def test_three_grids_and_known_unavailable_controls() -> None:
    """The fitted GS residual is tiny while the known derivative bias decreases as h²."""
    cases = {str(row["case"]): row for row in analytic()}
    assert len(cases) == 8
    previous = None
    for size in (33, 65, 129):
        row = cases["quartic_" + str(size)]
        diagnostics = _part(row, "diagnostics")
        field = _part(diagnostics, "field_on_comparison_region")
        assert field["source_rms_T"] == pytest.approx(row["known_discrete_source_rms_T"], rel=2e-12)
        assert cast(float, _part(diagnostics, "polynomial_gs")["ratio"]) < 1e-10
        assert _part(_part(diagnostics, "surface_gs"), "support")["full_support"] is True
        error = cast(float, row["known_pointwise_source_error_T"])
        assert error > 0 and cast(float, row["measured_source_rms_relative_difference"]) > 0
        if previous is not None:
            assert previous / error == pytest.approx(4, rel=1e-12)
        previous = error
    smaller = _part(cases["deformation_0.1"], "diagnostics")
    larger = _part(cases["deformation_0.4"], "diagnostics")
    assert (
        0
        < cast(float, _part(smaller, "reference_comparison")["bp_relative_l2"])
        < cast(float, _part(larger, "reference_comparison")["bp_relative_l2"])
    )
    assert cast(float, _part(larger, "polynomial_gs")["ratio"]) > 1e-3
    assert (
        _part(_part(cases["missing_mask"], "diagnostics"), "polynomial_gs")["reason"] == "missing_plasma_mask"
    )
    assert _part(_part(cases["zero_source"], "diagnostics"), "polynomial_gs")["ratio"] is None
    opened = _part(_part(cases["open_surfaces"], "diagnostics"), "surface_gs")
    assert _part(opened, "support")["reason"] == "no_usable_surface"
    json.dumps(cases, allow_nan=False)


def test_common_regions_and_known_reference_copies(tmp_path: Path) -> None:
    """All three maps share a stable per-frame region, and reference copies give zero error."""
    manifest, base, frames = _fixture(tmp_path)
    result = illustrations(manifest, base, frames)
    assert len(result) == 12
    assert len({str(row["region_identity"]) for row in result}) == 4
    expected_nodes = {"p50": 1276, "p90": 1268, "p99": 1390, "max": 921}
    for key in ("p50", "p90", "p99", "max"):
        chosen = [row for row in result if row["case"] == key]
        assert {row["map"] for row in chosen} == {"reference", "entry_after_finisher", "official_ridge"}
        assert len({str(row["region_identity"]) for row in chosen}) == 1
        assert all(row["reference_assisted_domain"] is True for row in chosen)
        assert all(row["requested_nodes"] == expected_nodes[key] for row in chosen)
        for row in chosen:
            if row["map"] != "entry_after_finisher":
                assert _part(_part(row, "diagnostics"), "reference_comparison")["bp_relative_l2"] == 0
    assert illustrations(manifest, base, frames) == result


@pytest.mark.parametrize(
    "fault", ["baseline_hash", "reference_hash", "missing", "stem", "time", "shape", "nan"]
)
def test_changed_illustrations_refused(tmp_path: Path, fault: str) -> None:
    """Checksum, all-example, identity and finite-grid guards are independently effective."""
    manifest, base, frames = _fixture(tmp_path)
    plan = json.loads(manifest.read_text())
    if fault == "baseline_hash":
        plan["baseline_frames_sha256"] = "0" * 64
    elif fault == "reference_hash":
        plan["reference_frames_sha256"] = "0" * 64
    elif fault == "missing":
        del plan["examples"]["max"]
    elif fault in ("stem", "time"):
        plan["examples"]["p50"]["stem" if fault == "stem" else "time_ms"] = "wrong"
    else:
        with np.load(base, allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in stored.files}
        if fault == "shape":
            arrays["p50_psi"] = np.zeros((3, 3))
        else:
            arrays["p50_psi"][0, 0] = np.nan
        np.savez_compressed(base, **arrays)
        plan["baseline_frames_sha256"] = checksum(base)
    manifest.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        illustrations(manifest, base, frames)


@pytest.mark.parametrize(
    "fault", ["size", "expanded", "duplicate", "version", "dtype", "elements", "empty", "v2"]
)
def test_archive_bounds_before_array_allocation(tmp_path: Path, fault: str) -> None:
    """Exercise actual byte snapshots and NPY headers, including supported version two."""
    manifest, base, frames = _fixture(tmp_path)
    if fault == "size":
        base.write_bytes(b"x" * (2**21 + 1))
    else:
        header: dict[str, object] = {"descr": "<f8", "fortran_order": False, "shape": (65, 65)}
        if fault == "dtype":
            header["descr"] = "|O"
        elif fault == "elements":
            header["shape"] = (2**21,)
        buffer = io.BytesIO()
        writer = npformat.write_array_header_2_0 if fault == "v2" else npformat.write_array_header_1_0
        writer(buffer, header)
        data = buffer.getvalue() + np.zeros((65, 65)).tobytes()
        if fault == "version":
            data = b"\x93NUMPY\x03\x00" + data[8:]
        with zipfile.ZipFile(base, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            if fault == "expanded":
                archive.writestr("large.npy", b"x" * (2**22 + 1))
            elif fault != "empty":
                for key in ("p50", "p90", "p99", "max"):
                    archive.writestr(key + "_psi.npy", data)
                if fault == "duplicate":
                    with pytest.warns(UserWarning, match="Duplicate"):
                        archive.writestr("p50_psi.npy", data)
    plan = json.loads(manifest.read_text())
    plan["baseline_frames_sha256"] = checksum(base)
    manifest.write_text(json.dumps(plan))
    if fault == "v2":
        assert len(illustrations(manifest, base, frames)) == 12
    else:
        with pytest.raises(ValueError):
            illustrations(manifest, base, frames)


def test_actual_cli_and_refusal(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The real command emits strict deterministic JSON and fixed path-free failures."""
    manifest, base, frames = _fixture(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--parallel-mode",
            "-m",
            "tools.audit_calibration",
            "--manifest",
            str(manifest),
            "--baseline",
            str(base),
            "--frames",
            str(frames),
        ],
        text=True,
        capture_output=True,
        timeout=90,
        check=True,
    )
    assert result.stderr == ""
    report = json.loads(result.stdout)
    assert len(report["analytic"]) == 8 and len(report["illustrations"]) == 12
    assert report["physical_force_balance_certified"] is False
    assert report["competition_score_computed"] is False
    assert main(["--manifest", str(tmp_path / "absent")]) == 2
    refusal = json.loads(capsys.readouterr().out)
    assert refusal["status"] == "refused" and str(tmp_path) not in refusal["message"]
    assert main(["--unknown"]) == 2


def test_sign_and_disjointness_controls(tmp_path: Path) -> None:
    """Common sign reversal preserves the diagnostics; train/evaluation overlap is refused."""
    manifest, base, frames = _fixture(tmp_path)
    original = illustrations(manifest, base, frames)
    with np.load(frames, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    for key in arrays:
        if key.endswith("_psi"):
            arrays[key] = -arrays[key]
    flipped = tmp_path / "opposite.npz"
    np.savez_compressed(flipped, **arrays)
    with np.load(base, allow_pickle=False) as stored:
        predictions = {key: -stored[key] for key in stored.files}
    np.savez_compressed(base, **predictions)
    plan = json.loads(manifest.read_text())
    plan["baseline_frames_sha256"], plan["reference_frames_sha256"] = checksum(base), checksum(flipped)
    manifest.write_text(json.dumps(plan))
    reverse = illustrations(manifest, base, flipped)
    for before, after in zip(original, reverse, strict=True):
        assert before["diagnostics"] == after["diagnostics"]
        assert before["axis_sign_from_reference"] == -cast(int, after["axis_sign_from_reference"])
    plan["training_shots"] = plan["excluded_shots"][:1]
    manifest.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="disjoint"):
        illustrations(manifest, base, flipped)


def test_actual_published_baseline_and_report() -> None:
    """Remeasure the real independently authored method's stored prediction maps."""
    manifest = DATA / "baseline_manifest_v1.json"
    plan = json.loads(manifest.read_text())
    assert len(plan["training_shots"]) == 512 and len(plan["excluded_shots"]) == 358
    assert not set(plan["training_shots"]) & set(plan["excluded_shots"])
    assert sum(row["finite_frames"] for row in plan["training_manifest"]) == 112374
    assert all(row["dropped_frames"] == 0 for row in plan["training_manifest"])
    assert plan["fit"]["selected_alpha"] == pytest.approx(2.154434690031882)
    expected = json.loads((DATA / "report_v1.json").read_text())["illustrations"]
    actual = illustrations(manifest, DATA / "baseline_frames_v1.npz", FRAMES)
    for observed, recorded in zip(actual, expected, strict=True):
        assert observed["case"] == recorded["case"] and observed["map"] == recorded["map"]
        assert observed["region_identity"] == recorded["region_identity"]
        left, right = _part(observed, "diagnostics"), recorded["diagnostics"]
        assert _part(left, "reference_comparison")["bp_relative_l2"] == pytest.approx(
            right["reference_comparison"]["bp_relative_l2"], rel=1e-8, abs=1e-12
        )
        assert _part(left, "polynomial_gs")["ratio"] == pytest.approx(
            right["polynomial_gs"]["ratio"], rel=1e-8, abs=1e-12
        )
        surface = _part(_part(left, "surface_gs"), "support")
        assert surface["ratio"] == pytest.approx(right["surface_gs"]["support"]["ratio"], rel=1e-8, abs=1e-12)
