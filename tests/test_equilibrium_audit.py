# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — public map audit command tests
"""Invoke the actual command on analytic fields and a published DIII-D map."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools.equilibrium_audit import REFUSAL, main
from tools.make_preview import bilinear, surfaces

ROOT = Path(__file__).resolve().parents[1]


def _args(path: Path) -> list[str]:
    return ["--map", str(path), "--length-unit", "m", "--flux-unit", "Wb/rad", "--axis-sign", "1"]


def test_real_published_map_cli(tmp_path: Path) -> None:
    """Audit an actual released DIII-D illustration with no private model import."""
    with np.load(ROOT / "tools/reference/d71_figure_frames.npz", allow_pickle=False) as data:
        shown = surfaces(ROOT / "tools/reference/d71_figure_frames.npz")
        phi = data["p50_margin_psi"].astype(float)
        radius, height = data["R"], data["Z"]
        lcfs = data["p50_margin_lcfs"]
        boundary = float(bilinear(radius, height, phi, lcfs).mean())
        axis = float(phi[shown.axis_index])
        sign = 1 if axis > boundary else -1
        path = tmp_path / "published.npz"
        np.savez(
            path,
            R=radius,
            Z=height,
            psi=sign * phi,
            plasma_mask=shown.inside,
            axis_flux=sign * axis,
            boundary_flux=sign * boundary,
        )
    command = [
        sys.executable,
        "-m",
        "coverage",
        "run",
        "--branch",
        "--parallel-mode",
        "-m",
        "tools.equilibrium_audit",
        *_args(path),
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True, timeout=60)
    report = json.loads(result.stdout)
    assert result.stderr == ""
    assert report["schema"] == "equilibrium-audit.v1"
    assert report["input"]["axis_origin"] == "supplied"
    assert report["grid"]["shape"] == [65, 65]
    assert report["diagnostics"]["polynomial_gs"]["available"] is True
    assert report["diagnostics"]["surface_gs"]["support"]["full_support"] is True
    assert report["physical_force_balance_certified"] is False
    assert report["competition_score_computed"] is False


def test_units_gauge_sign_and_fixed_reference_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Full-flux millimetre inputs agree after separately declared sign and gauge."""
    radius, height = np.linspace(1.0, 2.0, 33), np.linspace(-0.6, 0.6, 35)
    rr, zz = np.meshgrid(radius, height)
    phi = -((rr**2 - 1.5**2) ** 2) / 8 - zz**2
    mask = phi > -0.15
    path, reference, region = (tmp_path / name for name in ("map.npz", "reference.npz", "region.npz"))
    np.savez(
        path,
        R=radius * 1000,
        Z=height * 1000,
        psi=(phi + 3) * 2 * np.pi,
        plasma_mask=mask,
        axis_flux=3 * 2 * np.pi,
        boundary_flux=2.85 * 2 * np.pi,
    )
    np.savez(
        reference,
        R=radius * 1000,
        Z=height * 1000,
        psi=-phi * 2 * np.pi,
        plasma_mask=mask,
        axis_flux=0.0,
        boundary_flux=0.15 * 2 * np.pi,
    )
    np.savez(region, R=radius * 1000, Z=height * 1000, mask=~mask)
    args = [
        "--map",
        str(path),
        "--length-unit",
        "mm",
        "--flux-unit",
        "Wb-total",
        "--axis-sign",
        "1",
        "--reference",
        str(reference),
        "--reference-axis-sign",
        "-1",
        "--comparison-region",
        str(region),
        "--measured-current-a",
        "1000000",
    ]
    assert main(args) == 0
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert output.err == ""
    assert report["reference"]["axis_sign"] == -1
    assert report["region"]["policy"] == "fixed-input-region.v1"
    assert report["diagnostics"]["reference_comparison"]["bp_relative_l2"] < 1e-12
    assert report["diagnostics"]["measured_current_comparison"]["available"] is True
    assert main(args) == 0 and capsys.readouterr().out == output.out
    assert main(_args(path) + ["--reference", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["reference"]["axis_sign"] == 1


@pytest.mark.parametrize(
    "case",
    [
        "missing_file",
        "broken_zip",
        "missing_flag",
        "unit",
        "sign",
        "orphan_sign",
        "nan_current",
        "unknown_flag",
        "bad_float",
        "overflow",
        "bad_grid",
        "bad_shape",
    ],
)
def test_real_input_refusals(tmp_path: Path, capsys: pytest.CaptureFixture[str], case: str) -> None:
    """Caller-facing failures never print file paths, exceptions or tracebacks."""
    path = tmp_path / "private-name.npz"
    args = _args(path)
    if case == "broken_zip":
        path.write_bytes(b"not an archive")
    elif case == "missing_flag":
        args = args[:-2]
    elif case == "unit":
        args[3] = "inch"
    elif case == "sign":
        args[-1] = "0"
    elif case == "orphan_sign":
        args += ["--reference-axis-sign", "-1"]
    elif case == "unknown_flag":
        args += ["--bogus"]
    elif case == "bad_float":
        args += ["--measured-current-a", "not-a-number"]
    elif case != "missing_file":
        radius, height = np.linspace(1, 2, 17), np.linspace(-1, 1, 17)
        rr, zz = np.meshgrid(radius, height)
        phi = -((rr - 1.5) ** 2) - zz**2
        if case == "nan_current":
            args += ["--measured-current-a", "nan"]
        elif case == "overflow":
            phi *= 1e200
        elif case == "bad_grid":
            radius[5] = radius[4]
        elif case == "bad_shape":
            phi = phi[:-1]
        np.savez(path, R=radius, Z=height, psi=phi)
    assert main(args) == 2
    output = capsys.readouterr()
    code = (
        "INPUT_UNREADABLE"
        if case == "missing_file"
        else "INVALID_LENGTH_UNIT"
        if case == "unit"
        else "INVALID_ARGUMENT"
        if case in {"missing_flag", "sign", "orphan_sign", "unknown_flag", "bad_float"}
        else "NUMERIC_RANGE_EXCEEDED"
        if case == "overflow"
        else "AUDIT_REFUSED"
    )
    assert output.err == "" and json.loads(output.out) == {
        "schema": "equilibrium-audit-refusal.v1",
        "status": "refused",
        "code": code,
        "message": REFUSAL,
    }
    assert str(path) not in output.out and "Traceback" not in output.out


@pytest.mark.parametrize(
    "fault,code",
    [
        ("grid", "GRID_NONUNIFORM"),
        ("missing", "MISSING_ARRAY"),
        ("flux_unit", "INVALID_FLUX_UNIT"),
        ("overflow", "NUMERIC_RANGE_EXCEEDED"),
    ],
)
def test_actionable_refusal_codes_cli(tmp_path: Path, fault: str, code: str) -> None:
    """Identify the actual declaration fault without exposing private input details."""
    radius, height = np.linspace(1.0, 2.0, 17), np.linspace(-1.0, 1.0, 17)
    rr, zz = np.meshgrid(radius, height)
    phi = -((rr - 1.5) ** 2) - zz**2
    path = tmp_path / "private-map-name.npz"
    args = _args(path)
    if fault == "missing":
        np.savez(path, R=radius, Z=height)
    else:
        if fault == "grid":
            radius[5] += 0.01
        elif fault == "flux_unit":
            args[5] = "tesla"
        else:
            phi *= 1e200
        np.savez(path, R=radius, Z=height, psi=phi)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--branch",
            "--parallel-mode",
            "-m",
            "tools.equilibrium_audit",
            *args,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 2 and result.stderr == ""
    assert json.loads(result.stdout) == {
        "schema": "equilibrium-audit-refusal.v1",
        "status": "refused",
        "code": code,
        "message": REFUSAL,
    }
    assert str(path) not in result.stdout and "Traceback" not in result.stdout


def test_minimal_field_and_help(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A field-only map is usable and help describes the real required interface."""
    path = tmp_path / "flat.npz"
    np.savez(path, R=np.arange(1.0, 5.0), Z=np.arange(4.0), psi=np.zeros((4, 4)))
    assert main(_args(path)) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["input"]["plasma_nodes"] is None and report["reference"] is None
    assert report["diagnostics"]["polynomial_gs"]["reason"] == "missing_plasma_mask"
    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0
    assert "--length-unit" in capsys.readouterr().out
