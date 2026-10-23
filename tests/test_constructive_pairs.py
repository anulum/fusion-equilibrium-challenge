# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — synthetic field measurement controls
"""Check contour geometry, q and inductance against a circular analytic field."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from freegsnke.equilibrium_update import Equilibrium as NativeEquilibrium  # type: ignore[import-untyped]

from tools.constructive_model import Equilibrium, FixedPressureProfile, Profile, make_equilibrium
from tools.constructive_observations import compare_observations
from tools.constructive_pairs import SOURCE_PATHS, inner_boundary, measure

ROOT = Path(__file__).resolve().parents[1]


def test_recorded_case_ledger_preserves_all_states() -> None:
    """Verify the complete fixed search and recompute all available channel comparisons."""
    manifest = json.loads((ROOT / "tools/reference/constructive_calculation.json").read_text())
    entry = manifest["case_ledger"]
    path = ROOT / entry["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    record = json.loads(path.read_text())
    assert record["schema"] == "constructive_case_results_v4"
    assert record["status"] == "inconclusive_numerical_compatibility_gate_failed"
    assert record["controller_phase"] == "finished"
    assert record["source_sha256"] == manifest["source_sha256"]
    expected = ["reference_control"] + [
        f"n{size}_k{power:g}_i{initial}"
        for power in (2.0, 1.0, 1.5, 2.5, 3.0, 4.0)
        for size in (65, 129, 257)
        for initial in (0, 1)
    ]
    assert [row["id"] for row in record["cases"]] == expected
    assert record["expected_forward_cases"] == 36
    assert record["budget_seconds"] == 10800 and record["per_case_limit_seconds"] == 900
    assert Counter(row["process_status"] for row in record["cases"]) == {
        "returned": 26,
        "failed": 4,
        "timeout": 7,
    }
    assert Counter(row.get("report", {}).get("numerically_admissible") for row in record["cases"][1:]) == {
        True: 11,
        False: 14,
        None: 11,
    }
    cases = {row["id"]: row for row in record["cases"]}
    for row in cases.values():
        if row["process_status"] == "returned":
            assert row["report"]["status"] == "returned"
            assert "maps_sha256" in row
        elif row["process_status"] == "failed":
            assert row["report"]["status"] == "failed"
            assert "exception" in row["report"] and "maps_sha256" not in row
        else:
            assert row["process_status"] == "timeout" and "report" not in row and "maps_sha256" not in row
        if "report" in row:
            assert row["report"]["source_binding_sha256"] == manifest["original_source_binding_sha256"]
        if "observations" in row:
            values = np.asarray(row["observations"])
            assert values.shape == (196,) and np.isfinite(values).all() and np.all(values >= 0)
    pairs = record["all_fine_grid_pairings"]
    assert len(pairs) == 20
    assert Counter(pair["status"] for pair in pairs) == {
        "descriptive_only": 4,
        "unavailable_returned_state": 16,
    }
    expected_pairs = {
        (f"n257_k2_i{reference}", f"n257_k{power:g}_i{candidate}")
        for power in (1.0, 1.5, 2.5, 3.0, 4.0)
        for reference in (0, 1)
        for candidate in (0, 1)
    }
    assert {(pair["reference"], pair["candidate"]) for pair in pairs} == expected_pairs
    for pair in pairs:
        assert pair["primary_admitted"] is False
        reference, candidate = cases[pair["reference"]], cases[pair["candidate"]]
        if "observations" not in reference or "observations" not in candidate:
            assert pair["status"] == "unavailable_returned_state"
            continue
        assert pair["status"] == "descriptive_only"
        result = compare_observations(
            np.asarray(reference["observations"]), np.asarray(candidate["observations"])
        )
        assert pair["minimum_relative_with_floor"] == pytest.approx(
            result.minimum_relative, rel=1e-12, abs=1e-14
        )
        assert pair["minimum_relative_without_floor"] == pytest.approx(
            result.minimum_relative_without_floor, rel=1e-12, abs=1e-14
        )
        assert pair["all_channel_primary_compatible"] is result.primary_compatible
        assert pair["sensitivity"] == list(result.sensitivity)
        assert pair["li_difference"] == pytest.approx(
            abs(candidate["report"]["li"] - reference["report"]["li"]), rel=1e-12, abs=1e-14
        )
        assert pair["both_individually_admissible"] is (
            reference["report"]["numerically_admissible"] and candidate["report"]["numerically_admissible"]
        )


def test_recorded_coarse_solver_gate() -> None:
    """Recompute the failed coarse gate from the original public field arrays."""
    record = json.loads((ROOT / "tools/reference/constructive_calculation.json").read_text())
    bundle = record["coarse_bundle"]
    path = ROOT / bundle["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == bundle["sha256"]
    for relative in SOURCE_PATHS:
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == record["source_sha256"][relative]
    for relative, digest in record["runtime_locks"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
    boundary = record["boundary_control"]
    assert (
        hashlib.sha256((ROOT / boundary["maintained_reproducer_path"]).read_bytes()).hexdigest()
        == boundary["maintained_reproducer_sha256"]
    )
    with np.load(path, allow_pickle=False) as maps:
        np.testing.assert_array_equal(maps["inverse__R_m"], maps["forward__R_m"])
        np.testing.assert_array_equal(maps["inverse__Z_m"], maps["forward__Z_m"])
        difference = maps["forward__psi_Wb_per_rad"] - maps["inverse__psi_Wb_per_rad"]
        inverse = record["coarse_reports"]["inverse"]
        forward = record["coarse_reports"]["forward"]
        error = float(np.max(abs(difference - difference.mean())) / abs(inverse["flux_span_Wb_per_rad"]))
        measured = record["coarse_comparison"]
        assert measured["limits"]["max_flux_over_reference_span"] == 0.005
        assert error == pytest.approx(measured["max_flux_over_reference_span"], rel=2e-13)
        assert error > measured["limits"]["max_flux_over_reference_span"]
        assert record["result"] == "inconclusive_numerical_compatibility_gate_failed"
        axis_distance = np.linalg.norm(
            np.asarray(forward["axis"])[0, :2] - np.asarray(inverse["axis"])[0, :2]
        )
        assert axis_distance == pytest.approx(measured["axis_distance_m"], rel=2e-13)
        area = np.diff(maps["inverse__R_m"])[0] * np.diff(maps["inverse__Z_m"])[0]
        inverse_current = np.sum(maps["inverse__current_A_m2"]) * area
        forward_current = np.sum(maps["forward__current_A_m2"]) * area
        current_difference = abs(forward_current - inverse_current) / 1e6
        assert current_difference == pytest.approx(measured["rectangle_Ip_relative_difference"], abs=1e-15)
        assert forward["limited"] is inverse["limited"] is True
        np.testing.assert_array_equal(maps["frozen_coil_currents_A_per_turn"], inverse["currents_A"])


@pytest.fixture
def forward_binding(tmp_path: Path) -> Path:
    """Bind actual source files for the public forward entrypoint."""
    path = tmp_path / "binding.json"
    path.write_text(
        json.dumps(
            {"source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCE_PATHS}}
        )
    )
    return path


@pytest.fixture
def frozen_currents(tmp_path: Path) -> Path:
    """Read the original synthetic inverse output and serialize its 18 currents."""
    data = json.loads((ROOT / "tests/data/constructive_currents_v4.json").read_text())
    path = tmp_path / "currents.npy"
    np.save(path, np.asarray(data["currents_A"], dtype=np.float64), allow_pickle=False)
    return path


def run_forward(
    python: Path, binding: Path, currents: Path, output: Path, script: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Exercise the actual CLI, runtime, source binding and artifact export."""
    target = [str(script)] if script is not None else ["-m", "tools.constructive_pairs"]
    return subprocess.run(
        [
            str(python),
            "-m",
            "coverage",
            "run",
            "--parallel-mode",
            "--source=tools",
            *target,
            "--binding",
            str(binding),
            "--currents",
            str(currents),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )


@pytest.fixture
def circular_state() -> tuple[Equilibrium, FixedPressureProfile]:
    """Build a finite circular field with a declared current for diagnostic controls."""
    eq = make_equilibrium(129, 0, np.zeros(18))
    r0, a, coefficient = 1.7, 0.5, 1.2
    radius_squared = (eq.R - r0) ** 2 + eq.Z**2
    flux = coefficient * radius_squared
    eq.plasma_psi = flux
    eq.psi_axis, eq.psi_bndry = 0.0, coefficient * a**2
    eq.opt = np.asarray([[r0, 0.0, 0.0]])
    eq.xpt = np.empty((0, 3))
    profile = FixedPressureProfile(eq, Profile())
    mask = (radius_squared < a**2).astype(float)
    profile.Jtor_part2(eq.R, eq.Z, flux, eq.psi_axis, eq.psi_bndry, mask)
    profile.limiter_core_mask = mask
    profile.flag_limiter = True
    return eq, profile


def test_circular_field_diagnostics_and_residual(
    circular_state: tuple[Equilibrium, FixedPressureProfile],
) -> None:
    """Check exact circular q/li while detecting that this profile is not a GS solution."""
    eq, profile = circular_state
    r0, a, coefficient = 1.7, 0.5, 1.2
    flux = eq.psi()
    mask = profile.limiter_core_mask
    report, arrays = measure(eq, profile)
    f95 = float(profile.fpol(np.asarray([0.95]))[0])
    exact_q = f95 / (2 * coefficient * np.sqrt(r0**2 - a**2 * 0.95))
    q_actual = report["q95_model"]
    assert isinstance(q_actual, float)
    assert q_actual == pytest.approx(exact_q, rel=0.003)
    inner_a = a * np.sqrt(0.999)
    t = np.sqrt(r0**2 - inner_a**2)
    integral = 2 / 3 * r0**3 - r0**2 * t + t**3 / 3
    exact_li2 = 2 * t**2 / (r0 * inner_a**4) * integral
    li_actual, residual = report["li"], report["plasma_gs_rms_over_source"]
    assert isinstance(li_actual, float) and isinstance(residual, float)
    assert li_actual == pytest.approx(exact_li2, rel=0.02)
    assert residual > 0.1
    assert report["field_checks_passed"] is False
    assert arrays["inner_surface_m"].shape == (720, 2)
    assert arrays["observations"].shape == (196,)
    assert report["limited"] is True
    with pytest.raises(ValueError):
        inner_boundary(eq, 1.0)
    profile.limiter_core_mask = np.zeros_like(mask)
    with pytest.raises(ValueError, match="stencils"):
        measure(eq, profile)
    eq.plasma_psi = np.zeros_like(flux)
    with pytest.raises(ValueError, match="closed"):
        inner_boundary(eq)


def test_nonsquare_native_adapter_convention() -> None:
    """Verify the actual external (R, Z) ordering on an unequal-size grid."""
    base = make_equilibrium(17, 1, np.zeros(18))
    r, z = np.linspace(0.1, 2.8, 17), np.linspace(-1.8, 1.8, 33)
    rr, zz = np.meshgrid(r, z, indexing="ij")
    flux = 0.3 * (1 - ((rr - 1.7) / 0.5) ** 2 - (zz / 0.8) ** 2)
    native = NativeEquilibrium(
        tokamak=base.tokamak,
        Rmin=0.1,
        Rmax=2.8,
        Zmin=-1.8,
        Zmax=1.8,
        nx=17,
        ny=33,
        psi=flux,
        order=2,
    )
    np.testing.assert_array_equal(native.R, rr)
    np.testing.assert_array_equal(native.Z, zz)
    profile = FixedPressureProfile(native, Profile())
    mask = (flux > 0).astype(float)
    current = profile.Jtor_part2(native.R, native.Z, flux, 0.3, 0, mask)
    area = (r[1] - r[0]) * (z[1] - z[0])
    assert np.sum(current) * area == pytest.approx(1e6)
    limiter_current = profile.Jtor(native.R, native.Z, flux)
    assert np.sum(limiter_current) * area == pytest.approx(1e6)
    assert profile.limiter_core_mask.shape == (17, 33)
    assert profile.flag_limiter


def test_degenerate_current_diagnostic_refused(
    circular_state: tuple[Equilibrium, FixedPressureProfile],
) -> None:
    """Refuse a zero-source provider state rather than report a normalized residual."""
    eq, profile = circular_state
    profile.jtor = np.zeros_like(eq.R)
    with pytest.raises(ValueError, match="source is degenerate"):
        measure(eq, profile)


def test_overflowing_pressure_diagnostic_refused(
    circular_state: tuple[Equilibrium, FixedPressureProfile],
) -> None:
    """Prevent nonfinite diagnostic export from a corrupted pressure parameter."""
    eq, profile = circular_state
    old = profile.physical
    profile.physical = Profile(1e308, old.current, old.fvac, old.power, old.span, old.amplitude)
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="measured field quantity is nonfinite"):
            measure(eq, profile)


def test_surface_selection_excludes_secondary_magnetic_axis(
    circular_state: tuple[Equilibrium, FixedPressureProfile],
) -> None:
    """Select the primary island when a smooth field contains two closed islands."""
    eq, _ = circular_state
    first = (eq.R - 1.4) ** 2 + eq.Z**2
    second = (eq.R - 1.95) ** 2 + (eq.Z - 0.5) ** 2
    eq.plasma_psi = first * second
    eq.psi_axis, eq.psi_bndry = 0.0, 0.003
    eq.opt = np.asarray([[1.4, 0.0, 0.0], [1.95, 0.5, 0.0]])
    contour = inner_boundary(eq)
    assert contour[:, 0].min() < 1.4 < contour[:, 0].max() < 1.7
    assert contour[:, 1].min() < 0 < contour[:, 1].max() < 0.3


def test_open_vacuum_field_has_no_closed_surface(
    circular_state: tuple[Equilibrium, FixedPressureProfile],
) -> None:
    """Reject the open contours of a uniform vertical vacuum field."""
    eq, _ = circular_state
    eq.plasma_psi = eq.R**2 / 2
    eq.psi_axis, eq.psi_bndry = float(eq.plasma_psi.min()), float(eq.plasma_psi.max())
    with pytest.raises(ValueError, match="No closed near-boundary contour"):
        inner_boundary(eq)


@pytest.mark.integration
def test_forward_cli_converges_and_exports_actual_field(
    forward_binding: Path, frozen_currents: Path, tmp_path: Path
) -> None:
    """Solve the actual fixed-current model and recheck its saved numerical state."""
    output = tmp_path / "forward"
    result = run_forward(Path(sys.executable), forward_binding, frozen_currents, output)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "returned"
    assert report["numerically_admissible"] is True
    assert report["source_reevaluated_on_accepted_map"] is True
    assert report["source_refresh_max_absolute_A_m2"] == 0
    assert report["nonlinear_relative_max"] <= 1e-9
    assert report["plasma_gs_rms_over_source"] <= 1e-6
    assert report["current_A"] == pytest.approx(1e6, rel=1e-7)
    assert report["li"] == pytest.approx(1.6225724936664814, rel=0.001)
    with np.load(output / "maps.npz", allow_pickle=False) as maps:
        assert maps["psi_Wb_per_rad"].shape == (65, 65)
        assert maps["observations"].shape == (196,)
        area = np.diff(maps["R_m"])[0] * np.diff(maps["Z_m"])[0]
        assert np.sum(maps["current_A_m2"]) * area == pytest.approx(report["current_A"])


@pytest.mark.integration
def test_forward_refuses_stale_current_after_real_solver(
    forward_binding: Path, frozen_currents: Path, tmp_path: Path
) -> None:
    """Reject a native-provider current fault after genuine nonlinear convergence."""
    script = tmp_path / "stale_native_source.py"
    script.write_text(
        "from freegsnke.GSstaticsolver import NKGSsolver\n"
        "from tools.constructive_pairs import main\n"
        "original = NKGSsolver.forward_solve\n"
        "def solve_then_inject(self, *args, **kwargs):\n"
        "    original(self, *args, **kwargs)\n"
        "    evaluate = self.F_function\n"
        "    def stale_source(plasma_psi, tokamak_psi, profiles):\n"
        "        residual = evaluate(plasma_psi, tokamak_psi, profiles)\n"
        "        profiles.jtor = profiles.jtor.copy()\n"
        "        index = profiles.jtor.argmax()\n"
        "        profiles.jtor.flat[index] *= 1.01\n"
        "        return residual\n"
        "    self.F_function = stale_source\n"
        "NKGSsolver.forward_solve = solve_then_inject\n"
        "raise SystemExit(main())\n"
    )
    output = tmp_path / "stale-current"
    result = run_forward(Path(sys.executable), forward_binding, frozen_currents, output, script)
    assert result.returncode == 1, result.stdout + result.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed"
    assert "Accepted-map source differs" in report["exception"]
    assert not (output / "maps.npz").exists()


def test_forward_cli_refuses_bad_current_file(
    forward_binding: Path, frozen_currents: Path, tmp_path: Path
) -> None:
    """Retain an explicit refusal when the operator supplies an incomplete coil set."""
    np.save(frozen_currents, np.zeros(17), allow_pickle=False)
    output = tmp_path / "bad-current"
    result = run_forward(Path(sys.executable), forward_binding, frozen_currents, output)
    assert result.returncode == 1, result.stdout + result.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed"
    assert "18 finite coil currents" in report["exception"]
    assert not (output / "maps.npz").exists()


def test_forward_cli_refuses_modified_source(
    forward_binding: Path, frozen_currents: Path, tmp_path: Path
) -> None:
    """Refuse a corrupt source binding before writing an output directory."""
    binding = json.loads(forward_binding.read_text())
    binding["source_sha256"][SOURCE_PATHS[0]] = "0" * 64
    forward_binding.write_text(json.dumps(binding))
    output = tmp_path / "bad-binding"
    result = run_forward(Path(sys.executable), forward_binding, frozen_currents, output)
    assert result.returncode != 0
    assert "Source changed after binding" in result.stderr
    assert not output.exists()


def test_forward_cli_refuses_actual_unqualified_numpy(
    forward_binding: Path, frozen_currents: Path, tmp_path: Path
) -> None:
    """Reject a real installed NumPy version outside the qualified environment."""
    value = os.environ.get("CONSTRUCTIVE_WRONG_VERSION_PYTHON")
    if value is None:
        pytest.skip("Requires the hashed unqualified-version probe in the dedicated integration job.")
    output = tmp_path / "wrong-version"
    result = run_forward(Path(value), forward_binding, frozen_currents, output)
    assert result.returncode != 0
    assert "Unqualified package version: numpy" in result.stderr
    assert not output.exists()
