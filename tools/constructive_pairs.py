# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — fixed-coil synthetic equilibrium generation
"""Generate and retain individually identified static forward calculations.

This experiment does not train or change the competition submission. Its inner
surface distance, internal inductance and beta definitions are declared model
diagnostics, not the competition scorer or a proof about real DIII-D inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable
from contextlib import redirect_stdout
from dataclasses import asdict
from importlib.metadata import version
from io import StringIO
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from freegsnke.GSstaticsolver import NKGSsolver  # type: ignore[import-untyped]
from scipy.interpolate import RectBivariateSpline  # type: ignore[import-untyped]
from scipy.ndimage import binary_erosion  # type: ignore[import-untyped]
from shapely import contains_xy
from shapely.geometry import Point, Polygon
from skimage.measure import find_contours

from tools.constructive_model import (
    F64,
    MU0,
    Equilibrium,
    FixedPressureProfile,
    Profile,
    current_source,
    make_equilibrium,
)
from tools.constructive_observations import observe

VERSIONS = {
    "freegs": "0.8.2",
    "freegs4e": "0.15.0",
    "freegsnke": "3.1.1",
    "numpy": "2.3.5",
    "scipy": "1.18.1",
    "shapely": "2.2.0",
}
POWERS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
SURFACE = 0.999
SOURCE_PATHS = (
    "tools/constructive_model.py",
    "tools/constructive_observations.py",
    "tools/constructive_pairs.py",
    "tools/constructive_reference.py",
    "tools/constructive_geometry.py",
)


class Solver(Protocol):
    """Narrow interface to the independently distributed nonlinear solver."""

    relative_change: float
    norm_rel_change: list[float]

    def forward_solve(
        self,
        eq: Equilibrium,
        profiles: FixedPressureProfile,
        target_relative_tolerance: float,
        max_solving_iterations: int,
        suppress: bool,
        verbose: bool,
    ) -> None:
        """Solve at fixed metal currents with the declared profile."""
        ...

    def F_function(self, plasma_psi: F64, tokamak_psi: F64, profiles: FixedPressureProfile) -> F64:
        """Return the nonlinear free-boundary fixed-point residual."""
        ...

    def port_critical(self, eq: Equilibrium, profiles: FixedPressureProfile) -> None:
        """Attach final critical points and a profile copy to the equilibrium."""
        ...


def inner_boundary(eq: Equilibrium, level: float = SURFACE) -> F64:
    """Extract an interior flux surface without filtering contour vertices.

    Parameters
    ----------
    eq : Equilibrium
        Finite solved field and detected axis/boundary values.
    level : float
        Normalized flux in (0, 1), defaulting to the declared near-boundary proxy.

    Returns
    -------
    ndarray
        720 points at uniform arclength on the largest closed contour
        containing the primary axis. This is a near-boundary proxy, not D_LCFS.

    Raises
    ------
    ValueError
        No closed contour surrounds the primary axis.
    """
    if not 0 < level < 1:
        raise ValueError("An interior flux level in (0, 1) is required")
    normalized = (eq.psi() - eq.psi_axis) / (eq.psi_bndry - eq.psi_axis)
    contours = cast(Callable[[F64, float], list[F64]], find_contours)(normalized, level)
    polygons = []
    for contour in contours:
        if not np.allclose(contour[0], contour[-1], rtol=0, atol=1e-10):
            continue
        points = np.column_stack(
            [
                np.interp(contour[:, 0], np.arange(len(eq.R)), eq.R[:, 0]),
                np.interp(contour[:, 1], np.arange(eq.Z.shape[1]), eq.Z[0]),
            ]
        )
        polygon = Polygon(points)
        if polygon.is_valid and polygon.covers(Point(eq.opt[0, :2])):
            polygons.append(polygon)
    if not polygons:
        raise ValueError("No closed near-boundary contour contains the detected axis")
    selected = max(polygons, key=lambda p: p.area)
    ring = np.asarray(selected.exterior.coords)
    length = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ring, axis=0), axis=1))])
    samples = np.linspace(0, length[-1], 720, endpoint=False)
    return np.column_stack([np.interp(samples, length, ring[:, k]) for k in (0, 1)])


def measure(eq: Equilibrium, profile: FixedPressureProfile) -> tuple[dict[str, object], dict[str, F64]]:
    """Measure final arrays independently of the solver's convergence boolean.

    Parameters
    ----------
    eq : Equilibrium
        Final external solver field with current critical-point metadata.
    profile : FixedPressureProfile
        Limiter-aware current support and evaluated physical source.

    Returns
    -------
    metadata, arrays : dict, dict
        Physical quantities, explicit discrete residuals and original map arrays.

    Raises
    ------
    ValueError
        The field has no supported interior stencil or a valid flux function.
    """
    r, z = eq.R[:, 0], eq.Z[0]
    dr, dz = float(r[1] - r[0]), float(z[1] - z[0])
    mask = np.asarray(profile.limiter_core_mask, dtype=np.float64)
    plasma, flux = np.asarray(eq.plasma_psi), np.asarray(eq.psi())
    current = np.asarray(profile.jtor, dtype=np.float64)
    supported = np.asarray(binary_erosion(mask > 0, structure=np.ones((3, 3))))[1:-1, 1:-1]
    if not supported.any():
        raise ValueError("No eroded plasma stencils are available")

    def laplacian(values: F64) -> F64:
        centre = values[1:-1, 1:-1]
        return (
            (values[2:, 1:-1] - 2 * centre + values[:-2, 1:-1]) / dr**2
            - (values[2:, 1:-1] - values[:-2, 1:-1]) / (2 * dr * r[1:-1, None])
            + (values[1:-1, 2:] - 2 * centre + values[1:-1, :-2]) / dz**2
        )

    source = -MU0 * eq.R[1:-1, 1:-1] * current[1:-1, 1:-1]
    source_rms = float(np.sqrt(np.mean(source[supported] ** 2)))
    if not source_rms > 0:
        raise ValueError("The supported GS source is degenerate")
    plasma_residual = float(np.sqrt(np.mean((laplacian(plasma)[supported] - source[supported]) ** 2)))
    total_residual = float(np.sqrt(np.mean((laplacian(flux)[supported] - source[supported]) ** 2)))
    boundary = inner_boundary(eq)
    rgeom = float((boundary[:, 0].max() + boundary[:, 0].min()) / 2)
    minor_radius = float((boundary[:, 0].max() - boundary[:, 0].min()) / 2)
    spline = RectBivariateSpline(r, z, flux)
    bp_squared = (spline(r, z, dx=1) ** 2 + spline(r, z, dy=1) ** 2) / eq.R**2
    dv = 2 * np.pi * eq.R * dr * dz * mask
    li3 = float(2 * np.sum(bp_squared * dv) / (MU0**2 * rgeom * profile.physical.current**2))
    inside = np.asarray(contains_xy(Polygon(boundary), eq.R, eq.Z))
    surface_dv = 2 * np.pi * eq.R * dr * dz * inside
    perimeter_segments = np.linalg.norm(np.roll(boundary, -1, axis=0) - boundary, axis=1)
    br, bz = boundary[:, 0], boundary[:, 1]
    boundary_bp = np.hypot(spline.ev(br, bz, dx=1), spline.ev(br, bz, dy=1)) / br
    ampere_integral = float(np.sum((boundary_bp + np.roll(boundary_bp, -1)) * perimeter_segments / 2))
    perimeter = float(np.sum(perimeter_segments))
    mean_bp_squared = float(np.sum(bp_squared * surface_dv) / np.sum(surface_dv))
    li = mean_bp_squared / (ampere_integral / perimeter) ** 2
    s = np.clip((flux - eq.psi_axis) / (eq.psi_bndry - eq.psi_axis), 0, 1)
    p = profile.pressure(s)
    mean_pressure = float(np.sum(p * dv) / np.sum(dv))
    b0 = profile.physical.fvac / rgeom
    beta_n = float(2 * MU0 * mean_pressure / b0**2 * 100 * minor_radius * abs(b0))
    beta_n /= abs(profile.physical.current) / 1e6
    observations = observe(r, z, flux, mask, eq.psi_axis, eq.psi_bndry)
    surface95 = inner_boundary(eq, 0.95)
    rq, zq = surface95[:, 0], surface95[:, 1]
    gradient = np.hypot(spline.ev(rq, zq, dx=1), spline.ev(rq, zq, dy=1))
    integrand = float(profile.fpol(np.asarray([0.95]))[0]) / (2 * np.pi * rq * gradient)
    dl = np.linalg.norm(np.roll(surface95, -1, axis=0) - surface95, axis=1)
    q95 = float(np.sum((integrand + np.roll(integrand, -1)) * dl / 2))
    if not np.isfinite([li, beta_n, q95, plasma_residual, total_residual]).all():
        raise ValueError("A measured field quantity is nonfinite")
    gs_ratio = plasma_residual / source_rms
    relative_current_error = abs(float(np.sum(current) * dr * dz) / profile.physical.current - 1)
    metadata: dict[str, object] = {
        "profile": asdict(profile.physical),
        "axis": eq.opt.tolist(),
        "xpoints": eq.xpt.tolist(),
        "limited": bool(profile.flag_limiter),
        "li": li,
        "li_definition": "ITER li(2), spline Bp, volume within s=.999 contour, Ampere loop normalization",
        "li3_true_Ip": li3,
        "ampere_loop_current_A": ampere_integral / MU0,
        "betaN_model": beta_n,
        "betaN_geometry": {"Rgeo_m": rgeom, "minor_radius_m": minor_radius, "B0_T": b0},
        "q95_model": q95,
        "inner_surface_psiN": SURFACE,
        "inner_surface_sampling": "720 uniform arclength samples, no contour-vertex filtering",
        "plasma_gs_rms_over_source": gs_ratio,
        "total_gs_rms_over_source": total_residual / source_rms,
        "gs_stencil_nodes": int(supported.sum()),
        "current_A": float(np.sum(current) * dr * dz),
        "relative_current_error": relative_current_error,
        "flux_span_Wb_per_rad": eq.psi_bndry - eq.psi_axis,
        "field_checks_passed": bool(gs_ratio <= 1e-6 and relative_current_error <= 1e-7 and len(eq.opt) == 1),
    }
    return metadata, {
        "R_m": r,
        "Z_m": z,
        "psi_Wb_per_rad": flux,
        "plasma_psi_Wb_per_rad": plasma,
        "current_A_m2": current,
        "core_mask": mask,
        "observations": observations,
        "inner_surface_m": boundary,
    }


def solve_case(
    size: int, power: float, initial: int, currents: F64
) -> tuple[dict[str, object], dict[str, F64]]:
    """Solve one identified forward case with no replacement settings.

    Parameters
    ----------
    size, initial : int
        Grid and independent initial-guess index.
    power : float
        FF-prime exponent; pressure shape and amplitude remain fixed.
    currents : ndarray
        Frozen 18-coil current vector.

    Returns
    -------
    metadata, arrays : dict, dict
        Measured state even when nonlinear convergence fails; exceptions propagate.
    """
    eq = make_equilibrium(size, initial, currents)
    profile = FixedPressureProfile(eq, Profile(power=power))
    solver = cast(Solver, NKGSsolver(eq, seed=42, gs_operator_order=2))
    native_log = StringIO()
    try:
        with redirect_stdout(native_log):
            solver.forward_solve(eq, profile, 1e-9, max_solving_iterations=400, suppress=False, verbose=True)
    finally:
        print(native_log.getvalue(), end="")
    replaced_initial = "Default plasma_psi initialisation and adjustment invoked." in native_log.getvalue()
    coil_flux = eq.psi() - eq.plasma_psi
    residual = solver.F_function(eq.plasma_psi.ravel(), coil_flux.ravel(), profile)
    area = float((eq.R[1, 0] - eq.R[0, 0]) * (eq.Z[0, 1] - eq.Z[0, 0]))
    refreshed, evaluated = current_source(
        profile.physical,
        eq.R,
        eq.psi(),
        float(profile.opt[0, 2]),
        float(profile.psi_bndry),
        np.asarray(profile.limiter_core_mask, dtype=np.float64),
        area,
    )
    source_difference = float(np.max(np.abs(refreshed - np.asarray(profile.jtor))))
    if source_difference > 1e-10 * float(np.max(np.abs(refreshed))):
        raise ValueError("Accepted-map source differs from the solver's refreshed current")
    profile.physical = evaluated
    profile.fpol(np.asarray([0.0, 1.0]))
    solver.port_critical(eq, profile)
    metadata, arrays = measure(eq, profile)
    span = abs(eq.psi_bndry - eq.psi_axis)
    maximum = float(np.max(np.abs(residual))) / span
    metadata.update(
        {
            "nonlinear_relative_max": maximum,
            "reported_relative_change": float(solver.relative_change),
            "norm_relative_history": [float(x) for x in solver.norm_rel_change],
            "coil_currents_unchanged": bool(np.array_equal(eq.tokamak.current_vec, currents)),
            "converged": bool(maximum <= 1e-9),
            "native_initialization_rescales": native_log.getvalue().count(
                "Initial guess for plasma_psi failed"
            ),
            "native_default_initialization_used": replaced_initial,
            "source_reevaluated_on_accepted_map": True,
            "source_refresh_max_absolute_A_m2": source_difference,
            "numerically_admissible": bool(
                maximum <= 1e-9
                and metadata["field_checks_passed"] is True
                and not replaced_initial
                and np.array_equal(eq.tokamak.current_vec, currents)
            ),
        }
    )
    return metadata, arrays


def verify_binding(path: Path) -> str:
    """Verify the pre-calculation source binding and exact core package versions."""
    binding = json.loads(path.read_text())
    for relative in SOURCE_PATHS:
        actual = hashlib.sha256(Path(relative).read_bytes()).hexdigest()
        if binding["source_sha256"][relative] != actual:
            raise ValueError(f"Source changed after binding: {relative}")
    for package, expected in VERSIONS.items():
        if version(package) != expected:
            raise ValueError(f"Unqualified package version: {package}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    """Run one source-bound calculation, retaining an explicit failure receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--currents", type=Path, required=True)
    parser.add_argument("--size", type=int, choices=(65, 129, 257), default=65)
    parser.add_argument("--power", type=float, choices=POWERS, default=2)
    parser.add_argument("--initial", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    binding = verify_binding(args.binding)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report: dict[str, object] = {
        "schema": "constructive_static_v1",
        "source_binding_sha256": binding,
        "versions": {p: version(p) for p in VERSIONS},
        "size": args.size,
        "power": args.power,
        "initial": args.initial,
    }
    status = 0
    try:
        currents = np.asarray(np.load(args.currents, allow_pickle=False), dtype=np.float64)
        metadata, arrays = solve_case(args.size, args.power, args.initial, currents)
        report.update(metadata)
        np.savez_compressed(
            args.output / "maps.npz",
            allow_pickle=False,
            R_m=arrays["R_m"],
            Z_m=arrays["Z_m"],
            psi_Wb_per_rad=arrays["psi_Wb_per_rad"],
            plasma_psi_Wb_per_rad=arrays["plasma_psi_Wb_per_rad"],
            current_A_m2=arrays["current_A_m2"],
            core_mask=arrays["core_mask"],
            observations=arrays["observations"],
            inner_surface_m=arrays["inner_surface_m"],
        )
        report["status"] = "returned"
    except Exception as error:
        report.update(status="failed", exception=f"{type(error).__name__}: {error}")
        status = 1
    report["seconds"] = time.monotonic() - started
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
