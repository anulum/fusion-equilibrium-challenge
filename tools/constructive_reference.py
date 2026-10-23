# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — legacy synthetic reference coil generation
"""Generate one synthetic reference coil vector in an isolated legacy runtime.

FreeGS0.8.2 requires the pre-removal SciPy interp2d API for its limiter-aware
inverse-control path. This entrypoint shares no solver imports with the modern
forward-calculation runtime. It does not use any competition data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path

import freegs  # type: ignore[import-untyped]
import freegs.boundary  # type: ignore[import-untyped]
import freegs.control  # type: ignore[import-untyped]
import freegs.critical  # type: ignore[import-untyped]
import freegs.jtor  # type: ignore[import-untyped]
import freegs.picard  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import romb  # type: ignore[import-untyped]

from tools.constructive_geometry import limiter_boundary

F64 = NDArray[np.float64]
VERSIONS = {"freegs": "0.8.2", "numpy": "1.26.4", "scipy": "1.13.1"}


def reference_state() -> tuple[F64, dict[str, object], dict[str, F64]]:
    """Generate one inverse state and retain its currents, field and critical points."""
    legacy = freegs.machine.DIIID()
    vertices = limiter_boundary(np.column_stack([legacy.wall.R, legacy.wall.Z]))
    machine = freegs.machine.Machine(legacy.coils, wall=freegs.machine.Wall(vertices[:, 0], vertices[:, 1]))
    eq = freegs.Equilibrium(
        tokamak=machine,
        Rmin=0.1,
        Rmax=2.8,
        Zmin=-1.8,
        Zmax=1.8,
        nx=65,
        ny=65,
        boundary=freegs.boundary.freeBoundaryHagenow,
        order=2,
    )
    profile = freegs.jtor.ConstrainPaxisIp(eq, 20_000.0, 1_000_000.0, 3.4, alpha_m=1, alpha_n=2)
    control = freegs.control.constrain(
        xpoints=[(1.285, -1.176), (1.2, 1.0)],
        isoflux=[(1.285, -1.176, 1.2, 1.2)],
        gamma=1e-12,
    )
    freegs.picard.solve(
        eq, profile, control, rtol=1e-9, atol=1e-11, maxits=400, blend=0.5, check_limited=True
    )
    currents = np.asarray([coil.current for _, coil in machine.coils], dtype=np.float64)
    if not np.isfinite(currents).all() or np.max(np.abs(currents)) > 5e6:
        raise ValueError("Reference control did not produce finite bounded effective coil currents")
    flux = np.asarray(eq.psi(), dtype=np.float64)
    current = np.asarray(profile.Jtor(eq.R, eq.Z, flux), dtype=np.float64)
    opt, xpt = freegs.critical.find_critical(eq.R, eq.Z, flux)
    dr, dz = float(eq.R[1, 0] - eq.R[0, 0]), float(eq.Z[0, 1] - eq.Z[0, 0])
    metadata: dict[str, object] = {
        "axis": np.asarray(opt).tolist(),
        "xpoints": np.asarray(xpt).reshape(-1, 3).tolist(),
        "limited": bool(eq.is_limited),
        "flux_span_Wb_per_rad": float(eq.psi_bndry - eq.psi_axis),
        "current_rectangle_A": float(np.sum(current) * dr * dz),
        "current_romberg_A": float(romb(romb(current)) * dr * dz),
        "coil_labels": [label for label, _ in machine.coils],
        "coil_turns": [int(coil.turns) for _, coil in machine.coils],
        "coil_current_unit": "A per turn; effective ampere-turns = current times turns",
        "limiter_vertices_m": vertices.tolist(),
    }
    arrays = {
        "R_m": np.asarray(eq.R[:, 0], dtype=np.float64),
        "Z_m": np.asarray(eq.Z[0], dtype=np.float64),
        "psi_Wb_per_rad": flux,
        "plasma_psi_Wb_per_rad": np.asarray(eq.plasma_psi, dtype=np.float64),
        "current_A_m2": current,
        "core_mask": np.asarray(eq.mask, dtype=np.float64),
    }
    return currents, metadata, arrays


def main() -> int:
    """Run the complete reference path and retain its result or failure receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    binding = json.loads(args.binding.read_text())
    for relative, expected in binding["source_sha256"].items():
        if hashlib.sha256(Path(relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source changed after binding: {relative}")
    actual_versions = {p: version(p) for p in VERSIONS}
    if actual_versions != VERSIONS:
        raise ValueError(f"Unqualified reference environment: {actual_versions}")
    args.output.mkdir(parents=True, exist_ok=False)
    report: dict[str, object] = {
        "schema": "constructive_reference_v1",
        "source_binding_sha256": hashlib.sha256(args.binding.read_bytes()).hexdigest(),
        "versions": actual_versions,
    }
    started = time.monotonic()
    status = 0
    try:
        currents, metadata, arrays = reference_state()
        np.save(args.output / "currents.npy", currents, allow_pickle=False)
        np.savez_compressed(
            args.output / "maps.npz",
            R_m=arrays["R_m"],
            Z_m=arrays["Z_m"],
            psi_Wb_per_rad=arrays["psi_Wb_per_rad"],
            plasma_psi_Wb_per_rad=arrays["plasma_psi_Wb_per_rad"],
            current_A_m2=arrays["current_A_m2"],
            core_mask=arrays["core_mask"],
        )
        report.update(metadata)
        report.update(status="returned", currents_A=currents.tolist())
    except Exception as error:
        report.update(status="failed", exception=f"{type(error).__name__}: {error}")
        status = 1
    report["seconds"] = time.monotonic() - started
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
