# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — prescribed-current boundary calibration
"""Compare boundary operators with an independent continuous Green integral.

The source is prescribed, not a converged nonlinear equilibrium. Flux is Wb/rad,
coordinates are metres, current density is A/m² and the analytic current is 1 MA.
This separate calibration cannot change the constructive search's failed gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import romb  # type: ignore[import-untyped]
from scipy.special import ellipe, ellipk  # type: ignore[import-untyped]

F64 = NDArray[np.float64]
MU0 = 4e-7 * np.pi


def prescribed_source(radius: F64, height: F64) -> F64:
    """Evaluate the compact elliptical current on aligned coordinate arrays.

    Parameters
    ----------
    radius, height : ndarray
        Coordinates in metres, with equal or broadcastable shapes.

    Returns
    -------
    ndarray
        C³ source, centred at (1.7, 0), with semiaxes 0.35 and 0.55 m.
    """
    rho = ((radius - 1.7) / 0.35) ** 2 + (height / 0.55) ** 2
    return 5e6 / (np.pi * 0.35 * 0.55) * np.maximum(0, 1 - rho) ** 4


def edge_coordinates(size: int) -> tuple[F64, F64]:
    """Return box-edge points and the legacy operator's outward observation points.

    Parameters
    ----------
    size : int
        Number of grid points per direction on R=[0.1,2.8], Z=[-1.8,1.8].

    Returns
    -------
    points, shifted : ndarray
        Edge coordinates in NumPy row-major mask order. The 0.01 m offset
        follows the upstream Hagenow boundary's side and corner assignments.
    """
    rr, zz = np.meshgrid(np.linspace(0.1, 2.8, size), np.linspace(-1.8, 1.8, size), indexing="ij")
    mask = np.zeros((size, size), dtype=bool)
    mask[[0, -1], :] = True
    mask[:, [0, -1]] = True
    points = np.column_stack([rr[mask], zz[mask]])
    shifted = points.copy()
    # FreeGS 0.8.2, freegs/boundary.py:189–231: later radial sides overwrite corner observations.
    shifted[:, 1] += np.where(points[:, 1] == -1.8, -0.01, np.where(points[:, 1] == 1.8, 0.01, 0))
    vertical = (points[:, 0] == 0.1) | (points[:, 0] == 2.8)
    shifted[vertical, 1] = points[vertical, 1]
    shifted[vertical, 0] += np.where(points[vertical, 0] == 0.1, -0.01, 0.01)
    return points, shifted


def filament_green(radius: F64, height: F64, target_radius: F64, target_height: F64) -> F64:
    """Evaluate the axisymmetric filament kernel away from coincident points.

    Parameters
    ----------
    radius, height, target_radius, target_height : ndarray
        Broadcastable source and target coordinates in metres; positive R
        and noncoincident source/target points are required.

    Returns
    -------
    ndarray
        Flux in Wb/rad per ampere of filament current, with decay at infinity.
    """
    k2 = 4 * radius * target_radius / ((radius + target_radius) ** 2 + (height - target_height) ** 2)
    return cast(
        F64,
        MU0 / (2 * np.pi) * np.sqrt(radius * target_radius / k2) * ((2 - k2) * ellipk(k2) - 2 * ellipe(k2)),
    )


def continuous_edge_flux(points: F64, radial: int = 64, angular: int = 128) -> F64:
    """Integrate the prescribed source independently of a mesh boundary solver.

    Parameters
    ----------
    points : ndarray
        Nonempty (N,2) coordinates outside the compact source support.
    radial, angular : int
        Gauss-Legendre orders in squared elliptical radius and angle.

    Returns
    -------
    ndarray
        Continuous-source flux at each observation point in Wb/rad.
    """
    u, w = np.polynomial.legendre.leggauss(radial)
    u, w = (u + 1) / 2, w / 2
    theta, v = np.polynomial.legendre.leggauss(angular)
    theta, v = np.pi * (theta + 1), np.pi * v
    radius = 1.7 + 0.35 * np.sqrt(u[:, None]) * np.cos(theta)[None, :]
    height = 0.55 * np.sqrt(u[:, None]) * np.sin(theta)[None, :]
    weights = (
        (0.35 * 0.55 / 2 * w[:, None] * v[None, :]) * 5e6 / (np.pi * 0.35 * 0.55) * (1 - u[:, None]) ** 4
    )
    result = []
    for batch in np.array_split(points, max(1, int(np.ceil(len(points) / 8)))):
        kernel = filament_green(
            radius.ravel()[None, :], height.ravel()[None, :], batch[:, 0, None], batch[:, 1, None]
        )
        result.extend(np.sum(kernel * weights.ravel()[None, :], axis=1))
    return np.asarray(result, dtype=np.float64)


class PrescribedCurrent:
    """Provide the physical source to the actual FreeGSNKE boundary operator."""

    def Jtor(self, R: F64, Z: F64, psi: F64) -> F64:
        """Return the prescribed current, independent of the trial flux."""
        return prescribed_source(R, Z)


class BoundarySolver(Protocol):
    """Read only the public source-to-boundary calculation from FreeGSNKE."""

    psi_boundary: F64

    def freeboundary(self, plasma: F64, vacuum: F64, profile: PrescribedCurrent) -> None:
        """Calculate boundary values for the supplied physical source."""
        ...


def main() -> int:
    """Write one complete boundary calibration using a named numerical engine."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["reference", "legacy", "native"], required=True)
    parser.add_argument("--size", type=int, choices=[65, 129, 257], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    r, z = np.linspace(0.1, 2.8, args.size), np.linspace(-1.8, 1.8, args.size)
    rr, zz = np.meshgrid(r, z, indexing="ij")
    mask = np.zeros((args.size, args.size), dtype=bool)
    mask[[0, -1], :] = True
    mask[:, [0, -1]] = True
    points, shifted = edge_coordinates(args.size)
    current = prescribed_source(rr, zz)
    area = (r[1] - r[0]) * (z[1] - z[0])
    out: dict[str, object] = {
        "schema": "constructive_boundary_v1",
        "engine": args.engine,
        "size": args.size,
        "source": "J=5e6/(pi*a*b)*(1-u)^4 for u<1; u=((R-1.7)/a)^2+(Z/b)^2, a=.35,b=.55; integral1MA",
        "current_rectangle_A": float(current.sum() * area),
        "current_romberg_A": float(romb(romb(current)) * area),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {p: version(p) for p in ["numpy", "scipy"]},
    }
    if args.engine == "reference":
        value = continuous_edge_flux(points)
        displaced = continuous_edge_flux(shifted)
        probe = np.linspace(0, len(points) - 1, 17, dtype=int)
        fine = continuous_edge_flux(points[probe], 128, 256)
        out.update(
            psi_edge=value.tolist(),
            psi_at_legacy_outward_offset=displaced.tolist(),
            quadrature_probe_max_relative=float(np.max(abs(fine - value[probe])) / np.max(abs(value))),
        )
    elif args.engine == "legacy":
        import freegs  # type: ignore[import-untyped]  # Separate pinned reference runtime.
        import freegs.boundary  # type: ignore[import-untyped]

        eq = freegs.Equilibrium(
            tokamak=freegs.machine.DIIID(),
            Rmin=0.1,
            Rmax=2.8,
            Zmin=-1.8,
            Zmax=1.8,
            nx=args.size,
            ny=args.size,
            order=2,
        )
        values, direct = np.zeros_like(current), np.zeros_like(current)
        freegs.boundary.freeBoundaryHagenow(eq, current, values)
        freegs.boundary.freeBoundary(eq, current, direct)
        out.update(
            psi_edge=values[mask].tolist(),
            psi_edge_public_direct_romberg=direct[mask].tolist(),
            backend_versions={"freegs": version("freegs")},
        )
    else:
        from freegsnke.GSstaticsolver import NKGSsolver  # type: ignore[import-untyped]

        from tools.constructive_model import make_equilibrium

        eq_native = make_equilibrium(args.size, 0, np.zeros(18))
        solver = cast(BoundarySolver, NKGSsolver(eq_native, seed=42, gs_operator_order=2))
        solver.freeboundary(eq_native.plasma_psi.ravel(), np.zeros_like(current).ravel(), PrescribedCurrent())
        out.update(
            psi_edge=solver.psi_boundary[mask].tolist(),
            backend_versions={p: version(p) for p in ["freegs", "freegs4e", "freegsnke"]},
        )
    out["seconds"] = time.monotonic() - started
    with args.output.open("x") as stream:
        stream.write(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, list)}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
