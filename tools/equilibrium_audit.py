# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — public external-map audit command
"""Audit explicitly declared maps without importing an entrant's model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Never

import numpy as np

from tools.audit_map_input import InputRefusal, MapData, RefusalCode, load_map, load_region
from tools.field_diagnostics import measure

REFUSAL = "Map audit refused: check input files, units, grid and numerical range."


class Parser(argparse.ArgumentParser):
    """Report malformed command arguments through the same safe caller path."""

    def error(self, message: str) -> Never:
        """Refuse invalid arguments without printing interpreter-generated text."""
        raise InputRefusal("INVALID_ARGUMENT", REFUSAL)


def _identity(frame: MapData) -> dict[str, object]:
    return {
        "snapshot_sha256": frame.source_sha256,
        "input_length_unit": frame.length_unit,
        "input_flux_unit": frame.flux_unit,
        "axis_sign": frame.axis_sign,
        "axis_origin": frame.axis_origin,
        "axis_flux_Wb_per_rad": frame.axis,
        "boundary_flux_Wb_per_rad": frame.boundary,
        "plasma_nodes": None if frame.plasma is None else int(frame.plasma.sum()),
    }


def main(argv: list[str] | None = None) -> int:
    """Write a deterministic audit, or a coded refusal with fixed text and exit two.

    Parameters
    ----------
    argv : list of str or None
        Explicit command arguments; None reads the process argument vector.

    Returns
    -------
    int
        Zero for a completed audit, two for invalid input or numerical range.
        Unavailable diagnostics within a valid audit remain null with reasons.
    """
    parser = Parser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--length-unit", metavar="{m,cm,mm}", required=True)
    parser.add_argument("--flux-unit", metavar="{Wb/rad,Wb-total}", required=True)
    parser.add_argument("--axis-sign", type=int, choices=(-1, 1), required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-axis-sign", type=int, choices=(-1, 1))
    parser.add_argument("--comparison-region", type=Path)
    parser.add_argument("--measured-current-a", type=float)
    try:
        args = parser.parse_args(argv)
        if args.reference_axis_sign is not None and args.reference is None:
            raise InputRefusal("INVALID_ARGUMENT", REFUSAL)
        frame = load_map(args.map, args.length_unit, args.flux_unit, args.axis_sign)
        region = load_region(args.comparison_region, frame)
        reference = (
            None
            if args.reference is None
            else load_map(
                args.reference,
                args.length_unit,
                args.flux_unit,
                args.axis_sign if args.reference_axis_sign is None else args.reference_axis_sign,
            )
        )
        report = {
            "schema": "equilibrium-audit.v1",
            "physical_force_balance_certified": False,
            "competition_score_computed": False,
            "input": _identity(frame),
            "reference": None if reference is None else _identity(reference),
            "grid": {
                "shape": list(frame.phi.shape),
                "length_unit": "m",
                "dR": float(np.diff(frame.radius).mean()),
                "dZ": float(np.diff(frame.height).mean()),
                "relative_spacing_deviation_R": float(
                    np.max(np.abs(np.diff(frame.radius) / np.diff(frame.radius).mean() - 1))
                ),
                "relative_spacing_deviation_Z": float(
                    np.max(np.abs(np.diff(frame.height) / np.diff(frame.height).mean() - 1))
                ),
            },
            "region": {
                "policy": region.policy,
                "identity": region.identity,
                "snapshot_sha256": region.source_sha256,
                "requested_nodes": int(region.mask.sum()),
                "definition": "Fixed mask or all grid nodes, intersected with central-stencil support.",
            },
            "operators": {
                "version": "central-five-point.v1",
                "flux": "phi = axis_sign * input flux per radian",
                "field": "BR=-phi_Z/R; BZ=phi_R/R",
                "source": "Delta*=phi_RR-phi_R/R+phi_ZZ",
                "current": "Jphi=-Delta*/(mu0*R); node-area quadrature",
                "mu0_H_per_m": 4 * np.pi * 1e-7,
                "polynomial_basis": "[R^2*psiN^k, psiN^k], k=0..3; psiN=(axis-phi)/(axis-boundary)",
                "surface_weighting": "marching-squares vertices; own-map surfaces",
            },
            "diagnostics": measure(frame, region, reference, args.measured_current_a),
        }
        output = json.dumps(report, sort_keys=True, indent=2, allow_nan=False)
    except InputRefusal as exc:
        code: RefusalCode = exc.code
    except OSError:
        code = "INPUT_UNREADABLE"
    except (FloatingPointError, OverflowError):
        code = "NUMERIC_RANGE_EXCEEDED"
    except Exception:
        code = "AUDIT_REFUSED"
    else:
        print(output)
        return 0
    print(
        json.dumps(
            {"schema": "equilibrium-audit-refusal.v1", "status": "refused", "code": code, "message": REFUSAL},
            sort_keys=True,
        )
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
