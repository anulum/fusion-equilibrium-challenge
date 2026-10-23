# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — calibrated map-audit examples
"""Measure fixed analytic controls and reference-assisted public illustrations."""

from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import BinaryIO, cast

import numpy as np
from matplotlib.path import Path as Polygon
from numpy.lib import format as npformat
from numpy.typing import NDArray

from tools.audit_map_input import F64, MapData, Region
from tools.equilibrium_audit import Parser
from tools.field_diagnostics import measure
from tools.make_preview import bilinear

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments" / "audit_calibration"
FRAMES = ROOT / "tools" / "reference" / "d71_figure_frames.npz"
REFUSAL = "Calibration refused: check the published manifest and illustration files."
Header = tuple[tuple[int, ...], bool, np.dtype[np.generic]]


def _snapshot(path: Path, expected: object) -> dict[str, NDArray[np.generic]]:
    with path.open("rb") as stream:
        raw = stream.read(2**21 + 1)
    if len(raw) > 2**21 or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Published illustration checksum or size differs")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        if (
            len(members) > 64
            or len({item.filename for item in members}) != len(members)
            or sum(item.file_size for item in members) > 2**22
        ):
            raise ValueError("Invalid illustration archive members")
        readers = {(1, 0): npformat.read_array_header_1_0, (2, 0): npformat.read_array_header_2_0}
        for item in members:
            with archive.open(item) as stream:
                version = npformat.read_magic(stream)
                if version not in readers:
                    raise ValueError("Unsupported illustration array version")
                reader = cast(Callable[[BinaryIO], Header], readers[version])
                shape, _fortran, dtype = reader(cast(BinaryIO, stream))
            if math.prod(shape) > 2**20 or dtype.kind not in "fiuU" or dtype.itemsize > 80:
                raise ValueError("Unsupported illustration array shape or dtype")
    with np.load(io.BytesIO(raw), allow_pickle=False) as stored:
        return {key: cast(NDArray[np.generic], stored[key]) for key in stored.files}


def _frame(radius: F64, height: F64, phi: F64, plasma: NDArray[np.bool_] | None, boundary: float) -> MapData:
    selected = phi if plasma is None else phi[plasma]
    identity = hashlib.sha256(phi.astype("<f8").tobytes()).hexdigest()
    return MapData(
        radius,
        height,
        phi,
        plasma,
        float(selected.max()),
        boundary,
        "grid-extremum",
        "m",
        "Wb/rad",
        1,
        identity,
    )


def _region(frame: MapData, mask: NDArray[np.bool_], policy: str) -> Region:
    payload = (
        policy.encode()
        + frame.radius.astype("<f8").tobytes()
        + frame.height.astype("<f8").tobytes()
        + mask.tobytes()
    )
    return Region(mask, policy, hashlib.sha256(payload).hexdigest(), None)


def analytic() -> list[dict[str, object]]:
    """Measure three grid resolutions, two fixed deformations and unavailable cases.

    Returns
    -------
    list of dict
        Known continuous/discrete controls with complete measured diagnostics.
        The source RMS discrepancy compares two norms; the separately stated
        constant pointwise stencil error follows from the analytic quartic.
    """
    cases: list[dict[str, object]] = []
    for size in (33, 65, 129):
        radius, height = np.linspace(0.84, 2.54, size), np.linspace(-1.6, 1.6, size)
        rr, zz = np.meshgrid(radius, height)
        phi = -((rr**2 - 1.7**2) ** 2) / 8 - 0.9 * zz**2
        frame = _frame(radius, height, phi, phi >= -0.3, -0.3)
        frame = replace(frame, axis=0.0, axis_origin="supplied")
        region = _region(frame, np.ones(phi.shape, dtype=bool), "calibration-grid-interior.v1")
        result = measure(frame, region, frame)
        source = rr[1:-1, 1:-1] ** 2 + 1.8
        spacing = float(np.diff(radius).mean())
        continuous_rms = float(np.sqrt(np.mean(source**2)))
        observed = cast(dict[str, object], result["field_on_comparison_region"])
        cases.append(
            {
                "case": "quartic_" + str(size),
                "grid_shape": list(phi.shape),
                "known_continuous_source_rms_T": continuous_rms,
                "known_discrete_source_rms_T": float(np.sqrt(np.mean((source - spacing**2 / 4) ** 2))),
                "known_pointwise_source_error_T": spacing**2 / 4,
                "measured_source_rms_relative_difference": abs(
                    cast(float, observed["source_rms_T"]) / continuous_rms - 1
                ),
                "diagnostics": result,
            }
        )
        if size == 65:
            for alpha in (0.1, 0.4):
                altered = _frame(radius, height, phi + alpha * (rr - 1.7) ** 2 * zz**2, frame.plasma, -0.3)
                cases.append(
                    {
                        "case": "deformation_" + str(alpha),
                        "alpha": alpha,
                        "diagnostics": measure(altered, region, frame),
                    }
                )
            absent = _frame(radius, height, phi, None, -0.3)
            cases.append({"case": "missing_mask", "diagnostics": measure(absent, region)})
            opened = _frame(radius, height, rr, np.ones(phi.shape, dtype=bool), 0)
            cases.append({"case": "open_surfaces", "diagnostics": measure(opened, region)})
    radius, height = np.arange(1.0, 10.0), np.arange(-4.0, 5.0)
    rr, _zz = np.meshgrid(radius, height)
    vacuum = _frame(radius, height, rr**2, np.ones(rr.shape, dtype=bool), -100)
    region = _region(vacuum, np.ones(rr.shape, dtype=bool), "calibration-grid-interior.v1")
    cases.append({"case": "zero_source", "diagnostics": measure(vacuum, region)})
    return cases


def illustrations(manifest: Path, baseline: Path, frames: Path) -> list[dict[str, object]]:
    """Compare all four public frames under identical reference-dependent geometry.

    Parameters
    ----------
    manifest : Path
        Published fit manifest with both payload checksums and frame identities.
    baseline, frames : Path
        Small checksum-bound baseline and previously published D71 illustrations.

    Returns
    -------
    list of dict
        Twelve map audits. These biased illustrations are not a benchmark sample,
        a predicted boundary test, or an independent external replication.

    Raises
    ------
    ValueError
        Payload identity, finite maps or declared example metadata differ.
    """
    plan = cast(dict[str, object], json.loads(manifest.read_text()))
    examples = cast(dict[str, dict[str, object]], plan["examples"])
    training = cast(list[str], plan["training_shots"])
    excluded = cast(list[str], plan["excluded_shots"])
    if (
        not training
        or set(training) & set(excluded)
        or not {str(x["stem"]) for x in examples.values()} <= set(excluded)
    ):
        raise ValueError("Illustration training and evaluation shots must be disjoint")
    if set(examples) != {"p50", "p90", "p99", "max"}:
        raise ValueError("All four original illustrations are required")
    reports: list[dict[str, object]] = []
    base = _snapshot(baseline, plan["baseline_frames_sha256"])
    old = _snapshot(frames, plan["reference_frames_sha256"])
    if not base or not old:
        raise ValueError("Empty illustration archive")
    radius, height = old["R"].astype(np.float64), old["Z"].astype(np.float64)
    rr, zz = np.meshgrid(radius, height)
    for key, item in examples.items():
        if str(old[key + "_stem"]) != item["stem"] or float(old[key + "_time_ms"]) != item["time_ms"]:
            raise ValueError("Published frame identity differs")
        contour = old[key + "_reference_lcfs"].astype(np.float64)
        mask = Polygon(contour).contains_points(np.c_[rr.ravel(), zz.ravel()]).reshape(rr.shape)
        truth = old[key + "_reference_psi"].astype(np.float64)
        boundary = float(np.mean(bilinear(radius, height, truth, contour)))
        sign = 1 if abs(truth[mask].max() - boundary) > abs(truth[mask].min() - boundary) else -1
        reference = _frame(radius, height, sign * truth, mask, sign * boundary)
        region = _region(reference, mask, "calibration-reference-plasma.v1")
        for name, phi in (
            ("reference", truth),
            ("entry_after_finisher", old[key + "_margin_psi"].astype(np.float64)),
            ("official_ridge", base[key + "_psi"].astype(np.float64)),
        ):
            if phi.shape != truth.shape or not np.isfinite(phi).all():
                raise ValueError("Expected a finite co-registered illustration")
            own_boundary = float(np.mean(bilinear(radius, height, phi, contour)))
            candidate = _frame(radius, height, sign * phi, mask, sign * own_boundary)
            reports.append(
                {
                    "case": key,
                    "map": name,
                    "stem": item["stem"],
                    "time_ms": item["time_ms"],
                    "axis_sign_from_reference": sign,
                    "reference_assisted_domain": True,
                    "region_policy": region.policy,
                    "region_identity": region.identity,
                    "requested_nodes": int(mask.sum()),
                    "diagnostics": measure(candidate, region, reference),
                }
            )
    return reports


def main(argv: list[str] | None = None) -> int:
    """Write calibrated diagnostics without fitting a model or changing input files."""
    parser = Parser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DATA / "baseline_manifest_v1.json")
    parser.add_argument("--baseline", type=Path, default=DATA / "baseline_frames_v1.npz")
    parser.add_argument("--frames", type=Path, default=FRAMES)
    try:
        args = parser.parse_args(argv)
        report = {
            "schema": "audit-calibration.v1",
            "physical_force_balance_certified": False,
            "competition_score_computed": False,
            "external_independent_replication": False,
            "analytic": analytic(),
            "illustrations": illustrations(args.manifest, args.baseline, args.frames),
        }
        print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    except Exception:
        print(json.dumps({"status": "refused", "message": REFUSAL}, sort_keys=True))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
