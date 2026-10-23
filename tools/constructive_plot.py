# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — recorded coarse comparison figure
"""Plot the original failed coarse gate without changing its gauge or domain."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 - backend selection precedes pyplot.

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "tools/reference/constructive_coarse_v4.npz"
RECORD = ROOT / "tools/reference/constructive_calculation.json"
BUNDLE_SHA256 = "f37c4c55d9ceb740a6819e0626cdb038dc30513e27f8caa62034a82d59a4efff"
RECORD_SHA256 = "fe651f8cee3990f3297e934e80cea787687e1beea404286116856ed5b17e23ad"
F64 = NDArray[np.float64]


@dataclass(frozen=True)
class Comparison:
    """Original R,Z-ordered error, fixed reference core and closed box-edge trace."""

    radius: F64
    height: F64
    difference: F64
    core: NDArray[np.bool_]
    edge_distance: F64
    edge_error: F64
    report: dict[str, object]


def _snapshot(path: Path, expected: str) -> bytes:
    with path.open("rb") as stream:
        raw = stream.read(2**20 + 1)
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Recorded comparison input differs")
    return raw


def load_comparison(bundle: Path = BUNDLE, record: Path = RECORD) -> Comparison:
    """Recompute the comparison from the two hash-pinned public inputs.

    Parameters
    ----------
    bundle, record : Path
        Original coarse arrays and calculation record. Substitutions are refused.

    Returns
    -------
    Comparison
        Whole-grid mean-gauge error divided by the original inverse flux span,
        inverse limiter-core mask and counterclockwise edge trace from lower left.

    Raises
    ------
    OSError, ValueError
        A public input is unavailable or its bytes differ from the original.
    """
    original = json.loads(_snapshot(record, RECORD_SHA256))
    with np.load(io.BytesIO(_snapshot(bundle, BUNDLE_SHA256)), allow_pickle=False) as data:
        radius, height = data["inverse__R_m"], data["inverse__Z_m"]
        raw = data["forward__psi_Wb_per_rad"] - data["inverse__psi_Wb_per_rad"]
        core = data["inverse__core_mask"] > 0.5
    gauge = float(raw.mean())
    span = abs(float(original["coarse_reports"]["inverse"]["flux_span_Wb_per_rad"]))
    error = (raw - gauge) / span
    nr, nz = len(radius), len(height)
    ri = np.r_[np.arange(nr), np.full(nz - 1, nr - 1), np.arange(nr - 2, -1, -1), np.zeros(nz - 1, int)]
    zi = np.r_[np.zeros(nr, int), np.arange(1, nz), np.full(nr - 1, nz - 1), np.arange(nz - 2, -1, -1)]
    distance = np.r_[0.0, np.cumsum(np.hypot(np.diff(radius[ri]), np.diff(height[zi])))]
    edge_error = np.abs(error[ri, zi])
    maximum = np.unravel_index(np.argmax(np.abs(error)), error.shape)
    report: dict[str, object] = {
        "schema": "constructive-coarse-figure.v1",
        "bundle_sha256": BUNDLE_SHA256,
        "record_sha256": RECORD_SHA256,
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "global_mean_gauge_Wb_per_rad": gauge,
        "reference_flux_span_Wb_per_rad": span,
        "whole_domain_max_ratio": float(np.max(np.abs(error))),
        "inverse_core_max_ratio": float(np.max(np.abs(error[core]))),
        "inverse_core_nodes": int(core.sum()),
        "core_definition": "Original inverse limiter_core_mask > 0.5; descriptive subset only.",
        "max_location_RZ_m": [float(radius[maximum[0]]), float(height[maximum[1]])],
        "edge_distance_m": distance.tolist(),
        "edge_absolute_error_ratio": edge_error.tolist(),
        "edge_path": (
            "Lower left, lower right, upper right, upper left, lower left; corners once except closure."
        ),
        "original_whole_domain_limit": 0.005,
        "original_gate_passed": False,
        "new_equilibrium_solve": False,
    }
    return Comparison(radius, height, error, core, distance, edge_error, report)


def render(comparison: Comparison, output: Path) -> None:
    """Write a two-panel PNG, SVG and exact numerical report to a new directory.

    Parameters
    ----------
    comparison : Comparison
        Original hash-bound fields and their unchanged normalization.
    output : Path
        New directory; existing contents are never overwritten.

    Raises
    ------
    OSError, ValueError
        The output exists, cannot be written, or the renderer refuses it.
    """
    output.mkdir(parents=True, exist_ok=False)
    with plt.rc_context({"svg.hashsalt": "fec-original-coarse-v1", "font.family": "DejaVu Sans"}):
        figure = plt.figure(figsize=(11.5, 5.4), layout="constrained")
        left, right = figure.add_subplot(121), figure.add_subplot(122)
        limit = float(np.max(np.abs(comparison.difference))) * 100
        image = left.pcolormesh(
            comparison.radius,
            comparison.height,
            comparison.difference.T * 100,
            shading="nearest",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        left.contour(comparison.radius, comparison.height, comparison.core.T, levels=[0.5], colors="black")
        left.set(xlabel="R (m)", ylabel="Z (m)", title="Signed flux difference; inverse core outlined")
        left.set_aspect("equal")
        figure.colorbar(image, ax=left, label="Difference / inverse flux span (%)", shrink=0.8)
        right.plot(comparison.edge_distance, comparison.edge_error * 100, color="#235789")
        right.axhline(0.5, color="#a32525", linestyle="--", label="Original whole-domain limit: 0.5%")
        right.set(
            xlabel="Distance along box edge from lower-left corner (m)",
            ylabel="Absolute flux difference / inverse span (%)",
            title="Error along the closed computational boundary",
            xlim=(0, float(comparison.edge_distance[-1])),
            ylim=(0, limit * 1.12),
        )
        right.grid(alpha=0.2)
        right.legend(loc="lower right", fontsize=8)
        figure.suptitle("Original coarse inverse/forward comparison — initial 0; whole-domain gate failed")
        figure.savefig(
            output / "coarse_difference.png",
            dpi=180,
            metadata={
                "Author": "Anulum Fortis & Arcane Sapience",
                "Description": "One global mean-gauge correction; original reference span and gate.",
            },
        )
        figure.savefig(
            output / "coarse_difference.svg",
            metadata={"Creator": "Anulum Fortis & Arcane Sapience", "Date": None},
        )
        plt.close(figure)
    vector = output / "coarse_difference.svg"
    vector.write_text("\n".join(line.rstrip() for line in vector.read_text().splitlines()) + "\n")
    (output / "coarse_difference.json").write_text(
        json.dumps(comparison.report, indent=2, allow_nan=False) + "\n"
    )


def main() -> int:
    """Render the original comparison, or refuse an invalid input/output with exit two.

    Returns
    -------
    int
        Zero after export, two when inputs or the requested output are refused.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--record", type=Path, default=RECORD)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        comparison = load_comparison(args.bundle, args.record)
        render(comparison, args.out)
    except (OSError, ValueError):
        print("Unable to render the recorded coarse comparison.")
        return 2
    print(json.dumps(comparison.report, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
