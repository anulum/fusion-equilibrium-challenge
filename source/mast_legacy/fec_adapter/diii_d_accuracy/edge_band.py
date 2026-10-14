# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — where on the grid the plasma boundary usually runs

"""Map how often each grid cell lies near the last closed flux surface, over training shots.

A fifth of the score is read off the boundary of the predicted map — elongation,
triangularities, volume — and a tenth is the boundary's distance from the true
one. The fit's loss is the squared map error over the whole grid, where the band
the boundary runs through is a small share of the cells. This module measures
that band: for a sample of training frames it finds the last closed surface of
the true map and marks the cells whose normalised flux lies within
:data:`BAND` of it. The mean over frames is a weight map a fit can use to count
errors there more (:mod:`.fit_on_accelerators`, ``edge_emphasis``).

Only the training part of the fixed split is read, without the shots reserved
for evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from fec_adapter import split
from fec_adapter.closed_surface import closed_level
from fec_adapter.vessel import envelope

BAND = (0.8, 1.1)
"""Normalised flux, from inside the boundary to just outside it, that counts as near it."""


def near_boundary(oriented: NDArray[np.float64], interior: NDArray[np.bool_]) -> (
    NDArray[np.bool_] | None
):
    """Return the cells within :data:`BAND` of the last closed surface, or ``None`` without one.

    ``oriented`` has its maximum at the magnetic axis.
    """
    axis = int(np.argmax(np.where(interior, oriented, -np.inf)))
    top = float(oriented.ravel()[axis])
    closed = closed_level(oriented, interior, axis)
    if closed is None or top - closed[1] <= 0.0:
        return None
    found = closed[1]
    depth = (top - oriented) / (top - found)
    band: NDArray[np.bool_] = (depth > BAND[0]) & (depth < BAND[1]) & interior
    return band


def main() -> int:
    """Average the band over a sample of training frames and store it beside the tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted", type=Path, required=True)
    parser.add_argument("--reserved", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="the .npy file to write")
    parser.add_argument("--every-shot", type=int, default=10)
    parser.add_argument("--every-frame", type=int, default=8)
    args = parser.parse_args()

    reserved = frozenset(json.loads(args.reserved.read_text())["stems"])
    files = sorted(
        path for path in args.extracted.glob("*_shot_*.npz")
        if split.bucket_of(path.stem) == "training" and path.stem not in reserved
    )[:: args.every_shot]
    total, frames = None, 0
    for path in files:
        with np.load(path, allow_pickle=False) as stored:
            maps = stored["psi"][:: args.every_frame].astype(np.float64)
        interior = ndimage.binary_erosion(envelope("DIII-D"), iterations=1)
        for one in maps:
            band = near_boundary(-one, interior)  # stored with the axis at the minimum
            if band is None:
                continue
            total = band.astype(np.float64) if total is None else total + band
            frames += 1
    if total is None:
        raise SystemExit("no frame with a closed surface")
    np.save(args.out, total / frames)
    print(json.dumps({"shots": len(files), "frames": frames,
                      "cells_ever_in_the_band": int((total > 0).sum()),
                      "mean_share_of_the_grid": float((total / frames).mean())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
