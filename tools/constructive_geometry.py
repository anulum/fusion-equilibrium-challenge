# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — shared synthetic limiter geometry
"""Use one equal-area repaired limiter in both numerical runtimes."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from shapely import make_valid
from shapely.geometry import Polygon

F64 = NDArray[np.float64]


def limiter_boundary(vertices: F64) -> F64:
    """Remove retraced limiter segments while preserving polygon area and holes.

    Parameters
    ----------
    vertices : ndarray
        Finite physical (R, Z) vertices of one limiter without holes.

    Returns
    -------
    ndarray
        Closed exterior of a valid single polygon with the same area.

    Raises
    ------
    ValueError
        Vertices are invalid or repair changes polygon count, holes or area.
    """
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 4:
        raise ValueError("At least four physical limiter vertices are required")
    if not np.isfinite(vertices).all() or np.any(vertices[:, 0] <= 0):
        raise ValueError("Limiter coordinates must be finite with positive radius")
    polygon = Polygon(vertices)
    repaired = make_valid(polygon)
    if (
        not isinstance(repaired, Polygon)
        or len(repaired.interiors)
        or abs(repaired.area - polygon.area) > 1e-12
    ):
        raise ValueError("The declared limiter repair changed area or topology")
    return np.asarray(repaired.exterior.coords, dtype=np.float64)
