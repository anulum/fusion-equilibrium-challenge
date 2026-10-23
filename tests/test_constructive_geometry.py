# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — physical current-profile controls
"""Check equal-area limiter repair and refusal of topology changes."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

from tools.constructive_geometry import limiter_boundary


def test_limiter_retracing_and_topology() -> None:
    """Remove real retracing while refusing ambiguous multipart boundaries."""
    retraced = np.asarray([[1, 0], [1, 0.5], [1, 0.3], [1, 1], [2, 1], [2, 0], [1, 0]], float)
    result = limiter_boundary(retraced)
    assert Polygon(result).is_valid
    assert Polygon(result).area == 1
    for invalid in (
        np.zeros(4),
        np.zeros((3, 2)),
        np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], float),
        np.asarray([[1, 0], [2, 1], [1, 1], [2, 0]], float),
    ):
        with pytest.raises(ValueError):
            limiter_boundary(invalid)
