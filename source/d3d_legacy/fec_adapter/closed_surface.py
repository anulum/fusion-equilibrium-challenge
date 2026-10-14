# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — the last closed flux surface of a map

"""Find the last closed flux surface of a map without looking for X-points."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

BISECTIONS = 22


def closed_level(
    oriented: NDArray[np.float64], interior: NDArray[np.bool_], axis: int
) -> tuple[NDArray[np.bool_], float] | None:
    """Return the largest region around the axis whose flux surfaces stay inside, and its level.

    ``oriented`` has its maximum at the axis. The level is found by bisection: the region above
    a level either stays inside the interior (closed surfaces) or leaks to the wall — through an
    X-point or past a limiter, it makes no difference here.
    """
    high = float(oriented.ravel()[axis])
    low = float(oriented[interior].min())
    found = None
    for _ in range(BISECTIONS):
        level = 0.5 * (high + low)
        labels, _count = ndimage.label(oriented > level)
        label = labels.ravel()[axis]
        region = labels == label
        if label != 0 and not np.any(region & ~interior):
            found, high = region, level
        else:
            low = level
    return None if found is None else (found, high)
