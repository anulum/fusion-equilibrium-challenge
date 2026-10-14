# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — flux smoothing on the actual observation clock
"""Smooth flux coordinates within contiguous observation intervals."""

from __future__ import annotations

import numpy as np
from fec_adapter.temporal import smooth_in_time
from numpy.typing import NDArray


def smooth_observation_times(
    fields: NDArray[np.float64],
    times: NDArray[np.float64],
    half_window: int,
) -> NDArray[np.float64]:
    """Apply triangular local linear smoothing without bridging long gaps.

    Parameters
    ----------
    fields:
        Observations, with time on the first axis.
    times:
        Strictly increasing finite observation times in any consistent unit.
    half_window:
        Nominal half-window in median observation intervals.

    Returns
    -------
    numpy.ndarray
        Smoothed observations; regular segments retain the original algorithm.

    Notes
    -----
    Gaps above three median intervals split the record. Within each segment,
    a triangular kernel has radius ``(half_window + 1) * median_interval``.
    Odd reflection preserves endpoint trends. Local linear regression preserves
    affine trends even on an irregular clock. No target values set these choices.
    """
    values = np.asarray(fields, dtype=np.float64)
    clock = np.asarray(times, dtype=np.float64)
    if half_window < 0 or clock.ndim != 1 or len(clock) != len(values):
        raise ValueError("Invalid smoothing window or time shape")
    if not np.isfinite(clock).all() or np.any(np.diff(clock) <= 0):
        raise ValueError("Observation times must be finite and strictly increasing")
    if half_window == 0 or len(clock) < half_window + 2:
        return values.copy()
    cadence = float(np.median(np.diff(clock)))
    breaks = np.flatnonzero(np.diff(clock) > 3 * cadence) + 1
    bounds = np.r_[0, breaks, len(clock)]
    result = values.copy()
    for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
        segment = values[start:stop]
        t = clock[start:stop]
        if len(t) < half_window + 2:
            continue
        step = float(np.median(np.diff(t)))
        if np.allclose(np.diff(t), step, rtol=1e-10, atol=1e-10 * step):
            result[start:stop] = smooth_in_time(segment, half_window)
            continue
        radius = (half_window + 1) * step
        padded_t = np.r_[2 * t[0] - t[1:][::-1], t, 2 * t[-1] - t[:-1][::-1]]
        padded_y = np.concatenate(
            (
                2 * segment[:1] - segment[1:][::-1],
                segment,
                2 * segment[-1:] - segment[:-1][::-1],
            )
        )
        for index, centre in enumerate(t):
            selected = np.abs(padded_t - centre) < radius
            x = (padded_t[selected] - centre) / radius
            weights = 1 - np.abs(x)
            s0, s1, s2 = weights.sum(), np.dot(weights, x), np.dot(weights, x * x)
            determinant = s0 * s2 - s1 * s1
            if determinant <= np.finfo(float).eps * s0:
                continue
            equivalent = weights * (s2 - s1 * x) / determinant
            result[start + index] = np.tensordot(
                equivalent, padded_y[selected], axes=(0, 0)
            )
    return result
