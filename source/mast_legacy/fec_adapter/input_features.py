# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — per-frame quantities taken from the allowed inputs alone

"""Per-frame quantities built from the inputs the Challenge allows, and from nothing else.

Coil circuits, the plasma and toroidal-field current, the Thomson profiles and
the time base. Both entries draw on this module — the one that learns the
DIII-D reconstruction from the inputs directly and the one that corrects a
physics solve — which is why it sits beside the reader and not inside either.
It takes a :class:`Shot`, which the reader builds from input columns alone, so
no reconstruction column can reach it.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .shot_reader import Shot

CURRENT_FLOOR = 5.0e4
"""Amperes below which the plasma current is held in the ratio features.

The ratios divide by the plasma current, and the first frames of a discharge
carry a few kiloamperes of either sign or none at all. A floor of 50 kA — a
tenth of the smallest flat-top current in the dataset — keeps those frames
finite and of ordinary size without touching any frame that has a plasma.
"""

PHASE_HORIZON = 1.0
"""Seconds beyond which the distance to either end of the discharge stops counting.

The current ramps up and down within about a second; further from either end
the distance says nothing about the phase, only about how long the shot is.
"""


def circuit_currents(shot: Shot) -> NDArray[np.float64]:
    """Return the ampere-turns of each coil circuit, shape ``(T, N_circuits)``.

    Filaments that draw from one current column are one circuit; the columns
    are taken in order of first appearance, which is the machine's own order
    and therefore the same for every shot of a machine.
    """
    columns = list(dict.fromkeys(shot.coils.columns))
    membership = np.asarray(
        [[column == circuit for column in shot.coils.columns] for circuit in columns],
        dtype=np.float64,
    )
    summed: NDArray[np.float64] = shot.coil_current @ membership.T
    return summed


def profile_summary(samples: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ``[peak, mean, share of channels that fired]`` per frame, shape ``(T, 3)``."""
    fired = np.isfinite(samples) & (samples > 0.0)
    count = fired.sum(axis=1)
    kept = np.where(fired, samples, 0.0)
    mean = kept.sum(axis=1) / np.maximum(count, 1)
    return np.column_stack([kept.max(axis=1), mean, count / samples.shape[1]])


def phase_features(shot: Shot) -> NDArray[np.float64]:
    """Return where each frame of ``shot`` sits in its discharge; see :func:`discharge_phase`."""
    return discharge_phase(shot.times_ms, shot.plasma_current)


def discharge_phase(
    times_ms: NDArray[np.float64], plasma_current: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return where each frame sits in its discharge, shape ``(T, 5)``.

    Columns: the plasma current as a fraction of the shot's largest, its rate of
    change per second as a fraction of that largest current, the seconds since
    the first frame and to the last frame — both capped at
    :data:`PHASE_HORIZON` — and the elapsed fraction of the shot. All of it comes
    from the current trace and the time base, which
    are inputs for the whole shot.

    Why: without these a frame of the current ramp-down looks to the model like
    any other frame at that current. On twenty unseen shots the last twentieth
    of each discharge carried 30 % of the pooled internal-inductance error —
    the reconstruction's value climbs to 4.5 there while the model stopped at
    2.1 — from fewer than 5 % of the frames.
    """
    seconds = (times_ms - times_ms[0]) / 1.0e3
    magnitude = np.abs(plasma_current)
    largest = max(float(magnitude.max()), CURRENT_FLOOR)
    rate = np.gradient(magnitude, seconds) if times_ms.size > 1 else np.zeros(times_ms.size)
    duration = max(float(seconds[-1]), 1.0e-9)
    return np.column_stack([
        magnitude / largest,
        rate / largest,
        np.minimum(seconds, PHASE_HORIZON),
        np.minimum(seconds[-1] - seconds, PHASE_HORIZON),
        seconds / duration,
    ])
