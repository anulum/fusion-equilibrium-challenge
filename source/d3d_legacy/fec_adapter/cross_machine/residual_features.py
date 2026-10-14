# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — per-frame features for the learned correction

"""Build the per-frame features the learned correction is regressed on.

Two sources only: the inputs the Challenge allows (coil circuits, plasma and
toroidal-field current, the Thomson profiles — :mod:`fec_adapter.input_features`)
and what the physics solve
produced from them (its flux map, axis and boundary flux, confined
cross-section, how the frame was arrived at, the measured pressure). No
reconstruction column can reach this module: it takes a :class:`Shot`, which
the reader builds from input columns alone, and a
:class:`fec_adapter.physics_base.PhysicsBase`.

Every feature is finite by construction. A quantity that can be missing — a
boundary flux when no surface closed, a Thomson channel that did not fire — is
written as zero next to a flag that says so, because a linear model can use a
flag and cannot use a hole.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fec_adapter.input_features import (
    CURRENT_FLOOR,
    circuit_currents,
    phase_features,
    profile_summary,
)
from fec_adapter.physics_base import STATUS_ORDER, PhysicsBase
from fec_adapter.shot_reader import Shot

FIELD_SAMPLES = 5
"""The physics map is sampled on a ``FIELD_SAMPLES`` by ``FIELD_SAMPLES`` sub-grid."""

def _with_flag(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ``[value or 0, 1 if it was finite]`` per frame, shape ``(T, 2)``."""
    finite = np.isfinite(values)
    return np.column_stack([np.where(finite, values, 0.0), finite.astype(np.float64)])


def ratio_features(shot: Shot, base: PhysicsBase) -> NDArray[np.float64]:
    """Return the quantities the scalars depend on as ratios, shape ``(T, 8)``.

    The safety factor scales as the toroidal field over the plasma current
    times a shape factor, and the normalised beta as pressure times size and
    field over the current. A model that is linear in the currents cannot form
    a ratio of them, so the ratios are handed to it: measured on 64 held-out
    shots, a linear ``q95`` head went from R² 0.67 to 0.78 with these eight
    columns and nothing else changed.
    """
    magnitude = np.maximum(np.abs(shot.plasma_current), CURRENT_FLOOR)
    field_per_current = np.abs(shot.tf_current) / magnitude
    area = np.maximum(base.enclosed_area, 0.0)
    pressure = base.pressure.mean(axis=1)
    return np.column_stack([
        field_per_current,
        1.0e6 / magnitude,
        field_per_current * area,
        field_per_current * np.sqrt(area),
        pressure / magnitude**2 * 1.0e12,
        pressure * area / magnitude**2 * 1.0e12,
        pressure * np.abs(shot.tf_current) / magnitude**2 * 1.0e7,
        np.abs(shot.tf_current) / 1.0e5 * area,
    ])


OPTIONAL_BLOCKS = ("phase",)
"""Feature blocks a model may be fitted with or without; the rest are always present."""

DEFAULT_BLOCKS: tuple[str, ...] = ()
"""Optional blocks used when nothing is said. The phase block is not among them.

Measured twice with the scorer's own derived scalars, on sets that share no
shot. With the phase block the internal inductance improves (the ramp-down is
recognised) and so do elongation and volume a little, but the height of the
magnetic axis gets worse by more, and the two scalar heads do not gain:

============================  ==========  ==========
set                           without     with phase
============================  ==========  ==========
112 held-out shots, Consistency   0.5922      0.5806
  of which Z_axis                 0.540       0.338
  of which li                     0.571       0.620
20 reserved shots, composite S    0.9049      0.9022
============================  ==========  ==========

Capping the two time distances at
:data:`fec_adapter.input_features.PHASE_HORIZON` did not change that (0.9023
uncapped), so extrapolation on long shots — my first explanation — is not the
cause. Why the axis height suffers is not understood; until it is, the
block stays available and off.
"""


def frame_features(
    shot: Shot, base: PhysicsBase, blocks: tuple[str, ...] = DEFAULT_BLOCKS
) -> NDArray[np.float64]:
    """Return the feature matrix of one shot, shape ``(T, F)``, finite everywhere.

    ``blocks`` names the optional blocks to append. A fitted model records the
    blocks it was fitted with and asks for the same ones when it is applied, so
    adding a block to the package does not orphan the models fitted before it.
    """
    unknown = set(blocks) - set(OPTIONAL_BLOCKS)
    if unknown:
        raise ValueError(f"unknown feature blocks: {sorted(unknown)}")
    if base.psi.shape[0] != shot.n_frames:
        raise ValueError(f"base has {base.psi.shape[0]} frames, shot has {shot.n_frames}")
    rows = np.linspace(0, base.psi.shape[1] - 1, FIELD_SAMPLES + 2)[1:-1].round().astype(int)
    cols = np.linspace(0, base.psi.shape[2] - 1, FIELD_SAMPLES + 2)[1:-1].round().astype(int)
    field = base.psi[:, rows][:, :, cols].reshape(shot.n_frames, -1)
    status = np.eye(len(STATUS_ORDER))[base.status]
    features = np.column_stack([
        circuit_currents(shot),
        shot.plasma_current,
        shot.tf_current,
        profile_summary(shot.thomson.te),
        profile_summary(shot.thomson.ne),
        _with_flag(base.psi_axis),
        _with_flag(base.psi_boundary),
        base.enclosed_area,
        status,
        base.pressure[:, 0],
        base.pressure.mean(axis=1),
        base.pprime[:, 0],
        base.ffprime[:, 0],
        np.where(np.isfinite(field), field, 0.0),
        ratio_features(shot, base),
        *([phase_features(shot)] if "phase" in blocks else []),
    ])
    if not bool(np.all(np.isfinite(features))):
        raise ValueError(f"{shot.name}: non-finite input among the features")
    return features
