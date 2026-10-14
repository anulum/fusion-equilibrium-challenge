# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — per-frame features of the DIII-D accuracy entry

"""Turn a shot's aligned inputs into the per-frame features the learned model reads.

One function serves fitting and prediction. Fitting reads the aligned inputs the
extraction stored; prediction takes them from a shot the reader has just read;
both arrive here as :class:`AlignedInputs`, so the two cannot drift apart.

What the features say, and why each block is there:

oriented currents
    The reconstruction writes every flux map with the magnetic axis at its
    minimum, whichever way the plasma current runs, so the map follows the coil
    currents *times* the shot's orientation factor. Multiplying the currents by
    that factor makes one relation hold for both polarities, instead of asking
    the model to learn it twice, once mirrored.
rates of change
    Changing currents drive currents in the vessel, which the reconstruction
    sees and the inputs do not contain. The rate is the input that stands for
    them. Frames are 20 ms apart, so a difference over neighbouring frames is a
    fair rate.
currents over the plasma current
    The shape of an equilibrium depends on the coil currents relative to the
    plasma current, not on either alone.
discharge phase
    A frame of the ramp-down otherwise looks like any frame at that current.
Thomson profiles on fixed positions
    The core laser is vertical and the other one radial; the channels that fire
    and even their number change between campaigns. Averaging the channels into
    fixed height and radius bins gives every shot the same columns, and tells
    the model how far the hot plasma reaches — which is what the axis height
    and the elongation follow.

the same quantities a little earlier and a little later
    A frame alone does not say how the discharge came to it, and part of what
    the reconstruction reports is history: the current profile diffuses over
    hundreds of milliseconds, so the internal inductance at one instant follows
    the currents and temperatures of the instants before it. The whole shot is
    an input, so each frame also gets the mean of the slowly varying blocks —
    oriented currents and the Thomson profiles — over a short window around it
    and over two windows before it. The means count frames of whatever time
    base the inputs are on. On the reconstruction frames alone — the layout of
    every network fitted before 22 Sep 2026 — what happened before a record's
    first frame is not seen, and a record that is only the end of a ramp-down
    gets 200 ms of "history" where a longer record has a second of flat top.
    On the discharge record's base (:mod:`.history`) the same means reach back
    into the discharge before the record and across the gaps inside it, at one
    instant every 20 ms, and the phase columns are measured against the
    discharge (:class:`fec_adapter.discharge.DischargeWindow`).

Tried and left out: the vertical laser's channels stand 6 to 7 mm apart between 0.53 m
and 0.76 m, where the plasma edge crosses it, and the 8 cm bins average that away. A
block with the heights at which the temperature first reaches 50, 100 and 200 eV and
that stretch in 1.5 cm bins (84 columns) was fitted five times beside five fits without
it, same seeds and settings: validation flux R² 0.99983, 0.99976, 0.99976, 0.99975 and
one failed fit with it, against 0.99983, 0.99983, 0.99982, 0.99974, 0.99973 without;
the means of four were worse in every band of flux coordinates (0–4: 544 against 457).

Every feature is finite: a bin in which no channel fired is a zero beside a
flag that says so, and a shot whose currents are not finite is refused, not
patched.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fec_adapter.discharge import DischargeWindow
from fec_adapter.input_features import (
    CURRENT_FLOOR,
    circuit_currents,
    discharge_phase,
    profile_summary,
)
from fec_adapter.orientation import flux_orientation
from fec_adapter.shot_reader import Shot

HEIGHT_EDGES = np.linspace(-0.10, 0.86, 13)
"""Metres; bins of 8 cm covering the vertical laser's channels (measured span -0.07 to 0.83)."""

RADIUS_EDGES = np.linspace(1.65, 2.31, 7)
"""Metres; bins of 11 cm covering the radial laser's channels (measured span 1.67 to 2.27)."""

VERTICAL_TOLERANCE = 0.02
"""Metres within which a channel's radius counts as lying on the vertical laser."""

AROUND = 4
"""Frames either side of a frame in its centred mean: ±80 ms at 20 ms a frame."""

BEFORE = (16, 64)
"""Frames before a frame, itself included, in its two trailing means: 0.32 s and 1.3 s."""


@dataclass(frozen=True)
class AlignedInputs:
    """One shot's inputs on one time base, as plain arrays.

    The base is the reconstruction frames, or the discharge record's instants
    around and between them (:mod:`.history`).

    Attributes
    ----------
    times_ms:
        The instants, in milliseconds, shape ``(T,)``.
    circuits:
        Ampere-turns of each coil circuit, shape ``(T, C)``, in the machine's order.
    plasma_current, tf_current:
        Amperes, shape ``(T,)``.
    thomson_r, thomson_z:
        Channel positions in metres, shape ``(N,)``.
    thomson_te, thomson_ne:
        Electron temperature in eV and density in m⁻³, shape ``(T, N)``.
    """

    times_ms: NDArray[np.float64]
    circuits: NDArray[np.float64]
    plasma_current: NDArray[np.float64]
    tf_current: NDArray[np.float64]
    thomson_r: NDArray[np.float64]
    thomson_z: NDArray[np.float64]
    thomson_te: NDArray[np.float64]
    thomson_ne: NDArray[np.float64]

    @classmethod
    def from_shot(cls, shot: Shot) -> AlignedInputs:
        """Take the inputs from a shot the reader has aligned."""
        return cls(
            shot.times_ms, circuit_currents(shot), shot.plasma_current, shot.tf_current,
            shot.thomson.r, shot.thomson.z, shot.thomson.te, shot.thomson.ne,
        )

    @classmethod
    def from_stored(cls, stored: Mapping[str, NDArray[np.generic]]) -> AlignedInputs:
        """Take the inputs from a file the extraction wrote."""
        return cls(*(
            np.asarray(stored[name], dtype=np.float64)
            for name in (
                "times_ms", "circuits", "plasma_current", "tf_current",
                "thomson_r", "thomson_z", "thomson_te", "thomson_ne",
            )
        ))


def _rate(values: NDArray[np.float64], seconds: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the rate of change per second along time; zero for a single frame."""
    if seconds.size < 2:  # noqa: PLR2004 - a rate needs two instants
        return np.zeros_like(values)
    rate: NDArray[np.float64] = np.gradient(values, seconds, axis=0)
    return rate


def binned_profile(
    samples: NDArray[np.float64],
    positions: NDArray[np.float64],
    member: NDArray[np.bool_],
    edges: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Average the fired channels of one laser into fixed position bins.

    Returns ``(means, fired)``, both ``(T, bins)``: the mean over the channels
    of a bin that fired in that frame, zero where none did, and the flag.
    """
    bins = edges.size - 1
    which = np.digitize(positions, edges) - 1
    usable = member & (which >= 0) & (which < bins)
    fired = np.isfinite(samples) & (samples > 0.0) & usable[np.newaxis, :]
    kept = np.where(fired, samples, 0.0)
    assignment = np.zeros((bins, positions.size))
    assignment[which[usable], np.flatnonzero(usable)] = 1.0
    totals = kept @ assignment.T
    counts = fired.astype(np.float64) @ assignment.T
    means = totals / np.maximum(counts, 1.0)
    return means, (counts > 0.0).astype(np.float64)


def thomson_features(inputs: AlignedInputs) -> NDArray[np.float64]:
    """Return both lasers' profiles on fixed bins, with flags and overall summaries."""
    te_kev, ne_19 = inputs.thomson_te / 1.0e3, inputs.thomson_ne / 1.0e19
    if inputs.thomson_r.size == 0:
        vertical = np.zeros(0, dtype=bool)
    else:
        on_laser = np.abs(inputs.thomson_r - np.median(inputs.thomson_r)) < VERTICAL_TOLERANCE
        vertical = np.asarray(on_laser, dtype=bool)
    blocks = []
    for member, positions, edges in (
        (vertical, inputs.thomson_z, HEIGHT_EDGES),
        (~vertical, inputs.thomson_r, RADIUS_EDGES),
    ):
        te_mean, fired = binned_profile(te_kev, positions, member, edges)
        ne_mean, _ = binned_profile(ne_19, positions, member, edges)
        blocks += [te_mean, ne_mean, fired]
    if inputs.thomson_r.size:
        blocks += [profile_summary(te_kev), profile_summary(ne_19)]
    else:
        blocks += [np.zeros((inputs.times_ms.size, 6))]
    return np.column_stack(blocks)


def window_mean(values: NDArray[np.float64], before: int, after: int) -> NDArray[np.float64]:
    """Return per frame the mean over ``before`` frames back to ``after`` frames ahead.

    The window is cut where the shot begins or ends, so the first frame's trailing
    mean is that frame and nothing is invented beyond the record.
    """
    frames = values.shape[0]
    running = np.concatenate([np.zeros((1, *values.shape[1:])), np.cumsum(values, axis=0)])
    index = np.arange(frames)
    first = np.maximum(index - before, 0)
    last = np.minimum(index + after + 1, frames)
    span = (last - first).reshape(-1, *([1] * (values.ndim - 1)))
    mean: NDArray[np.float64] = (running[last] - running[first]) / span
    return mean


def with_context(slow: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the centred and the trailing means of the slowly varying columns, side by side."""
    return np.column_stack([
        window_mean(slow, AROUND, AROUND),
        *(window_mean(slow, length - 1, 0) for length in BEFORE),
    ])


def frame_features(
    inputs: AlignedInputs, window: DischargeWindow | None = None
) -> NDArray[np.float64]:
    """Return the feature matrix of one shot, shape ``(T, F)``, finite everywhere.

    ``F`` depends on the number of coil circuits only, which is fixed for a machine,
    and is the same in both layouts; only what the phase columns measure against
    differs.

    Parameters
    ----------
    inputs:
        The shot's inputs on some time base: the reconstruction frames alone, or
        the discharge record's longer base (:mod:`.history`).
    window:
        The discharge's own span and peak current. Given, the phase columns are
        measured against the discharge (:meth:`DischargeWindow.phase`); omitted,
        against the instants in ``inputs`` — the record-relative layout every
        network fitted before 22 Sep 2026 reads, kept so that those networks and
        the archives submitted with them can be rebuilt exactly.
    """
    seconds = (inputs.times_ms - inputs.times_ms[0]) / 1.0e3
    orientation = flux_orientation(inputs.plasma_current)
    circuits = orientation * inputs.circuits / 1.0e6
    plasma = orientation * inputs.plasma_current / 1.0e6
    magnitude = np.maximum(np.abs(inputs.plasma_current), CURRENT_FLOOR) / 1.0e6
    toroidal = np.abs(inputs.tf_current) / 1.0e5
    thomson = thomson_features(inputs)
    features = np.column_stack([
        circuits,
        plasma,
        toroidal,
        _rate(circuits, seconds),
        _rate(plasma, seconds),
        circuits / magnitude[:, np.newaxis],
        toroidal / magnitude,
        1.0 / magnitude,
        discharge_phase(inputs.times_ms, inputs.plasma_current) if window is None
        else window.phase(inputs.times_ms, inputs.plasma_current),
        thomson,
        with_context(np.column_stack([circuits, plasma, thomson])),
    ])
    if not bool(np.all(np.isfinite(features))):
        columns = np.flatnonzero(~np.all(np.isfinite(features), axis=0))
        raise ValueError(f"non-finite input behind feature columns {columns.tolist()}")
    return features
