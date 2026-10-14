# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — a shot's inputs on its discharge's time base

"""A shot's inputs on a time base that reaches into the discharge around the record.

The features of a frame include means over the frames before it and around it.
Taken over the reconstruction frames alone, those means stop at the record's
first frame, so a record that holds only the last few hundred milliseconds of a
ramp-down is described as a young discharge; and because they count frames, a
record with frames missing inside it — 5174 of the 6669 labelled DIII-D records
with more than one frame have a step longer than 30 ms, the corpus keeping
diverted frames only — averages over more time where frames are missing.

This module aligns the inputs on the record's own frames *and* on instants of
the discharge at :data:`CADENCE_MS` around and between them: back as far as the
longest trailing mean reaches, ahead as far as the centred one, and into every
gap inside the record, never outside the discharge. The features are computed
on that base and read at the reconstruction frames, so a frame's features do not
depend on which other frames of its discharge were shipped with it: a frame is
described by its discharge, not by the record it came in.

The layout this produces is :data:`~.layout.DISCHARGE_LAYOUT`; the record-relative
one, which every network fitted before it reads, is :data:`~.layout.RECORD_LAYOUT`.
"""
from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from fec_adapter.diii_d_accuracy.features import AROUND, BEFORE, AlignedInputs, frame_features
from fec_adapter.diii_d_accuracy.layout import (
    DISCHARGE_LAYOUT,
    RECORD_LAYOUT,
    FeatureLayout,
    layout_of,
)
from fec_adapter.discharge import DischargeWindow, discharge_window
from fec_adapter.shot_reader import Shot, aligned_shot, read_row

REACH_BEFORE: Final = BEFORE[-1] - 1
"""Instants before the first frame that the longest trailing mean reaches."""

REACH_AFTER: Final = AROUND
"""Instants after the last frame that the centred mean reaches."""

CADENCE_MS: Final = 20.0
"""Milliseconds between the instants of the discharge base.

The window lengths of :mod:`.features` are stated at this cadence (±80 ms, 0.32 s,
1.3 s), and it is the median spacing of 6657 of the 6669 labelled DIII-D records
with more than one frame (census of the extracted set, 22 Sep 2026: six records
are 40 ms apart, four others 30 ms to 2.6 s; fourteen records hold one frame).
"""


def extended_times(
    record_ms: NDArray[np.float64], window: DischargeWindow
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Return the discharge record's instants and where the record's frames sit in them.

    The record's own instants are kept exactly. Instants :data:`CADENCE_MS` apart
    are added: up to :data:`REACH_BEFORE` before the first frame and up to
    :data:`REACH_AFTER` after the last, only while inside the discharge window,
    and after every frame that is followed by a gap, until the next frame is
    less than one and a half cadences away.

    Parameters
    ----------
    record_ms:
        The reconstruction times in milliseconds, increasing, shape ``(T,)``.
    window:
        The shot's discharge window.

    Returns
    -------
    times, frames:
        The instants, increasing, and the positions of the record's frames among
        them, shape ``(T,)``.
    """
    record = np.asarray(record_ms, dtype=np.float64)
    if record.size == 0:
        raise ValueError("a record without frames has no time base")
    if record.size > 1 and not bool(np.all(np.diff(record) > 0.0)):
        raise ValueError("the reconstruction times do not increase")
    before = record[0] - CADENCE_MS * np.arange(REACH_BEFORE, 0, -1, dtype=np.float64)
    before = before[before >= window.start_ms]
    after = record[-1] + CADENCE_MS * np.arange(1, REACH_AFTER + 1, dtype=np.float64)
    after = after[after <= window.end_ms]
    between = [
        start + CADENCE_MS * np.arange(1, int(np.ceil((end - start) / CADENCE_MS - 1.5)) + 1)
        for start, end in itertools.pairwise(record)
        if end - start >= 1.5 * CADENCE_MS
    ]
    times = np.concatenate([before, record, *between, after])
    times.sort(kind="stable")
    frames = np.searchsorted(times, record).astype(np.intp)
    return times, frames


@dataclass(frozen=True)
class DischargeRecord:
    """One shot's inputs on its discharge record's base, and where its frames are.

    Attributes
    ----------
    inputs:
        The inputs aligned on the extended instants of :func:`extended_times`.
    frames:
        Positions of the reconstruction frames among those instants, shape ``(T,)``.
    window:
        The discharge window the extension and the phase columns are measured against.
    """

    inputs: AlignedInputs
    frames: NDArray[np.intp]
    window: DischargeWindow

    def features(self) -> NDArray[np.float64]:
        """Return the discharge-layout features of the reconstruction frames, ``(T, F)``."""
        return frame_features(self.inputs, self.window)[self.frames]

    def arrays(self) -> dict[str, NDArray[np.generic]]:
        """Return the record as named arrays for storage, inputs in single precision."""
        inputs = self.inputs
        return {
            "times_ms": inputs.times_ms,
            "circuits": inputs.circuits.astype(np.float32),
            "plasma_current": inputs.plasma_current.astype(np.float32),
            "tf_current": inputs.tf_current.astype(np.float32),
            "thomson_r": inputs.thomson_r.astype(np.float32),
            "thomson_z": inputs.thomson_z.astype(np.float32),
            "thomson_te": inputs.thomson_te.astype(np.float32),
            "thomson_ne": inputs.thomson_ne.astype(np.float32),
            "frames": self.frames.astype(np.int64),
            "window": np.asarray(
                [self.window.start_ms, self.window.end_ms, self.window.peak_current]
            ),
        }

    @classmethod
    def from_stored(cls, stored: Mapping[str, NDArray[np.generic]]) -> DischargeRecord:
        """Read a record from the arrays :meth:`arrays` produced."""
        start, end, peak = (float(value) for value in np.asarray(stored["window"]))
        return cls(
            AlignedInputs.from_stored(stored),
            np.asarray(stored["frames"], dtype=np.intp),
            DischargeWindow(start, end, peak),
        )

    @classmethod
    def from_row(cls, row: dict[str, Any], name: str) -> DischargeRecord:
        """Build the record of one shot row from its input columns."""
        window = discharge_window(row)
        times, frames = extended_times(np.asarray(row["efit_times"], dtype=np.float64), window)
        return cls(AlignedInputs.from_shot(aligned_shot(row, name, times)), frames, window)


def read_discharge(path: Path) -> DischargeRecord:
    """Read one shot file into its discharge record."""
    return DischargeRecord.from_row(read_row(path), path.stem)


def save_record(record: DischargeRecord, path: Path) -> None:
    """Store a record under ``path``, written whole under a hidden name and then renamed."""
    partial = path.with_name(f".{path.stem}.partial.npz")
    np.savez_compressed(partial, allow_pickle=False, **record.arrays())
    partial.rename(path)


def load_record(path: Path) -> DischargeRecord:
    """Read a record :func:`save_record` stored."""
    with np.load(path, allow_pickle=False) as stored:
        return DischargeRecord.from_stored(stored)


def layout_features(
    row: dict[str, Any], shot: Shot, layouts: Iterable[FeatureLayout]
) -> dict[FeatureLayout, NDArray[np.float64]]:
    """Return one shot's features of the reconstruction frames in each layout asked for.

    Parameters
    ----------
    row:
        The shot's columns, as :func:`~fec_adapter.shot_reader.read_row` returns them.
    shot:
        The same shot aligned on its reconstruction times
        (:func:`~fec_adapter.shot_reader.aligned_shot`).
    layouts:
        The layouts the networks at hand were fitted on.
    """
    wanted = {layout_of(layout) for layout in layouts}
    features: dict[FeatureLayout, NDArray[np.float64]] = {}
    if RECORD_LAYOUT in wanted:
        features[RECORD_LAYOUT] = frame_features(AlignedInputs.from_shot(shot))
    if DISCHARGE_LAYOUT in wanted:
        features[DISCHARGE_LAYOUT] = DischargeRecord.from_row(row, shot.name).features()
    return features
