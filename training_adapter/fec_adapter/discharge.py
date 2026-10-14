# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — the discharge's own start, end and peak current

"""Where a discharge begins and ends, read from the whole plasma-current record.

The reconstruction frames a shot ships are a selection of its discharge. Since
the release that keeps diverted frames only, a record can be the last few
hundred milliseconds of a two-second discharge: on the validation part six
records lie wholly in the final 40–320 ms of the current ramp-down, at 0.2–0.6
of the discharge's peak current. Anything measured against the record — the
current as a share of the record's largest, the time since the record's first
frame, a trailing mean over the record's own frames — then describes the
selection and not the plasma, and a frame at the end of a ramp-down reads as
the start of a discharge at full current.

The plasma current is an input over the whole acquisition, labelled shot or
not, so the discharge's start, end and peak are known for every shot. They are
taken from that record here, on the clock :func:`machine.plasma_current_times`
corrects, and nowhere from a reconstruction column.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from . import machine

RUNNING_SHARE: Final = 0.05
"""Share of the peak current above which the discharge counts as running.

Below it the record holds the acquisition's noise floor before breakdown and the
tail after the quench. The six end-of-ramp-down records of the validation part
end 20–180 ms before the current falls through this share (22 Sep census,
inputs only); across the part the record's end lies 0.34 s before it in the
median and 0.02 s at the fifth percentile.
"""

PEAK_PERCENTILE: Final = 99.5
"""Percentile of the current's magnitude taken as the discharge's peak.

A percentile rather than the maximum, so that one acquisition spike does not set
the scale of every ratio; it is the same statistic the clock correction uses.
"""

PHASE_HORIZON: Final = 1.0
"""Seconds beyond which the distance to either end of the discharge stops counting.

The current ramps up and down within about a second; further from either end
the distance says nothing about the phase, only about how long the shot is.
"""


@dataclass(frozen=True)
class DischargeWindow:
    """The span over which a discharge carries current, and its peak current.

    Attributes
    ----------
    start_ms, end_ms:
        First and last instant, in milliseconds on the corrected clock, at which
        the current's magnitude exceeds :data:`RUNNING_SHARE` of the peak.
    peak_current:
        The discharge's peak current magnitude in amperes (:data:`PEAK_PERCENTILE`).
    """

    start_ms: float
    end_ms: float
    peak_current: float

    def __post_init__(self) -> None:
        """Refuse a window that is empty, reversed or without current."""
        if not (np.isfinite(self.start_ms) and np.isfinite(self.end_ms)):
            raise ValueError(f"discharge window {self.start_ms}–{self.end_ms} ms is not finite")
        if self.end_ms < self.start_ms:
            raise ValueError(f"discharge ends at {self.end_ms} ms before it starts")
        if not self.peak_current > 0.0:
            raise ValueError(f"discharge peak current {self.peak_current} A is not positive")

    def phase(
        self, times_ms: NDArray[np.float64], plasma_current: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return where each instant sits in the discharge, shape ``(T, 5)``.

        Columns: the current as a share of the discharge's peak; its rate of change
        per second as a share of the peak; the seconds since the discharge started
        and the seconds until it ends, each clipped to ±:data:`PHASE_HORIZON`; and
        the elapsed share of the discharge, clipped to ``[0, 1]``. The columns mean
        what the record-relative ones of :func:`input_features.discharge_phase`
        were meant to mean, measured against the discharge instead of the frames
        a record happens to hold.

        Parameters
        ----------
        times_ms:
            Instants in milliseconds, increasing, shape ``(T,)``.
        plasma_current:
            The plasma current in amperes at those instants, shape ``(T,)``.
        """
        seconds = np.asarray(times_ms, dtype=np.float64) / 1.0e3
        magnitude = np.abs(np.asarray(plasma_current, dtype=np.float64))
        rate = np.gradient(magnitude, seconds) if seconds.size > 1 else np.zeros(seconds.size)
        start, end = self.start_ms / 1.0e3, self.end_ms / 1.0e3
        duration = max(end - start, 1.0e-9)
        return np.column_stack([
            magnitude / self.peak_current,
            rate / self.peak_current,
            np.clip(seconds - start, -PHASE_HORIZON, PHASE_HORIZON),
            np.clip(end - seconds, -PHASE_HORIZON, PHASE_HORIZON),
            np.clip((seconds - start) / duration, 0.0, 1.0),
        ])


def discharge_window(row: dict[str, Any]) -> DischargeWindow:
    """Return the discharge window of one shot row, from its plasma-current input.

    Raises
    ------
    ValueError
        If the row holds no finite current sample above zero.
    """
    times = np.asarray(machine.plasma_current_times(row), dtype=np.float64)
    current = np.abs(np.asarray(row["magnetics_plasma_current"], dtype=np.float64)) * machine.KILO
    finite = np.isfinite(times) & np.isfinite(current)
    times, current = times[finite], current[finite]
    if current.size == 0 or not float(current.max()) > 0.0:
        raise ValueError("the plasma-current record holds no current")
    order = np.argsort(times, kind="stable")
    times, current = times[order], current[order]
    peak = float(np.percentile(current, PEAK_PERCENTILE))
    running = np.flatnonzero(current > RUNNING_SHARE * peak)
    return DischargeWindow(float(times[running[0]]), float(times[running[-1]]), peak)
