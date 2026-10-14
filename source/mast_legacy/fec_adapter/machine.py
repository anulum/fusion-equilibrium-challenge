# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — machine description for the Challenge dataset

"""Translate the Challenge dataset's machine description into solver inputs.

The dataset ships coil geometry and coil currents in the conventions of each
machine's own reconstruction code, and those conventions differ in ways that
silently ruin a Green's-function calculation if they are assumed rather than
read. This module holds the conversions in one place, each with the reason it
is what it is, so that a wrong field strength is a visible edit rather than a
buried factor.

DIII-D current conventions, as documented by the dataset:

``magnetics_F1A`` ... ``magnetics_F9B``
    Kilo-ampere-turns. The `mhdin.dat` turn counts (58 or 55) are already
    folded in, so these are total ampere-turns per rectangle once scaled by
    1000 — exactly the quantity a filament Green's function wants, and no
    further turn multiplier belongs here.

``magnetics_ECOILA``
    Kilo-amperes, *not* kilo-ampere-turns. The dataset states the turn
    convention for this group is unresolved because `ECOILB` — a second
    co-located group — is not shipped. Applying a turn count here would be
    inventing one.

``magnetics_bcoil``
    Kilo-amperes in the toroidal field circuit. It has no poloidal-plane
    rectangle and contributes nothing to the poloidal flux; it sets the
    vacuum toroidal field function, which is what ``q95`` needs.

MAST current conventions, as documented by the dataset and checked on its files:

``magnetics_p2l_current`` ... ``magnetics_p6u_current``
    Kilo-ampere-turns of the whole coil. The coil table lists every conductor
    turn as a row of its own (P2 20, P3 8, P4 23, P5 23, P6 4 rows per column),
    so each row carries the column's value divided by the number of rows that
    share the column.

``magnetics_sol_current``
    Kilo-amperes in the solenoid feed. Its 656 rows are single turns, and each
    carries :data:`MAST_SOLENOID_ROW_SHARE` of the feed current — a measured
    machine constant, not a documented one.

``magnetics_tf_current``, ``magnetics_efps_current``
    Kilo-amperes; neither has a poloidal-plane row in the coil table.

``magnetics_plasma_current``
    Kilo-amperes, on the shared ``magnetics_time`` — MAST ships no separate
    clock for it, and the DIII-D clock erratum does not exist there. On 114
    early shots ``magnetics_time`` is the union of two acquisition grids and
    every column is ``nan`` on the other grid's samples; the interpolation
    below drops non-finite samples per column, which is the masking the dataset
    asks for.

Times are milliseconds throughout the dataset, on both machines.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

KILO: Final = 1.0e3
"""Scale from the dataset's kilo-units to SI amperes."""

ECOIL_COLUMN: Final = "magnetics_ECOILA"
"""The one DIII-D current column shipped in amperes rather than ampere-turns."""

ECOIL_FILAMENTS: Final = 48
"""Filaments the solenoid rectangle is divided into along its height.

The rectangle is three and a half metres tall. A single filament at its centre
gets the flux pattern of a solenoid badly wrong; EFIT's own machine file uses
48 elements over the same envelope, and the same count is used here.
"""

ECOIL_EFFECTIVE_TURNS: Final = 148.7
"""Ampere-turns the ohmic solenoid circuit delivers per ampere shipped.

The dataset ships this one column in amperes and says its turn convention is
unresolved, because a second co-located winding group is not shipped at all.
Left at one turn, the solenoid's flux — which outside a long solenoid is nearly
a constant, and which therefore shifts the whole flux map — is simply missing,
and the prediction carries an offset of a few tenths of a weber that tracks the
solenoid current through the shot.

The value is a machine constant, measured once: the wall-flux closure (shipped
shaping-coil currents, plus the plasma current, against the reconstruction on
the computational wall) was fitted for this single number, shot by shot, on
forty shots of the calibration bucket in ``splits.json``. The per-shot values
have a median of 148.6 with an interquartile range of 147.9 to 149.7, and a
trimmed mean of 148.7 +/- 1.0 — a spread that small across different
discharges is what a machine constant looks like. On shots outside the fit it
explains a median 93 % of the wall residual that a one-turn model leaves (which
explains 1 %). DIII-D's ohmic coil has of the order of 140 turns, so the number
is also the physically expected one. It uses no information from the shot being
predicted.
"""

TF_COLUMN: Final = "magnetics_bcoil"
"""DIII-D's toroidal field circuit; carries no poloidal-plane geometry."""


MAST_SOLENOID_ROW_SHARE: Final = 0.5
"""Share of the solenoid feed current that each of MAST's 656 solenoid rows carries.

The dataset documents the column as the feed current in kilo-amperes and lists
656 single-turn rows in four layers of 164 (radii 0.119, 0.132, 0.145 and
0.157 m). Read as 656 turns in series, the solenoid comes out twice too strong.

Measured on the three labelled MAST demonstration shots of the organisers'
starter kit, the only MAST reconstructions that exist: the *change* of the
reconstructed flux between flat-top frames, on the grid cells outside the vessel
envelope, regressed on the change of the shaping coils', the solenoid's and a
plasma filament's vacuum flux. The solenoid sweeps by 45 to 49 kA within each
flat top while the plasma current moves by 60 kA, so its coefficient is well
determined: 0.513, 0.512 and 0.512 on the three shots (R² 0.983 to 0.989), and
holding it at 1 lowers the pooled R² from 0.986 to 0.763. Changes are used
rather than the fields themselves because currents induced in the vessel, which
the dataset omits, are nearly constant during a steady sweep and drop out of a
difference; fitted on the fields they bias the coefficient to 0.3–0.4.

A half is what two parallel paths of two series layers give. The labels are
used here for a constant of the machine, as DIII-D's effective solenoid turns
were, and for nothing that depends on a shot being predicted.
"""


@dataclass(frozen=True)
class Conventions:
    """How one machine's row names and scales its currents.

    Attributes
    ----------
    solenoid_column:
        Current column of the ohmic solenoid.
    tf_column:
        Current column of the toroidal field circuit.
    plasma_current_clock:
        Column holding the plasma current's own time base, or ``None`` where it
        shares ``magnetics_time``.
    rows_are_turns:
        Whether the coil table lists one row per conductor turn (MAST) rather
        than one lumped rectangle per coil (DIII-D).
    solenoid_row_share:
        Where rows are turns: the share of the solenoid feed current each
        solenoid row carries.
    """

    solenoid_column: str
    tf_column: str
    plasma_current_clock: str | None
    rows_are_turns: bool
    solenoid_row_share: float = 1.0


CONVENTIONS: Final = {
    "DIII-D": Conventions(
        solenoid_column=ECOIL_COLUMN,
        tf_column=TF_COLUMN,
        plasma_current_clock="magnetics_plasma_current_times",
        rows_are_turns=False,
    ),
    "MAST": Conventions(
        solenoid_column="magnetics_sol_current",
        tf_column="magnetics_tf_current",
        plasma_current_clock=None,
        rows_are_turns=True,
        solenoid_row_share=MAST_SOLENOID_ROW_SHARE,
    ),
}


def conventions(row: dict[str, Any]) -> Conventions:
    """Return the conventions of the machine a row says it comes from."""
    source = str(row["source"])
    if source not in CONVENTIONS:
        raise ValueError(f"no current conventions recorded for machine {source!r}")
    return CONVENTIONS[source]


@dataclass(frozen=True)
class CoilSet:
    """Filament representation of the poloidal field coils.

    Attributes
    ----------
    names:
        Coil identifiers, in the order the current columns are stacked.
    columns:
        The current column each filament draws from.
    r, z:
        Filament positions in metres — the conductor-rectangle centres, except
        for the solenoid, which is spread along its height.
    turns:
        Factor from the shipped current to the filament's ampere-turns. Unity
        where the dataset has already folded the turn count in.
    solenoid_column:
        The current column of the ohmic solenoid on this machine.
    """

    names: tuple[str, ...]
    columns: tuple[str, ...]
    r: NDArray[np.float64]
    z: NDArray[np.float64]
    turns: NDArray[np.float64]
    solenoid_column: str = ECOIL_COLUMN

    @property
    def is_solenoid(self) -> NDArray[np.bool_]:
        """Return which filaments belong to the ohmic solenoid."""
        return np.asarray([column == self.solenoid_column for column in self.columns], dtype=bool)

    def __len__(self) -> int:
        """Return the number of filaments."""
        return len(self.names)


def coil_set(row: dict[str, Any]) -> CoilSet:
    """Build the filament set from one shot's coil table.

    Each shaping-coil rectangle becomes a single filament at its centre. The
    dataset notes that six DIII-D F-coils are parallelograms, and that the skew
    matters only when the conductor cross-section is integrated over rather
    than lumped; lumping is what EFIT's own machine file does at this
    granularity, so the skew is carried in the data and deliberately not used.

    The solenoid is the exception. It is divided into
    :data:`ECOIL_FILAMENTS` filaments along its height, sharing
    :data:`ECOIL_EFFECTIVE_TURNS` between them.

    Where the table lists one row per conductor turn, every row is a filament
    as it stands; see :func:`_turn_rows`.
    """
    if conventions(row).rows_are_turns:
        return _turn_rows(row)
    names: list[str] = []
    columns: list[str] = []
    r: list[float] = []
    z: list[float] = []
    turns: list[float] = []
    heights = row.get("coil_height") or [0.0] * len(row["coil_name"])
    for name, column, centre_r, centre_z, height in zip(
        row["coil_name"], row["coil_input_column"], row["coil_R"], row["coil_Z"], heights,
        strict=True,
    ):
        if str(column) != ECOIL_COLUMN:
            names.append(str(name))
            columns.append(str(column))
            r.append(float(centre_r))
            z.append(float(centre_z))
            turns.append(1.0)
            continue
        offsets = (np.arange(ECOIL_FILAMENTS) - (ECOIL_FILAMENTS - 1) / 2.0) / ECOIL_FILAMENTS
        for index, offset in enumerate(offsets):
            names.append(f"{name}_{index:02d}")
            columns.append(str(column))
            r.append(float(centre_r))
            z.append(float(centre_z) + float(offset) * float(height))
            turns.append(ECOIL_EFFECTIVE_TURNS / ECOIL_FILAMENTS)
    return CoilSet(
        names=tuple(names),
        columns=tuple(columns),
        r=np.asarray(r, dtype=np.float64),
        z=np.asarray(z, dtype=np.float64),
        turns=np.asarray(turns, dtype=np.float64),
    )


def _turn_rows(row: dict[str, Any]) -> CoilSet:
    """Build the filament set of a machine whose coil table lists every conductor turn.

    A shaping-coil column is the ampere-turns of the whole coil, so each of its
    rows carries that value divided by the number of rows sharing the column.
    The solenoid column is the feed current, of which each solenoid row carries
    the machine's measured share.
    """
    solenoid = conventions(row).solenoid_column
    share = conventions(row).solenoid_row_share
    columns = tuple(str(column) for column in row["coil_input_column"])
    rows_per_column = {column: columns.count(column) for column in set(columns)}
    return CoilSet(
        names=tuple(str(name) for name in row["coil_name"]),
        columns=columns,
        r=np.asarray(row["coil_R"], dtype=np.float64),
        z=np.asarray(row["coil_Z"], dtype=np.float64),
        turns=np.asarray(
            [share if column == solenoid else 1.0 / rows_per_column[column] for column in columns],
            dtype=np.float64,
        ),
        solenoid_column=solenoid,
    )


def coil_currents_si(
    row: dict[str, Any], coils: CoilSet, times_ms: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return coil currents in amperes at each requested time.

    Parameters
    ----------
    row:
        One shot's columns.
    coils:
        The filament set the currents are assigned to.
    times_ms:
        Times to sample at, in milliseconds — normally ``efit_times``.

    Returns
    -------
    (T, N_filaments) array of ampere-turns per filament. The shaping coils ship
    with their turns folded in; the solenoid ships in amperes and is scaled to
    ampere-turns here through :attr:`CoilSet.turns`.

    Notes
    -----
    The current columns are sampled on ``magnetics_time``, which runs far
    outside the reconstruction window at a much finer cadence. Sampling is a
    linear interpolation; values outside the recorded span are held at the
    nearest recorded value rather than extrapolated, because a coil current
    extrapolated past the end of its record is fiction.
    """
    base = np.asarray(row["magnetics_time"], dtype=np.float64)
    out = np.empty((times_ms.size, len(coils)), dtype=np.float64)
    for index, column in enumerate(coils.columns):
        series = np.asarray(row[column], dtype=np.float64)
        out[:, index] = _interpolate(times_ms, base, series) * KILO * coils.turns[index]
    return out


def toroidal_current_si(row: dict[str, Any], times_ms: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the toroidal field circuit current in amperes at each time."""
    base = np.asarray(row["magnetics_time"], dtype=np.float64)
    series = np.asarray(row[conventions(row).tf_column], dtype=np.float64)
    return _interpolate(times_ms, base, series) * KILO


BREAKDOWN_WINDOW_MS: Final = (-500.0, 1000.0)
"""Window around breakdown the plasma-current clock must start in.

DIII-D takes ``t = 0`` to be breakdown, so a current record whose first
significant rise lands seconds away from zero is on the wrong clock.
"""

BREAKDOWN_FRACTION: Final = 0.10
"""Share of the peak current that counts as the discharge having started."""


def plasma_current_times(row: dict[str, Any]) -> NDArray[np.float64]:
    """Return the plasma current's time base, corrected where it is wrong.

    The release ships every DIII-D shot with the same plasma-current clock.
    It is right for roughly three shots in ten and about three seconds early
    on the rest, where reading the current at the reconstruction times returns
    the pre-plasma noise floor — a couple of kiloamperes instead of a
    megaampere. The values are sound throughout; only the timestamps are not.

    The decision is made per shot, because shifting the shots that are already
    correct would break them. A shot is left alone when its first rise past a
    tenth of the peak current lands near breakdown; otherwise the clock is
    re-origined onto the start of the magnetics record, which is where the
    affected acquisitions actually began.

    This reproduces the correction the organisers published for the erratum,
    against which it is checked in the tests, rather than inventing a second
    convention for the same fault.

    A machine whose plasma current shares ``magnetics_time`` has no clock of its
    own to be wrong, and that base is returned as it is.
    """
    clock = conventions(row).plasma_current_clock
    if clock is None:
        return np.asarray(row["magnetics_time"], dtype=np.float64)
    times = np.asarray(row[clock], dtype=np.float64)
    current = np.abs(np.asarray(row["magnetics_plasma_current"], dtype=np.float64))
    if times.size == 0 or current.size == 0:
        return times
    threshold = BREAKDOWN_FRACTION * float(np.nanpercentile(current, 99.5))
    breakdown = float(times[int(np.argmax(current > threshold))])
    low, high = BREAKDOWN_WINDOW_MS
    if low <= breakdown <= high:
        return times
    origin = float(row["magnetics_time"][0])
    shifted: NDArray[np.float64] = origin + (times - times[0])
    return shifted


def plasma_current_si(row: dict[str, Any], times_ms: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the measured plasma current in amperes at each time.

    The plasma current has its own acquisition, coarser than the coil one and
    on a clock that the release got wrong for most shots, so it cannot share
    the coil sampling path.
    """
    base = plasma_current_times(row)
    series = np.asarray(row["magnetics_plasma_current"], dtype=np.float64)
    return _interpolate(times_ms, base, series) * KILO


GRID_UNIFORMITY_TOLERANCE: Final = 1.0e-4
"""Relative spread of cell spacings still counted as a uniform grid.

The grids ship as single-precision coordinates, so their spacings differ in
the seventh decimal place — about a ten-millionth of a metre on a two-and-a-
half-metre machine. That is the storage format, not the geometry. The
tolerance is set well above that rounding and well below any real grading, so
a genuinely graded grid is still rejected.
"""


def grid(row: dict[str, Any]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the reconstruction grid as ``(R, Z)`` in metres, regularised.

    The solver takes its cell size from the first two coordinates and assumes
    it holds everywhere. Handing it the stored coordinates directly would make
    that assumption false by a rounding error at every other cell, so the grid
    is rebuilt as an exact uniform span between the recorded endpoints — the
    grid the machine file describes, rather than its single-precision image.

    A grid whose spacing genuinely varies is rejected: it would need a
    different discretisation, and silently regularising it would move every
    interior point.
    """
    r = _regularise("R", np.asarray(row["efit_grid_R"], dtype=np.float64))
    z = _regularise("Z", np.asarray(row["efit_grid_Z"], dtype=np.float64))
    return r, z


def _regularise(axis: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return an exactly uniform grid spanning the same range."""
    spacing = np.diff(values)
    if spacing.size == 0:
        raise ValueError(f"{axis} grid has fewer than two points")
    spread = float(spacing.max() - spacing.min())
    if spread > GRID_UNIFORMITY_TOLERANCE * abs(float(spacing.mean())):
        raise ValueError(f"{axis} grid is not uniform; the solver assumes a uniform grid")
    return np.linspace(float(values[0]), float(values[-1]), values.size, dtype=np.float64)


def _interpolate(
    at_ms: NDArray[np.float64],
    base_ms: NDArray[np.float64],
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Interpolate ``values`` onto ``at_ms``, holding the ends flat.

    Non-finite samples are dropped before interpolating: a single NaN in a
    long diagnostic record would otherwise poison every frame that straddles
    it, which is a much larger failure than the one missing sample.
    """
    finite = np.isfinite(base_ms) & np.isfinite(values)
    if not finite.any():
        return np.zeros_like(at_ms)
    clean_base = base_ms[finite]
    clean_values = values[finite]
    order = np.argsort(clean_base, kind="stable")
    sampled: NDArray[np.float64] = np.interp(at_ms, clean_base[order], clean_values[order])
    return sampled
