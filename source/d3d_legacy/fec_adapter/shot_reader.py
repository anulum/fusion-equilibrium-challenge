# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — shot reader for the Challenge dataset

"""Assemble one shot's inputs onto the reconstruction time base.

The dataset gives every diagnostic on its own clock: coil currents at roughly
40 kHz over a window that starts seconds before the plasma, the plasma current
on a coarser base, and Thomson profiles at a few hundred hertz. Predictions are
required at ``efit_times`` and nowhere else, so everything is brought onto that
base once, here, and the rest of the pipeline sees aligned frames only.

Reading the targets is deliberately not offered. The training split carries
``efit_*`` ground truth in the same row as the inputs, and a predictor that can
see it scores a number that means nothing. The reader exposes inputs; the
targets stay in the scorer's hands.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

import numpy as np
import pyarrow.parquet as pq
from numpy.typing import NDArray

from . import machine

TARGET_PREFIX: Final = "efit_"
"""Columns under this prefix are reconstruction outputs, not inputs."""

INPUT_TARGET_EXCEPTIONS: Final = frozenset({"efit_times", "efit_grid_R", "efit_grid_Z"})
"""Target-prefixed columns that are genuinely inputs: the clock and the grid."""

PROFILE_DIMENSIONS: Final = 2
"""A profile block is laser pulses by channels."""

DERIVED_NON_INPUT: Final = frozenset({"magnetics_dsep", "magnetics_dsep_times"})
"""Computed from the target equilibrium; neither an input nor a scored target."""


@dataclass(frozen=True)
class ThomsonProfile:
    """Electron temperature and density samples at known positions.

    Attributes
    ----------
    r, z:
        Measurement positions in metres, one per channel.
    te:
        Electron temperature in electronvolts, shape ``(T, channels)``.
    ne:
        Electron density in inverse cubic metres, shape ``(T, channels)``.
    """

    r: NDArray[np.float64]
    z: NDArray[np.float64]
    te: NDArray[np.float64]
    ne: NDArray[np.float64]


@dataclass(frozen=True)
class Shot:
    """One shot's inputs, aligned to the reconstruction time base.

    Attributes
    ----------
    name:
        The shot file's stem, used to label outputs.
    source:
        Machine identifier as the dataset records it.
    times_ms:
        The reconstruction times, in milliseconds.
    r_grid, z_grid:
        The uniform reconstruction grid, in metres.
    coils:
        Filament positions and the columns they draw current from.
    coil_current:
        Ampere-turns per filament, shape ``(T, N_coils)``.
    plasma_current:
        Measured plasma current in amperes, shape ``(T,)``.
    tf_current:
        Toroidal field circuit current in amperes, shape ``(T,)``.
    thomson:
        Electron profiles sampled at the reconstruction times.
    """

    name: str
    source: str
    times_ms: NDArray[np.float64]
    r_grid: NDArray[np.float64]
    z_grid: NDArray[np.float64]
    coils: machine.CoilSet
    coil_current: NDArray[np.float64]
    plasma_current: NDArray[np.float64]
    tf_current: NDArray[np.float64]
    thomson: ThomsonProfile

    @property
    def n_frames(self) -> int:
        """Return the number of reconstruction frames."""
        return int(self.times_ms.size)


def select_frames(shot: Shot, keep: NDArray[np.intp] | slice) -> Shot:
    """Return the same shot restricted to some of its frames.

    Every per-frame series is cut by the same selection; the grid, the coils
    and the Thomson geometry are per shot and stay as they are.
    """
    return replace(
        shot,
        times_ms=shot.times_ms[keep],
        coil_current=shot.coil_current[keep],
        plasma_current=shot.plasma_current[keep],
        tf_current=shot.tf_current[keep],
        thomson=ThomsonProfile(
            r=shot.thomson.r, z=shot.thomson.z,
            te=shot.thomson.te[keep], ne=shot.thomson.ne[keep],
        ),
    )


def read_row(path: Path) -> dict[str, Any]:
    """Read one shot file into a column mapping.

    Each file holds exactly one row; anything else means the file is not what
    this reader was written against, and guessing would be worse than stopping.
    """
    table = pq.read_table(path)
    if table.num_rows != 1:
        raise ValueError(f"{path.name}: expected 1 row, found {table.num_rows}")
    return {name: table.column(name)[0].as_py() for name in table.schema.names}


def input_columns(row: dict[str, Any]) -> list[str]:
    """Return the columns a predictor is allowed to read.

    This exists so the separation can be asserted in a test rather than merely
    intended: the training rows carry their own answers, and the only reliable
    guard against reading them is an explicit list.
    """
    allowed: list[str] = []
    for name in row:
        if name in DERIVED_NON_INPUT:
            continue
        if name.startswith(TARGET_PREFIX) and name not in INPUT_TARGET_EXCEPTIONS:
            continue
        allowed.append(name)
    return allowed


def read_shot(path: Path) -> Shot:
    """Read one shot and align every input to the reconstruction time base."""
    row = read_row(path)
    return aligned_shot(row, path.stem, np.asarray(row["efit_times"], dtype=np.float64))


def aligned_shot(row: dict[str, Any], name: str, times_ms: NDArray[np.float64]) -> Shot:
    """Align every input of one shot row to the given instants.

    :func:`read_shot` asks for the reconstruction times; the accuracy entry's
    discharge record also asks for instants before and after them, on the same
    spacing, so that its trailing means see the discharge rather than the record.
    Only input columns are read, whatever the instants.

    Parameters
    ----------
    row:
        One shot's columns, as :func:`read_row` returns them.
    name:
        The shot file's stem.
    times_ms:
        Instants in milliseconds, increasing, shape ``(T,)``.
    """
    times = np.asarray(times_ms, dtype=np.float64)
    r_grid, z_grid = machine.grid(row)
    coils = machine.coil_set(row)
    return Shot(
        name=name,
        source=str(row["source"]),
        times_ms=times,
        r_grid=r_grid,
        z_grid=z_grid,
        coils=coils,
        coil_current=machine.coil_currents_si(row, coils, times),
        plasma_current=machine.plasma_current_si(row, times),
        tf_current=machine.toroidal_current_si(row, times),
        thomson=_thomson(row, times),
    )


def _thomson(row: dict[str, Any], times_ms: NDArray[np.float64]) -> ThomsonProfile:
    """Merge the core and edge Thomson systems onto the reconstruction times.

    DIII-D's core system is a vertical laser: every channel sits at the same
    radius and the informative coordinate is the height, which the profile
    columns do not carry. The chord table does, under the ``TS_core_`` names,
    in the same channel order — verified on the data by the temperature rising
    monotonically towards the channel nearest the magnetic axis rather than
    scattering. The edge system ships its heights directly.

    MAST's two systems are horizontal lasers on the midplane: the informative
    coordinate is the radius, and the chord table carries it for both under the
    ``TS_core_`` and ``TS_edge_`` names, at height zero. Its core block ships
    one channel more than it has positions for — the first data channel has no
    coordinate and is empty on every shot, which the dataset's errata record —
    so that channel is dropped before positions are paired.
    """
    names = [str(name) for name in row["thomson_chord_name"]]
    chord_r = np.asarray(row["thomson_chord_R"], dtype=np.float64)
    chord_z = np.asarray(row["thomson_chord_Z"], dtype=np.float64)

    core_te = np.asarray(row["thomson_core_Te"], dtype=np.float64)
    core_ne = np.asarray(row["thomson_core_ne"], dtype=np.float64)
    core_times = np.asarray(row["thomson_core_times"], dtype=np.float64)
    if core_te.ndim == PROFILE_DIMENSIONS and core_te.shape[1] == len(row["thomson_core_R"]) + 1:
        core_te, core_ne = core_te[:, 1:], core_ne[:, 1:]
    core_r, core_z = _core_positions(names, chord_r, chord_z, core_te.shape[1], row)

    edge_te = np.asarray(row["thomson_edge_Te"], dtype=np.float64)
    edge_ne = np.asarray(row["thomson_edge_ne"], dtype=np.float64)
    edge_times = np.asarray(row["thomson_edge_times"], dtype=np.float64)
    edge_r, edge_z = _edge_positions(
        names, chord_r, chord_z, np.asarray(row["thomson_edge_spatial"], dtype=np.float64)
    )

    te = np.hstack(
        [_resample(times_ms, core_times, core_te), _resample(times_ms, edge_times, edge_te)]
    )
    ne = np.hstack(
        [_resample(times_ms, core_times, core_ne), _resample(times_ms, edge_times, edge_ne)]
    )
    return ThomsonProfile(
        r=np.concatenate([core_r, edge_r]),
        z=np.concatenate([core_z, edge_z]),
        te=te,
        ne=ne,
    )


def _core_positions(
    names: list[str],
    chord_r: NDArray[np.float64],
    chord_z: NDArray[np.float64],
    channels: int,
    row: dict[str, Any],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return core-channel positions, falling back to the shipped radius.

    The chord table is per shot and its layout genuinely varies between
    campaigns, so a mismatch in channel count is expected rather than
    exceptional. When it happens the heights are unknown and the channels are
    placed on the midplane at the shipped radius, which is wrong but bounded —
    the alternative, silently pairing the wrong height to a temperature, is
    neither.
    """
    index = [i for i, name in enumerate(names) if name.startswith("TS_core_")]
    if len(index) == channels:
        return chord_r[index], chord_z[index]
    radius = np.asarray(row["thomson_core_R"], dtype=np.float64)
    if radius.size != channels:
        radius = np.full(channels, float(np.median(radius)) if radius.size else float("nan"))
    return radius, np.zeros(channels, dtype=np.float64)


def _edge_positions(
    names: list[str],
    chord_r: NDArray[np.float64],
    chord_z: NDArray[np.float64],
    spatial: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the radius and height of each edge channel.

    Where the chord table lists the edge system under ``TS_edge_`` it carries
    both coordinates and is used as it is. Otherwise the edge system is
    DIII-D's tangential laser: its heights ship with the profile as the spatial
    coordinate, and its radius only appears in the chord table.
    """
    channels = int(spatial.size)
    listed = [i for i, name in enumerate(names) if name.startswith("TS_edge_")]
    if listed and len(listed) == channels:
        return chord_r[listed], chord_z[listed]
    index = [i for i, name in enumerate(names) if name.startswith("TS_tangential_")]
    if len(index) == channels:
        return chord_r[index], spatial
    fallback = float(np.median(chord_r)) if chord_r.size else float("nan")
    return np.full(channels, fallback, dtype=np.float64), spatial


def _resample(
    at_ms: NDArray[np.float64],
    base_ms: NDArray[np.float64],
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Sample a channel-by-time profile block at the requested times.

    Nearest-sample selection is used rather than interpolation between laser
    pulses: a Thomson profile is a snapshot of the whole plasma at one instant,
    and averaging two of them across a transient invents a profile that was
    never measured.
    """
    if base_ms.size == 0 or values.size == 0:
        return np.zeros((at_ms.size, values.shape[1] if values.ndim == PROFILE_DIMENSIONS else 0))
    order = np.argsort(base_ms, kind="stable")
    sorted_base = base_ms[order]
    pick = np.searchsorted(sorted_base, at_ms)
    pick = np.clip(pick, 1, sorted_base.size - 1)
    left = sorted_base[pick - 1]
    right = sorted_base[pick]
    nearer = np.where(np.abs(at_ms - left) <= np.abs(right - at_ms), pick - 1, pick)
    return np.asarray(values[order][nearer], dtype=np.float64)
