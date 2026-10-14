# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — bend a predicted map until its boundary follows a predicted contour

"""Write the scorer's boundary as radii about the axis, and bend a map until it follows them.

A tenth of the score is the boundary distance: the symmetric Hausdorff distance
between the boundary the scorer extracts from the submitted map and the one it
extracts from the true map, over the major radius. The shape edit
(:mod:`.shape_edit`) moves four scalars of the boundary; this module moves the
whole curve. The boundary is written as its radius about the magnetic axis at
:data:`CONTOUR_ANGLES` equally spaced angles (it is star-shaped about the axis),
which a network can predict as further outputs, and a map is bent radially
about its axis — each direction stretched by the ratio of the wanted radius to
the map's own, fading to no stretch a share of the local radius beyond the
boundary — until the scorer's re-read boundary follows the wanted one.

Measured with the true boundaries as the wanted ones (a perfect head), on 538
frames of 24 validation shots of recipe P1: the boundary distance fell from
0.0115 to 0.0063 with 64 angles (0.0071 with 32, 0.0080 with 16), and the
squared map error fell too, from 20.6 to 15.5 (24 Sep 2026,
``contour_oracle_probe.py``).
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from fec_adapter.diii_d_accuracy.shape_edit import Reader, Reading, ReadingBackend

CONTOUR_ANGLES: Final = 64
"""Directions about the axis at which the boundary's radius is written."""

FADE: Final = 0.25
"""Share of the local boundary radius beyond the boundary over which the bend fades out."""

ROUNDS: Final = 3
"""Corrections of the bend; each re-reads the boundary the scorer sees."""

SHARE: Final = 1.0
"""Share of the way from the map's boundary to the wanted one that the first bend goes."""

TOLERANCE: Final = 1.0e-4
"""Largest radius difference, in metres, at which a boundary counts as in place (a grid
cell is about 2.6 cm); a map already there is not resampled at all."""

STRETCH_LIMIT: Final = 0.2
"""Largest relative change of the radius in any direction; a wild output cannot tear the map."""


def label_names(count: int = CONTOUR_ANGLES) -> tuple[str, ...]:
    """Return the table's names of the boundary radii, one per direction."""
    return tuple(f"lcfs_r_{index:02d}" for index in range(count))


def angles(count: int = CONTOUR_ANGLES) -> NDArray[np.float64]:
    """Return the directions, from -pi (inboard midplane) counter-clockwise, as angles."""
    result: NDArray[np.float64] = np.linspace(-np.pi, np.pi, count, endpoint=False)
    return result


def _periodic(values: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the angles and ``values`` repeated once on either side, for periodic lookup."""
    grid = angles(values.size)
    return (np.concatenate([grid - 2.0 * np.pi, grid, grid + 2.0 * np.pi]),
            np.tile(values, 3))


def radii(contour: NDArray[np.float64], axis: tuple[float, float],
          count: int = CONTOUR_ANGLES) -> NDArray[np.float64]:
    """Return a closed contour's radius about ``axis`` at the :func:`angles` directions."""
    theta = np.arctan2(contour[:, 1] - axis[1], contour[:, 0] - axis[0])
    radius = np.hypot(contour[:, 0] - axis[0], contour[:, 1] - axis[1])
    order = np.argsort(theta)
    theta, radius = theta[order], radius[order]
    theta = np.concatenate([theta[-1:] - 2.0 * np.pi, theta, theta[:1] + 2.0 * np.pi])
    radius = np.concatenate([radius[-1:], radius, radius[:1]])
    result: NDArray[np.float64] = np.interp(angles(count), theta, radius)
    return result


def contour_radii(reader: Reader, flux: NDArray[np.float64],
                  count: int = CONTOUR_ANGLES) -> NDArray[np.float64]:
    """Return the radii of the scorer's boundary of ``flux`` about its axis, NaN without one."""
    reading = reader.read(flux)
    if reading.contour is None:
        return np.full(count, np.nan)
    return radii(reading.contour, reading.axis, count)


def bent(reader: Reader, flux: NDArray[np.float64], axis: tuple[float, float],
         have: NDArray[np.float64], want: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ``flux`` stretched radially about ``axis`` so a radius ``have`` becomes ``want``.

    Inside the boundary each direction is stretched by ``want / have`` (held within
    :data:`STRETCH_LIMIT`); beyond it the stretch fades to none over :data:`FADE` of
    the local radius. The new map at a point is the old map at the point carried
    there, sampled bilinearly.
    """
    theta = np.arctan2(reader.zz - axis[1], reader.rr - axis[0])
    radius = np.hypot(reader.rr - axis[0], reader.zz - axis[1])
    ext, have_ext = _periodic(have)
    _, want_ext = _periodic(want)
    local_have = np.interp(theta, ext, have_ext)
    ratio = np.clip(np.interp(theta, ext, want_ext) / local_have,
                    1.0 - STRETCH_LIMIT, 1.0 + STRETCH_LIMIT)
    beyond = np.clip(radius / local_have - 1.0, 0.0, None) / FADE
    stretch = 1.0 + np.exp(-(beyond**2)) * (ratio - 1.0)
    source_r = axis[0] + (reader.rr - axis[0]) / stretch
    source_z = axis[1] + (reader.zz - axis[1]) / stretch
    step_r = float(reader.r_grid[1] - reader.r_grid[0])
    step_z = float(reader.z_grid[1] - reader.z_grid[0])
    sampled: NDArray[np.float64] = ndimage.map_coordinates(
        flux, [(source_z - reader.z_grid[0]) / step_z, (source_r - reader.r_grid[0]) / step_r],
        order=1, mode="nearest")
    return sampled


def edited_frame(reader: Reader, flux: NDArray[np.float64], wanted: NDArray[np.float64],
                 share: float = SHARE, rounds: int = ROUNDS,
                 ) -> tuple[NDArray[np.float64], Reading]:
    """Return one map bent until its boundary follows ``wanted`` radii, and what it then reads.

    The first bend goes ``share`` of the way from the map's radii to ``wanted``; the
    later rounds correct towards that same aim. A frame without a boundary, or with
    a wanted radius that is not finite, is left as it was; a round that loses the
    boundary is undone.
    """
    reading = reader.read(flux)
    if reading.contour is None or not np.all(np.isfinite(wanted)) or np.any(wanted <= 0.0):
        return flux, reading
    start = radii(reading.contour, reading.axis, wanted.size)
    aim = wanted if share == SHARE else start + share * (wanted - start)
    result = flux
    for _ in range(rounds):
        assert reading.contour is not None  # noqa: S101 - checked before the loop and below
        have = radii(reading.contour, reading.axis, wanted.size)
        if float(np.max(np.abs(aim - have))) < TOLERANCE:
            break
        trial = bent(reader, result, reading.axis, have, aim)
        trial_reading = reader.read(trial)
        if trial_reading.contour is None:
            break
        result, reading = trial, trial_reading
    return result, reading


def edited(maps: NDArray[np.floating], targets: Mapping[str, NDArray[np.floating]],
           machine: str, backend: ReadingBackend | None = None, share: float = SHARE,
           rounds: int = ROUNDS) -> NDArray[np.float64]:
    """Return every frame of a shot bent until its boundary follows its wanted radii.

    ``targets`` maps each of :func:`label_names` to one value per frame.

    Raises
    ------
    ValueError
        If a label is missing, ``share`` is not in (0, 1], or ``rounds`` is below one.
    """
    names = label_names(len(targets))
    missing = [name for name in names if name not in targets]
    if missing or not targets:
        raise ValueError(f"the contour targets lack {missing or 'every direction'}")
    if not 0.0 < share <= 1.0 or rounds < 1:
        raise ValueError("the contour edit needs a share in (0, 1] and one round or more")
    reader = Reader(machine, backend)
    wanted = np.column_stack([np.asarray(targets[name], dtype=np.float64) for name in names])
    out = np.empty(maps.shape, dtype=np.float64)
    for frame in range(maps.shape[0]):
        out[frame], _reading = edited_frame(
            reader, np.asarray(maps[frame], dtype=np.float64), wanted[frame], share, rounds)
    return out
