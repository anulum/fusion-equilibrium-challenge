# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — bend a predicted map until the scorer reads the shape the network says

"""Edit a predicted flux map so that its boundary and interior carry the network's shape outputs.

The scorer reads elongation, the two triangularities, the volume and the
internal inductance off the submitted map and compares them with the same
readings of the true map. Outputs fitted to those readings directly are the
better answer (see :mod:`.shape_labels`), and the only way to submit them is to
put them into the map. Each is moved by a smooth deformation that has that
scalar as its lever and leaves the rest of the map, in particular the axis and
the flux far from the plasma, where it was:

- **elongation and volume** — the map is stretched about the magnetic axis,
  vertically by ``s_z`` and radially by ``s_r``, so the boundary's height and
  width scale with them: ``κ`` goes as ``s_z / s_r``. Around a fixed axis, the volume response includes
  the shifted area centroid; its local isotropic exponent is ``3 - R_axis/R_centroid``;
- **triangularity** — the map is sheared in ``R`` within a band of height
  :data:`TIP_WIDTH` about the boundary's top (bottom), which moves that point
  in ``R`` by the lever times the minor radius and nothing on the midplane;
- **internal inductance** — the flux inside the boundary is reparametrised,
  ``x → x + a·x²(1 - x)`` in normalised flux, which leaves the boundary, its
  level and the field on it unchanged and scales the field inside, so
  ``li = ⟨B_p²⟩ / B_p,boundary²`` moves with ``a``.

The deformation is weighted to one inside the plasma and fades to zero a
share of the minor radius beyond it (:data:`FADE` unless told otherwise), so
the coil regions are not touched. Every
scalar is measured with the organisers' own derivation after each step and
the levers are corrected by fixed-point iteration, so what is matched is what
the scorer will read. A frame whose boundary the scorer cannot extract, or
whose target is missing, is left as it was.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from fec_adapter.scoring_kit import (
    AXIS_SIGN,
    CONS_SCALARS,
    derive_frame,
    extract_lcfs,
    internal_inductance,
    points_in_polygon,
)
from fec_adapter.vessel import envelope, grids

try:
    import fec_native
except ImportError:  # an accelerator only: the organisers' Python is the reference reading
    NATIVE_READING = False
else:
    NATIVE_READING = True

ReadingBackend = Literal["native", "python"]
"""Who reads a map: the compiled port (``fec_native``) or the organisers' Python."""

DEFAULT_BACKEND: Final[ReadingBackend] = "native" if NATIVE_READING else "python"
"""The compiled reading where it is installed, else the organisers' Python.

The two give the same bits: 7001 frames of the submission-10 archive (every thirtieth
public-test shot), all seven scalars and every boundary point equal, the compiled one
11 times faster on the workstation (22 Sep 2026; ``tests/diii_d_accuracy/test_native_reading.py``
holds the equality on synthetic maps).
"""

ROUNDS = 4
"""Fixed-point corrections of the geometric levers; the closest round is kept."""

DAMPING = 0.7
"""Share of each correction applied; the boundary's lowest point is an X-point, whose
position answers a stretch unevenly, and an undamped step can hop over the target."""

INDUCTANCE_ROUNDS = 3
"""Secant steps on the inductance lever."""

FADE = 0.25
"""Share of the minor radius over which the deformation fades out beyond the boundary.

A second edit after the X-point move raised the squared map error of the validation part
by 40 %, and 60 % of that rise lay outside the true boundary, most of it outside the
vessel interior (23 Sep 2026): the stretch carries the vacuum flux along with the plasma.
A shorter fade is the lever that keeps it where it was."""

SCALE_LIMIT = 0.15
"""Largest relative stretch in either direction; a wild output cannot tear the map."""

SHEAR_LIMIT = 0.15
"""Largest triangularity change, in units of the minor radius."""

TIP_WIDTH = 0.35
"""Share of the boundary's half height over which the shear at its top or bottom fades."""

INDUCTANCE_LIMIT = 0.6
"""Largest magnitude of the reparametrisation lever."""

SHARE = 1.0
"""Share of the way from what a map reads to the network's target that an edit goes.

Every edit costs flux, roughly as the square of its lever, while the reading's error
falls linearly with it; the sum of the two is least short of the whole way. One share
for the four geometric scalars and one for the inductance can be set per build."""

CONTOUR_POINTS = 512

INDUCTANCE_MIN_POINTS = 8
"""Boundary points below which the scorer does not derive li (``derive.internal_inductance``)."""

EDITED = ("kappa", "tri_top", "tri_bot", "volume", "li")
"""The scorer's names of what this module can move."""

GEOMETRIC = EDITED[:4]
"""The ones moved by the stretch and the shear; the inductance is moved afterwards."""


@dataclass(frozen=True)
class Reading:
    """What the scorer reads off one map, and the boundary it read it from.

    Attributes
    ----------
    values:
        The seven derived scalars by the scorer's names, ``NaN`` where it counts one undefined.
    contour:
        The boundary, ``(N, 2)`` in metres, or ``None`` where no closed surface was found.
    axis:
        The magnetic axis ``(R, Z)`` in metres.
    li_unclamped:
        The internal inductance before the scorer's plausible range (0.1–5) is applied. The
        scorer counts a map whose li reads 5.2 as undefined, which costs it the fold's mean;
        the edit needs the 5.2 to bring the map back into range.
    """

    values: dict[str, float]
    contour: NDArray[np.float64] | None
    axis: tuple[float, float]
    li_unclamped: float = float("nan")

    @property
    def usable(self) -> bool:
        """Say whether the boundary was found and the shape is finite."""
        return self.contour is not None and all(
            np.isfinite(self.values[name]) for name in EDITED
        )

    @property
    def geometry_usable(self) -> bool:
        """Say whether the boundary was found and its four geometric scalars are finite."""
        return self.contour is not None and all(
            np.isfinite(self.values[name]) for name in GEOMETRIC
        )


class Reader:
    """The organisers' derivation bound to one machine's grid and envelope.

    Parameters
    ----------
    machine:
        The machine whose grid, envelope and flux orientation the maps are read with.
    backend:
        ``"native"`` or ``"python"``; omitted, :data:`DEFAULT_BACKEND`.

    Raises
    ------
    ImportError
        If the compiled reading is asked for and not installed.
    """

    def __init__(self, machine: str, backend: ReadingBackend | None = None) -> None:
        """Load the machine's grid and vessel envelope once."""
        self.machine = machine
        self.r_grid, self.z_grid = grids(machine)
        self.coarse = envelope(machine)
        self.mask = self.coarse.astype(np.float64)
        self.rr, self.zz = np.meshgrid(self.r_grid, self.z_grid)
        self.backend: ReadingBackend = DEFAULT_BACKEND if backend is None else backend
        self._native: fec_native.Reader | None = None
        if self.backend == "native":
            if not NATIVE_READING:
                raise ImportError("the compiled reading (fec_native) is not installed")
            self._native = fec_native.Reader(
                np.asarray(self.r_grid, dtype=np.float64),
                np.asarray(self.z_grid, dtype=np.float64),
                np.ascontiguousarray(self.coarse, dtype=bool), float(AXIS_SIGN[machine]),
                CONTOUR_POINTS,
            )

    def read(self, flux: NDArray[np.float64]) -> Reading:
        """Return the seven derived scalars of a map and the boundary they came from."""
        if self._native is not None:
            scalars, boundary = self._native.read(np.ascontiguousarray(flux, dtype=np.float64))
            return Reading(dict(zip(CONS_SCALARS, scalars[:7], strict=True)), boundary,
                           (scalars[0], scalars[1]), scalars[7])
        contour = extract_lcfs(flux, self.r_grid, self.z_grid, self.machine, self.coarse,
                               self.mask, n_points=CONTOUR_POINTS)
        values = derive_frame(flux, self.r_grid, self.z_grid, self.machine, self.coarse,
                              self.mask, contour=contour)
        li_unclamped = float(values["li"])
        if (not np.isfinite(li_unclamped) and contour is not None
                and len(contour) >= INDUCTANCE_MIN_POINTS):
            li_unclamped = float(internal_inductance(flux, self.r_grid, self.z_grid, self.machine,
                                                     contour))
        return Reading({name: float(values[name]) for name in CONS_SCALARS}, contour,
                       (values["R_axis"], values["Z_axis"]), li_unclamped)


@dataclass(frozen=True)
class Levers:
    """The deformation of one frame."""

    s_r: float = 1.0
    s_z: float = 1.0
    shear_top: float = 0.0
    shear_bot: float = 0.0
    inductance: float = 0.0

    def limited(self) -> Levers:
        """Return the levers held within their limits."""
        return Levers(
            float(np.clip(self.s_r, 1.0 - SCALE_LIMIT, 1.0 + SCALE_LIMIT)),
            float(np.clip(self.s_z, 1.0 - SCALE_LIMIT, 1.0 + SCALE_LIMIT)),
            float(np.clip(self.shear_top, -SHEAR_LIMIT, SHEAR_LIMIT)),
            float(np.clip(self.shear_bot, -SHEAR_LIMIT, SHEAR_LIMIT)),
            float(np.clip(self.inductance, -INDUCTANCE_LIMIT, INDUCTANCE_LIMIT)),
        )


def _weight(reader: Reader, axis: tuple[float, float], half_width: float,
            half_height: float, fade: float = FADE) -> NDArray[np.float64]:
    """Return one inside the boundary's ellipse, fading to zero ``fade`` beyond it."""
    distance = np.hypot((reader.rr - axis[0]) / half_width, (reader.zz - axis[1]) / half_height)
    beyond = np.clip(distance - 1.0, 0.0, None) / fade
    weight: NDArray[np.float64] = np.exp(-(beyond**2))
    return weight


def warped(
    reader: Reader, flux: NDArray[np.float64], levers: Levers, axis: tuple[float, float],
    half_width: float, half_height: float, fade: float = FADE,
) -> NDArray[np.float64]:
    """Return the map stretched and sheared about the axis, as the levers say.

    The new map at ``(R, Z)`` is the old map at the point that is carried there,
    sampled bilinearly, so a stretch by ``s`` makes every surface ``s`` times as
    large in that direction. ``fade`` is the share of the minor radius over which
    the deformation fades out beyond the boundary.
    """
    weight = _weight(reader, axis, half_width, half_height, fade)
    dr, dz = self_steps(reader)
    height = (reader.zz - axis[1]) / half_height
    shear = half_width * (
        levers.shear_top * np.exp(-(((height - 1.0) / TIP_WIDTH) ** 2))
        + levers.shear_bot * np.exp(-(((height + 1.0) / TIP_WIDTH) ** 2))
    )
    source_r = reader.rr + weight * ((reader.rr - axis[0]) * (1.0 / levers.s_r - 1.0) + shear)
    source_z = reader.zz + weight * (reader.zz - axis[1]) * (1.0 / levers.s_z - 1.0)
    rows = (source_z - reader.z_grid[0]) / dz
    columns = (source_r - reader.r_grid[0]) / dr
    sampled: NDArray[np.float64] = ndimage.map_coordinates(
        flux, [rows, columns], order=1, mode="nearest"
    )
    return sampled


def self_steps(reader: Reader) -> tuple[float, float]:
    """Return the grid steps in ``R`` and ``Z``."""
    return (float(reader.r_grid[1] - reader.r_grid[0]),
            float(reader.z_grid[1] - reader.z_grid[0]))


def reparametrised(
    flux: NDArray[np.float64], inside: NDArray[np.bool_], axis_level: float,
    boundary_level: float, lever: float,
) -> NDArray[np.float64]:
    """Return the map with the flux inside the boundary moved by ``x → x + a·x²(1 - x)``.

    ``x`` is the normalised flux, zero on the boundary and one on the axis; the
    boundary's level and the map's gradient on it are unchanged.
    """
    span = axis_level - boundary_level
    if span == 0.0:
        return flux
    x = np.clip((flux - boundary_level) / span, 0.0, 1.0)
    moved = flux + np.where(inside, span * lever * x**2 * (1.0 - x), 0.0)
    return np.asarray(moved, dtype=np.float64)


def _inside(reader: Reader, contour: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Return the cells inside a closed boundary polygon."""
    inside: NDArray[np.bool_] = np.asarray(
        points_in_polygon(reader.rr.ravel(), reader.zz.ravel(), contour), dtype=bool
    ).reshape(reader.rr.shape)
    return inside


def _extent(contour: NDArray[np.float64]) -> tuple[float, float]:
    return (0.5 * float(np.ptp(contour[:, 0])), 0.5 * float(np.ptp(contour[:, 1])))



def volume_kappa_exponent(contour: NDArray[np.float64], axis_r: float) -> float:
    """Log response of V/kappa to common radial/vertical scaling at a fixed axis.

    V(t)/V(1) = t^2 [a + (1-a)t], with a = R_axis/R_area_centroid.
    This is the geometric local response before discrete resampling and fading.
    """
    radius, height = contour[:, 0], contour[:, 1]
    following_r, following_z = np.roll(radius, -1), np.roll(height, -1)
    cross = radius*following_z-following_r*height
    area_twice = np.sum(cross)
    if not np.isfinite(area_twice) or area_twice == 0:
        raise ValueError("Invalid contour area for fixed-axis volume response")
    centroid = np.sum((radius+following_r)*cross)/(3*area_twice)
    exponent = 3-axis_r/centroid
    if not np.isfinite(exponent) or exponent <= 0:
        raise ValueError("Invalid fixed-axis volume response")
    return float(exponent)

def edited_frame(
    reader: Reader, flux: NDArray[np.float64], target: dict[str, float], fade: float = FADE,
    geometric_share: float = SHARE, li_share: float = SHARE,
    scalar_shares: Mapping[str, float] | None = None,
) -> tuple[NDArray[np.float64], Reading]:
    """Return one map bent towards ``target`` (the scorer's names) and what it then reads.

    Targets that are missing or not finite are not pursued; the corresponding
    lever stays at rest. ``fade`` is passed to :func:`warped`. The geometric
    targets are taken ``geometric_share`` of the way from the map's first reading,
    the inductance ``li_share`` of the way from its reading after the geometric step.
    ``scalar_shares`` overrides ``geometric_share`` for the geometric scalars it names.
    """
    first = reader.read(flux)
    if not first.geometry_usable or first.contour is None:
        return flux, first
    axis = first.axis
    half_width, half_height = _extent(first.contour)
    levers = Levers()
    reading = first
    result = flux
    want = {name: target.get(name, np.nan) for name in EDITED}
    shares = dict.fromkeys(GEOMETRIC, geometric_share) | dict(scalar_shares or {})
    for name in GEOMETRIC:
        if shares[name] != SHARE:  # the whole way stays bit for bit the target itself
            want[name] = first.values[name] + shares[name] * (want[name] - first.values[name])
    best = (flux, first, _miss(want, first))
    rounds = ROUNDS if any(np.isfinite(want[name]) for name in GEOMETRIC) else 0
    for _ in range(rounds):
        ratio_kappa = _ratio(want["kappa"], reading.values["kappa"]) ** DAMPING
        ratio_volume = _ratio(want["volume"], reading.values["volume"]) ** DAMPING
        common = (ratio_volume / ratio_kappa) ** (1.0 / volume_kappa_exponent(reading.contour, axis[0]))
        levers = Levers(
            levers.s_r * common, levers.s_z * ratio_kappa * common,
            levers.shear_top + DAMPING * _difference(want["tri_top"], reading.values["tri_top"]),
            levers.shear_bot + DAMPING * _difference(want["tri_bot"], reading.values["tri_bot"]),
        ).limited()
        result = warped(reader, flux, levers, axis, half_width, half_height, fade)
        reading = reader.read(result)
        if not reading.geometry_usable:
            break
        if _miss(want, reading) < best[2]:
            best = (result, reading, _miss(want, reading))
    result, reading, _ = best
    if (np.isfinite(want["li"]) and reading.contour is not None
            and np.isfinite(reading.li_unclamped)):
        wanted = want["li"]
        if li_share != SHARE:
            wanted = reading.li_unclamped + li_share * (wanted - reading.li_unclamped)
        result, reading = _with_inductance(reader, result, reading, wanted)
    return result, reading


def _miss(want: dict[str, float], reading: Reading) -> float:
    """Return the summed relative distance of a reading from the geometric targets pursued."""
    total = 0.0
    for name in GEOMETRIC:
        if np.isfinite(want[name]) and want[name] != 0.0:
            total += abs(reading.values[name] - want[name]) / abs(want[name])
    return total


def _with_inductance(
    reader: Reader, flux: NDArray[np.float64], reading: Reading, wanted: float,
) -> tuple[NDArray[np.float64], Reading]:
    """Move the internal inductance to ``wanted`` by secant steps on the lever.

    The steps follow li before the scorer's plausible range is applied, so that a map
    whose li reads above it — which the scorer counts as undefined — is pulled back.
    """
    assert reading.contour is not None  # noqa: S101 - checked by the caller
    inside = _inside(reader, reading.contour)
    if not inside.any():
        return flux, reading
    axis_level = float(_at(reader, flux, reading.axis))
    boundary_level = float(np.mean(_at_points(reader, flux, reading.contour)))
    lever, value = 0.0, reading.li_unclamped
    trial_lever = 0.1
    best = (flux, reading)
    for _ in range(INDUCTANCE_ROUNDS):
        trial = reparametrised(flux, inside, axis_level, boundary_level, trial_lever)
        trial_reading = reader.read(trial)
        if not trial_reading.geometry_usable or not np.isfinite(trial_reading.li_unclamped):
            break
        trial_value = trial_reading.li_unclamped
        if abs(trial_value - wanted) < abs(best[1].li_unclamped - wanted):
            best = (trial, trial_reading)
        if trial_lever == lever:
            break
        slope = (trial_value - value) / (trial_lever - lever)
        if slope == 0.0 or not np.isfinite(slope):
            break
        lever, value = trial_lever, trial_value
        trial_lever = float(np.clip(lever + (wanted - value) / slope,
                                    -INDUCTANCE_LIMIT, INDUCTANCE_LIMIT))
    return best


def _at(reader: Reader, flux: NDArray[np.float64], point: tuple[float, float]) -> float:
    return float(_at_points(reader, flux, np.asarray([point]))[0])


def _at_points(reader: Reader, flux: NDArray[np.float64],
               points: NDArray[np.float64]) -> NDArray[np.float64]:
    dr, dz = self_steps(reader)
    rows = (points[:, 1] - reader.z_grid[0]) / dz
    columns = (points[:, 0] - reader.r_grid[0]) / dr
    values: NDArray[np.float64] = ndimage.map_coordinates(flux, [rows, columns], order=1,
                                                          mode="nearest")
    return values


def _ratio(wanted: float, current: float) -> float:
    if not np.isfinite(wanted) or not np.isfinite(current) or current == 0.0:
        return 1.0
    return wanted / current


def _difference(wanted: float, current: float) -> float:
    if not np.isfinite(wanted) or not np.isfinite(current):
        return 0.0
    return wanted - current


def edited(
    maps: NDArray[np.floating], targets: Mapping[str, NDArray[np.floating]], machine: str,
    backend: ReadingBackend | None = None, fade: float = FADE,
    geometric_share: float = SHARE, li_share: float = SHARE,
    scalar_shares: Mapping[str, float] | None = None,
) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    """Return every frame of a shot bent towards its targets, and what the scorer then reads.

    ``targets`` maps a scorer name in :data:`EDITED` to one value per frame; ``backend``
    chooses who reads the maps (:class:`Reader`); ``fade`` is the share of the minor
    radius over which the deformation fades out beyond the boundary; the two shares
    are passed to :func:`edited_frame`, and so are the per-scalar ``scalar_shares``.

    Raises
    ------
    ValueError
        If ``fade`` is not a positive finite number, a share is not in (0, 1], or
        ``scalar_shares`` names a scalar outside the geometric four.
    """
    if not (np.isfinite(fade) and fade > 0.0):
        raise ValueError("the deformation's fade must be a positive share of the minor radius")
    named = dict(scalar_shares or {})
    if not set(named) <= set(GEOMETRIC):
        raise ValueError("per-scalar shares are for the geometric scalars only")
    if not all(0.0 < share <= 1.0 for share in (geometric_share, li_share, *named.values())):
        raise ValueError("a share of the way to the target must lie in (0, 1]")
    reader = Reader(machine, backend)
    out = np.empty(maps.shape, dtype=np.float64)
    read = {name: np.full(maps.shape[0], np.nan) for name in CONS_SCALARS}
    for frame in range(maps.shape[0]):
        target = {name: float(values[frame]) for name, values in targets.items()}
        out[frame], reading = edited_frame(reader, np.asarray(maps[frame], dtype=np.float64),
                                           target, fade, geometric_share, li_share, named)
        for name in CONS_SCALARS:
            read[name][frame] = reading.values[name]
    return out, read
