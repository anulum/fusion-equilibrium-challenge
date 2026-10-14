# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — where a map's X-points are, and a local move that puts them elsewhere

"""Find the X-points at the ends of a map's boundary, and move them.

On the validation part the scorer's boundary distance is lost almost entirely
at the X-points: in 79 % of frames the point of the symmetric Hausdorff
distance lies on the lower or upper X-point side, carrying 85 % of the summed
distance, and the frame's distance correlates at 0.95 with the error of the
boundary's extreme point (ledger, 23 Sep). A stretch of the whole map cannot
correct that — matching the true extreme heights by an up/down stretch made the
distance worse — because the loss is the local geometry around the saddle.

An X-point is a saddle of the flux. Near it the flux is, to second order,
``ψ(x) = ψ₀ + ½ (x - x₀)ᵀ H (x - x₀)`` with ``det H < 0``. Adding a linear field
``c · (x - x₀)`` under a window that is one at ``x₀`` moves the saddle to
``x₀ - H⁻¹ c``; ``c = H (x₀ - x_wanted)`` puts it at ``x_wanted`` to first order.
The window is Gaussian, :data:`WINDOW_WIDTH` wide, so the flux far from the
X-point is unchanged. With the true X-points put into our maps this way the
mean boundary distance of the validation part fell from 0.0133 to 0.0100
(an oracle bound, widths 0.05–0.25 m scanned; 0.15 m best).

The saddle is found where the scorer finds the boundary's ends: within
:data:`SEARCH_CELLS` cells of the boundary's lowest and highest point, at the
grid minimum of ``|∇ψ|²``, refined by one Newton step on the gradient.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

from fec_adapter.diii_d_accuracy.shape_edit import Reader, Reading

End = Literal["lower", "upper"]

ENDS: Final[tuple[End, End]] = ("lower", "upper")
"""The two ends of the boundary an X-point can sit at, lowest point first."""

LABELS: Final[tuple[str, ...]] = ("x_lower_r", "x_lower_z", "x_upper_r", "x_upper_z")
"""Table and network names of the two X-points' positions, in metres."""

SEARCH_CELLS: Final = 4
"""Half width, in cells, of the square about a boundary end searched for the saddle."""

WINDOW_WIDTH: Final = 0.15
"""Width, in metres, of the Gaussian window of the move (validation oracle scan, 23 Sep)."""

MOVE_ROUNDS: Final = 3
"""Corrections of one X-point's move; each re-finds the saddle."""

MOVE_TOLERANCE: Final = 1.0e-4
"""Distance, in metres, below which an X-point counts as in place."""

MOVE_LIMIT: Final = 0.05
"""Largest move, in metres, asked of one X-point; a wild target cannot tear the map."""

SHARE: Final = 1.0
"""Share of the way from the map's saddle to the network's X-point that the move goes.

The shape edits gained on every term by going part of the way to the networks' noisy
targets (validation, 23–24 Sep); the same holds out the question for the X-points."""


@dataclass(frozen=True)
class Saddle:
    """One X-point: where it is, and the flux's curvature there.

    Attributes
    ----------
    position:
        ``(R, Z)`` in metres.
    hessian:
        The second derivatives ``[[ψ_RR, ψ_RZ], [ψ_RZ, ψ_ZZ]]`` there; ``det < 0``.
    """

    position: NDArray[np.float64]
    hessian: NDArray[np.float64]


def saddle_near(reader: Reader, flux: NDArray[np.float64],
                near: NDArray[np.float64]) -> Saddle | None:
    """Return the saddle of ``flux`` near the point ``near`` ``(R, Z)``, or ``None``.

    ``None`` when the flattest cell of the searched square is on the grid's edge, is
    not a saddle (``det H ≥ 0``), or the Newton step leaves the cell.
    """
    r_grid, z_grid = reader.r_grid, reader.z_grid
    step_r = float(r_grid[1] - r_grid[0])
    step_z = float(z_grid[1] - z_grid[0])
    grad_z, grad_r = (np.asarray(part, dtype=np.float64)
                      for part in np.gradient(flux, step_z, step_r))
    row = round((float(near[1]) - float(z_grid[0])) / step_z)
    column = round((float(near[0]) - float(r_grid[0])) / step_r)
    rows = slice(max(1, row - SEARCH_CELLS), min(len(z_grid) - 1, row + SEARCH_CELLS + 1))
    columns = slice(max(1, column - SEARCH_CELLS),
                    min(len(r_grid) - 1, column + SEARCH_CELLS + 1))
    if rows.start >= rows.stop or columns.start >= columns.stop:
        return None
    flatness = grad_r[rows, columns] ** 2 + grad_z[rows, columns] ** 2
    local_row, local_column = np.unravel_index(int(np.argmin(flatness)), flatness.shape)
    i, j = rows.start + int(local_row), columns.start + int(local_column)
    h_rr = (flux[i, j + 1] - 2.0 * flux[i, j] + flux[i, j - 1]) / step_r**2
    h_zz = (flux[i + 1, j] - 2.0 * flux[i, j] + flux[i - 1, j]) / step_z**2
    h_rz = (flux[i + 1, j + 1] - flux[i + 1, j - 1] - flux[i - 1, j + 1]
            + flux[i - 1, j - 1]) / (4.0 * step_r * step_z)
    hessian = np.array([[h_rr, h_rz], [h_rz, h_zz]], dtype=np.float64)
    if not np.all(np.isfinite(hessian)) or np.linalg.det(hessian) >= 0.0:
        return None
    gradient = np.array([grad_r[i, j], grad_z[i, j]], dtype=np.float64)
    newton = np.linalg.solve(hessian, gradient)
    if abs(newton[0]) > step_r or abs(newton[1]) > step_z:
        return None
    position = np.array([r_grid[j], z_grid[i]], dtype=np.float64) - newton
    return Saddle(position, hessian)


def boundary_end(contour: NDArray[np.float64], end: End) -> NDArray[np.float64]:
    """Return the boundary's lowest (``"lower"``) or highest (``"upper"``) point."""
    index = int(np.argmin(contour[:, 1])) if end == "lower" else int(np.argmax(contour[:, 1]))
    return np.asarray(contour[index], dtype=np.float64)


def boundary_saddles(reader: Reader, flux: NDArray[np.float64],
                     reading: Reading | None = None) -> dict[End, Saddle | None]:
    """Return the saddle at each end of the scorer's boundary of ``flux``, ``None`` where none."""
    reading = reader.read(flux) if reading is None else reading
    if reading.contour is None:
        return dict.fromkeys(ENDS)
    return {end: saddle_near(reader, flux, boundary_end(reading.contour, end)) for end in ENDS}


def x_point_positions(reader: Reader, flux: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the two X-points of a map as :data:`LABELS`, ``NaN`` where an end has none."""
    out = np.full(len(LABELS), np.nan)
    saddles = boundary_saddles(reader, flux)
    for index, end in enumerate(ENDS):
        saddle = saddles[end]
        if saddle is not None:
            out[2 * index: 2 * index + 2] = saddle.position
    return out


def moved(reader: Reader, flux: NDArray[np.float64], saddle: Saddle,
          wanted: NDArray[np.float64], width: float = WINDOW_WIDTH) -> NDArray[np.float64]:
    """Return ``flux`` with the saddle moved towards ``wanted`` under the Gaussian window.

    The move is first order: the window's own curvature shortens it by a fraction
    that grows with the distance (a 5 cm move lands about 4 cm away), which
    :func:`with_x_points` corrects in further rounds. The move asked for is held
    within :data:`MOVE_LIMIT`.
    """
    shift = np.asarray(wanted, dtype=np.float64) - saddle.position
    length = float(np.hypot(shift[0], shift[1]))
    if length > MOVE_LIMIT:
        shift *= MOVE_LIMIT / length
    push = -(saddle.hessian @ shift)
    offset_r = reader.rr - saddle.position[0]
    offset_z = reader.zz - saddle.position[1]
    window = np.exp(-(offset_r**2 + offset_z**2) / width**2)
    result: NDArray[np.float64] = flux + window * (push[0] * offset_r + push[1] * offset_z)
    return result


def with_x_points(
    reader: Reader, flux: NDArray[np.float64], wanted: dict[End, NDArray[np.float64]],
    share: float = SHARE, width: float = WINDOW_WIDTH, rounds: int = MOVE_ROUNDS,
) -> tuple[NDArray[np.float64], Reading]:
    """Return ``flux`` with each boundary X-point moved to ``wanted`` and what the scorer reads.

    An end whose target is missing or not finite, or where either the map has no
    saddle, is left as it is; a round that loses the boundary is undone. Each
    X-point goes ``share`` of the way from where the map first has it, under a
    window ``width`` wide, in at most ``rounds`` corrections.
    """
    reading = reader.read(flux)
    for end in ENDS:
        target = wanted.get(end)
        if target is None or not np.all(np.isfinite(target)):
            continue
        if share != SHARE:  # the whole way stays bit for bit the target itself
            if reading.contour is None:
                continue
            start = saddle_near(reader, flux, boundary_end(reading.contour, end))
            if start is None:
                continue
            target = start.position + share * (np.asarray(target, dtype=np.float64)
                                               - start.position)
        for _ in range(rounds):
            if reading.contour is None:
                break
            saddle = saddle_near(reader, flux, boundary_end(reading.contour, end))
            if saddle is None or float(np.hypot(*(saddle.position - target))) < MOVE_TOLERANCE:
                break
            trial = moved(reader, flux, saddle, target, width)
            trial_reading = reader.read(trial)
            if trial_reading.contour is None:
                break
            flux, reading = trial, trial_reading
    return flux, reading


def edited_x_points(
    maps: NDArray[np.floating], targets: dict[str, NDArray[np.floating]], machine: str,
    share: float = SHARE, width: float = WINDOW_WIDTH, rounds: int = MOVE_ROUNDS,
) -> NDArray[np.float64]:
    """Return every frame of a shot with its boundary X-points moved to the targets.

    ``targets`` maps each of :data:`LABELS` to one value per frame. A frame whose
    map has no saddle at an end, or whose target there is not finite, keeps that end
    as it is, so a plasma the network reads as limited is not given an X-point.
    ``share``, ``width`` and ``rounds`` are passed to :func:`with_x_points`.

    Raises
    ------
    ValueError
        If ``share`` is not in (0, 1], ``width`` is not positive and finite, or
        ``rounds`` is below one.
    """
    if not 0.0 < share <= 1.0:
        raise ValueError("a share of the way to the X-point must lie in (0, 1]")
    if not (np.isfinite(width) and width > 0.0) or rounds < 1:
        raise ValueError("the X-point window must be a positive width and one round or more")
    reader = Reader(machine)
    out = np.empty(maps.shape, dtype=np.float64)
    for frame in range(maps.shape[0]):
        wanted: dict[End, NDArray[np.float64]] = {
            end: np.array([targets[f"x_{end}_r"][frame], targets[f"x_{end}_z"][frame]],
                          dtype=np.float64)
            for end in ENDS
        }
        out[frame], _reading = with_x_points(
            reader, np.asarray(maps[frame], dtype=np.float64), wanted, share, width, rounds
        )
    return out
