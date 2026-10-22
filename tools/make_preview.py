# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — repository preview image
"""Render the README preview image from one published reconstruction frame.

The drawing shows predicted flux surfaces of one real frame: the frame of
``tools/reference/d71_figure_frames.npz`` nearest to the median boundary
distance after the finisher (the "typical" row of the reconstruction figure).
Only the predicted map is drawn. The scores in the text are those of
Development submission 986068 on the public DIII-D test fold. The text uses
Inter when it is installed and DejaVu Sans otherwise, so the pixels depend on
the installed fonts; the drawn data do not.

Usage: ``python tools/make_preview.py [--data NPZ] [--out PNG]``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 - the backend is chosen before pyplot is imported
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.path import Path as Polygon  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "reference" / "d71_figure_frames.npz"
OUT = HERE.parent / "docs" / "assets" / "fusion-equilibrium-challenge-preview.png"
BG, FG, BODY, MUTED = "#0b1220", "#eef2f8", "#c3cde0", "#8b9bb8"
GOLD, RULE, OPEN = "#ffcf5c", "#273248", "#3a4762"
LEVELS = np.linspace(0.1, 0.9, 9)
F64 = NDArray[np.float64]


@dataclass(frozen=True)
class Surfaces:
    """Normalised flux of one map and the data drawn from it."""

    radius: F64
    height: F64
    psin: F64
    inside: NDArray[np.bool_]
    lcfs: F64
    axis_index: tuple[int, int]
    distance_mm: float


def bilinear(radius: F64, height: F64, field: F64, points: F64) -> F64:
    """Interpolate a ``(len(height), len(radius))`` field linearly at ``(R, Z)`` points.

    Parameters
    ----------
    radius, height : ndarray
        Strictly increasing grid coordinates in metres.
    field : ndarray
        Values on the grid, indexed ``[z, r]``.
    points : ndarray
        ``(n, 2)`` array of ``(R, Z)`` positions inside the grid.

    Returns
    -------
    ndarray
        Interpolated values, one per point.

    Raises
    ------
    ValueError
        A point lies outside the grid.
    """
    r, z = points[:, 0], points[:, 1]
    if np.any((r < radius[0]) | (r > radius[-1]) | (z < height[0]) | (z > height[-1])):
        raise ValueError("Interpolation point outside the grid")
    i = np.clip(np.searchsorted(radius, r, side="right") - 1, 0, len(radius) - 2)
    j = np.clip(np.searchsorted(height, z, side="right") - 1, 0, len(height) - 2)
    tr = (r - radius[i]) / (radius[i + 1] - radius[i])
    tz = (z - height[j]) / (height[j + 1] - height[j])
    return np.asarray(
        field[j, i] * (1 - tz) * (1 - tr)
        + field[j, i + 1] * (1 - tz) * tr
        + field[j + 1, i] * tz * (1 - tr)
        + field[j + 1, i + 1] * tz * tr,
        dtype=np.float64,
    )


def surfaces(data: Path) -> Surfaces:
    """Read the typical frame and normalise its flux between axis and boundary.

    Parameters
    ----------
    data : Path
        The published reconstruction frames (``d71_figure_frames.npz``).

    Returns
    -------
    Surfaces
        Grid, normalised flux (0 on the axis, 1 on the boundary), plasma mask,
        boundary, axis grid index and the frame's boundary distance in millimetres.

    Raises
    ------
    ValueError
        The frame has no plasma pixels or a degenerate flux.
    """
    with np.load(data, allow_pickle=False) as stored:
        radius, height = stored["R"].astype(np.float64), stored["Z"].astype(np.float64)
        psi = stored["p50_margin_psi"].astype(np.float64)
        lcfs = stored["p50_margin_lcfs"].astype(np.float64)
        distance_mm = 1000 * float(stored["p50_distance_m"])
    boundary = float(np.mean(bilinear(radius, height, psi, lcfs)))
    rr, zz = np.meshgrid(radius, height)
    inside = Polygon(lcfs).contains_points(np.c_[rr.ravel(), zz.ravel()]).reshape(psi.shape)
    if not inside.any():
        raise ValueError("No plasma pixel inside the boundary")
    hi, lo = float(psi[inside].max()), float(psi[inside].min())
    axis = hi if abs(hi - boundary) > abs(lo - boundary) else lo
    if axis == boundary:
        raise ValueError("Degenerate flux between axis and boundary")
    psin = (psi - axis) / (boundary - axis)
    iz, ir = np.unravel_index(int(np.nanargmin(np.where(inside, psin, np.nan))), psin.shape)
    return Surfaces(radius, height, psin, inside, lcfs, (int(iz), int(ir)), distance_mm)


def font(candidates: Sequence[str] | None = None) -> str:
    """Return Inter when one of the font files is Inter, else DejaVu Sans.

    Parameters
    ----------
    candidates : Sequence[str] or None
        Font file paths; the installed system fonts when omitted.

    Returns
    -------
    str
        Font family used for every text of the image.
    """
    paths = font_manager.findSystemFonts() if candidates is None else candidates
    inter = [path for path in paths if "inter" in Path(path).stem.lower()]
    list(map(font_manager.fontManager.addfont, inter))
    return "Inter" if inter else "DejaVu Sans"


def render(data: Path, out: Path) -> Surfaces:
    """Draw the 1280 x 640 preview image.

    Parameters
    ----------
    data : Path
        The published reconstruction frames.
    out : Path
        PNG file to write.

    Returns
    -------
    Surfaces
        The data drawn, for checking against the published frames.
    """
    shown = surfaces(data)
    radius, height = shown.radius, shown.height
    family = font()
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor=BG)

    def text(x: float, y: float, s: str, size: float, color: str = FG, weight: str = "normal") -> None:
        fig.text(
            x,
            y,
            s,
            fontsize=size,
            color=color,
            fontweight=weight,
            family=family,
            va="baseline",
            transform=fig.transFigure,
        )

    text(0.055, 0.815, "FUSION EQUILIBRIUM CHALLENGE  ·  NeurIPS 2026", 15, MUTED, "semibold")
    text(0.055, 0.705, "Tokamak equilibrium", 40, weight="semibold")
    text(0.055, 0.615, "without magnetic probes", 40, weight="semibold")
    text(0.055, 0.51, "The Anulum entry: DIII-D and MAST flux maps from coil currents,", 17, BODY)
    text(0.055, 0.465, "measured plasma current and Thomson profiles.", 17, BODY)
    text(0.055, 0.365, "DIII-D  S 0.99636   ·   flux R² 0.9999", 21, GOLD, "semibold")
    text(0.055, 0.318, "Development phase, public test fold (submission 986068)", 13, MUTED)
    text(0.055, 0.272, "Code · weights · study · every experiment, including the failures", 14, MUTED)
    fig.add_artist(Line2D([0.055, 0.548], [0.22, 0.22], color=RULE, lw=1, transform=fig.transFigure))
    text(0.055, 0.15, "github.com/anulum/fusion-equilibrium-challenge", 17)
    text(0.055, 0.102, "Anulum  ·  Arcane Sapience", 13, MUTED)

    ax = fig.add_axes((0.66, 0.1, 0.31, 0.82), facecolor=BG)
    cmap = plt.get_cmap("plasma")
    ax.contour(
        radius,
        height,
        np.where(shown.inside, shown.psin, np.nan),
        levels=LEVELS,
        colors=[cmap(0.15 + 0.7 * level) for level in LEVELS],
        linewidths=1.3,
    )
    ax.contour(radius, height, shown.psin, levels=[1.08, 1.18, 1.3], colors=[OPEN], linewidths=0.8)
    ax.plot(shown.lcfs[:, 0], shown.lcfs[:, 1], color=GOLD, lw=2.6, solid_capstyle="round")
    iz, ir = shown.axis_index
    ax.plot(radius[ir], height[iz], marker="+", color=GOLD, ms=13, mew=1.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(shown.lcfs[:, 0].min() - 0.12, shown.lcfs[:, 0].max() + 0.12)
    ax.set_ylim(shown.lcfs[:, 1].min() - 0.12, shown.lcfs[:, 1].max() + 0.12)
    fig.text(
        0.815,
        0.045,
        "Predicted flux surfaces, one DIII-D frame\n"
        f"(median boundary error, {shown.distance_mm:.1f} mm from EFIT)",
        fontsize=11.5,
        color=MUTED,
        family=family,
        ha="center",
        va="baseline",
        transform=fig.transFigure,
        linespacing=1.3,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=BG, dpi=100)
    plt.close(fig)
    return shown


def main(argv: Sequence[str] | None = None) -> int:
    """Render the preview image.

    Parameters
    ----------
    argv : Sequence[str] or None
        Command arguments, or the process arguments when omitted.

    Returns
    -------
    int
        Zero after writing the image; one when the frames cannot be read or drawn.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    try:
        shown = render(args.data, args.out)
    except (OSError, KeyError, ValueError):
        print("Unable to read or draw the reconstruction frame.")
        return 1
    print(f"Wrote {args.out} (boundary distance {shown.distance_mm:.1f} mm)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
