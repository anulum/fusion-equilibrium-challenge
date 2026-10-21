# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — repository preview image
"""python tools/make_preview.py  (writes docs/assets/fusion-equilibrium-challenge-preview.png)

The drawing shows predicted flux surfaces of one real frame: the frame of tools/reference/d71_figure_frames.npz
nearest to the median boundary distance after the finisher (the "typical" row of fig6). Only the predicted map is
drawn. The scores in the text are those of Development submission 986068 on the public DIII-D test fold. The text
uses Inter when it is installed and falls back to DejaVu Sans otherwise.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.path import Path as Polygon
from scipy.interpolate import RegularGridInterpolator

HERE = Path(__file__).resolve().parent
DATA = HERE / "reference" / "d71_figure_frames.npz"
OUT = HERE.parent / "docs" / "assets" / "fusion-equilibrium-challenge-preview.png"
BG, FG, BODY, MUTED, GOLD, RULE, OPEN = "#0b1220", "#eef2f8", "#c3cde0", "#8b9bb8", "#ffcf5c", "#273248", "#3a4762"


def font() -> str:
    """Return Inter if installed, else DejaVu Sans."""
    for path in font_manager.findSystemFonts():
        if "inter" in Path(path).stem.lower():
            font_manager.fontManager.addfont(path)
    return "Inter" if any(f.name == "Inter" for f in font_manager.fontManager.ttflist) else "DejaVu Sans"


def main() -> None:
    """Render the 1280 x 640 preview image."""
    z = np.load(DATA, allow_pickle=False)
    R, Z = z["R"], z["Z"]
    psi, lcfs = z["p50_margin_psi"].astype(np.float64), z["p50_margin_lcfs"]
    distance_mm = 1000 * float(z["p50_distance_m"])
    boundary = float(np.mean(RegularGridInterpolator((Z, R), psi)(lcfs[:, ::-1])))
    rr, zz = np.meshgrid(R, Z)
    inside = Polygon(lcfs).contains_points(np.c_[rr.ravel(), zz.ravel()]).reshape(psi.shape)
    hi, lo = psi[inside].max(), psi[inside].min()
    axis = hi if abs(hi - boundary) > abs(lo - boundary) else lo
    psin = (psi - axis) / (boundary - axis)

    family = font()
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor=BG)

    def text(x: float, y: float, s: str, size: float, color: str = FG, weight: str = "normal") -> None:
        fig.text(x, y, s, fontsize=size, color=color, fontweight=weight, family=family, va="baseline",
                 transform=fig.transFigure)

    text(.055, .815, "FUSION EQUILIBRIUM CHALLENGE  ·  NeurIPS 2026", 15, MUTED, "semibold")
    text(.055, .705, "Tokamak equilibrium", 40, weight="semibold")
    text(.055, .615, "without magnetic probes", 40, weight="semibold")
    text(.055, .51, "The Anulum entry: DIII-D and MAST flux maps from coil currents,", 17, BODY)
    text(.055, .465, "measured plasma current and Thomson profiles.", 17, BODY)
    text(.055, .365, "DIII-D  S 0.99636   ·   flux R² 0.9999", 21, GOLD, "semibold")
    text(.055, .318, "Development phase, public test fold (submission 986068)", 13, MUTED)
    text(.055, .272, "Code · weights · study · every experiment, including the failures", 14, MUTED)
    fig.add_artist(plt.Line2D([.055, .548], [.22, .22], color=RULE, lw=1, transform=fig.transFigure))
    text(.055, .15, "github.com/anulum/fusion-equilibrium-challenge", 17)
    text(.055, .102, "Anulum  ·  Arcane Sapience", 13, MUTED)

    ax = fig.add_axes((.66, .1, .31, .82), facecolor=BG)
    cmap = plt.get_cmap("plasma")
    levels = np.linspace(.1, .9, 9)
    ax.contour(R, Z, np.where(inside, psin, np.nan), levels=levels,
               colors=[cmap(.15 + .7 * level) for level in levels], linewidths=1.3)
    ax.contour(R, Z, psin, levels=[1.08, 1.18, 1.3], colors=[OPEN], linewidths=.8)
    ax.plot(lcfs[:, 0], lcfs[:, 1], color=GOLD, lw=2.6, solid_capstyle="round")
    iz, ir = np.unravel_index(np.nanargmin(np.where(inside, psin, np.nan)), psin.shape)
    ax.plot(R[ir], Z[iz], marker="+", color=GOLD, ms=13, mew=1.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(lcfs[:, 0].min() - .12, lcfs[:, 0].max() + .12)
    ax.set_ylim(lcfs[:, 1].min() - .12, lcfs[:, 1].max() + .12)
    fig.text(.815, .045, f"Predicted flux surfaces, one DIII-D frame\n(median boundary error, {distance_mm:.1f} mm from EFIT)",
             fontsize=11.5, color=MUTED, family=family, ha="center", va="baseline", transform=fig.transFigure,
             linespacing=1.3)
    fig.savefig(OUT, facecolor=BG, dpi=100)


if __name__ == "__main__":
    main()
