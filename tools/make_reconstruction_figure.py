# SPDX-License-Identifier: AGPL-3.0-or-later
# Fusion Equilibrium Challenge entry — reference, prediction and difference on four development frames (D71)
"""python tools/make_reconstruction_figure.py  (writes docs/figures/fig6_reconstruction.png)

The four frames were chosen before any D71 result, by a fixed rule: pooled over all 78 875 frames of the 358
development shots, the frames nearest to the 50th, 90th and 99th percentile of the submitted map's boundary
distance, and the largest. The evaluated networks were not fitted on these shots; the shots had been used before
in model comparisons. Reference maps and inputs: organisers' DIII-D data, CC BY 4.0.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "reference" / "d71_figure_frames.npz"
OUT = HERE.parent / "docs" / "figures" / "fig6_reconstruction.png"
INK, MUTED, REF, PRED, PRE = "#1b2333", "#5b6577", "#1b2333", "#c2410c", "#2563eb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK})
LABELS = {"p50": "typical (median)", "p90": "90th percentile", "p99": "99th percentile", "max": "largest"}


def main() -> None:
    z = np.load(DATA, allow_pickle=False)
    R, Z = z["R"], z["Z"]
    rows = [str(k) for k in z["order"]]
    fig = plt.figure(figsize=(11.5, 4.5 * len(rows) + .6), layout="constrained")
    subfigures = fig.subfigures(len(rows), 1)
    axes = np.array([sub.subplots(1, 4) for sub in subfigures])
    for i, key in enumerate(rows):
        ref, pre, post = z[f"{key}_reference_psi"], z[f"{key}_smoothed_psi"], z[f"{key}_margin_psi"]
        c_ref, c_pre, c_post = z[f"{key}_reference_lcfs"], z[f"{key}_smoothed_lcfs"], z[f"{key}_margin_lcfs"]
        levels = np.linspace(min(ref.min(), post.min()), max(ref.max(), post.max()), 26)
        span = float(np.nanmax(np.abs(np.concatenate([(post - ref).ravel(), (pre - ref).ravel()]))))
        title = (f"{LABELS[key]}: {str(z[f'{key}_stem'])[9:]}, t = {float(z[f'{key}_time_ms']):.0f} ms, "
                 f"boundary distance {1000 * float(z[f'{key}_distance_m']):.1f} mm")
        for j, (field, name) in enumerate(((ref, "reference (EFIT)"), (post, "submitted map"))):
            ax = axes[i, j]
            ax.contour(R, Z, field, levels=levels, cmap="plasma", linewidths=.8)
            ax.set_title(name, fontsize=9.5, color=INK)
        for j, (field, name) in enumerate(((post - ref, "submitted − reference"),
                                           (pre - ref, "before the finisher − reference")), start=2):
            ax = axes[i, j]
            image = ax.pcolormesh(R, Z, field, cmap="RdBu_r", vmin=-span, vmax=span, shading="auto")
            ax.set_title(name, fontsize=9.5, color=INK)
            if j == 3:
                subfigures[i].colorbar(image, ax=ax, shrink=.8, label="Δψ [Wb/rad]")
        for ax in axes[i]:
            ax.plot(c_ref[:, 0], c_ref[:, 1], color=REF, lw=1.6, label="reference boundary")
            if len(c_post):
                ax.plot(c_post[:, 0], c_post[:, 1], color=PRED, lw=1.3, ls="--", label="submitted boundary")
            if len(c_pre) and ax is axes[i, 3]:
                ax.plot(c_pre[:, 0], c_pre[:, 1], color=PRE, lw=1.1, ls=":", label="boundary before the finisher")
            ax.set_aspect("equal"); ax.set_xlabel("R [m]")
        axes[i, 0].set_ylabel("Z [m]")
        subfigures[i].suptitle(title, x=.01, ha="left", fontsize=10.5, color=INK)
    handles, labels = [], []
    for ax in axes[0]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h); labels.append(lab)
    fig.suptitle("DIII-D development shots: reference, submitted map and their difference (no magnetic measurement used)",
                 color=INK, fontsize=12.5, x=.01, ha="left")
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
