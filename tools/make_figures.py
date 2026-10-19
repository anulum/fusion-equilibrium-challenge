# SPDX-License-Identifier: AGPL-3.0-or-later
# Fusion Equilibrium Challenge entry — figures of the study, drawn from numbers recorded in docs/EXPERIMENT_LEDGER.md
"""python tools/make_figures.py  (writes docs/figures/*.png). Every number below cites its ledger row or study section."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
INK, MUTED, ACCENT, GOOD, BAD, GRID = "#1b2333", "#5b6577", "#c2410c", "#15803d", "#b91c1c", "#e5e7eb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# 1. Public DIII-D score, step by step (Codabench Development submissions; study §3, §4.6-§4.23, §6)
steps = [("v25 recipe, 65 networks", 949223, .995960), ("+ all-data ensemble (130)", 952730, .995988),
         ("195 networks", 953621, .995990), ("+ boosted-tree heads", 963357, .996025),
         ("sub-frame input features, 32 networks", 965153, .996223), ("+ tree blend", 966820, .996247),
         ("70 networks (second seeds)", 968291, .996263), ("69 networks", 970217, .996274),
         ("+ q95/βN networks", 975851, .996300), ("+ X-point networks", 981856, .996342),
         ("scalar shares settled", 986068, .996359)]
fig, ax = plt.subplots(figsize=(9.5, 4.2))
x = range(len(steps)); y = [s[2] for s in steps]
ax.plot(x, y, color=ACCENT, lw=2, marker="o", ms=6, zorder=3)
for i, (label, sub, s) in enumerate(steps):
    ax.annotate(f"{label}\n#{sub}", (i, s), textcoords="offset points", xytext=(0, -32 if i % 2 else 12),
                ha="center", fontsize=7.6, color=MUTED)
ax.set_xticks([]); ax.set_ylabel("DIII-D composite S (public fold)")
ax.set_ylim(.99580, .99648); ax.grid(axis="y", color=GRID)
ax.set_title("What moved the public DIII-D score, 28 September to 14 October 2026", loc="left", color=INK, fontsize=12)
save(fig, "fig1_score_progression.png")

# 2. Frozen-protocol experiments: effect and interval in µS (ledger rows 2.40-2.85; study §4.15-§4.26)
rows = [("X-point networks, 238 unscored shots", 45.5, 22.2, 75.5, True, "99 %"),
        ("X-point networks, 357 reserved shots", 36.4, 12.0, 62.4, True, "99 %"),
        ("tree heads blended (proxy, 5936 shots)", 31.0, 20.0, 45.0, True, "95 %"),
        ("q95/βN networks, 358 shots", 20.0, 5.0, 37.0, True, "interval"),
        ("least-squares scalar shares, 358 shots", 14.0, -6.0, 39.0, None, "99 %"),
        ("finisher re-tune, best of 26 settings", 11.3, -1.9, 52.5, False, "99 %"),
        ("X-move by member spread", 3.1, -26.8, 28.6, False, "99 %"),
        ("whole-discharge attention vs control", -6.7, -63.0, 62.0, False, "95 %"),
        ("input clip 50 instead of 8 (heads)", 32.0, -71.0, 133.0, False, "95 %"),
        ("Thomson availability context (heads)", -129.0, -205.0, -18.0, False, "95 %"),
        ("boundary-loss radius readout", -166.1, -208.9, -127.9, False, "99 %"),
        ("residual branch, coarse Thomson bins", -157.3, -240.6, -93.3, False, "99 %"),
        ("residual branch, raw Thomson channels", -197.9, -299.7, -118.2, False, "99 %")]
fig, ax = plt.subplots(figsize=(9.5, 5.6))
for i, (label, m, lo, hi, kept, ci) in enumerate(rows[::-1]):
    c = GOOD if kept else (MUTED if kept is None else BAD)
    ax.plot([lo, hi], [i, i], color=c, lw=2.2, solid_capstyle="round")
    ax.plot(m, i, "o", color=c, ms=6)
    ax.text(hi + 6, i, ci, va="center", fontsize=7.5, color=MUTED)
ax.axvline(0, color=INK, lw=.8); ax.axvline(10, color=ACCENT, lw=.8, ls="--"); ax.set_ylim(-1.4, len(rows) - .4)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=8.6)
ax.set_xlabel("change of the composite score, µS (1 µS = 10⁻⁶), point estimate and interval")
ax.text(13, -0.9, "usual admission threshold +10 µS", color=ACCENT, fontsize=7.8, ha="left")
ax.grid(axis="x", color=GRID)
ax.set_title("Experiments under frozen rules: kept (green), rejected (red), settled on the public fold (grey)", loc="left", fontsize=11.5)
save(fig, "fig2_frozen_experiments.png")

# 3. Where the remaining DIII-D score sits (358 held-out shots, D70 stage 1; study §4.26)
parts = [("flux map", 61), ("q95 and βN", 438), ("boundary distance", 828), ("geometric consistency", 2573)]
fig, ax = plt.subplots(figsize=(8.5, 2.9))
left = 0
for (label, v), col in zip(parts, ["#94a3b8", "#60a5fa", ACCENT, "#7c3aed"]):
    ax.barh(0, v, left=left, color=col, height=.55)
    if v > 600:
        ax.text(left + v / 2, 0, f"{label}\n{v} µS", ha="center", va="center", color="white", fontsize=8.5)
    else:
        ax.annotate(f"{label}: {v} µS", (left + v / 2, .28), xytext=(left + v / 2 + (0 if v > 100 else 120), .55 + (0 if v > 100 else .12)),
                    ha="left", fontsize=8.3, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=.7))
    left += v
ax.set_ylim(-.45, .8); ax.set_yticks([]); ax.set_xlabel("score not yet earned, µS (of 3900 in total on these shots)")
ax.set_title("87 % of the remaining DIII-D deficit is boundary and geometry, not the flux map", loc="left", fontsize=11.5)
save(fig, "fig3_remaining_deficit.png")

# 4. What a given accuracy buys (dose-response, ledger rows 2.56 and 2.59; study §4.20)
dose = [("axis position", 11.0), ("internal inductance", 10.6), ("X-point positions", 7.0), ("elongation", 4.9),
        ("volume", 2.1), ("triangularities", 1.1)]
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.barh([d[0] for d in dose][::-1], [d[1] for d in dose][::-1], color=ACCENT)
for i, (_, v) in enumerate(dose[::-1]):
    ax.text(v + .2, i, f"{v}", va="center", fontsize=9)
ax.set_xlabel("µS of composite score per percent of the target's error removed")
ax.set_title("What a given accuracy buys: measured with true-map targets", loc="left", fontsize=11.5)
ax.grid(axis="x", color=GRID)
save(fig, "fig4_dose_response.png")

# 5. MAST versions on the public fold (ledger rows 5.7, 5.34-5.39, 2.51; study §5)
mast = [("v7  physics solve", .7296, "excluded: demo-shot checks"), ("v8b Thomson-edge warp", .7122, "rejected: breaks current closure"),
        ("v10 vessel + coil freedom", .7246, "excluded with v7"), ("v11 clean rebuild", .7190, "used organisers' envelope file"),
        ("v11f FreeGS default profile", .6011, "rejected"), ("v12 submitted", .7145, "submitted")]
fig, ax = plt.subplots(figsize=(8.5, 3.4))
cols = [MUTED, MUTED, MUTED, MUTED, MUTED, ACCENT]
ax.barh([m[0] for m in mast][::-1], [m[1] for m in mast][::-1], color=cols[::-1])
for i, (_, v, note) in enumerate(mast[::-1]):
    ax.text(v + .003, i, f"{v:.4f}   {note}", va="center", fontsize=8.3, color=INK)
ax.set_xlim(.55, .86); ax.set_xlabel("MAST composite S (public fold, zero-shot)")
ax.set_title("MAST: the admissible reconstruction cost about 0.015 against the best excluded one", loc="left", fontsize=11.5)
save(fig, "fig5_mast_versions.png")
print("written", sorted(p.name for p in OUT.glob("*.png")))
