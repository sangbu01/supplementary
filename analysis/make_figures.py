# -*- coding: utf-8 -*-
"""Regenerate the two wide figures from the audited per-condition counts."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

FIG = os.path.dirname(os.path.abspath(__file__))

rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COL_EN = "#4C72B0"
COL_NON = "#C44E52"
COL_ZH = "#DD8452"
COL_GREY = "#8C8C8C"


def wilson(k, n, z=1.959963984540054):
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def barpanel(ax, labels, ks, ns, colors, title, ylabel=False, rot=0):
    vals = [100 * k / n for k, n in zip(ks, ns)]
    lo = [max(0.0, 100 * (v / 100 - wilson(k, n)[0])) for v, k, n in zip(vals, ks, ns)]
    hi = [max(0.0, 100 * (wilson(k, n)[1] - v / 100)) for v, k, n in zip(vals, ks, ns)]
    x = np.arange(len(labels))
    ax.bar(x, vals, width=0.68, color=colors, edgecolor="black", linewidth=0.4,
           yerr=[lo, hi], capsize=1.8, error_kw={"linewidth": 0.6})
    for xi, v, k, n in zip(x, vals, ks, ns):
        ax.text(xi, min(v + 2.5, 100), f"{v:.1f}", ha="center", va="bottom", fontsize=6.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rot, ha="center" if rot == 0 else "right")
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title(title, pad=3)
    if ylabel:
        ax.set_ylabel("Task success (%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2)


# ------------------------------------------------------------------ Figure 1
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.05), gridspec_kw={"wspace": 0.30})

barpanel(axes[0], ["orig.", "para.", "P1", "P2", "P3", "P4"],
         [27, 2, 16, 19, 6, 19], [45] * 6,
         [COL_EN, COL_NON, COL_GREY, COL_GREY, COL_GREY, COL_GREY],
         "(a)  Hammer: same-language rewording", ylabel=True)

barpanel(axes[1], ["en", "para.", "es", "zh", "sw", "hi", "ar"],
         [45, 44, 17, 19, 19, 23, 24], [45] * 7,
         [COL_EN, COL_EN, COL_NON, COL_NON, COL_NON, COL_NON, COL_NON],
         "(b)  Akita: instruction language")

barpanel(axes[2], ["en", "empty", "irrelevant"],
         [39, 39, 39], [75, 75, 75],
         [COL_EN, COL_GREY, COL_GREY],
         "(c)  Control suite: no instruction")
axes[2].text(1.0, 8, "identical success\n(39/75 each)", ha="center", va="bottom", fontsize=6.3)

fig.savefig(os.path.join(FIG, "fig1_two_patterns.pdf"), bbox_inches="tight", pad_inches=0.02)
fig.savefig(os.path.join(FIG, "fig1_two_patterns.png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

# ------------------------------------------------------------------ Figure 2
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.15), gridspec_kw={"wspace": 0.32})

# (a) frame x relation
labels = ["EN rel", "ZH rel", "EN rel", "ZH rel"]
ks = [30, 29, 20, 21]
ns = [30, 30, 30, 30]
colors = [COL_EN, COL_EN, COL_ZH, COL_ZH]
ax = axes[0]
barpanel(ax, labels, ks, ns, colors, "(a)  Frame $\\times$ relation language", ylabel=True)
ax.axvline(1.5, color="0.4", linewidth=0.6, linestyle="--")
ax.text(0.5, 112, "English frame", ha="center", fontsize=6.5)
ax.text(2.5, 112, "Chinese frame", ha="center", fontsize=6.5)

# (b) frame x head verb
labels = ["EN verb", "ZH verb", "EN verb", "ZH verb"]
ks = [29, 60, 42, 30]
ns = [30, 60, 60, 60]
ax = axes[1]
barpanel(ax, labels, ks, ns, colors, "(b)  Frame $\\times$ head-verb language")
ax.axvline(1.5, color="0.4", linewidth=0.6, linestyle="--")
ax.text(0.5, 112, "English frame", ha="center", fontsize=6.5)
ax.text(2.5, 112, "Chinese frame", ha="center", fontsize=6.5)
for xi, (k, n) in enumerate(zip(ks, ns)):
    ax.text(xi, 3, f"{k}/{n}", ha="center", va="bottom", fontsize=5.8, color="white")

# (c) soda
labels = ["English", "Chinese", "+EN nouns", "+EN dest."]
ks = [29, 18, 21, 17]
ns = [30] * 4
ax = axes[2]
barpanel(ax, labels, ks, ns, [COL_EN, COL_ZH, COL_ZH, COL_ZH],
         "(c)  Soda: no look-alike object", rot=20)

fig.savefig(os.path.join(FIG, "fig2_frame_mechanism.pdf"), bbox_inches="tight", pad_inches=0.02)
fig.savefig(os.path.join(FIG, "fig2_frame_mechanism.png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("figures written")
