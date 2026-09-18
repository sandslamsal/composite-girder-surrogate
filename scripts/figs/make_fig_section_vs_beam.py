#!/usr/bin/env python
"""Figure: AASHTO stiffness deviation under the two partial-composite
representations (width-scaled section model vs discrete-connector beam
model), by eta_c bin and load regime.

Restyle of the old Fig. 11 with src/utils/figstyle.py.  Message: the
beam model imposes nothing about stiffness reduction (eta_c emerges
from connector equilibrium), yet it independently reproduces the
positive width-scaling deviation at low eta_c and, at service load, its
decay with eta_c toward a small floor.

Bins are the TARGET-eta_c assignment, i.e. the upper block of
tables/tab_sensitivity.tex; the emergent-eta_c reassignment (lower
block, reports/section_vs_beam_revision/rebin_by_emergent.csv) is not
plotted.  The x-axis label says "target" so the figure is unambiguous
against that table.

Marks: three bar series (legend); capped error bars, +/- 1
section-clustered bootstrap standard error, and the grey Delta = 0
reference line (both named in the caption, "grey line, $\Delta = 0$").
The light y-grid is not a data mark.  The beam-model coverage of rows is
quoted in the text of Section 4.4, not in the panels, so the value lives
in one place only.

Data
----
reports/section_vs_beam_revision/sensitivity_summary.csv
    per (eta_bin, regime, metric) mean_pct / se_pct / coverage, written
    by paper2-beam-level/src/validation/sensitivity_sweep.py.

Output
------
paper/revision_2/submission/sources/figures/fig_section_vs_beam.pdf
figures/preview/fig_section_vs_beam.png   (preview only)

Usage
-----
/opt/anaconda3/envs/ops_x86/bin/python scripts/figs/make_fig_section_vs_beam.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.utils import figstyle as FS  # noqa: E402

SUMMARY = REPO / "reports/section_vs_beam_revision/sensitivity_summary.csv"
OUT = REPO / "paper/revision_2/submission/sources/figures/fig_section_vs_beam.pdf"

# metric key -> (legend label, bar kwargs)
SERIES = (
    ("section_matched", "section model, width-scaled $\\eta_c$",
     dict(facecolor=FS.color("opensees"), edgecolor=FS.color("opensees"))),
    ("beam_defl", "beam model, deflection $EI_\\delta$",
     dict(facecolor=FS.color("beam"), edgecolor=FS.color("beam"))),
    ("beam_curv", "beam model, curvature $EI_\\varphi$",
     dict(facecolor="white", edgecolor=FS.color("beam"), hatch="////")),
)

# Headings carry no mathematics: mathtext ignores fontweight, so a
# "$M/M_p \\leq 0.4$" in a heading prints at regular weight inside an
# otherwise bold title.  The load-level condition goes in the caption.
# FS.panel() capitalises the first letter, so these print as
# "Service load" / "Extended elastic".
REGIME_PANEL = (
    ("service", "(a)", "service load"),
    ("extended", "(b)", "extended elastic"),
)

# the bins are assigned by the section's target eta_c (see docstring)
X_LABEL = r"target degree of composite action $\eta_c$ (%)"

# Legend type: every label carries a mathtext subscript, which renders at
# 0.7 of the parent size, so the legend must be at least FS.MATH_MIN_FS
# (9.29 pt) for the subscripts to clear the 6.5 pt floor.
LEGEND_FS = FS.FS_LABEL

# Error bars: near-black with a thin white halo, so the half of each bar
# that lies inside a black section-model bar stays visible.
ERR_KW = dict(elinewidth=1.0, capsize=2.2, capthick=1.0, ecolor="0.15")
ERR_HALO = [pe.Stroke(linewidth=2.0, foreground="white"), pe.Normal()]


def load() -> pd.DataFrame:
    df = pd.read_csv(SUMMARY)
    df["eta_bin"] = pd.Categorical(df.eta_bin, categories=FS.ETA_BINS,
                                   ordered=True)
    return df


def draw_panel(ax, df, regime, letter, title):
    sub = df[df.regime == regime]
    pos = np.arange(len(FS.ETA_BINS), dtype=float)
    w = 0.26
    for k, (metric, _label, kw) in enumerate(SERIES):
        s = (sub[sub.metric == metric]
             .set_index("eta_bin").reindex(FS.ETA_BINS))
        bars = ax.bar(pos + (k - 1) * w, s.mean_pct.to_numpy(), width=w,
                      yerr=s.se_pct.to_numpy(), error_kw=ERR_KW,
                      zorder=3, **kw)
        eb = bars.errorbar
        for art in list(eb.lines[1]) + list(eb.lines[2]):
            art.set_path_effects(ERR_HALO)
            art.set_zorder(4)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.90", lw=0.6, zorder=0)
    # Delta = 0 reference line; named in the caption ("grey line").
    ax.axhline(0.0, color="0.35", lw=0.9, zorder=2)
    FS.eta_bin_axis(ax, positions=pos, label=False)
    ax.set_xlabel(X_LABEL)
    ax.set_xlim(-0.55, len(pos) - 0.45)
    FS.panel(ax, letter, title)
    return sub


def main() -> None:
    FS.apply()
    df = load()

    fig, axes = plt.subplots(1, 2, figsize=(FS.FIG_W, 2.95), sharey=True)
    for ax, (regime, letter, title) in zip(axes, REGIME_PANEL):
        draw_panel(ax, df, regime, letter, title)

    axes[0].set_ylabel("AASHTO stiffness deviation $\\Delta$ (%)")
    # Upper limit must clear the tallest bar plus its error bar. In the
    # extended-elastic panel the section-model means reach 24.9 +/- 4.8
    # (cap at 29.6) in the 25-50 % bin and 19.1 +/- 8.8 (cap at 27.9) in
    # the 90-100 % bin.  With the in-panel coverage note gone, 32 clears
    # both caps without the extra headroom the note needed.
    axes[0].set_ylim(-6.0, 32.0)
    axes[0].yaxis.set_ticks(np.arange(-5, 31, 5))

    # Shared legend above both panels.  It is attached to axes[0] (placed
    # in figure coordinates) rather than to the figure, so FS.audit, which
    # only inspects axes legends, checks its type size.
    handles = [Patch(**kw) for _m, _l, kw in SERIES]
    labels = [lab for _m, lab, _kw in SERIES]
    axes[0].legend(handles, labels, loc="lower center", ncol=3,
                   bbox_to_anchor=(0.5, 0.955),
                   bbox_transform=fig.transFigure, frameon=False,
                   fontsize=LEGEND_FS, handlelength=1.5, columnspacing=1.2,
                   handletextpad=0.45)

    fig.subplots_adjust(left=0.085, right=0.995, top=0.845, bottom=0.155,
                        wspace=0.09)

    probs = FS.audit(fig)
    if probs:
        print(f"[audit] {len(probs)} problem(s)")
    else:
        print("[audit] clean")
    FS.save(fig, OUT)
    print(f"[out] {OUT}")


if __name__ == "__main__":
    main()
