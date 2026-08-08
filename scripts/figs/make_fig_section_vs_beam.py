#!/usr/bin/env python
"""Figure: AASHTO stiffness deviation under the two partial-composite
representations (width-scaled section model vs discrete-connector beam
model), by eta_c bin and load regime.

Restyle of the old Fig. 11 with src/utils/figstyle.py.  Message: the
beam model imposes nothing about stiffness reduction -- eta_c emerges
from connector equilibrium -- yet it independently reproduces the sign
and the monotonic decay of the width-scaling deviation.

Data
----
reports/section_vs_beam_revision/sensitivity_summary.csv
    per (eta_bin, regime, metric) mean_pct / se_pct / coverage, written
    by paper2-beam-level/src/validation/sensitivity_sweep.py.

Output
------
paper/revision_1/submission/sources/figures/fig_section_vs_beam.{png,pdf}

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

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.utils import figstyle as FS  # noqa: E402

SUMMARY = REPO / "reports/section_vs_beam_revision/sensitivity_summary.csv"
OUT = REPO / "paper/revision_1/submission/sources/figures/fig_section_vs_beam.png"

# metric key -> (legend label, bar kwargs)
SERIES = (
    ("section_matched", "section model, width-scaled $\\eta_c$",
     dict(facecolor=FS.color("opensees"), edgecolor=FS.color("opensees"))),
    ("beam_defl", "beam model, deflection $EI_\\delta$",
     dict(facecolor=FS.color("beam"), edgecolor=FS.color("beam"))),
    ("beam_curv", "beam model, curvature $EI_\\varphi$",
     dict(facecolor="white", edgecolor=FS.color("beam"), hatch="////")),
)

REGIME_PANEL = (
    ("service", "(a)", "service load, $M/M_p \\leq 0.4$"),
    ("extended", "(b)", "extended elastic, $M/M_p \\leq 0.6$"),
)


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
        ax.bar(pos + (k - 1) * w, s.mean_pct.to_numpy(), width=w,
               yerr=s.se_pct.to_numpy(), error_kw=dict(elinewidth=1.0,
                                                       capsize=2.0,
                                                       ecolor="0.25"),
               zorder=3, **kw)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.90", lw=0.6, zorder=0)
    ax.axhline(0.0, color="0.35", lw=0.9, zorder=2)
    FS.eta_bin_axis(ax, positions=pos)
    ax.set_xlim(-0.55, len(pos) - 0.45)
    FS.panel(ax, letter, title)
    return sub


def main() -> None:
    FS.apply()
    df = load()

    fig, axes = plt.subplots(1, 2, figsize=(FS.FIG_W, 2.95), sharey=True)
    covers = {}
    for ax, (regime, letter, title) in zip(axes, REGIME_PANEL):
        sub = draw_panel(ax, df, regime, letter, title)
        cov = (sub[sub.metric == "beam_defl"]
               .set_index("eta_bin").reindex(FS.ETA_BINS).coverage * 100.0)
        covers[regime] = (float(cov.min()), float(cov.max()))

    axes[0].set_ylabel("AASHTO stiffness deviation $\\Delta$ (%)")
    # Upper limit must clear the tallest bar plus its error bar. In the
    # extended-elastic panel the section-model means reach 24.9 +/- 4.8
    # in the 25-50 % bin and 19.1 +/- 8.8 in the 90-100 % bin, so 23 (the
    # limit used before the support-constraint correction raised beam
    # coverage and with it the matched-row population) now clips them.
    axes[0].set_ylim(-6.0, 34.0)
    axes[0].yaxis.set_ticks(np.arange(-5, 31, 5))

    # honest coverage note, above every bar and error bar
    for ax, (regime, _l, _t) in zip(axes, REGIME_PANEL):
        lo, hi = covers[regime]
        FS.inside_label(ax, 3.45, 33.4,
                        "beam-level analyses cover\n"
                        f"{lo:.0f}–{hi:.0f} % of rows",
                        ha="right", va="top", fontsize=FS.FS_ANNOT,
                        color="0.30", linespacing=1.25)

    handles = [Patch(**kw) for _m, _l, kw in SERIES]
    labels = [lab for _m, lab, _kw in SERIES]
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.985), frameon=False,
               fontsize=FS.FS_LEGEND, handlelength=1.5, columnspacing=1.6,
               handletextpad=0.5)

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
