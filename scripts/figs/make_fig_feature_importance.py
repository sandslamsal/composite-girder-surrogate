#!/usr/bin/env python
"""Restyled permutation-feature-importance figure (paper Fig. 6).

Values are read from the cached JSON written by
``scripts/feature_importance.py`` (paper/revision_1/submission/sources/figures/
fig_feature_importance.json), so the plotted ordering and magnitudes are
byte-identical to the ones already quoted in the manuscript.  Nothing is
recomputed here; this script only restyles.

    python scripts/figs/make_fig_feature_importance.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.utils import figstyle as FS  # noqa: E402

FIGDIR = REPO / "paper" / "revision_1" / "submission" / "sources" / "figures"
JSON_IN = FIGDIR / "fig_feature_importance.json"
OUT = FIGDIR / "fig_feature_importance.png"

# ---------------------------------------------------------------- grouping
# Physical grouping of the features (same partition as
# scripts/feature_importance.py), keyed by the JSON display label.
CATEGORY = {
    r"$M/M_p$  (moment ratio)": "applied load",
    r"$t_w$  (web thickness)": "steel section",
    r"$f_y$  (steel yield)": "steel section",
    r"$b_f$  (flange width)": "steel section",
    r"$d_s$  (steel depth)": "steel section",
    r"$t_f$  (flange thickness)": "steel section",
    r"$b_{\mathrm{eff}}$  (deck width)": "concrete deck",
    r"$f_c'$  (deck strength)": "concrete deck",
    r"$t_s$  (deck thickness)": "concrete deck",
    r"$\eta_c$  (composite action)": "composite section",
    r"$d_{\mathrm{total}}$  (total depth)": "composite section",
    r"$S$  (girder spacing)": "global geometry",
    r"$L$  (span)": "global geometry",
    "section type (one-hot)": "section type",
}

# Feature groups never share a panel with the ENTITY registry, so the
# Okabe-Ito constants are re-used here as a four-way categorical ramp.
CAT_ORDER = ["applied load", "steel section", "concrete deck",
             "composite section"]
CAT_COLOR = {
    "applied load": FS.ORANGE,
    "steel section": FS.BLUE,
    "concrete deck": FS.GREEN,
    "composite section": FS.PURPLE,
    "global geometry": FS.GRAY,
    "section type": FS.GRAY,
}


def tint(c, f=0.42):
    """Blend `c` toward white by fraction `f` (large fills print softer)."""
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(c)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def main() -> None:
    blob = json.loads(JSON_IN.read_text())
    imp = blob["importance"]
    base_r2 = float(blob["baseline_r2"])

    # JSON preserves the descending order written by feature_importance.py;
    # re-sort anyway so the figure cannot silently drift from the values.
    shown = [k for k, v in imp.items() if v["shown"]]
    shown.sort(key=lambda k: imp[k]["mean_r2_drop"], reverse=True)
    omitted = [k for k, v in imp.items() if not v["shown"]]
    means = np.array([imp[k]["mean_r2_drop"] for k in shown])
    errs = np.array([imp[k]["std"] for k in shown])
    colors = [CAT_COLOR[CATEGORY[k]] for k in shown]

    FS.apply()
    n = len(shown)
    fig, ax = plt.subplots(figsize=(FS.FIG_W, 0.246 * n + 0.92))
    fig.subplots_adjust(left=0.283, right=0.995, top=0.985, bottom=0.145)

    xmax = float(means.max()) * 1.145
    y = np.arange(n)[::-1]

    ax.grid(axis="x", color="0.90", linewidth=0.6, zorder=0)
    ax.barh(y, means, xerr=errs, height=0.62,
            color=[tint(c) for c in colors], edgecolor=colors, linewidth=0.9,
            zorder=3,
            error_kw=dict(linewidth=0.9, ecolor="0.25", capsize=2.0,
                          zorder=4))

    # value at the bar end -- replaces reading positions off a busy axis
    for yi, m in zip(y, means):
        ax.text(m + xmax * 0.012, yi, f"{m:.2f}", va="center", ha="left",
                fontsize=FS.FS_LEGEND, color="0.20", zorder=5, clip_on=False)

    ax.set_yticks(y)
    # mathtext subscripts render at 0.7 of the declared size, so the tick
    # labels must sit at FS_LABEL (9.5 pt -> 6.65 pt) to clear the floor
    ax.set_yticklabels(shown, fontsize=FS.FS_LABEL)
    ax.set_ylim(-0.68, n - 0.32)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, xmax)
    ax.set_xticks([0.0, 0.4, 0.8, 1.2, 1.6])
    # the axis names the quantity; what shuffling is, and the baseline it
    # is measured against, are stated in the caption and the text
    ax.set_xlabel(r"drop in curvature $R^2$")
    ax.spines["left"].set_visible(True)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=tint(CAT_COLOR[c]),
                             ec=CAT_COLOR[c], lw=0.9) for c in CAT_ORDER]
    leg = ax.legend(handles, CAT_ORDER, loc="lower right",
                    frameon=True, framealpha=0.95, edgecolor="0.85",
                    borderpad=0.55, handlelength=1.25, handleheight=1.05,
                    labelspacing=0.36,
                    bbox_to_anchor=(0.999, 0.012))
    leg.set_zorder(6)

    # The unshuffled baseline R^2 is NOT written on the panel: the text
    # that discusses this figure already gives it, and it decodes nothing
    # the reader can see.  It is echoed on stdout instead.
    probs = FS.audit(fig)
    print(f"[audit] {len(probs)} problem(s)")
    FS.save(fig, OUT)
    print(f"[done] {OUT}")
    print(f"[note] unshuffled baseline curvature R2 = {base_r2:.3f}")
    print("[note] omitted (importance < %.2f): %s"
          % (blob["min_show"], ", ".join(omitted)))


if __name__ == "__main__":
    main()
