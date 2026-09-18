#!/usr/bin/env python
"""Restyled permutation-feature-importance figure (paper Fig. 5,
label fig:importance).

Values are read from the cached JSON written by
``scripts/feature_importance.py`` (paper/revision_2/submission/sources/figures/
fig_feature_importance.json), so the plotted ordering and magnitudes are
byte-identical to the ones already quoted in the manuscript.  Nothing is
recomputed here; this script only restyles.

Revision 2 changes (Reviewer 2, legends vs images):
  * each feature group carries a hatch as well as a colour, on the bars
    and on the legend swatches alike, so the legend decodes in greyscale;
  * the +/-1 sd error bars are no longer drawn: the sd over the five
    permutations is at most 0.006, which rendered as a cap tick at the bar
    end rather than as a readable error bar.  The largest sd is printed on
    stdout so the sentence quoting it in the text can be checked;
  * bar-end values carry three decimals, so d_s (0.912) and t_f (0.907)
    no longer print as a tie and b_f (0.929) is not read as the baseline
    R^2 (0.932);
  * the tick labels name the quantity the deployed checkpoint permutes
    (absolute inches for b_eff, b_f, t_f, t_w; see normalize.py).

    python scripts/figs/make_fig_feature_importance.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.utils import figstyle as FS  # noqa: E402

FIGDIR = REPO / "paper" / "revision_2" / "submission" / "sources" / "figures"
JSON_IN = FIGDIR / "fig_feature_importance.json"
OUT = FIGDIR / "fig_feature_importance.pdf"

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

# Printed tick label where it differs from the JSON key.  "steel yield"
# read as a yield of steel; the table and the text call f_y the steel
# yield strength.
DISPLAY = {
    r"$f_y$  (steel yield)": r"$f_y$  (yield strength)",
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
# Greyscale code, independent of hue: the four tinted fills sit within
# 47 grey levels of each other (green and magenta within 7), so each group
# also carries its own hatch.  Hatches draw in the edge colour.
CAT_HATCH = {
    "applied load": "",
    "steel section": "////",
    "concrete deck": "....",
    "composite section": "xxx",
    "global geometry": "",
    "section type": "",
}


def tint(c, f=0.42):
    """Blend `c` toward white by fraction `f` (large fills print softer)."""
    r, g, b = mc.to_rgb(c)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def shade(c, f=0.28):
    """Blend `c` toward black by fraction `f`, so the edge and hatch strokes
    stand clear of the tinted fill in greyscale."""
    r, g, b = mc.to_rgb(c)
    return (r * (1 - f), g * (1 - f), b * (1 - f))


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
    sds = np.array([imp[k]["std"] for k in shown])
    cats = [CATEGORY[k] for k in shown]

    FS.apply()
    n = len(shown)
    fig, ax = plt.subplots(figsize=(FS.FIG_W, 0.246 * n + 0.92))
    fig.subplots_adjust(left=0.283, right=0.995, top=0.985, bottom=0.145)

    xmax = float(means.max()) * 1.16
    y = np.arange(n)[::-1]

    ax.grid(axis="x", color="0.90", linewidth=0.6, zorder=0)
    bars = ax.barh(y, means, height=0.62,
                   color=[tint(CAT_COLOR[c]) for c in cats],
                   edgecolor=[shade(CAT_COLOR[c]) for c in cats],
                   linewidth=0.9, zorder=3)
    for patch, c in zip(bars.patches, cats):
        patch.set_hatch(CAT_HATCH[c])

    # value at the bar end -- replaces reading positions off a busy axis
    for yi, m in zip(y, means):
        ax.text(m + xmax * 0.010, yi, f"{m:.3f}", va="center", ha="left",
                fontsize=FS.FS_LEGEND, color="0.20", zorder=5, clip_on=False)

    ax.set_yticks(y)
    # mathtext subscripts render at 0.7 of the declared size, so the tick
    # labels must sit at FS_LABEL (9.5 pt -> 6.65 pt) to clear the floor
    ax.set_yticklabels([DISPLAY.get(k, k) for k in shown],
                       fontsize=FS.FS_LABEL)
    ax.set_ylim(-0.68, n - 0.32)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, xmax)
    ax.set_xticks([0.0, 0.4, 0.8, 1.2, 1.6])
    # the axis names the quantity; what shuffling is, and the baseline it
    # is measured against, are stated in the caption and the text
    ax.set_xlabel(r"drop in curvature $R^2$")
    ax.spines["left"].set_visible(True)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=tint(CAT_COLOR[c]),
                             ec=shade(CAT_COLOR[c]), lw=0.9,
                             hatch=CAT_HATCH[c]) for c in CAT_ORDER]
    leg = ax.legend(handles, CAT_ORDER, loc="lower right",
                    frameon=True, framealpha=0.95, edgecolor="0.85",
                    borderpad=0.55, handlelength=2.0, handleheight=1.15,
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
    print(f"[note] sd over {blob['repeats']} permutations: "
          f"{sds.min():.4f} to {sds.max():.4f} (not drawn)")
    print(f"[note] drops above the baseline (shuffled R2 < 0): "
          + ", ".join(f"{k} {m:.3f}" for k, m in zip(shown, means)
                      if m > base_r2))
    print("[note] omitted (importance < %.2f): %s"
          % (blob["min_show"], ", ".join(
              f"{k} {imp[k]['mean_r2_drop']:+.5f}" for k in omitted)))


if __name__ == "__main__":
    main()
