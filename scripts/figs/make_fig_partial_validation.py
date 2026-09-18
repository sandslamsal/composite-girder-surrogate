#!/usr/bin/env python
"""Figure: measured against predicted stiffness for published solid-slab
tests at partial shear connection (Revision 2, Reviewer 2 comment 2).

One panel, read row by row. Each row is one prediction (full-composite
transformed section; the same section reduced by R_EI; the beam-level model
with the connector data of each test series), headed by its name. Within a row, one
strip per deck type, cast-in-place and precast: a dot per specimen, a thin
line over the full range, a tinted bar over the middle half of the specimens
and a dark bar at the median with its value. A ratio below 1 means the
prediction is stiffer than the test.

Every prediction carries the same web-shear compliance
(scripts/partial_connection_deflection_basis.py). Test series, degrees of
shear connection and the profiled-slab girder are in Table
tab:partial-validation, not in the figure, so that the figure carries one
message.

Data
----
reports/model_validation/partial_connection/deflection_basis.csv

Output
------
paper/revision_2/submission/sources/figures/fig_partial_validation.pdf
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
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

from src.utils import figstyle as FS  # noqa: E402

DATA = REPO / "reports/model_validation/partial_connection/deflection_basis.csv"
OUT = REPO / "paper/revision_2/submission/sources/figures/fig_partial_validation.pdf"

# validated categorical pair (dataviz validator, light surface)
DECK = [("cast", "cast-in-place decks", "#2a78d6"),
        ("precast", "precast decks", "#eb6834")]
SHORT = {"cast": "cast-in-place", "precast": "precast"}
# deep step of each deck hue: median bar and its value flag (white text
# on these clears 6.8:1 and 5.3:1)
DEEP = {"cast": "#1f5aa6", "precast": "#b8461a"}
INK = "#111111"
INK2 = "#4a4a48"
INK3 = "#2b2b2a"
AXIS = "0.35"
BAND = "#f2f4f8"          # +/-10 % band
GRID = "#e3e6ea"
REF = "0.30"

PRED = [("meas_over_full", "Full-composite section", "(AASHTO)"),
        ("meas_over_REI", r"Section reduced by $R_{EI}$", "(this study)"),
        ("meas_over_beam", "Beam-level model", "(measured connectors)")]

# test programme -> (legend label, marker, relative marker area)
PROG = [("Kwon", "Kwon et al. (2007)", "o", 1.00),
        ("Viest", "Viest et al. (1952)", "s", 0.85),
        ("Slutter", "Culver et al. (1960, 1961)", "^", 1.15),
        ("McGarraugh", "McGarraugh and Baldwin (1971)", "v", 1.15),
        ("Provines", "Provines et al. (2019)", "D", 0.80),
        ("Suwaed", "Suwaed and Karavasilis (2020)", "P", 1.25)]
MARKER = {key: (mk, area) for key, _, mk, area in PROG}
DOT_AREA = 38.0
LEGEND_GREY = "#6b7079"

XLIM = (0.56, 1.36)
ROW_H = 1.0
Y_HEAD = 0.34         # header offset above the row centre
Y_STRIP = (0.07, -0.29)   # cast-in-place, precast strip offsets
JITTER = 0.045


def jitter(values: np.ndarray) -> np.ndarray:
    """Deterministic spread: dots that sit close in x alternate above and
    below the strip line, so none hides another."""
    order = np.argsort(values)
    offs = np.zeros(len(values))
    pattern = [0.0, 1.0, -1.0, 0.5, -0.5]
    last_x, k = None, 0
    for i in order:
        k = k + 1 if (last_x is not None and abs(values[i] - last_x) < 0.03) else 0
        offs[i] = pattern[k % len(pattern)] * JITTER
        last_x = values[i]
    return offs


def main() -> None:
    FS.apply()
    d = pd.read_csv(DATA)
    part = d[d.group.notna()].copy()
    part["deck"] = np.where(part.group == "precast solid slab", "precast", "cast")
    part["prog"] = part.programme.map(
        lambda name: next((k for k, *_ in PROG if name.startswith(k)), ""))
    assert (part.prog != "").all(), "specimen without a programme marker"

    n_rows = len(PRED)
    FIG_H = 3.95
    fig, ax = plt.subplots(figsize=(FS.FIG_W, FIG_H))
    # plot area fixed in inches (bottom 0.49 in, top 2.93 in) so the legend
    # block above it sets only the figure height
    fig.subplots_adjust(left=0.265, right=0.975, bottom=0.49 / FIG_H,
                        top=2.93 / FIG_H)
    blend = blended_transform_factory(ax.transAxes, ax.transData)
    # one left edge, in figure coordinates, shared by the row titles, the
    # deck labels and the legend frame
    LEFT, RIGHT = 0.02, 0.975
    fblend = blended_transform_factory(fig.transFigure, ax.transData)
    # deck-colour swatch: a filled rectangle, 1.6:1, SWATCH_MS pt wide
    SWATCH = [(-1.6, -1.0), (1.6, -1.0), (1.6, 1.0), (-1.6, 1.0), (-1.6, -1.0)]
    SWATCH_MS = 8.0
    sw_half = SWATCH_MS / 2 / 72.0 / FS.FIG_W   # half width, figure fraction

    y_lo, y_hi = -0.55, n_rows - 0.45
    # vertical grid, then the band and the reference line
    for gx in np.arange(0.6, 1.31, 0.1):
        ax.axvline(gx, color=GRID, lw=0.5, zorder=0.2)
    ax.axvspan(0.9, 1.1, facecolor=BAND, edgecolor="none", zorder=0.3)
    ax.axvline(1.0, color=REF, lw=0.9, ls=(0, (3.6, 2.2)), zorder=1)

    for r, (col, name, tag) in enumerate(PRED):
        yc = n_rows - 1 - r
        if r:
            ax.plot([LEFT, RIGHT], [yc + ROW_H / 2] * 2, transform=fblend,
                    color="#cfd4da", lw=0.7, zorder=0.1, clip_on=False)
        # row header on the shared left edge: name, then a quieter tag
        head = ax.text(LEFT, yc + Y_HEAD, name, transform=fblend,
                       ha="left", va="center", fontsize=9.8,
                       fontweight="bold", color=INK, clip_on=False)
        ax.annotate(tag, xy=(1.0, 0.5), xycoords=head, xytext=(4, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=FS.FS_ANNOT, color=INK2)

        for (deck, label, colour), dy in zip(DECK, Y_STRIP):
            y = yc + dy
            strip = part[part.deck == deck]
            vals = strip[col].to_numpy()
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            # strip label on the shared left edge: colour swatch, then text
            ax.plot([LEFT + sw_half], [y], transform=fblend, ls="none",
                    marker=SWATCH, ms=SWATCH_MS, mfc=colour, mec=colour,
                    mew=0.0, clip_on=False)
            ax.text(LEFT + 2 * sw_half + 0.008, y,
                    f"{SHORT[deck]}, {len(vals)} beams", transform=fblend,
                    ha="left", va="center", fontsize=8.2, color=INK3,
                    clip_on=False)
            # range, middle half, specimens, median
            ax.plot([vals.min(), vals.max()], [y, y], color=colour, lw=1.0,
                    alpha=0.45, solid_capstyle="round", zorder=2)
            ax.plot([q1, q3], [y, y], color=colour, lw=8.5, alpha=0.20,
                    solid_capstyle="round", zorder=2)
            offs = jitter(vals)
            progs = strip.prog.to_numpy()
            for key in dict.fromkeys(progs):
                sel = progs == key
                mk, area = MARKER[key]
                ax.scatter(vals[sel], y + offs[sel], s=DOT_AREA * area,
                           marker=mk, color=colour, edgecolor="white",
                           linewidth=0.7, zorder=3, clip_on=False)
            ax.plot([med, med], [y - 0.11, y + 0.11], color=DEEP[deck],
                    lw=2.4, solid_capstyle="butt", zorder=4)
            ax.annotate(f"{med:.2f}", (med, y + 0.11), xytext=(0, 0),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=7.6, color="white", zorder=5,
                        bbox=dict(boxstyle="square,pad=0.28",
                                  fc=DEEP[deck], ec="none"))

    ax.set_ylim(y_lo, y_hi)
    ax.set_yticks([])
    ax.set_xlim(*XLIM)
    ax.set_xticks(np.arange(0.6, 1.31, 0.1))
    ax.set_xticklabels([f"{v:.1f}" for v in np.arange(0.6, 1.31, 0.1)])
    ax.set_xlabel("measured stiffness / predicted stiffness", fontsize=9.5,
                  color=INK)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="x", colors=AXIS, labelcolor=INK3, labelsize=8.5)

    # legend: three untitled groups side by side (deck colour, test series
    # shape, summary marks), each with only its own entries
    grey = "#8a8f98"
    deck_h = [Line2D([], [], ls="none", marker=SWATCH, ms=SWATCH_MS, mfc=c,
                     mec=c, mew=0.0) for _, _, c in DECK]
    deck_l = [lab for _, lab, _ in DECK]
    prog_h = [Line2D([], [], ls="none", marker=mk, ms=6.2 * np.sqrt(area),
                     mfc=LEGEND_GREY, mec="white", mew=0.6)
              for _, _, mk, area in PROG]
    prog_l = [lab for _, lab, _, _ in PROG]
    mark_h = [Line2D([], [], color="#3d4148", lw=2.4),
              Line2D([], [], color=grey, lw=6.5, alpha=0.35,
                     solid_capstyle="butt"),
              Line2D([], [], color=grey, lw=1.0, alpha=0.6),
              Patch(facecolor=BAND, edgecolor="none"),
              Line2D([], [], color=REF, lw=0.9, ls=(0, (3.6, 2.2)))]
    mark_l = ["median (value in flag)", "middle half of specimens",
              "full range", r"within $\pm$10 %",
              "1.0: prediction equals test"]
    common = dict(frameon=False, fontsize=7.2, handlelength=1.7,
                  handletextpad=0.55, labelspacing=0.45, borderpad=0.0,
                  alignment="left")
    y_leg = 1.0 - 0.1025 / FIG_H
    PAD = 0.014
    legs = [fig.legend(deck_h, deck_l, loc="upper left", ncol=1,
                       bbox_to_anchor=(LEFT + PAD, y_leg), **common),
            fig.legend(prog_h, prog_l, loc="upper left", ncol=2,
                       bbox_to_anchor=(0.3, y_leg), columnspacing=1.1,
                       **common),
            fig.legend(mark_h, mark_l, loc="upper left", ncol=1,
                       bbox_to_anchor=(0.7, y_leg), **common)]
    for leg in legs:
        for t in leg.get_texts():
            t.set_color(INK3)
    # frame from the shared left edge to the plot's right edge; the first
    # group sits PAD inside the left edge, the last PAD inside the right,
    # and the middle group is centred in the space between them
    fig.canvas.draw()
    rr = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    w = [lg.get_window_extent(rr).transformed(inv).width for lg in legs]
    x_first_end = LEFT + PAD + w[0]
    x_last = RIGHT - PAD - w[2]
    x_mid = x_first_end + (x_last - x_first_end - w[1]) / 2
    legs[1].set_bbox_to_anchor((x_mid, y_leg), transform=fig.transFigure)
    legs[2].set_bbox_to_anchor((x_last, y_leg), transform=fig.transFigure)
    fig.canvas.draw()
    boxes = [lg.get_window_extent(rr).transformed(inv) for lg in legs]
    y0 = min(b.y0 for b in boxes) - PAD
    y1 = max(b.y1 for b in boxes) + PAD * 0.7
    fig.patches.append(Rectangle((LEFT, y0), RIGHT - LEFT, y1 - y0,
                                 transform=fig.transFigure, facecolor="none",
                                 edgecolor="#d5d9df", lw=0.6, zorder=0))

    probs = FS.audit(fig)
    print("[audit] clean" if not probs else f"[audit] {len(probs)} problem(s)")
    for p in probs:
        print("   ", p)
    for col, name, _ in PRED:
        print(f"[data] {name}: cast "
              f"{part.loc[part.deck == 'cast', col].median():.3f}, precast "
              f"{part.loc[part.deck == 'precast', col].median():.3f}")
    FS.save(fig, OUT)
    print(f"[out] {OUT}")


if __name__ == "__main__":
    main()
