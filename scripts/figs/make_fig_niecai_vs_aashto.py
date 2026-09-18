#!/usr/bin/env python
"""Restyled Nie & Cai vs AASHTO cross-validation figure (compiled Fig. 7).

Message: the lab-calibrated Nie & Cai slip correction shifts the AASHTO
bin-mean deviation by a nearly constant 7-9 percentage points in every
composite-action bin, at both deck-reinforcement levels, but it does not
close the remaining cracking / neutral-axis-migration gap.

Panel (a) plots the bin-mean curvature deviation from the OpenSeesPy
fiber-section reference for the AASHTO closed form and the Nie & Cai
analytical formula, at rho_l = 0 and rho_l = 0.7 %.  The two shaded
bands split the total AASHTO deviation at rho_l = 0 into the part the
slip correction recovers (between the AASHTO and Nie & Cai curves) and
the residual it leaves (between the Nie & Cai curve and the zero line).
Panel (b) plots the shift in bin-mean deviation the slip correction
produces, Nie & Cai minus AASHTO, in percentage points.  It is a signed
shift, not a reduction in deviation magnitude: in the top bin at
rho_l = 0.7 % the Nie & Cai mean overshoots zero (+2.9 %), so the shift
(8.7 points) exceeds the drop in |deviation| (2.9 points).

Legend rules (Revision 2): every drawn mark has an entry and every entry
glyph is the drawn mark.  Panel (a) carries two keys, each in space its
own data leaves empty: the four series (predictor x reinforcement, each
with its exact line style, marker and fill) in the bottom-right corner
below the AASHTO rho_l = 0 lower envelope, and the zero line plus the two
bands in the strip above the zero line.  Panel (b) keys itself in its
bottom-left corner.  The clearance of each key from the plotted ink is
measured by free_check() on every run.

Encoding: in (a) the marker names the predictor (diamond AASHTO, triangle
Nie & Cai) and solid/filled vs dashed/open names the reinforcement level;
(b) keeps the same solid/filled vs dashed/open code on circles, so no
glyph means one thing in (a) and another in (b), and every series stays
separable in greyscale.  The slip band is hatched as well as tinted.

All values come from reports/niecai/niecai_summary.csv and
reports/niecai_rebar007/niecai_summary.csv (the extended-elastic set,
M/Mp <= 0.6, 427 025 rows at rho_l = 0), which are the same numbers as
Table 8 of the manuscript.

    python scripts/figs/make_fig_niecai_vs_aashto.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.utils import figstyle as FS  # noqa: E402

SUM_RHO0 = REPO / "reports" / "niecai" / "niecai_summary.csv"
SUM_RHO07 = REPO / "reports" / "niecai_rebar007" / "niecai_summary.csv"
OUT = REPO / "paper" / "revision_2" / "submission" / "sources" / "figures" / "fig_niecai_vs_aashto.pdf"

# ---------------------------------------------------------------- band fills
# One dict per shaded region, used BOTH to draw the band and to build its
# legend patch, so the key swatch is the band's actual fill.  The slip
# band carries a hatch because a pale blue and a pale grey collapse to
# within a few grey levels of each other in a greyscale print.
BAND_SLIP = dict(facecolor="#DCEEFC", edgecolor=FS.SKY, hatch="\\" * 3,
                 linewidth=0.0)
BAND_RESIDUAL = dict(facecolor="0.885", edgecolor="none", linewidth=0.0)
BAND_REDUCTION = dict(facecolor="0.90", edgecolor="none", linewidth=0.0)

# the zero line of (a), drawn and keyed from the same dict
ZERO_LINE = dict(color="black", lw=1.0, ls="-")

# key entries are labels, not sentences: the paragraph carries the argument
LBL_ZERO = "OpenSeesPy reference (zero deviation)"
LBL_SLIP = "recovered by the slip correction"
LBL_RESIDUAL = "residual to the reference"
LBL_REDUCTION = "7–9 point range"
LBL_RHO = {"rho0": "no deck reinforcement",
           "rho07": "deck reinforcement 0.7 %"}
LBL_PRED = {"aashto": "AASHTO", "niecai": "Nie & Cai"}

# per reinforcement level: line style and marker fill, shared by (a) and (b)
LS_RHO = {"rho0": "-", "rho07": (0, (5, 1.8))}
MK_RHO = {"rho0": {}, "rho07": dict(mfc="white")}
LW, MS, MEW = 1.9, 5.2, 1.4

# in-panel keys are set one step below the axis labels, the sanctioned
# size for type that sits inside the data area (mathtext subscripts would
# fall below the size floor here, so reinforcement is named in words)
FS_KEY = FS.FS_ANNOT


def load(path: Path) -> pd.DataFrame:
    """Summary rows ordered by the sanctioned eta_c bin order."""
    df = pd.read_csv(path).set_index("eta_bin")
    return df.loc[list(FS.ETA_BINS)]


def _dense(v: np.ndarray, n: int = 64) -> np.ndarray:
    """Subdivide a polyline, so a check sees the segments, not just the
    vertices.  A four-point curve clears any box if only its vertices are
    tested; what has to clear the box is the line drawn between them."""
    v = np.asarray(v, float)
    if len(v) < 2:
        return v
    t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
    seg = v[:-1][None] * (1 - t[:, None]) + v[1:][None] * t[:, None]
    return np.vstack([seg.reshape(-1, 2), v[-1:]])


def free_check(ax, lg, name: str) -> float:
    """Vertical gap, in axes fractions, between a legend and the data.

    Every line (the zero line included), band polygon and span patch is
    densified and tested against the legend's rendered box.  A key in the
    lower half is measured to the lowest ink above it, a key in the upper
    half to the highest ink below it; the return value is that clearance,
    negative if anything reaches into the box.
    """
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    to_ax = ax.transAxes.inverted()
    bb = lg.get_window_extent(r)
    (x0, y0), (x1, y1) = to_ax.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])

    pts = []
    for ln in ax.lines:                            # curves, markers, axhline
        d = np.asarray(ln.get_xydata(), float)
        if len(d):
            pts.append(to_ax.transform(ln.get_transform().transform(_dense(d))))
    for art in list(ax.collections) + list(ax.patches):   # bands, spans
        tr = art.get_transform()
        paths = art.get_paths() if hasattr(art, "get_paths") else [art.get_path()]
        for p in paths:
            pts.append(to_ax.transform(tr.transform(_dense(p.vertices))))
    pts = np.vstack([p for p in pts if len(p)])
    pts = pts[np.isfinite(pts).all(axis=1)]

    over = pts[(pts[:, 0] >= x0 - 0.01) & (pts[:, 0] <= x1 + 0.01)]
    inside = int(((over[:, 1] >= y0) & (over[:, 1] <= y1)).sum())
    if 0.5 * (y0 + y1) < 0.5:                      # key at the bottom
        above = over[over[:, 1] >= y0]
        gap = float(above[:, 1].min() - y1) if len(above) else 1.0
    else:                                          # key at the top
        below = over[over[:, 1] <= y1]
        gap = float(y0 - below[:, 1].max()) if len(below) else 1.0
    if inside:
        gap = min(gap, -0.001)
    print(f"[free] {name} key box x {x0:.3f}-{x1:.3f}, y {y0:.3f}-{y1:.3f}; "
          f"clearance {gap:+.3f} of panel height, {inside} ink samples inside")
    return gap


def main() -> None:
    d0, d7 = load(SUM_RHO0), load(SUM_RHO07)
    x = np.arange(len(FS.ETA_BINS), dtype=float)

    series = {}          # (predictor, rho) -> bin-mean deviation, %
    for tag, d in (("rho0", d0), ("rho07", d7)):
        series[("aashto", tag)] = d["aashto_err_mean_pct"].to_numpy()
        series[("niecai", tag)] = d["niecai_err_mean_pct"].to_numpy()
    # signed shift in bin-mean deviation from the slip correction, points
    shift = {t: series[("niecai", t)] - series[("aashto", t)]
             for t in ("rho0", "rho07")}

    for t in ("rho0", "rho07"):
        print(f"[{t}] AASHTO    " + "  ".join(f"{v:+6.1f}" for v in series[("aashto", t)]))
        print(f"[{t}] Nie&Cai   " + "  ".join(f"{v:+6.1f}" for v in series[("niecai", t)]))
        print(f"[{t}] shift     " + "  ".join(f"{v:+6.1f}" for v in shift[t]))
    print(f"[n] rho0 rows = {int(d0['n_rows'].sum())}")

    FS.apply()
    fig = plt.figure(figsize=(FS.FIG_W, 3.95))
    # the left margin holds the two-line y label of (a) and the right
    # margin the bold heading of (b) inside the canvas, so the tight crop
    # is no wider than FIG_W and \includegraphics applies scale 1.0 (type
    # then prints at its declared size)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.42, 1.0], wspace=0.32,
                          left=0.095, right=0.970, top=0.935, bottom=0.124)
    axa, axb = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    XLO, XHI = -0.32, 3.32

    # -------------------------------------------------- (a) deviations
    a0, n0 = series[("aashto", "rho0")], series[("niecai", "rho0")]
    # the two shares, as the vertical gaps they occupy; the bands run the
    # full width so they do not read as free-floating rectangles.  Both
    # are measured from the same zero line, so slip band + residual band =
    # the whole AASHTO deviation at rho_l = 0, bin by bin
    xf = np.concatenate(([XLO], x, [XHI]))
    ef = lambda v: np.concatenate(([v[0]], v, [v[-1]]))  # noqa: E731
    axa.fill_between(xf, ef(a0), ef(n0), zorder=1, **BAND_SLIP)
    axa.fill_between(xf, ef(n0), 0.0, zorder=0, **BAND_RESIDUAL)
    axa.axhline(0.0, zorder=2, **ZERO_LINE)

    def series_kw(pred, tag, **extra):
        return FS.style(pred, ls=LS_RHO[tag], label=False, lw=LW, ms=MS,
                        mew=MEW, **MK_RHO[tag], **extra)

    for tag in FS.REINFORCEMENT:
        for pred in ("aashto", "niecai"):
            axa.plot(x, series[(pred, tag)], zorder=4, **series_kw(pred, tag))

    # the strip above the zero line is left free for the band key
    axa.set_ylim(-60, 17)
    axa.set_yticks([-60, -50, -40, -30, -20, -10, 0, 10])
    FS.eta_bin_axis(axa, positions=x)
    axa.set_xlim(XLO, XHI)
    axa.set_ylabel("bin-mean curvature deviation\nfrom OpenSeesPy (%)")

    # series key: one entry per drawn series, glyph = the drawn series
    keys_series = [Line2D([], [], **series_kw(pred, tag))
                   for pred in ("aashto", "niecai")
                   for tag in FS.REINFORCEMENT]
    for h, (pred, tag) in zip(keys_series,
                              [(p, t) for p in ("aashto", "niecai")
                               for t in FS.REINFORCEMENT]):
        h.set_label(f"{LBL_PRED[pred]}, {LBL_RHO[tag]}")
    lga = axa.legend(handles=keys_series, loc="lower right", frameon=False,
                     handlelength=2.2, handleheight=1.00, handletextpad=0.45,
                     borderaxespad=0.35, labelspacing=0.26, fontsize=FS_KEY)
    lga.set_zorder(6)
    axa.add_artist(lga)

    # reference and band key, in the strip above the zero line
    keys_ref = [Line2D([], [], label=LBL_ZERO, **ZERO_LINE),
                Patch(label=LBL_SLIP, **BAND_SLIP),
                Patch(label=LBL_RESIDUAL, **BAND_RESIDUAL)]
    lga2 = axa.legend(handles=keys_ref, loc="upper left", frameon=False,
                      handlelength=2.2, handleheight=1.00, handletextpad=0.45,
                      borderaxespad=0.35, labelspacing=0.26, fontsize=FS_KEY)
    for lg in (lga, lga2):
        if hasattr(lg, "set_alignment"):
            lg.set_alignment("left")

    # -------------------------------------------------- (b) shift
    axb.axhspan(7.0, 9.0, zorder=0, **BAND_REDUCTION)
    for tag in FS.REINFORCEMENT:
        axb.plot(x, shift[tag], zorder=3,
                 **FS.style(tag, label=False, lw=LW, ms=MS, mew=MEW,
                            marker="o", **MK_RHO[tag]))
    axb.set_ylim(0, 11.0)
    axb.set_yticks([0, 2, 4, 6, 8, 10])
    FS.eta_bin_axis(axb, positions=x)
    axb.set_xlim(XLO, XHI)
    axb.set_ylabel("shift in bin-mean deviation,\nNie & Cai $-$ AASHTO\n"
                   "(percentage points)")
    keys_b = [FS.handle(tag, lw=LW, ms=5.0, mew=MEW, marker="o",
                        fontsize=FS_KEY, **MK_RHO[tag])
              for tag in FS.REINFORCEMENT]
    for h, tag in zip(keys_b, FS.REINFORCEMENT):
        h.set_label(LBL_RHO[tag])
    keys_b.append(Patch(label=LBL_REDUCTION, **BAND_REDUCTION))
    lgb = axb.legend(handles=keys_b, loc="lower left", frameon=False,
                     handlelength=2.0, handleheight=1.00, handletextpad=0.5,
                     borderaxespad=0.5, labelspacing=0.34, fontsize=FS_KEY)
    if hasattr(lgb, "set_alignment"):
        lgb.set_alignment("left")

    FS.panel(axa, "(a)", "Deviation from the OpenSeesPy reference", dy=1.030)
    FS.panel(axb, "(b)", "Shift due to the slip correction", dy=1.030)

    gaps = [free_check(axa, lga, "(a) series"),
            free_check(axa, lga2, "(a) reference and bands"),
            free_check(axb, lgb, "(b)")]

    # judged at printed size: the figure is included at width=\linewidth
    scale = FS.printed_scale(fig, 1.0)
    probs = FS.audit(fig, scale=scale)
    print(f'[audit] {"clean" if not probs else f"{len(probs)} problem(s)"} '
          f"(printed scale {scale:.3f})")
    if min(gaps) < 0:
        print("[free] WARNING: a key overlaps plotted ink")
    FS.save(fig, OUT)
    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
