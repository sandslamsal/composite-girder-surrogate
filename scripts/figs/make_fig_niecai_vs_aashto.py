#!/usr/bin/env python
"""Restyled Nie & Cai vs AASHTO cross-validation figure (compiled Fig. 7).

Message: the lab-calibrated Nie & Cai slip correction removes a nearly
constant 7-9 percentage-point slice of the AASHTO deviation in every
composite-action bin, at both deck-reinforcement levels, but it does not
close the remaining cracking / neutral-axis-migration gap.

Panel (a) plots the bin-mean curvature deviation from the OpenSeesPy
fiber-section reference for the AASHTO closed form and the Nie & Cai
analytical formula, at rho_l = 0 and rho_l = 0.7 %.  The two shaded
bands split the total AASHTO deviation at rho_l = 0 into the part the
slip correction recovers (AASHTO down to Nie & Cai) and the residual it
leaves (Nie & Cai down to the reference).  Panel (b) plots the reduction
in deviation the slip correction achieves, AASHTO minus Nie & Cai, in
percentage points.

Every key sits inside the panel it decodes.  Panel (a) keys itself in
its own bottom-right corner, which the plotted data leaves empty across
the whole x range (the AASHTO rho_l = 0 curve is the lower envelope of
every curve and both bands, and it clears the legend box by the margin
free_check() prints on each run); panel (b) keys itself in its own
bottom-left corner.  There is no shared block above the panels, so no
entry has to be tagged with the panel it belongs to and nothing is
stated twice.

The entries are labels, not sentences.  What the decomposition means,
that the bands are drawn for the unreinforced series only, and that the
zero line is the OpenSeesPy reference are all said once, in the
manuscript caption and the y-axis label, and are therefore not repeated
on the panels.  The slip band is separated from the residual band by
hatching as well as hue so the two survive greyscale printing.

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
OUT = REPO / "paper" / "revision_1" / "submission" / "sources" / "figures" / "fig_niecai_vs_aashto.png"

# ---------------------------------------------------------------- band fills
# One dict per shaded region, used BOTH to draw the band and to build its
# legend patch, so the key swatch is the band's actual fill.  The slip
# band carries a hatch because a pale blue and a pale grey collapse to
# within a few grey levels of each other in a greyscale print.
BAND_SLIP = dict(facecolor="#DCEEFC", edgecolor=FS.SKY, hatch="\\" * 3,
                 linewidth=0.0)
BAND_RESIDUAL = dict(facecolor="0.885", edgecolor="none", linewidth=0.0)
BAND_REDUCTION = dict(facecolor="0.90", edgecolor="none", linewidth=0.0)

# key entries are labels, not sentences: the caption carries the argument
LBL_SLIP = "recovered by slip correction"
LBL_RESIDUAL = "residual: cracking and\nneutral-axis migration"
LBL_REDUCTION = "7–9 point range"

# in-panel keys are set one step below the axis labels, the sanctioned
# size for type that sits inside the data area
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

    The legend of (a) is placed in a corner the curves leave empty, so
    the claim that it covers nothing is measured rather than assumed.
    Every curve and both band polygons are densified and tested against
    the legend's rendered box; the return value is the smallest
    clearance above the box, negative if anything reaches into it.
    """
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    tr = ax.transData + ax.transAxes.inverted()
    bb = lg.get_window_extent(r)
    (x0, y0), (x1, y1) = ax.transAxes.inverted().transform(
        [[bb.x0, bb.y0], [bb.x1, bb.y1]])

    pts = []
    for ln in ax.lines:                            # curves and markers
        if ln.get_transform() is not ax.transData:  # axhline: blended
            continue
        d = np.asarray(ln.get_xydata(), float)
        if len(d):
            pts.append(tr.transform(_dense(d)))
    for col in ax.collections:                     # the shaded bands
        for p in col.get_paths():
            pts.append(tr.transform(_dense(p.vertices)))
    pts = np.vstack([p for p in pts if len(p)])
    pts = pts[np.isfinite(pts).all(axis=1)]

    over = pts[(pts[:, 0] >= x0 - 0.01) & (pts[:, 0] <= x1 + 0.01)]
    gap = float(over[:, 1].min() - y1) if len(over) else 1.0
    inside = int(((over[:, 1] >= y0) & (over[:, 1] <= y1)).sum())
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
    # reduction in deviation delivered by the slip correction, in points
    reduction = {t: series[("niecai", t)] - series[("aashto", t)]
                 for t in ("rho0", "rho07")}

    for t in ("rho0", "rho07"):
        print(f"[{t}] AASHTO    " + "  ".join(f"{v:+6.1f}" for v in series[("aashto", t)]))
        print(f"[{t}] Nie&Cai   " + "  ".join(f"{v:+6.1f}" for v in series[("niecai", t)]))
        print(f"[{t}] reduction " + "  ".join(f"{v:+6.1f}" for v in reduction[t]))
    print(f"[n] rho0 rows = {int(d0['n_rows'].sum())}")

    FS.apply()
    # no legend block above the panels any more, so the panels themselves
    # take the height the block used to occupy
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
    # the two mechanisms, as the vertical gaps they occupy; the bands run
    # the full width so they do not read as free-floating rectangles.
    # both are measured from the same zero line, so slip band + residual
    # band = the whole AASHTO deviation, bin by bin
    xf = np.concatenate(([XLO], x, [XHI]))
    ef = lambda v: np.concatenate(([v[0]], v, [v[-1]]))  # noqa: E731
    axa.fill_between(xf, ef(a0), ef(n0), zorder=1, **BAND_SLIP)
    axa.fill_between(xf, ef(n0), 0.0, zorder=0, **BAND_RESIDUAL)
    axa.axhline(0.0, color="black", lw=1.0, zorder=2)

    for tag, ls in (("rho0", "-"), ("rho07", (0, (5, 1.8)))):
        for pred in ("aashto", "niecai"):
            axa.plot(x, series[(pred, tag)], zorder=4,
                     **FS.style(pred, ls=ls, label=False, lw=1.9,
                                ms=5.2, mfc="white" if tag == "rho07" else None,
                                mew=1.4))

    axa.set_ylim(-56, 9)
    axa.set_yticks([-50, -40, -30, -20, -10, 0])
    FS.eta_bin_axis(axa, positions=x)
    axa.set_xlim(XLO, XHI)
    axa.set_ylabel("curvature deviation from\nOpenSeesPy (%)")

    # (a) keys itself in the corner its own data leaves empty.  The zero
    # line needs no entry: the y axis already says what zero is.
    keys_a = [FS.handle("aashto", ls="-", lw=1.9, ms=5.0, fontsize=FS_KEY),
              FS.handle("niecai", ls="-", lw=1.9, ms=5.0, fontsize=FS_KEY),
              Line2D([], [], color="0.35", ls="-", lw=1.5, marker="o", ms=5.0,
                     label=FS.entity_label("rho0", fontsize=FS_KEY)),
              Line2D([], [], color="0.35", ls=(0, (5, 1.8)), lw=1.5,
                     marker="o", ms=5.0, mfc="white", mew=1.4,
                     label=FS.entity_label("rho07", fontsize=FS_KEY)),
              Patch(label=LBL_SLIP, **BAND_SLIP),
              Patch(label=LBL_RESIDUAL, **BAND_RESIDUAL)]
    loc_a = FS.legend_loc(axa, size=(0.58, 0.30),
                          candidates=["lower right", "lower left",
                                      "upper left", "upper right"])
    lga = axa.legend(handles=keys_a, loc=loc_a, frameon=False,
                     handlelength=1.8, handleheight=1.00, handletextpad=0.45,
                     borderaxespad=0.5, labelspacing=0.26, fontsize=FS_KEY)
    if hasattr(lga, "set_alignment"):
        lga.set_alignment("left")

    # -------------------------------------------------- (b) reduction
    axb.axhspan(7.0, 9.0, zorder=0, **BAND_REDUCTION)
    for tag in FS.REINFORCEMENT:
        open_mk = dict(mfc="white", mew=1.4) if tag == "rho07" else {}
        axb.plot(x, reduction[tag], zorder=3,
                 **FS.style(tag, label=False, lw=1.9, ms=5.2, **open_mk))
    axb.set_ylim(0, 11.0)
    axb.set_yticks([0, 2, 4, 6, 8, 10])
    FS.eta_bin_axis(axb, positions=x)
    axb.set_xlim(XLO, XHI)
    axb.set_ylabel("reduction in deviation,\nAASHTO $-$ Nie & Cai\n"
                   "(percentage points)")
    # (b) keys itself in its own black and magenta styles
    loc_b = FS.legend_loc(axb, size=(0.74, 0.15),
                          candidates=["lower left", "lower right",
                                      "upper left", "upper right"])
    lgb = axb.legend(handles=[FS.handle("rho0", lw=1.9, ms=5.0,
                                        fontsize=FS_KEY),
                              FS.handle("rho07", lw=1.9, ms=5.0, mfc="white",
                                        mew=1.4, fontsize=FS_KEY),
                              Patch(label=LBL_REDUCTION, **BAND_REDUCTION)],
                     loc=loc_b, frameon=False, handlelength=2.0,
                     handleheight=1.00, handletextpad=0.5, borderaxespad=0.5,
                     labelspacing=0.34, fontsize=FS_KEY)
    if hasattr(lgb, "set_alignment"):
        lgb.set_alignment("left")

    FS.panel(axa, "(a)", "deviation from the OpenSeesPy reference", dy=1.030)
    FS.panel(axb, "(b)", "reduction by the slip correction", dy=1.030)

    print(f"[key] (a) -> {loc_a}, (b) -> {loc_b}")
    free_check(axa, lga, "(a)")
    free_check(axb, lgb, "(b)")

    # judged at printed size: the figure is included at width=\linewidth
    scale = FS.printed_scale(fig, 1.0)
    probs = FS.audit(fig, scale=scale)
    print(f'[audit] {"clean" if not probs else f"{len(probs)} problem(s)"} '
          f"(printed scale {scale:.3f})")
    FS.save(fig, OUT)
    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
