#!/usr/bin/env python
"""Figure: comparison against published composite-beam tests.

(a) Moment-curvature.  The 19 measured moment-curvature points of the
    three published test series that carry a digitised curvature,
    overlaid on the computed fiber-section curve of each specimen that
    contributes a point.  Curves and points share the colour and line
    style of their test series, so a point can be read against its own
    specimen's curve.  Each curve is drawn over the whole analysis, rising
    branch, peak and post-peak branch, because the statistic the text
    quotes is the moment read at the measured curvature, and several
    measured curvatures lie past the computed peak.

    Hollow markers: the measured moment exceeds the computed peak moment
    (flag ``section_beyond_model_peak`` in per_specimen.csv, set by
    ``interp_at_moment`` in scripts/validate_models_experimental.py).
    The flag withholds only the curvature read at the measured moment;
    the moment read at the measured curvature exists for every plotted
    point, hollow or filled, and every moment-at-curvature statistic in
    the text uses all 19.

(b) Partial-interaction check.  Interface end slip against applied load
    on the Sheehan, Dai and Lam (2018) beam at a degree of shear
    connection of 0.33, with the connector stiffness fixed at the
    70 kN/mm that the parent DISCCO project published a priori, so
    neither prediction has a free parameter.  Two independent
    partial-interaction formulations, the Newmark closed form that DISCCO
    itself uses and this study's beam-level model, are drawn against the
    measured slip.  The inset shows mid-span deflection on the same load
    cycles for the measurement and both formulations.

    This panel is not a validation of interface slip.  At service load
    both formulations over-predict the measured end slip by nearly an
    order of magnitude while predicting the deflection at the same load
    to within a few per cent.

Data
----
data/experimental/literature_tests.csv        measured M-phi points
reports/model_validation/per_specimen.csv     flags for panel (a)
reports/model_validation/sheehan_partial_interaction.csv
    measured (Sheehan Table 2 slip, Table 1 deflection), closed-form and
    beam-model slip and deflection, written by
    scripts/validate_sheehan_partial_interaction.py
reports/model_validation/exp_curves.npz       M-phi curves, written by
    scripts/figs/cache_experimental_curves.py

Output
------
paper/revision_2/submission/sources/figures/fig_experimental_validation.pdf
figures/preview/fig_experimental_validation.png   (preview only)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.utils import figstyle as FS  # noqa: E402

TESTS = REPO / "data/experimental/literature_tests.csv"
PER = REPO / "reports/model_validation/per_specimen.csv"
SHEEHAN = REPO / "reports/model_validation/sheehan_partial_interaction.csv"
OUT = REPO / "paper/revision_2/submission/sources/figures/fig_experimental_validation.pdf"

# test series -> (legend label, colour, marker, curve line style).
# Chapman is blue, not vermilion: vermilion is the closed-form predictor in
# panel (b), and one hue must not mean a measurement on the left and a
# prediction on the right.  Line styles differ so the Nie and Cai curve,
# which runs through the Ansourian cluster, separates in greyscale.
PROG = {
    "Chapman_Balakrishnan_1964": ("Chapman & Balakrishnan (1964)",
                                  FS.BLUE, "o", "-"),
    "Nie_Cai_2003": ("Nie & Cai (2003)", FS.SKY, "^", (0, (5.0, 1.6, 1.2, 1.6))),
    "Ansourian_1982": ("Ansourian (1982)", FS.GREEN, "s", (0, (3.2, 1.5))),
}
CURVE_LW = 0.95
MS_A, MEW_A = 5.2, 1.1
# a filled point closer than this to a hollow point of the same series is
# drawn smaller, at its own position, inside the ring (curvature in
# 1e-3 1/in, moment in kip-in).  2.0 pt keeps a white gap to the ring's
# inner edge for the 0.9 pt offset of AN-CTB6 above AN-CTB3; 2.5 pt merged
# with the top edge.
NEST_DX, NEST_DY = 0.03, 80.0
MS_NEST = 2.0

# panel (b) identities.  Measured data is black; the closed form is the
# vermilion dotted diamond of a code-style closed-form predictor and the
# beam-level model keeps its registry identity (magenta, long dash, down
# triangle).
C_MEAS = "black"
C_CF = FS.VERM
C_BM = FS.color("beam")
LS_CF = (0, (1.4, 1.3))
LS_BM = (0, (5, 2))
MS_MEAS, MS_CF, MS_BM = 4.6, 3.6, 4.2

# The DISCCO EUR 28458 EN Section 4.4 end slip for this beam at 5 kN/m2
# (0.53 mm, same k) is quoted in the text, not drawn: at this scale it
# falls between the closed-form diamond (0.565 mm) and the beam-model
# triangle (0.476 mm), less than 2 pt from either, and a marker there
# covered both and captured the gap bracket.


# --------------------------------------------------------------- panel (a)

def plotted_points(tests, per):
    """Rows that carry a measured curvature, with their hollow flag."""
    rows = []
    for prog in PROG:
        sub = tests[tests.source == prog]
        for _, row in sub.iterrows():
            if not np.isfinite(row.measured_curvature_1_per_in):
                continue
            flags = str(per.loc[row.test_id, "flags"])
            rows.append((prog, row, "section_beyond_model_peak" in flags))
    return rows


def panel_mphi(ax, tests, per, curves, index):
    pts = plotted_points(tests, per)

    # one curve per specimen that contributes a plotted point, whole analysis
    seen = set()
    for prog, row, _ in pts:
        sp = index[row.test_id]
        if sp in seen:
            continue
        seen.add(sp)
        _, col, _, ls = PROG[prog]
        m, phi = curves[f"{sp}_sec_M"], curves[f"{sp}_sec_phi"]
        # The analysis records its first increment, not the unloaded state;
        # every curve starts from the origin, so draw it from there.
        if phi.size and phi[0] > 0.0:
            m, phi = np.concatenate(([0.0], m)), np.concatenate(([0.0], phi))
        ax.plot(phi * 1e3, m, color=col, ls=ls, lw=CURVE_LW, marker="",
                zorder=2, solid_capstyle="round", dash_capstyle="butt")

    # filled markers first, hollow above them.  A filled point that sits
    # under a hollow one of its own series (AN-CTB6 under AN-CTB3: same
    # curvature, 36 kip-in apart, under 1 pt at this scale) would vanish
    # behind the white face of the ring, so it is drawn instead as a smaller
    # filled marker inside that ring, and both points stay readable.
    hollow_xy = [(p, r.measured_curvature_1_per_in * 1e3,
                  r.measured_moment_kip_in) for p, r, b in pts if b]
    n_nested = 0
    for prog, row, beyond in pts:
        _, col, mk, _ = PROG[prog]
        x = row.measured_curvature_1_per_in * 1e3
        y = row.measured_moment_kip_in
        covered = (not beyond) and any(
            p == prog and abs(x - hx) < NEST_DX and abs(y - hy) < NEST_DY
            for p, hx, hy in hollow_xy)
        if covered:
            n_nested += 1
            ax.plot(x, y, ls="none", marker=mk, ms=MS_NEST, mew=0.0,
                    mfc=col, mec=col, zorder=6)
            continue
        ax.plot(x, y, ls="none", marker=mk, ms=MS_A, mew=MEW_A,
                mfc="white" if beyond else col, mec=col,
                zorder=5 if beyond else 4)
    ax._n_nested = n_nested

    ax.set_xlabel(r"curvature $\varphi$ ($10^{-3}$ 1/in)")
    ax.set_ylabel("moment $M$ (kip-in)")
    ax.set_xlim(0.0, 4.0)
    ax.set_ylim(0.0, 7000.0)
    ax.set_yticks(np.arange(0, 6001, 2000))
    return pts, len(seen)


def legend_mphi(ax, pts):
    """One row per test series: its curve segment beside its marker, the
    marker filled only if a filled instance is drawn; then one row for the
    hollow convention showing exactly the hollow marks that occur."""
    handles, labels = [], []
    hollow = []
    for prog, (label, col, mk, ls) in PROG.items():
        mine = [b for p, _, b in pts if p == prog]
        if not mine:
            continue
        filled = not all(mine)
        seg = Line2D([], [], color=col, ls=ls, lw=CURVE_LW)
        mark = Line2D([], [], ls="none", marker=mk, ms=MS_A, mew=MEW_A,
                      mfc=col if filled else "white", mec=col)
        handles.append((seg, mark))
        labels.append(label)
        if any(mine):
            hollow.append(Line2D([], [], ls="none", marker=mk, ms=MS_A,
                                 mew=MEW_A, mfc="white", mec=col))
    handles.append(tuple(hollow))
    labels.append("hollow: measured $M$\nabove model peak")
    # Glyphs after the labels, flush with the right spine: the free space
    # is the upper right, and glyph-first rows put a hollow circle glyph
    # beside the hollow Chapman points at 1.3e-3 to 1.5e-3 1/in, where it
    # read as one more data point.
    ax.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.02, 1.0),
              frameon=False, fontsize=FS.FS_SMALL, handlelength=3.0,
              handletextpad=0.5, labelspacing=0.30, borderpad=0.1,
              markerfirst=False, alignment="right",
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.35)})


# --------------------------------------------------------------- panel (b)

def panel_slip(ax, sh):
    """End slip against applied load, three decades, log ordinate.

    The measured quantity is drawn as the band between the two rows
    Sheehan Table 2 prints (the maximum reached within each load cycle
    and the cumulative maximum including the residual slip carried
    forward), with markers on the cumulative row, the row the parent
    project reads when it describes this test.
    """
    q = sh.load_kn_per_m2.to_numpy()
    fail = sh.is_failure_cycle.to_numpy(bool)
    lo = sh.slip_cycle_max_mm.to_numpy()
    hi = sh.slip_cumulative_mm.to_numpy()

    ax.fill_between(q, lo, hi, facecolor="0.55", alpha=0.30, lw=0.0, zorder=1)
    ax.plot(q, hi, color=C_MEAS, ls="-", lw=1.1, marker="", zorder=4)
    ax.plot(q[~fail], hi[~fail], ls="none", marker="o", ms=MS_MEAS, mew=1.0,
            mfc=C_MEAS, mec=C_MEAS, zorder=5)
    ax.plot(q[fail], hi[fail], ls="none", marker="o", ms=MS_MEAS, mew=1.0,
            mfc="white", mec=C_MEAS, zorder=5)

    ax.plot(q, sh.cf_end_slip_mm, color=C_CF, ls=LS_CF, lw=1.3,
            marker="D", ms=MS_CF, mfc=C_CF, mec=C_CF, zorder=3)
    ax.plot(q, sh.bm_end_slip_mm, color=C_BM, ls=LS_BM, lw=1.3,
            marker="v", ms=MS_BM, mfc=C_BM, mec=C_BM, zorder=3)

    ax.set_yscale("log")
    ax.set_xlim(0.9, 19.6)
    ax.set_ylim(0.004, 90.0)
    ax.set_xticks([5, 10, 15])
    ax.set_yticks([0.01, 0.1, 1.0, 10.0])
    ax.set_yticklabels(["0.01", "0.1", "1", "10"])
    ax.set_xlabel("applied load (kN/m$^2$)")
    ax.set_ylabel("interface end slip (mm)")
    return q, hi, lo


def inset_deflection(ax, sh):
    """Mid-span deflection on the same cycles (per-cycle maximum row of
    Sheehan Table 1), with both formulations, on its own axes."""
    ins = ax.inset_axes([0.175, 0.690, 0.320, 0.270])
    keep = ~sh.is_failure_cycle.to_numpy(bool)
    q = sh.load_kn_per_m2.to_numpy()[keep]
    ins.plot(q, sh.defl_cycle_max_mm.to_numpy()[keep], color=C_MEAS, ls="-",
             lw=1.0, marker="o", ms=3.0, mfc=C_MEAS, mec=C_MEAS, zorder=4)
    ins.plot(q, sh.cf_deflection_mm.to_numpy()[keep], color=C_CF, ls=LS_CF,
             lw=1.1, marker="D", ms=2.6, mfc=C_CF, mec=C_CF, zorder=2)
    ins.plot(q, sh.bm_deflection_mm.to_numpy()[keep], color=C_BM, ls=LS_BM,
             lw=1.1, marker="v", ms=3.2, mfc=C_BM, mec=C_BM, zorder=3)
    ins.set_xlim(1.5, 16.5)
    ins.set_ylim(0.0, 88.0)
    ins.set_xticks([5, 10, 15])
    ins.set_yticks([0, 40, 80])
    ins.tick_params(labelsize=FS.FS_SMALL, length=2.0, width=0.7, pad=1.6)
    ins.set_ylabel("deflection (mm)", fontsize=FS.FS_SMALL, labelpad=1.0)
    ins.set_xlabel("load (kN/m$^2$)", fontsize=FS.FS_SMALL, labelpad=0.8)
    for s in ins.spines.values():
        s.set_linewidth(0.7)
    return ins


def label_clearance(fig, ax, text, min_pt=1.0):
    """FS.audit checks text against text only.  Check the gap label against
    every line and marker on the main panel axes: the clearance, in points,
    between the text's box and each drawn line (sampled densely along its
    segments) and each marker's square extent."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb = text.get_window_extent(renderer=r)
    px_per_pt = fig.dpi / 72.0
    worst = np.inf
    for ln in ax.get_lines():
        xy = ax.transData.transform(np.column_stack(ln.get_data()))
        xy = xy[np.all(np.isfinite(xy), axis=1)]
        if len(xy) == 0:
            continue
        half_w = 0.5 * ln.get_linewidth() * px_per_pt
        if ln.get_linestyle() not in ("None", "none", "", " ") and len(xy) > 1:
            t = np.linspace(0.0, 1.0, 200)[:, None]
            seg = [a + t * (b - a) for a, b in zip(xy[:-1], xy[1:])]
            samp = np.vstack(seg)
            dx = np.maximum(np.maximum(bb.x0 - samp[:, 0], samp[:, 0] - bb.x1), 0)
            dy = np.maximum(np.maximum(bb.y0 - samp[:, 1], samp[:, 1] - bb.y1), 0)
            worst = min(worst, (np.hypot(dx, dy).min() - half_w) / px_per_pt)
        if ln.get_marker() not in ("None", "none", "", " ", None):
            h = 0.5 * (ln.get_markersize() + ln.get_markeredgewidth()) * px_per_pt
            dx = np.maximum(np.maximum(bb.x0 - (xy[:, 0] + h),
                                       (xy[:, 0] - h) - bb.x1), 0)
            dy = np.maximum(np.maximum(bb.y0 - (xy[:, 1] + h),
                                       (xy[:, 1] - h) - bb.y1), 0)
            worst = min(worst, np.hypot(dx, dy).min() / px_per_pt)
    print(f"[clearance] gap label to nearest line or marker: {worst:.2f} pt")
    if worst < min_pt:
        return [f"gap label within {worst:.2f} pt of a line or marker"]
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(
        REPO / "reports/model_validation/exp_curves.npz"))
    args = ap.parse_args()

    FS.apply()
    tests = pd.read_csv(TESTS)
    per = pd.read_csv(PER).set_index("test_id")
    sh = pd.read_csv(SHEEHAN)
    curves = np.load(args.cache)
    index = json.loads(Path(args.cache).with_suffix(".index.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(FS.FIG_W, 3.15),
                             gridspec_kw=dict(width_ratios=[1.18, 1.0]))
    ax_a, ax_b = axes

    pts, n_curves = panel_mphi(ax_a, tests, per, curves, index)
    q, hi, lo = panel_slip(ax_b, sh)
    inset_deflection(ax_b, sh)

    # headings carry no mathematics (mathtext does not print bold); the
    # degree of shear connection of the panel (b) beam is in the caption
    FS.panel(ax_a, "(a)", "moment–curvature")
    FS.panel(ax_b, "(b)", "partial-interaction check")

    legend_mphi(ax_a, pts)

    # ---- panel (b) legend, in the empty lower-right corner.  Every mark
    # on the panel gets exactly one entry, glyphs drawn as the marks are.
    h_b = [
        Line2D([], [], color=C_MEAS, ls="-", lw=1.1, marker="o", ms=MS_MEAS,
               mfc=C_MEAS, mec=C_MEAS),
        Line2D([], [], ls="none", marker="o", ms=MS_MEAS, mew=1.0,
               mfc="white", mec=C_MEAS),
        Patch(facecolor="0.55", alpha=0.30, lw=0.0),
        Line2D([], [], color=C_CF, ls=LS_CF, lw=1.3, marker="D", ms=MS_CF,
               mfc=C_CF, mec=C_CF),
        Line2D([], [], color=C_BM, ls=LS_BM, lw=1.3, marker="v", ms=MS_BM,
               mfc=C_BM, mec=C_BM),
    ]
    l_b = ["measured, Sheehan et al. (2018)",
           "failure cycle",
           "per-cycle to cumulative range",
           "Newmark closed form, $k$ = 70 kN/mm",
           "beam model (this study), same $k$"]
    ax_b.legend(h_b, l_b, loc="lower right", bbox_to_anchor=(1.02, 0.005),
                frameon=False, fontsize=FS.FS_SMALL, handlelength=2.0,
                handletextpad=0.45, labelspacing=0.24, borderpad=0.1)

    # ---- the gap at 5 kN/m2, measured to beam model, in absolute mm
    # (0.41 mm, the over-prediction the text quotes for this study's
    # model).  The closed-form diamond sits 3 pt above the beam-model
    # triangle, too close for a second arrowhead, so the bracket ends at
    # the nearer prediction and its label names both ends.  The label
    # hangs to the left of the arrow, in the space under the prediction
    # lines and above the measured line.
    slip_meas = float(sh.slip_cumulative_mm[1])
    slip_cf = float(sh.cf_end_slip_mm[1])
    slip_bm = float(sh.bm_end_slip_mm[1])
    x_cal = 5.0
    ax_b.annotate("", xy=(x_cal, slip_meas), xytext=(x_cal, slip_bm),
                  arrowprops=dict(arrowstyle="<->", lw=1.0, color="0.25",
                                  shrinkA=MS_BM / 2 + 1.2,
                                  shrinkB=MS_MEAS / 2 + 0.8,
                                  mutation_scale=9),
                  zorder=5)
    # The label sits 6 pt below the geometric mean of the arrow's ends:
    # centred on it, the top of "0.06 to" ran into the lower tip of the
    # beam-model triangle at 3 kN/m2.  The measured line below leaves room.
    y_mid = float(np.sqrt(slip_meas * slip_bm))
    gap_label = ax_b.annotate(f"{slip_meas:.2f} to\n{slip_bm:.2f} mm",
                              xy=(x_cal, y_mid), xycoords="data",
                              xytext=(-4, -6), textcoords="offset points",
                              fontsize=FS.FS_ANNOT, color="0.20", ha="right",
                              va="center", linespacing=1.1)

    fig.subplots_adjust(left=0.083, right=0.995, top=0.915, bottom=0.155,
                        wspace=0.30)

    probs = FS.audit(fig)
    probs += label_clearance(fig, ax_b, gap_label)
    print("[audit] clean" if not probs
          else f"[audit] {len(probs)} problem(s)")
    for p in probs:
        print("   ", p)
    n_hollow = sum(b for _, _, b in pts)
    print(f"[data] panel a: {len(pts)} points, {n_hollow} hollow "
          f"(measured M above model peak), {n_curves} curves, "
          f"{ax_a._n_nested} filled nested inside a ring")
    print("[data] panel b: Sheehan et al. (2018), eta = 0.33, "
          "k = 70 kN/mm published by DISCCO")
    print(f"   measured cumulative end slip {hi.min():.3f}-{hi.max():.2f} mm "
          f"over {q.min():.0f}-{q.max():.0f} kN/m2")
    print(f"   at 5 kN/m2: measured {slip_meas:.3f} mm | "
          f"closed form {slip_cf:.2f} mm | beam model {slip_bm:.2f} mm")
    serv = sh[sh.load_kn_per_m2 <= 15.0]
    r = serv.bm_defl_over_measured_cycle
    print(f"   deflection ratio 3-15 kN/m2: {r.min():.2f} to {r.max():.2f} "
          "x measured")
    FS.save(fig, OUT)
    print(f"[out] {OUT}")


if __name__ == "__main__":
    main()
