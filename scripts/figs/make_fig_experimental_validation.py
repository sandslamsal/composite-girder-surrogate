#!/usr/bin/env python
"""Figure: comparison against published composite-beam tests.

(a) measured moment-curvature points of the three published test series
    overlaid on the computed fiber-section response.  Points whose
    measured moment exceeds the model peak are drawn hollow: the
    prediction there is flagged, never extrapolated.  UNCHANGED.

(b) interface end slip against applied load on the Sheehan, Dai and Lam
    (2018) beam at a degree of shear connection of 0.33, with the
    connector stiffness fixed at the 70 kN/mm that the parent DISCCO
    project published a priori, so neither prediction has a free
    parameter.  Two independent partial-interaction formulations - the
    Newmark closed form that DISCCO itself uses, and this study's
    beam-level model - are drawn against the measured slip.  The inset
    shows mid-span deflection on the same load cycles.

    This panel is NOT a validation of interface slip and must not be
    described as one.  It shows that at service load both formulations
    over-predict the measured end slip by nearly an order of magnitude
    while predicting the deflection at the same load to within a few per
    cent: end slip at service load is a badly conditioned target because
    the interface has not debonded and the measured slip is governed by
    chemical bond, friction and the mechanical keying of the profiled
    deck, none of which a partial-interaction formulation carries.  The
    flexural stiffness this study is about is well conditioned.

Data
----
data/experimental/literature_tests.csv        measured M-phi points
reports/model_validation/per_specimen.csv     flags for panel (a)
reports/model_validation/sheehan_partial_interaction.csv
    measured, closed-form and beam-model slip and deflection, written by
    scripts/validate_sheehan_partial_interaction.py
<cache>/exp_curves.npz                        M-phi curves, written by
    scripts/figs/cache_experimental_curves.py

Output
------
paper/revision_1/submission/sources/figures/fig_experimental_validation.{png,pdf}
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.utils import figstyle as FS  # noqa: E402

TESTS = REPO / "data/experimental/literature_tests.csv"
PER = REPO / "reports/model_validation/per_specimen.csv"
SHEEHAN = REPO / "reports/model_validation/sheehan_partial_interaction.csv"
OUT = REPO / "paper/revision_1/submission/sources/figures/fig_experimental_validation.png"

# test series -> (short label, colour, marker)
PROG = {
    "Chapman_Balakrishnan_1964": ("Chapman & Balakrishnan (1964)",
                                  FS.VERM, "o"),
    "Nie_Cai_2003": ("Nie & Cai (2003)", FS.SKY, "^"),
    "Ansourian_1982": ("Ansourian (1982)", FS.GREEN, "s"),
}

# panel (b) identities.  Measured data is black; the two predictions keep
# the registry identities they carry elsewhere in the paper (the
# beam-level model is magenta with a long dash and a down triangle) or,
# for the closed form, the vermilion dotted diamond of a code-style
# closed-form predictor.
C_MEAS = "black"
C_CF = FS.VERM
C_BM = FS.color("beam")

# DISCCO EUR 28458 EN Section 4.4, printed for this beam at 5 kN/m2 with
# the same k = 70 kN/mm: end slip 0.53 mm.  Plotted hollow so the
# reimplementation of their closed form can be seen to reproduce it.
DISCCO_Q, DISCCO_SLIP = 5.0, 0.53


def rising(m, *others):
    """Strictly increasing prefix of the moment history, to its peak."""
    if m.size == 0:
        return (m,) + others
    i = int(np.argmax(m))
    mm = m[: i + 1]
    keep = np.zeros(mm.size, bool)
    run = -np.inf
    for k, v in enumerate(mm):
        if v > run:
            keep[k] = True
            run = v
    return (mm[keep],) + tuple(o[: i + 1][keep] for o in others)


# --------------------------------------------------------------- panel (a)

def panel_mphi(ax, tests, per, curves, index):
    seen = set()
    for _, row in tests.iterrows():
        sp = index[row.test_id]
        if sp in seen:
            continue
        seen.add(sp)
        m, phi = rising(curves[f"{sp}_sec_M"], curves[f"{sp}_sec_phi"])
        ax.plot(phi * 1e3, m, color=FS.color("opensees"), ls="-", lw=1.0,
                marker="", zorder=2, solid_capstyle="round")

    n_pt = n_hollow = 0
    for prog, (label, col, mk) in PROG.items():
        sub = tests[tests.source == prog]
        for _, row in sub.iterrows():
            phi_m = row.measured_curvature_1_per_in
            if not np.isfinite(phi_m):
                continue
            flags = str(per.loc[row.test_id, "flags"])
            beyond = "section_beyond_model_peak" in flags
            n_pt += 1
            n_hollow += int(beyond)
            ax.plot(phi_m * 1e3, row.measured_moment_kip_in, ls="none",
                    marker=mk, ms=5.2, mew=1.1,
                    mfc="white" if beyond else col, mec=col, zorder=4)
    ax.set_xlabel(r"curvature $\varphi$ ($\times10^{-3}$ 1/in)")
    ax.set_ylabel("moment $M$ (kip-in)")
    ax.set_xlim(0.0, 4.0)
    ax.set_ylim(0.0, 6600.0)
    ax.set_yticks(np.arange(0, 6001, 2000))
    return n_pt, n_hollow


# --------------------------------------------------------------- panel (b)

def panel_slip(ax, sh):
    """End slip against applied load, three decades, log ordinate.

    The measured quantity is drawn as the band between the two rows
    Sheehan Table 2 prints (the maximum reached within each load cycle
    and the cumulative maximum including the residual slip carried
    forward), with markers on the cumulative row - the row the parent
    project reads when it describes this test.
    """
    q = sh.load_kn_per_m2.to_numpy()
    fail = sh.is_failure_cycle.to_numpy(bool)
    lo = sh.slip_cycle_max_mm.to_numpy()
    hi = sh.slip_cumulative_mm.to_numpy()

    ax.fill_between(q, lo, hi, facecolor="0.55", alpha=0.30, lw=0.0, zorder=1)
    ax.plot(q, hi, color=C_MEAS, ls="-", lw=1.1, marker="", zorder=4)
    ax.plot(q[~fail], hi[~fail], ls="none", marker="o", ms=4.6, mew=1.0,
            mfc=C_MEAS, mec=C_MEAS, zorder=5)
    ax.plot(q[fail], hi[fail], ls="none", marker="o", ms=4.6, mew=1.0,
            mfc="white", mec=C_MEAS, zorder=5)

    ax.plot(q, sh.cf_end_slip_mm, color=C_CF, ls=(0, (1.4, 1.3)), lw=1.3,
            marker="D", ms=3.6, mfc=C_CF, mec=C_CF, zorder=3)
    ax.plot(q, sh.bm_end_slip_mm, color=C_BM, ls=(0, (5, 2)), lw=1.3,
            marker="v", ms=4.2, mfc=C_BM, mec=C_BM, zorder=3)
    ax.plot([DISCCO_Q], [DISCCO_SLIP], ls="none", marker="D", ms=6.4,
            mew=1.2, mfc="white", mec=C_CF, zorder=6)

    ax.set_yscale("log")
    ax.set_xlim(1.6, 19.6)
    ax.set_ylim(0.008, 90.0)
    ax.set_xticks([5, 10, 15])
    ax.set_yticks([0.01, 0.1, 1.0, 10.0])
    ax.set_yticklabels(["0.01", "0.1", "1", "10"])
    ax.set_xlabel("applied load (kN/m$^2$)")
    ax.set_ylabel("interface end slip (mm)")
    return q, hi, lo


def inset_deflection(ax, sh):
    """Mid-span deflection on the same cycles: the well-conditioned
    counterpart of the slip panel, on its own axes rather than a second
    ordinate."""
    ins = ax.inset_axes([0.165, 0.655, 0.355, 0.300])
    keep = ~sh.is_failure_cycle.to_numpy(bool)
    q = sh.load_kn_per_m2.to_numpy()[keep]
    ins.plot(q, sh.defl_cycle_max_mm.to_numpy()[keep], color=C_MEAS, ls="-",
             lw=1.0, marker="o", ms=3.0, mfc=C_MEAS, mec=C_MEAS, zorder=3)
    ins.plot(q, sh.bm_deflection_mm.to_numpy()[keep], color=C_BM,
             ls=(0, (5, 2)), lw=1.1, marker="v", ms=3.2, mfc=C_BM, mec=C_BM,
             zorder=2)
    ins.set_xlim(1.5, 16.5)
    ins.set_ylim(0.0, 88.0)
    ins.set_xticks([5, 15])
    ins.set_yticks([0, 40, 80])
    ins.tick_params(labelsize=FS.FS_SMALL, length=2.0, width=0.7, pad=1.6)
    ins.set_ylabel("deflection (mm)", fontsize=FS.FS_SMALL, labelpad=1.0)
    for s in ins.spines.values():
        s.set_linewidth(0.7)
    return ins


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(
        Path("/private/tmp/claude-501/-Users-sandeshlamsal-Desktop-"
             "CompositeGirder/1b666f82-1e84-4f9e-a35c-afb843f2b292/"
             "scratchpad/exp_curves.npz")))
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

    n_pt, n_hollow = panel_mphi(ax_a, tests, per, curves, index)
    q, hi, lo = panel_slip(ax_b, sh)
    inset_deflection(ax_b, sh)

    FS.panel(ax_a, "(a)", "moment–curvature")
    FS.panel(ax_b, "(b)", "end slip, $\\eta$ = 0.33")

    # ---- shared legend for panel (a), above the panel row, never over data
    handles = [Line2D([], [], color=FS.color("opensees"), lw=1.0)]
    labels = ["OpenSeesPy fiber section"]
    for _p, (label, col, mk) in PROG.items():
        handles.append(Line2D([], [], ls="none", marker=mk, ms=5.2, mew=1.1,
                              mfc=col, mec=col))
        labels.append(label)
    handles.append(Line2D([], [], ls="none", marker="o", ms=5.2, mew=1.1,
                          mfc="white", mec="0.35"))
    # the count is a sample size, which the caption and the text give;
    # the legend entry only has to name the mark
    labels.append("beyond model peak")
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.978), frameon=False,
               fontsize=FS.FS_LEGEND, handlelength=1.4, handletextpad=0.5,
               columnspacing=1.6, labelspacing=0.34)

    # how many of the 24 points are plotted, and why the rest are not, is
    # stated where the figure is discussed; the panel does not repeat it

    # ---- panel (b) legend, in the empty lower-right corner.  Every mark
    # on the panel gets exactly one entry, including the grey band, whose
    # decoding used to live in the caption.
    h_b = [
        Line2D([], [], color=C_MEAS, ls="-", lw=1.1, marker="o", ms=4.0,
               mfc=C_MEAS, mec=C_MEAS),
        Line2D([], [], ls="none", marker="o", ms=4.0, mew=1.0, mfc="white",
               mec=C_MEAS),
        Patch(facecolor="0.55", alpha=0.30, lw=0.0),
        Line2D([], [], color=C_CF, ls=(0, (1.4, 1.3)), lw=1.3, marker="D",
               ms=3.4, mfc=C_CF, mec=C_CF),
        Line2D([], [], ls="none", marker="D", ms=4.6, mew=1.1, mfc="white",
               mec=C_CF),
        Line2D([], [], color=C_BM, ls=(0, (5, 2)), lw=1.3, marker="v",
               ms=4.0, mfc=C_BM, mec=C_BM),
    ]
    l_b = ["measured, Sheehan et al. (2018)",
           "failure cycle",
           "per-cycle to cumulative range",
           "Newmark closed form, $k$ = 70 kN/mm",
           "DISCCO published, 0.53 mm",
           "beam model (this study), same $k$"]
    ax_b.legend(h_b, l_b, loc="lower right", bbox_to_anchor=(1.02, 0.005),
                frameon=False, fontsize=FS.FS_SMALL, handlelength=2.0,
                handletextpad=0.45, labelspacing=0.24, borderpad=0.1)

    # ---- the finding, stated on the panel
    ratio_cf = float(sh.cf_end_slip_mm[1] / sh.slip_cumulative_mm[1])
    ratio_bm = float(sh.bm_end_slip_mm[1] / sh.slip_cumulative_mm[1])
    # Span the two points with a double-headed arrow and hang the label
    # off it with a leader, so the text is unambiguously attached to the
    # gap it describes. A bare arrow with the caption floating beside it
    # left the reader to guess which pair of points was being compared.
    x_cal = 5.0
    y_lo = float(sh.slip_cumulative_mm[1])
    y_hi = float(sh.cf_end_slip_mm[1])
    ax_b.annotate("", xy=(x_cal, y_lo), xytext=(x_cal, y_hi),
                  arrowprops=dict(arrowstyle="<->", lw=1.1, color="0.25",
                                  shrinkA=2.0, shrinkB=2.0,
                                  mutation_scale=11))
    # short serifs closing the span, so it reads as a measured interval
    for yy in (y_lo, y_hi):
        ax_b.plot([x_cal * 0.955, x_cal * 1.045], [yy, yy],
                  color="0.25", lw=0.8, solid_capstyle="butt", zorder=5)
    y_mid = (y_lo * y_hi) ** 0.5          # geometric mean: log-axis centre
    # Label to the LEFT of the bracket. To its right the measured curve
    # climbs steeply through exactly this height and overprinted the
    # digits; to its left the curve is still an order of magnitude below.
    # The halo is belt and braces: figstyle.audit compares text bounding
    # boxes against other TEXT, so it cannot see text-over-line overlap
    # and did not catch the original collision.
    ax_b.annotate(f"{ratio_bm:.0f}$-${ratio_cf:.0f}$\\times$",
                  xy=(x_cal, y_mid), xycoords="data",
                  xytext=(-6, 0), textcoords="offset points",
                  fontsize=FS.FS_ANNOT, color="0.20", ha="right", va="center",
                  bbox=dict(boxstyle="square,pad=0.12", fc="white",
                            ec="none", alpha=0.85))
    # That the connector stiffness was published a priori, and that
    # neither prediction therefore carries a free parameter, is a claim
    # about the method: it belongs to the text and the caption, not to
    # the panel, and is not repeated here.

    fig.subplots_adjust(left=0.083, right=0.995, top=0.845, bottom=0.155,
                        wspace=0.30)

    probs = FS.audit(fig)
    print("[audit] clean" if not probs
          else f"[audit] {len(probs)} problem(s)")
    for p in probs:
        print("   ", p)
    print(f"[data] panel a: {n_pt} points, {n_hollow} beyond model peak")
    print("[data] panel b: Sheehan et al. (2018), eta = 0.33, "
          "k = 70 kN/mm published by DISCCO")
    print(f"   measured cumulative end slip {hi.min():.3f}-{hi.max():.2f} mm "
          f"over {q.min():.0f}-{q.max():.0f} kN/m2")
    print(f"   at 5 kN/m2: measured {sh.slip_cumulative_mm[1]:.3f} mm | "
          f"closed form {sh.cf_end_slip_mm[1]:.2f} mm ({ratio_cf:.1f}x) | "
          f"beam model {sh.bm_end_slip_mm[1]:.2f} mm ({ratio_bm:.1f}x)")
    serv = sh[sh.load_kn_per_m2 <= 15.0]
    r = serv.bm_defl_over_measured_cycle
    print(f"   deflection ratio 3-15 kN/m2: {r.min():.2f} to {r.max():.2f} "
          "x measured")
    FS.save(fig, OUT)
    print(f"[out] {OUT}")


if __name__ == "__main__":
    main()
