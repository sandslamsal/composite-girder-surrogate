#!/usr/bin/env python
"""Consolidated SURROGATE ACCURACY figure (replaces old Figs 3, 4 and 5).

Four panels, 2x2, drawn at the printed width of the manuscript
(``figstyle.FIG_W`` = 6.5 in, included at ``\\linewidth``):

    (a) training / validation MSE loss vs epoch, log y, the epoch of
        lowest validation loss (the released checkpoint) starred
    (b) neutral-axis-depth parity on the test split, 1:1 line
    (c) relative-error distribution of both outputs on the test split
    (d) curvature parity on the test split, 1:1 line

Split names follow the manuscript: "training", "validation", "test";
"held-out" means the test split only, so the validation curve of (a) is
never called held-out.  Panel (a) is drawn in neutral tones so that its
two split curves cannot be read as the two target colours of (b)-(d).

R^2 and MAPE are printed on the parity panels; the best epoch and the
inter-decile ranges are given in the text, so on the panels they would
only restate it; every one of them is still written to stdout by
:func:`main`.  The relative error of panel (c) and the MAPE of (b) and
(d) use ONE definition, 100 (surrogate - OpenSeesPy) / |OpenSeesPy|
(:func:`rel_error`), so the figure and Table tab:heldout cannot disagree.

Marks decoded in the caption rather than in a legend: the hexbin
shading of (b) and (d) (test rows per cell, logarithmic, darker for
more), the dashed y = x line of (b) and (d), and the dotted zero-error
line of (c).  Percent signs are set after a space throughout the figure
("MAPE 11.9 %", "1 % bin"), matching the manuscript's siunitx spacing.

The right column holds both parity plots, so the two 1:1
panels are read as a pair instead of straddling the diagonal of the
figure.  The four panel boxes are placed by hand on an inch grid and
are SQUARE: a parity panel carries ``set_aspect('equal')``, which
shrinks the axes inside its subplot cell whenever that cell is not
square, and it was exactly that shrinking that made the lower row
print shorter than the upper one under ``tight_layout``.  With square
cells the aspect constraint is already satisfied, so every panel keeps
the height it was given and the two row gaps are identical.

The reference everywhere is the OpenSeesPy fiber-section model; the
words 'truth' and 'ground truth' appear nowhere in the figure.

Everything is computed from the DEPLOYED checkpoint ``weights/best.pt``
and its ``weights/history.json`` -- the 15-feature retrained model whose
metrics are the ones tabulated in the manuscript (R^2 = 0.982 / 0.932,
662,530 parameters, 4891 held-out sections / 391,280 rows).  The
superseded 17-feature run (``checkpoints/run_2output_final``) is NOT
compatible with the current 15-feature normaliser and its best epoch
(287) belongs to a model that is no longer the paper's surrogate; the
deployed run peaks at epoch 296.

Usage:
    python scripts/figs/make_fig_surrogate_accuracy.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator

from src.utils import figstyle as FS

DATA = REPO_ROOT / "data" / "raw" / "full_50k.parquet"
CKPT = REPO_ROOT / "weights" / "best.pt"
HISTORY = REPO_ROOT / "weights" / "history.json"
OUT = REPO_ROOT / "paper" / "revision_2" / "submission" / "sources" / "figures" / "fig_surrogate_accuracy.pdf"
# repo-local and git-ignored (**/data/processed/); delete or pass
# --refresh after retraining so the panels never show stale predictions
CACHE = REPO_ROOT / "data" / "processed" / "heldout_pred.npz"

# split reproduction (identical to scripts/make_figures.py::_split_by_sample
# and scripts/revision_common.py, seed from configs/training.yaml)
SEED = 20260513
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}

CURV_SCALE = 1e3          # plot curvature in 1e-3 1/in (house unit)
ERR_CLIP = 50.0           # panel (c) x range; the caption states it and
                          # the text gives the share of rows beyond it

# panel (a) is neutral: the split curves must not borrow the target colours
# registry colours of the two loss curves (FS.ENTITY 'train' / 'heldout'),
# with their distinct dash patterns so they also separate in greyscale
TRAIN_KW = dict(color=FS.ORANGE, ls="-", lw=1.6)
VAL_KW = dict(color=FS.BLUE, ls=(0, (1, 1.1)), lw=1.5)

# ---- panel grid, in inches on a FIG_W-wide canvas.  Margins are sized
# for the widest tick labels and axis titles of each column / row; the
# panel box itself is square (see the module docstring).
M_LEFT = 0.58             # y tick labels + y label of the left column
M_COL = 0.62              # gap between the columns (right column's y axis)
M_RIGHT = 0.10
M_BOTTOM = 0.44           # x tick labels + x label of the lower row
M_ROW = 0.72              # x axis of the upper row + panel title of the lower
M_TOP = 0.26              # panel title of the upper row


# --------------------------------------------------------------- data
def _split_by_sample(df: pd.DataFrame, fracs: dict, seed: int):
    ids = df["sample_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(fracs["train"] * n))
    n_val = int(round(fracs["val"] * n))
    return (df[df["sample_id"].isin(set(ids[:n_train]))].reset_index(drop=True),
            df[df["sample_id"].isin(set(ids[n_train:n_train + n_val]))]
            .reset_index(drop=True),
            df[df["sample_id"].isin(set(ids[n_train + n_val:]))]
            .reset_index(drop=True))


def r2(y, yh):
    y, yh = np.asarray(y, float), np.asarray(yh, float)
    ss_res = np.sum((y - yh) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot


def rel_error(y, yh, eps=1e-6):
    """Signed relative error (%), 100 (yh - y) / |y|, |y| floored at eps.

    The single definition behind panel (c) AND the MAPE on (b), (d) and
    in Table tab:heldout.  (Panel (c) once floored |y| at 1e-3 of its
    mean instead, which moved the neutral-axis MAPE from 11.9 % to 8.4 %
    through 163 rows with |y_na| < 0.012 in; the percentiles quoted in
    the text are identical under both floors.)
    """
    y, yh = np.asarray(y, float), np.asarray(yh, float)
    denom = np.where(np.abs(y) > eps, np.abs(y), eps)
    return 100.0 * (yh - y) / denom


def mape(y, yh, eps=1e-6):
    return float(np.mean(np.abs(rel_error(y, yh, eps))))


def heldout_arrays(refresh: bool = False):
    """(true, pred) for the two outputs on the full held-out test split."""
    if CACHE.exists() and not refresh:
        z = np.load(CACHE)
        return {k: z[k] for k in z.files}
    from src.models.inference import SurrogatePredictor

    df = pd.read_parquet(DATA)
    _, _, test_df = _split_by_sample(df, SPLITS, SEED)
    print(f"[data] held out: {len(test_df):,} rows / "
          f"{test_df['sample_id'].nunique():,} sections", flush=True)
    predictor = SurrogatePredictor.load(CKPT)
    pred = predictor.predict(test_df)
    out = {
        "y_na_true": test_df["neutral_axis_in"].to_numpy(float),
        "y_na_pred": pred["neutral_axis_in"].to_numpy(float),
        "curv_true": test_df["curvature_1_per_in"].to_numpy(float),
        "curv_pred": pred["curvature_1_per_in"].to_numpy(float),
        "moment_ratio": test_df["moment_ratio"].to_numpy(float),
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **out)
    return out


# --------------------------------------------------------------- panels
def _seq_cmap(hex_color):
    """Light tint -> entity colour -> near-black sequential hexbin map.

    Starting from a tint rather than pure white keeps the single-count
    hexes visible for the pale (amber) target as well as the blue one.
    """
    import matplotlib.colors as mcolors
    rgb = np.array(mcolors.to_rgb(hex_color))
    tint = 1.0 - 0.26 * (1.0 - rgb)          # 26 % of the hue on white
    return LinearSegmentedColormap.from_list(
        "seq", [tuple(tint), hex_color, "#101C30"], N=256)


def _nice_ticks(lo, hi, n=5, edge=0.02):
    """Round ticks strictly inside (lo, hi), clear of the axes corners."""
    from matplotlib.ticker import MaxNLocator
    span = hi - lo
    t = MaxNLocator(nbins=n, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
    return [v for v in t if lo + edge * span <= v <= hi - edge * span]


def panel_loss(ax, history):
    epochs = np.array([h["epoch"] for h in history], float)
    train = np.array([h["train"]["total"] for h in history], float)
    val = np.array([h["val"]["total"] for h in history], float)
    best = int(np.argmin(val))

    ax.semilogy(epochs, train, label="training", zorder=3, **TRAIN_KW)
    ax.semilogy(epochs, val, label="validation", zorder=4, **VAL_KW)
    ax.plot([epochs[best]], [val[best]], marker="*", ms=12, mfc=FS.BLUE,
            mec="black", mew=0.9, ls="none", zorder=6, clip_on=False,
            label="lowest validation loss")

    ax.set_xlabel("epoch")
    ax.set_ylabel(r"MSE loss (normalised, $\times10^{-3}$)")
    ax.set_xlim(-4, 304)
    ax.set_ylim(1.22e-3, 8.2e-3)
    ticks = np.array([1.5e-3, 2e-3, 3e-3, 5e-3, 8e-3])
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=60))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _p: f"{v * 1e3:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _p: ""))
    ax.grid(True, which="major", axis="both", lw=0.6, color="0.9")

    # the star is decoded by the legend; its epoch (296) is given in the
    # text, so no annotation repeats it on the panel
    return int(epochs[best])


def panel_parity(ax, true, pred, key, axis_label, gridsize=52):
    lo = float(min(true.min(), pred.min()))
    hi = float(max(true.max(), pred.max()))
    pad = 0.03 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    hb = ax.hexbin(true, pred, gridsize=gridsize, bins="log", mincnt=1,
                   cmap=_seq_cmap(FS.color(key)), linewidths=0.0,
                   extent=(lo, hi, lo, hi))
    ax.plot([lo, hi], [lo, hi], color="black", ls=(0, (4, 2.2)), lw=1.0,
            zorder=4)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ticks = _nice_ticks(lo, hi)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel(f"OpenSeesPy {axis_label}")
    ax.set_ylabel(f"surrogate {axis_label}")
    # R^2 and MAPE ARE printed on the panel. They were removed once as
    # "redundant with the text", which was wrong: a parity cloud cannot be
    # judged without them, and a reader should not have to hold two numbers
    # from a distant paragraph in their head to read the panel.
    # Use THIS module's r2/mape, the same definitions that produce the
    # numbers tabulated in the manuscript and (through rel_error) the
    # distribution of panel (c).
    # Set at MATH_MIN_FS: a superscript renders at 0.7 of its parent, so
    # anything smaller puts "R^2" under the 6.5 pt floor and audit fails.
    FS.inside_label(ax, 0.045, 0.945,
                    f"$R^2 = {r2(true, pred):.3f}$\n"
                    f"MAPE {mape(true, pred):.1f} %",
                    transform=ax.transAxes, fontsize=FS.MATH_MIN_FS,
                    color="0.15", ha="left", va="top", linespacing=1.35)
    return hb


def panel_errors(ax, rel_errors):
    """Overlaid relative-error densities for the two outputs (log density,
    so the near-zero peak and the tails are both readable)."""
    edges = np.linspace(-ERR_CLIP, ERR_CLIP, 101)
    centres = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    for key in FS.TARGETS:
        rel = rel_errors[key]
        h, _ = np.histogram(rel, bins=edges)
        frac = 100.0 * h / rel.size / width          # % of rows per 1 % bin
        ax.semilogy(centres, np.where(frac > 0, frac, np.nan),
                    **FS.style(key, marker="", lw=1.7))
    ax.axvline(0.0, color="0.55", lw=0.7, ls=":", zorder=0)
    ax.set_xlim(-ERR_CLIP, ERR_CLIP)
    ax.set_ylim(5e-3, 4e2)
    ax.set_xticks([-50, -25, 0, 25, 50])
    ax.yaxis.set_major_locator(FixedLocator([1e-2, 1e-1, 1e0, 1e1, 1e2]))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=60))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _p: f"{v:g}" if v >= 0.1 else f"{v:.2f}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _p: ""))
    # the label spells out the quantity (rel_error), so it cannot be read
    # as an absolute difference given in percent; positive = over-prediction
    ax.set_xlabel("(surrogate − OpenSeesPy)/|OpenSeesPy| (%)")
    ax.set_ylabel("share of test rows (% per 1 % bin)")

    # A key that names the two curves and nothing else.  The inter-decile
    # ranges it used to carry are given in the results paragraph, and a
    # panel must not restate the text.  Drawn as a legend rather than as
    # coloured text so the dash pattern carries the identity too, which
    # is what keeps the panel readable in greyscale.  Upper right, at the
    # legend size of panel (a), clear of the curves and of the dotted
    # zero-error line (which the caption decodes).
    ax.legend(loc="upper right", frameon=False, fontsize=FS.FS_LEGEND,
              handlelength=2.2, handletextpad=0.55, labelspacing=0.3,
              borderaxespad=0.25)
    return centres


# --------------------------------------------------------------- figure
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="recompute the held-out predictions cache")
    args = ap.parse_args()

    FS.apply()
    d = heldout_arrays(refresh=args.refresh)
    history = json.loads(HISTORY.read_text())

    r2_na = r2(d["y_na_true"], d["y_na_pred"])
    r2_cu = r2(d["curv_true"], d["curv_pred"])
    mape_na = mape(d["y_na_true"], d["y_na_pred"])
    mape_cu = mape(d["curv_true"], d["curv_pred"])
    rmse_na = float(np.sqrt(np.mean((d["y_na_pred"] - d["y_na_true"]) ** 2)))
    rmse_cu = float(np.sqrt(np.mean((d["curv_pred"] - d["curv_true"]) ** 2)))
    print(f"[metrics] y_na  R2={r2_na:.4f} RMSE={rmse_na:.4g} in "
          f"MAPE={mape_na:.2f}%")
    print(f"[metrics] curv  R2={r2_cu:.4f} RMSE={rmse_cu:.4g} 1/in "
          f"MAPE={mape_cu:.2f}%")
    for lim in (0.4, 0.6):
        m = d["moment_ratio"] <= lim
        print(f"[metrics] M/Mp<={lim}: n={int(m.sum())} "
              f"curv R2={r2(d['curv_true'][m], d['curv_pred'][m]):.4f} "
              f"MAPE={mape(d['curv_true'][m], d['curv_pred'][m]):.2f}%")

    rel = {}
    for key, t, p in (("y_na", d["y_na_true"], d["y_na_pred"]),
                      ("curvature", d["curv_true"], d["curv_pred"])):
        r = rel_error(t, p)
        rel[key] = r[np.isfinite(r)]
        p10, p50, p90 = np.percentile(rel[key], [10, 50, 90])
        above = 100.0 * np.mean(rel[key] > ERR_CLIP)
        below = 100.0 * np.mean(rel[key] < -ERR_CLIP)
        print(f"[err] {key}: median={p50:.2f}%  P10={p10:.2f}%  "
              f"P90={p90:.2f}%  MAPE={np.mean(np.abs(rel[key])):.2f}%  "
              f"off-scale: {above + below:.2f}% of rows "
              f"({above:.2f}% > +50, {below:.2f}% < -50)")

    # ---- asymmetry about y = x that the results paragraph describes:
    # no net offset (mean residual << RMSE), but the large residuals of
    # both outputs lie mostly above the line; for the curvature they sit
    # at small reference values, and above 1e-3 1/in most rows fall
    # slightly below the line
    for key, t, p, big, unit in (
            ("y_na", d["y_na_true"], d["y_na_pred"], 2.0, "in"),
            ("curv", d["curv_true"], d["curv_pred"], 0.5e-3, "1/in")):
        res = p - t
        over, under = int(np.sum(res > big)), int(np.sum(res < -big))
        print(f"[asym] {key}: mean residual={res.mean():.3g} {unit} "
              f"(RMSE {np.sqrt(np.mean(res ** 2)):.3g}); |res|>{big:g}: "
              f"{over} above y=x, {under} below "
              f"({100.0 * over / max(over + under, 1):.1f}% above)")
    res_cu = d["curv_pred"] - d["curv_true"]
    small = d["curv_true"] < 1e-3
    print(f"[asym] curv: rows >0.5e-3 above y=x with reference <1e-3: "
          f"{100.0 * np.mean(small[res_cu > 0.5e-3]):.1f}% "
          f"(base rate {100.0 * small.mean():.1f}%); reference >=1e-3: "
          f"{100.0 * np.mean(res_cu[~small] < 0):.1f}% below y=x, median "
          f"rel. error {np.median(rel_error(d['curv_true'][~small], d['curv_pred'][~small])):.2f}%")

    # ---- square panel boxes on an inch grid, so the equal-aspect parity
    # panels fill their cell and both rows print at the same height with
    # the same gap between them
    pw = (FS.FIG_W - M_LEFT - M_COL - M_RIGHT) / 2.0
    ph = pw
    fig_h = M_BOTTOM + 2.0 * ph + M_ROW + M_TOP
    fig = plt.figure(figsize=(FS.FIG_W, fig_h))

    def cell(col, row):
        """Axes at (col, row); row 0 is the upper row."""
        x0 = M_LEFT + col * (pw + M_COL)
        y0 = M_BOTTOM + (1 - row) * (ph + M_ROW)
        return fig.add_axes((x0 / FS.FIG_W, y0 / fig_h,
                             pw / FS.FIG_W, ph / fig_h))

    ax_a, ax_b = cell(0, 0), cell(1, 0)
    ax_c, ax_d = cell(0, 1), cell(1, 1)

    panel_loss(ax_a, history)
    # y_na is recorded below the fiber-area centroid (hence the negative
    # values); the datum is named on the axis, where those values are met
    panel_parity(ax_b, d["y_na_true"], d["y_na_pred"], "y_na",
                 r"$y_{na}$ below centroid (in)")
    panel_errors(ax_c, rel)
    panel_parity(ax_d, d["curv_true"] * CURV_SCALE, d["curv_pred"] * CURV_SCALE,
                 "curvature", r"curvature $\varphi$ ($10^{-3}$ 1/in)")

    # headings are printed in sentence case by FS.panel; the caption's bold
    # titles repeat them word for word
    FS.panel(ax_a, "(a)", "Loss history")
    FS.panel(ax_b, "(b)", "Neutral-axis-depth parity")
    FS.panel(ax_c, "(c)", "Relative-error distribution")
    FS.panel(ax_d, "(d)", "Curvature parity")

    FS.place_legend(ax_a, ncol=1)

    probs = FS.audit(fig)
    print(f"[audit] {len(probs)} problem(s)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    FS.save(fig, OUT)
    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
