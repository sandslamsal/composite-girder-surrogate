#!/usr/bin/env python
"""Consolidated SURROGATE ACCURACY figure (replaces old Figs 3, 4 and 5).

Four panels, 2x2, drawn at the printed width of the manuscript
(``figstyle.FIG_W`` = 6.5 in, included at ``\\linewidth``):

    (a) training / held-out MSE loss vs epoch, log y, best epoch starred
    (b) parity for the neutral-axis depth, 1:1 line
    (c) held-out relative-error distribution for both outputs
    (d) parity for the curvature, 1:1 line

R^2 and MAPE are printed on the parity panels and the best epoch on
the loss panel; the inter-decile ranges appear in
Table tab:heldout, so on the panels they would only restate the text;
every one of them is still written to stdout by :func:`main`.

The right column therefore holds both parity plots, so the two 1:1
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
OUT = REPO_ROOT / "paper" / "revision_1" / "submission" / "sources" / "figures" / "fig_surrogate_accuracy.png"
CACHE = Path("/private/tmp/claude-501/-Users-sandeshlamsal-Desktop-CompositeGirder/"
             "1b666f82-1e84-4f9e-a35c-afb843f2b292/scratchpad/heldout_pred.npz")

# split reproduction (identical to scripts/make_figures.py::_split_by_sample
# and scripts/revision_common.py, seed from configs/training.yaml)
SEED = 20260513
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}

CURV_SCALE = 1e4          # plot curvature in 1e-4 1/in
ERR_CLIP = 50.0           # panel (c) x range, per the manuscript caption

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


def mape(y, yh, eps=1e-6):
    y, yh = np.asarray(y, float), np.asarray(yh, float)
    denom = np.where(np.abs(y) > eps, np.abs(y), eps)
    return float(100.0 * np.mean(np.abs((yh - y) / denom)))


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

    ax.semilogy(epochs, train, **FS.style("train", lw=1.5))
    ax.semilogy(epochs, val, **FS.style("heldout", lw=1.6))
    ax.plot([epochs[best]], [val[best]], marker="*", ms=11,
            color=FS.color("heldout"), mec="black", mew=0.7, ls="none",
            zorder=6, clip_on=False)

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

    ax.annotate(f"best held-out\nepoch {int(epochs[best])}",
                xy=(epochs[best], val[best]), xytext=(-14, 26),
                textcoords="offset points", fontsize=FS.FS_ANNOT,
                color="0.20", ha="right", va="bottom",
                arrowprops=dict(arrowstyle="-", lw=0.7, color="0.45",
                                shrinkA=1.0, shrinkB=3.0))
    # the star is decoded by the caption ('star, best held-out epoch') and
    # the epoch number is given in the text, so no annotation is drawn here
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
    # numbers tabulated in the manuscript. An independently written MAPE
    # with a different denominator gave 8.4 % against the reported 11.9 %,
    # which would have put a figure and a table in open disagreement.
    # Set at MATH_MIN_FS: a superscript renders at 0.7 of its parent, so
    # anything smaller puts "R^2" under the 6.5 pt floor and audit fails.
    FS.inside_label(ax, 0.045, 0.945,
                    f"$R^2 = {r2(true, pred):.3f}$\n"
                    f"MAPE {mape(true, pred):.1f}%",
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
    ax.set_xlabel("relative error, surrogate − OpenSeesPy (%)")
    ax.set_ylabel("share of held-out rows (% per 1 % bin)")

    # A key that names the two curves and nothing else.  The inter-decile
    # ranges it used to carry are given in the results paragraph, and a
    # panel must not restate the text.  Drawn as a legend rather than as
    # coloured text so the dash pattern carries the identity too, which
    # is what keeps the panel readable in greyscale.
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False,
              fontsize=FS.FS_ANNOT, handlelength=2.2, handletextpad=0.55,
              labelspacing=0.3, borderaxespad=0.25)
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
        denom = np.maximum(np.abs(t), np.abs(t).mean() * 1e-3)
        r = 100.0 * (p - t) / denom
        rel[key] = r[np.isfinite(r)]
        p10, p50, p90 = np.percentile(rel[key], [10, 50, 90])
        inside = 100.0 * np.mean(np.abs(rel[key]) <= ERR_CLIP)
        print(f"[err] {key}: median={p50:.2f}%  P10={p10:.2f}%  "
              f"P90={p90:.2f}%  |e|<=50%: {inside:.1f}% of rows")

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
    panel_parity(ax_b, d["y_na_true"], d["y_na_pred"], "y_na",
                 r"$y_{na}$ (in)")
    panel_errors(ax_c, rel)
    panel_parity(ax_d, d["curv_true"] * CURV_SCALE, d["curv_pred"] * CURV_SCALE,
                 "curvature", r"$\varphi$ ($10^{-4}$ 1/in)")

    FS.panel(ax_a, "(a)", "loss history")
    FS.panel(ax_b, "(b)", "neutral-axis depth")
    FS.panel(ax_c, "(c)", "relative-error distribution")
    FS.panel(ax_d, "(d)", "curvature")

    FS.place_legend(ax_a, ncol=1)

    probs = FS.audit(fig)
    print(f"[audit] {len(probs)} problem(s)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    FS.save(fig, OUT)
    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
