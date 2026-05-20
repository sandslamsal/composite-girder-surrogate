#!/usr/bin/env python
"""Generate all manuscript figures from a trained surrogate checkpoint.

Output: ``paper/figures/fig_*.png`` (and matching ``.pdf`` for LaTeX).

Usage:
    python scripts/make_figures.py \\
        --checkpoint checkpoints/full_300/best.pt \\
        --data data/raw/full_50k.parquet \\
        --history checkpoints/full_300/history.json \\
        --aashto reports/aashto_full/aashto_comparison.parquet \\
        --out paper/figures/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt

from src.models.inference import SurrogatePredictor
from src.utils.plotting import (
    apply_paper_style, savefig, color_cycle, COLORS,
    COL_SINGLE_IN, COL_DOUBLE_IN,
)
from src.utils.normalize import TARGET_COLUMNS

TARGET_LABELS = {
    "neutral_axis_in": "Neutral axis depth $y_{na}$ (in)",
    "curvature_1_per_in": r"Curvature $\varphi$ (1/in)",
    "moment_kip_in": "Moment $M$ (kip-in)",
    "slip_in": r"Interface slip $\delta$ (in)",
}


# ---------------------------------------------------------------------------
# Figure 1 — training curves
# ---------------------------------------------------------------------------
def fig_training_curves(history_path: Path, out: Path) -> None:
    """Single-panel log-loss training curves.

    Clean redesign: solid train + smoothed validation (light raw +
    bold rolling mean), star marker on the best-validation epoch, no
    intrusive vertical guide line, legend in lower-left (empty
    region after monotone decay)."""
    history = json.loads(history_path.read_text())
    epochs = np.array([h["epoch"] for h in history])
    train_total = np.array([h["train"]["total"] for h in history])
    val_total = np.array([h["val"]["total"] for h in history])

    best_idx = int(np.argmin(val_total))
    best_epoch = int(epochs[best_idx])
    best_val = float(val_total[best_idx])

    # Smooth val curve (5-epoch centred rolling mean) to suppress
    # epoch-to-epoch jitter; raw stays as a thin guideline.
    w = 5
    val_smooth = np.convolve(val_total, np.ones(w) / w, mode="same")
    # avoid edge artefacts of valid convolution at the boundaries
    val_smooth[: w // 2] = val_total[: w // 2]
    val_smooth[-(w // 2):] = val_total[-(w // 2):]

    train_color = "#1f77b4"   # blue
    val_color = "#d62728"     # red
    best_color = "#2ca02c"    # green for best-epoch star

    fig, ax = plt.subplots(figsize=(COL_SINGLE_IN, COL_SINGLE_IN * 0.78))
    ax.semilogy(epochs, train_total, color=train_color, linewidth=1.7,
                label="train")
    # raw val (faint)
    ax.semilogy(epochs, val_total, color=val_color, linewidth=0.7,
                alpha=0.45)
    # smoothed val (bold)
    ax.semilogy(epochs, val_smooth, color=val_color, linewidth=1.7,
                label="validation (5-epoch smoothed)")
    # best-epoch star
    ax.scatter([best_epoch], [best_val], marker="*",
               s=140, color=best_color, edgecolor="black", linewidth=0.8,
               zorder=6, label=f"best @ epoch {best_epoch}")

    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel(r"MSE loss (normalised, $\times\,10^{-3}$)", fontsize=9)
    ax.tick_params(axis="both", which="major", labelsize=8)

    # Minor grid for log decade midpoints
    ax.grid(True, which="major", linestyle="-", linewidth=0.5,
            color="#cccccc", alpha=0.8)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4,
            color="#dddddd", alpha=0.6)

    # Display y values as plain decimals scaled by 1e3, with the scale
    # factor declared once in the y-label. Explicit ticks at 0.5, 1,
    # 2, 3, 5 (units of 1e-3) give a readable, uncluttered axis.
    from matplotlib.ticker import FuncFormatter, FixedLocator, LogLocator
    tick_vals = np.array([3e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3])
    ax.yaxis.set_major_locator(FixedLocator(tick_vals))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1,
                                          numticks=120))
    def _scaled(v, _pos):
        s = v * 1000.0
        if s >= 10:
            return f"{s:.0f}"
        if s >= 1:
            return f"{s:.1f}".rstrip("0").rstrip(".")
        return f"{s:.2f}".rstrip("0").rstrip(".")
    ax.yaxis.set_major_formatter(FuncFormatter(_scaled))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, p: ""))

    # Tight x range, small y-axis padding
    ax.set_xlim(epochs.min() - 1, epochs.max() + 1)
    ymin = min(train_total.min(), val_total.min()) * 0.85
    ymax = max(train_total[0], val_total[0]) * 1.15
    ax.set_ylim(ymin, ymax)

    # Legend in the upper-right "empty" region above the asymptote.
    # The early-epoch curves spike up top-left, so right side is open.
    ax.legend(loc="upper right", bbox_to_anchor=(0.985, 0.985),
              frameon=True, framealpha=0.95, edgecolor="#888",
              fontsize=8, handlelength=1.8, handletextpad=0.5,
              borderpad=0.5)

    fig.tight_layout()
    savefig(fig, out / "fig_training_curves.png")


# ---------------------------------------------------------------------------
# Figure 2 — Tier-1 parity plots (4 panels)
# ---------------------------------------------------------------------------
def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot


def fig_parity_plots(
    predictor: SurrogatePredictor, df_test: pd.DataFrame, out: Path,
) -> None:
    pred = predictor.predict(df_test)

    # Only the two non-trivial predictions are plotted. The moment and
    # slip channels are arithmetic pass-throughs (moment follows from the
    # M/M_p input feature; slip is a closed-form analytical function), so
    # they carry no independent accuracy information.
    panel_cols = ["neutral_axis_in", "curvature_1_per_in"]
    fig, axes = plt.subplots(1, 2, figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.50))
    for j, col in enumerate(panel_cols):
        ax = axes[j]
        true_vals = df_test[col].to_numpy()
        pred_vals = pred[col].to_numpy()
        r2 = _r2(true_vals, pred_vals)
        # 2D hexbin avoids visual saturation on 100k+ points
        hb = ax.hexbin(
            true_vals, pred_vals,
            gridsize=60, cmap="viridis", mincnt=1, bins="log",
        )
        lo = min(true_vals.min(), pred_vals.min())
        hi = max(true_vals.max(), pred_vals.max())
        ax.plot([lo, hi], [lo, hi], color="#d62728", linestyle="--",
                linewidth=1.2)
        ax.set_xlabel(f"OpenSeesPy {TARGET_LABELS[col]}")
        ax.set_ylabel(f"Predicted {TARGET_LABELS[col]}")
        ax.text(
            0.04, 0.96, f"$R^2 = {r2:.3f}$",
            transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="black", alpha=0.85),
        )
    fig.tight_layout()
    savefig(fig, out / "fig_parity.png")


# ---------------------------------------------------------------------------
# Figure 3 — AASHTO transformed-section error (the headline Tier-2 figure)
# ---------------------------------------------------------------------------
def fig_aashto_error(aashto_parquet: Path, out: Path) -> None:
    df = pd.read_parquet(aashto_parquet)
    bins = ["25-50%", "50-70%", "70-90%", "90-100%"]
    df = df[df["eta_bin"].isin(bins)].copy()
    # NEGATIVE phi_error_pct means AASHTO under-predicts curvature (overstiff).
    # Flip to a "stiffness overestimation" sign so the figure reads as
    # "how much AASHTO is too stiff" — that's the manuscript framing.
    df["stiffness_overest_pct"] = -df["phi_error_pct"]

    # Single-panel: boxplot with mean diamond + sample count above each box.
    # Trend is monotone and clearly visible; the CI is negligible compared
    # to the spread so a separate CI panel was misleading.
    data = [df.loc[df["eta_bin"] == b, "stiffness_overest_pct"].to_numpy()
            for b in bins]
    means = np.array([d.mean() for d in data])

    fig, ax = plt.subplots(figsize=(COL_SINGLE_IN, COL_SINGLE_IN * 0.85))
    bp = ax.boxplot(
        data, tick_labels=bins, widths=0.55, showfliers=False,
        medianprops=dict(color="black", linewidth=1.6),
        boxprops=dict(color="black", linewidth=0.9),
        whiskerprops=dict(color="black", linewidth=0.9),
        capprops=dict(color="black", linewidth=0.9),
        patch_artist=True,
    )
    bin_colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
    for patch, c in zip(bp["boxes"], bin_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)

    # Overlay mean as a filled white diamond
    xs = np.arange(1, len(bins) + 1)
    ax.plot(xs, means, "D", markerfacecolor="white",
            markeredgecolor="black", markeredgewidth=1.0, markersize=6,
            zorder=4, label="mean")

    # Sample counts ABOVE the boxes
    for x, d in zip(xs, data):
        n = len(d)
        n_label = f"n = {n/1000:.0f}k" if n >= 10_000 else f"n = {n}"
        ax.text(x, 110, n_label, ha="center", va="center",
                fontsize=8, color="#333")

    ax.axhline(0.0, color="black", linewidth=0.6, linestyle=":")
    ax.set_xlabel(r"Degree of composite action $\eta_c$")
    ax.set_ylabel(r"AASHTO Stiffness $\Delta$ (%)")
    ax.set_ylim(-22, 118)

    # Compact legend with only the mean-diamond glyph. The colour
    # mapping is implicit from the x-axis bin labels, so per-bin
    # swatches would be redundant.
    h_mean_proxy = plt.Line2D([], [], marker="D", color="black",
                              markerfacecolor="white", markersize=6,
                              lw=0, label="mean")
    ax.legend(
        handles=[h_mean_proxy],
        loc="upper right", bbox_to_anchor=(0.99, 0.88),
        frameon=True, framealpha=0.92,
        fontsize=8, ncol=1, handlelength=1.0, handletextpad=0.5,
        borderpad=0.4,
    )
    fig.tight_layout()
    savefig(fig, out / "fig_aashto_error.png")


# ---------------------------------------------------------------------------
# R_EI continuous design chart
# ---------------------------------------------------------------------------
def fig_rei_curve(aashto_parquet: Path, out: Path) -> None:
    """Continuous stiffness-reduction factor R_EI versus the degree of
    composite action, for the service-load and extended-elastic regimes.
    R_EI = 1 - Delta and is capped at 1.0 (AASHTO stiffness is not
    amplified). Provides a continuous reading of the binned Table values."""
    df = pd.read_parquet(aashto_parquet)
    # phi_error_pct < 0 means AASHTO is stiffer; Delta = -phi_error_pct
    # and R_EI = 1 - Delta/100 = 1 + phi_error_pct/100.
    df = df.assign(rei=1.0 + df["phi_error_pct"] / 100.0)

    edges = np.linspace(0.25, 1.0, 17)
    centres = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(COL_SINGLE_IN, COL_SINGLE_IN * 0.82))
    regimes = [
        (r"Service load ($M/M_p \leq 0.4$)",
         df["moment_ratio"].to_numpy() <= 0.4, "#1f77b4"),
        (r"Extended elastic ($M/M_p \leq 0.6$)",
         df["moment_ratio"].to_numpy() <= 0.6, "#d62728"),
    ]
    for label, mask, color in regimes:
        eta = df["composite_action"].to_numpy()[mask]
        rei = df["rei"].to_numpy()[mask]
        idx = np.clip(np.digitize(eta, edges) - 1, 0, len(centres) - 1)
        mean = np.array([rei[idx == k].mean() if (idx == k).any() else np.nan
                         for k in range(len(centres))])
        mean = np.minimum(mean, 1.0)   # R_EI capped at 1.0
        ax.plot(centres, mean, color=color, linewidth=1.9, label=label)

    ax.axhline(1.0, color="black", linewidth=0.6, linestyle=":")
    ax.set_xlabel(r"Degree of composite action $\eta_c$")
    ax.set_ylabel(r"Stiffness reduction factor $R_{EI}$")
    ax.set_xlim(0.25, 1.0)
    ax.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.92)
    fig.tight_layout()
    savefig(fig, out / "fig_rei_curve.png")


# ---------------------------------------------------------------------------
# Deviation as a continuous function of moment ratio
# ---------------------------------------------------------------------------
def fig_deviation_vs_moment(aashto_parquet: Path, out: Path) -> None:
    """AASHTO stiffness over-prediction Delta as a continuous function of
    the moment ratio M/M_p, stratified by degree of composite action. The
    service-load and extended-elastic regime limits are marked."""
    df = pd.read_parquet(aashto_parquet)
    df = df.assign(delta=-df["phi_error_pct"])
    bins = ["25-50%", "50-70%", "70-90%", "90-100%"]
    bin_colors = {"25-50%": "#d62728", "50-70%": "#ff7f0e",
                  "70-90%": "#2ca02c", "90-100%": "#1f77b4"}

    edges = np.linspace(float(df["moment_ratio"].min()), 0.6, 26)
    centres = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(COL_SINGLE_IN, COL_SINGLE_IN * 0.82))
    for b in bins:
        sub = df[df["eta_bin"] == b]
        mr = sub["moment_ratio"].to_numpy()
        d = sub["delta"].to_numpy()
        idx = np.clip(np.digitize(mr, edges) - 1, 0, len(centres) - 1)
        mean = np.array([d[idx == k].mean() if (idx == k).any() else np.nan
                         for k in range(len(centres))])
        ax.plot(centres, mean, color=bin_colors[b], linewidth=1.7,
                label=rf"$\eta_c$ {b}")

    for cut in (0.4, 0.6):
        ax.axvline(cut, color="black", linewidth=0.7, linestyle="--")
    ax.set_xlabel(r"Moment ratio $M/M_p$")
    ax.set_ylabel(r"AASHTO stiffness $\Delta$ (%)")
    ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.92)
    fig.tight_layout()
    savefig(fig, out / "fig_deviation_vs_moment.png")


# ---------------------------------------------------------------------------
# Nie & Cai cross-validation figure
# ---------------------------------------------------------------------------
def fig_niecai_vs_aashto(niecai_parquet: Path, out: Path,
                         surrogate_json: Path | None = None) -> None:
    """Grouped bar chart: AASHTO vs Nie & Cai vs proposed surrogate,
    each compared against the OpenSeesPy reference per eta_c bin.
    Demonstrates the progression AASHTO -> Nie&Cai -> surrogate
    (proposed model) which essentially matches OpenSeesPy."""
    import json

    df = pd.read_parquet(niecai_parquet)
    bins = ["25-50%", "50-70%", "70-90%", "90-100%"]
    df = df[df["eta_bin"].isin(bins)].copy()
    # Sign convention: positive = predictor over-predicts stiffness
    # (under-predicts curvature). Matches the tab:aashto framing.
    aashto_over = []
    niecai_over = []
    aashto_med = []
    niecai_med = []
    counts = []
    for b in bins:
        sub = df[df["eta_bin"] == b]
        aashto_over.append(-sub["aashto_error_pct_vs_opensees"].mean())
        niecai_over.append(-sub["niecai_error_pct_vs_opensees"].mean())
        aashto_med.append(-sub["aashto_error_pct_vs_opensees"].median())
        niecai_med.append(-sub["niecai_error_pct_vs_opensees"].median())
        counts.append(len(sub))

    # Optional third bar: proposed surrogate vs OpenSeesPy
    surrogate_over = surrogate_med = None
    if surrogate_json is not None and surrogate_json.exists():
        sj = json.loads(surrogate_json.read_text())
        surrogate_over = [sj[b]["mean"] for b in bins]
        surrogate_med = [sj[b]["median"] for b in bins]

    fig, ax = plt.subplots(figsize=(COL_SINGLE_IN, COL_SINGLE_IN * 0.80))
    x = np.arange(len(bins))
    c_aashto = "#d62728"   # red
    c_niecai = "#1f77b4"   # blue
    c_prop = "#2ca02c"     # green

    if surrogate_over is None:
        w = 0.38
        ax.bar(x - w / 2, aashto_over, w, color=c_aashto, alpha=0.80,
               edgecolor="black", linewidth=0.6, label="AASHTO (no slip)")
        ax.bar(x + w / 2, niecai_over, w, color=c_niecai, alpha=0.80,
               edgecolor="black", linewidth=0.6,
               label="Nie & Cai (analytical, calibrated)")
        ax.plot(x - w / 2, aashto_med, "D", markerfacecolor="white",
                markeredgecolor="black", markersize=4.5, lw=0, zorder=5)
        ax.plot(x + w / 2, niecai_med, "D", markerfacecolor="white",
                markeredgecolor="black", markersize=4.5, lw=0, zorder=5,
                label="median")
    else:
        w = 0.26
        ax.bar(x - w, aashto_over, w, color=c_aashto, alpha=0.80,
               edgecolor="black", linewidth=0.6, label="AASHTO (no slip)")
        ax.bar(x,     niecai_over, w, color=c_niecai, alpha=0.80,
               edgecolor="black", linewidth=0.6,
               label="Nie & Cai (analytical)")
        ax.bar(x + w, surrogate_over, w, color=c_prop, alpha=0.80,
               edgecolor="black", linewidth=0.6,
               label="Proposed model")
        ax.plot(x - w, aashto_med, "D", markerfacecolor="white",
                markeredgecolor="black", markersize=4.5, lw=0, zorder=5)
        ax.plot(x,     niecai_med, "D", markerfacecolor="white",
                markeredgecolor="black", markersize=4.5, lw=0, zorder=5)
        ax.plot(x + w, surrogate_med, "D", markerfacecolor="white",
                markeredgecolor="black", markersize=4.5, lw=0, zorder=5,
                label="median")

    # Sample counts above the highest bar of each group
    bar_lists = [aashto_over, niecai_over]
    if surrogate_over is not None:
        bar_lists.append(surrogate_over)
    tops = [max(vals) for vals in zip(*bar_lists)]
    for xi, n, hi in zip(x, counts, tops):
        n_lab = f"n={n/1000:.0f}k" if n >= 10_000 else f"n={n}"
        ax.text(xi, hi + 2.5, n_lab, ha="center", va="bottom",
                fontsize=7.5, color="#333")

    ax.axhline(0.0, color="black", linewidth=0.6, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_xlabel(r"Degree of composite action $\eta_c$")
    ax.set_ylabel(r"Stiffness over-prediction $\Delta$ (%)")
    ax.set_ylim(-8, max(aashto_over) * 1.28)
    ax.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.92,
              handlelength=1.4, handletextpad=0.5, borderpad=0.4)
    fig.tight_layout()
    savefig(fig, out / "fig_niecai_vs_aashto.png")


# ---------------------------------------------------------------------------
# Figure 4 — representative M-phi curves
# ---------------------------------------------------------------------------
def _pick_representative_sections(df: pd.DataFrame, n: int = 4) -> list[int]:
    """Choose ``n`` sections spanning the design space (section_type,
    composite_action, span). Returns sample_ids."""
    picks = []
    per_section_type = max(1, n // 2)
    for stype in ("W", "plate"):
        candidates = df[df["section_type"] == stype].drop_duplicates("sample_id")
        # quantile picks on composite_action
        qs = np.linspace(0.2, 0.8, per_section_type)
        for q in qs:
            target_eta = float(candidates["composite_action"].quantile(q))
            idx = (candidates["composite_action"] - target_eta).abs().idxmin()
            picks.append(int(candidates.loc[idx, "sample_id"]))
    return picks[:n]


def fig_moment_curvature(
    predictor, df_full: pd.DataFrame, out: Path,
) -> None:
    """Four-panel M-phi comparison with a *shared* legend below the
    subplots so no panel's legend can cover its data."""
    ids = _pick_representative_sections(df_full, n=4)
    fig, axes = plt.subplots(
        2, 2, figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.78),
        sharex=False,
    )
    line_handles = None
    for ax, sid in zip(axes.flat, ids):
        sub = df_full[df_full["sample_id"] == sid].sort_values("step_index")
        if sub.empty:
            continue
        pred = predictor.predict(sub)
        # Moment is the applied load level (an input); the curves compare
        # OpenSeesPy curvature and predicted curvature at those levels.
        M_applied = sub["moment_kip_in"].to_numpy() / 12.0
        phi_true = sub["curvature_1_per_in"].to_numpy() * 1e3
        phi_pred = pred["curvature_1_per_in"].to_numpy() * 1e3
        order = np.argsort(phi_pred)
        h0, = ax.plot(phi_true, M_applied, color=COLORS[0], linewidth=1.6,
                      marker="o", markersize=3, markevery=8,
                      label="OpenSeesPy")
        h1, = ax.plot(phi_pred[order], M_applied[order], color=COLORS[1],
                      linestyle="--", linewidth=1.6,
                      marker="s", markersize=3, markevery=8,
                      label="Proposed model")
        line_handles = (h0, h1)
        eta = float(sub["composite_action"].iloc[0])
        stype = sub["section_type"].iloc[0]
        d = float(sub["total_depth_in"].iloc[0])
        ax.set_title(rf"{stype}, $d={d:.0f}$ in, $\eta_c={eta:.2f}$",
                     fontsize=9)
        ax.set_xlabel(r"Curvature $\varphi \times 10^3$ (1/in)")
        ax.set_ylabel("Moment (kip-ft)")
        ax.set_xlim(left=0)
        ax.set_ylim(0, M_applied.max() * 1.08)
    if line_handles is not None:
        fig.legend(line_handles, ["OpenSeesPy", "Proposed model"],
                   loc="lower center", bbox_to_anchor=(0.5, -0.02),
                   ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    savefig(fig, out / "fig_moment_curvature.png")


# ---------------------------------------------------------------------------
# Figure 5 — neutral-axis migration with curvature (AASHTO assumes fixed NA)
# ---------------------------------------------------------------------------
def fig_neutral_axis_migration(
    predictor: SurrogatePredictor, df_full: pd.DataFrame,
    aashto_parquet: Path, out: Path,
) -> None:
    # Pick one mid-size section with moderate composite action.
    ids = _pick_representative_sections(df_full, n=4)
    sid = ids[1]
    sub = df_full[df_full["sample_id"] == sid].sort_values("step_index")
    pred = predictor.predict(sub)

    aashto = pd.read_parquet(aashto_parquet)
    row = aashto[aashto["sample_id"] == sid].head(1)
    y_na_aashto = float(row["y_na_aashto_in"].iloc[0]) if not row.empty else np.nan

    # Sort by predicted curvature so the dashed surrogate line is monotone
    phi_true = sub["curvature_1_per_in"].to_numpy() * 1e3
    y_true = sub["neutral_axis_in"].to_numpy()
    phi_pred = pred["curvature_1_per_in"].to_numpy() * 1e3
    y_pred = pred["neutral_axis_in"].to_numpy()
    order = np.argsort(phi_pred)

    fig, ax = plt.subplots(figsize=(COL_SINGLE_IN, COL_SINGLE_IN * 0.85))
    ax.plot(phi_true, y_true, color=COLORS[0], linewidth=1.8,
            marker="o", markersize=3, markevery=8, label="OpenSeesPy")
    ax.plot(phi_pred[order], y_pred[order], color=COLORS[1],
            linestyle="--", linewidth=1.8,
            marker="s", markersize=3, markevery=8, label="Proposed model")
    if not np.isnan(y_na_aashto):
        ax.axhline(y_na_aashto, color=COLORS[2], linestyle=":",
                   linewidth=1.6, label="AASHTO (fixed)")
    ax.set_xlabel(r"Curvature $\varphi \times 10^3$ (1/in)")
    ax.set_ylabel(r"Neutral axis $y_{na}$ (in)")
    ax.set_xlim(left=0)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    savefig(fig, out / "fig_neutral_axis_migration.png")


# ---------------------------------------------------------------------------
# Figure 6 — MC-Dropout uncertainty bands
# ---------------------------------------------------------------------------
def fig_uncertainty_band(
    predictor, df_full: pd.DataFrame, out: Path,
    n_dropout: int = 50,
) -> None:
    """Uncertainty-band figure using MC-Dropout. The surrogate predicts
    curvature, so the +/-1.96-sigma band is on the curvature axis."""
    ids = _pick_representative_sections(df_full, n=4)
    sid = ids[0]
    sub = df_full[df_full["sample_id"] == sid].sort_values("step_index")
    unc = predictor.predict_with_uncertainty(sub, n_samples=n_dropout)
    uncertainty_source = f"MC-Dropout, T = {n_dropout}"

    # Moment is the applied load level (the y-axis); curvature is the
    # predicted quantity carrying the dropout uncertainty.
    M = sub["moment_kip_in"].to_numpy() / 12.0
    phi_true = sub["curvature_1_per_in"].to_numpy() * 1e3
    phi_mean = unc["curvature_1_per_in_mean"].to_numpy() * 1e3
    phi_std = unc["curvature_1_per_in_std"].to_numpy() * 1e3
    # Sort by applied moment so all lines are monotone
    order = np.argsort(M)
    M_o = M[order]
    phi_true_o = phi_true[order]
    phi_mean_o = phi_mean[order]
    phi_std_o = phi_std[order]

    # Single landscape panel; the M-phi response and its +/-1.96-sigma
    # band (horizontal, on curvature) share one axes.
    fig, ax = plt.subplots(figsize=(COL_SINGLE_IN, COL_SINGLE_IN * 0.85))

    h0, = ax.plot(phi_true_o, M_o, color=COLORS[0], linewidth=1.8, marker="o",
                  markersize=3, markevery=6, label="OpenSeesPy")
    h1, = ax.plot(phi_mean_o, M_o, color=COLORS[1], linestyle="--",
                  linewidth=1.8, label="Proposed model (mean)")
    h2 = ax.fill_betweenx(
        M_o, phi_mean_o - 1.96 * phi_std_o, phi_mean_o + 1.96 * phi_std_o,
        color=COLORS[1], alpha=0.25, label=r"$\pm\,1.96\sigma$",
    )
    ax.set_xlabel(r"Curvature $\varphi \times 10^3$ (1/in)")
    ax.set_ylabel("Moment (kip-ft)")
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    # Lower-right of the axes is empty (curve plateaus across the top,
    # rises on the left), so the legend with the uncertainty title fits
    # without occluding any data.
    ax.legend(
        handles=[h0, h1, h2], loc="lower right",
        frameon=True, fontsize=8, framealpha=0.92,
        title=uncertainty_source, title_fontsize=8,
    )

    fig.tight_layout()
    savefig(fig, out / "fig_uncertainty_band.png")


# ---------------------------------------------------------------------------
# Figure 7 — error distributions on the test set
# ---------------------------------------------------------------------------
def fig_error_distribution(
    predictor, df_test: pd.DataFrame, out: Path,
) -> None:
    """Per-target relative-error violin with mean/median markers and the
    P10-P90 inter-decile range annotated. Tails beyond the 1st-99th
    percentile are dropped (rather than clipped to a flat plateau) so
    the violin shape reflects the actual distribution density."""
    pred = predictor.predict(df_test)
    fig, ax = plt.subplots(figsize=(COL_SINGLE_IN, COL_SINGLE_IN * 0.85))
    rel_errors = []
    labels = []
    stats = []
    for col in TARGET_COLUMNS:
        true_vals = df_test[col].to_numpy()
        pred_vals = pred[col].to_numpy()
        denom = np.maximum(np.abs(true_vals), np.abs(true_vals).mean() * 1e-3)
        rel = 100.0 * (pred_vals - true_vals) / denom
        rel = rel[np.isfinite(rel)]
        # Drop tails outside [P1, P99] so the violin shape is meaningful
        p1, p99 = np.percentile(rel, [1, 99])
        rel = rel[(rel >= p1) & (rel <= p99)]
        rel_errors.append(rel)
        labels.append({
            "neutral_axis_in": r"$y_{na}$",
            "curvature_1_per_in": r"$\varphi$",
            "moment_kip_in": r"$M$",
            "slip_in": r"$\delta$",
        }[col])
        p10, p50, p90 = np.percentile(rel, [10, 50, 90])
        stats.append((p10, p50, p90, rel.mean()))

    parts = ax.violinplot(rel_errors, showmeans=False, showmedians=False,
                          showextrema=False, widths=0.78)
    target_colors = [COLORS[0], COLORS[1], COLORS[2], COLORS[3]]
    for body, c in zip(parts["bodies"], target_colors):
        body.set_facecolor(c)
        body.set_edgecolor("black")
        body.set_alpha(0.55)
    # Median (black bar), mean (white circle), and P10/P90 whiskers
    for i, (p10, p50, p90, m) in enumerate(stats, start=1):
        ax.plot([i - 0.18, i + 0.18], [p50, p50], color="black",
                linewidth=1.6, zorder=5)
        ax.plot(i, m, "o", color="white", markeredgecolor="black",
                markeredgewidth=1.0, markersize=5, zorder=5)
        ax.plot([i, i], [p10, p90], color="black",
                linewidth=0.7, alpha=0.6)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Relative error (%)")
    ax.axhline(0.0, color="black", linewidth=0.5, linestyle=":")
    # Tighter y-range based on the union of per-target P10/P90 so the
    # violin bodies dominate the visual area. Symmetrise about zero
    # for readability since the errors are roughly centred.
    half = max(abs(np.percentile(r, 10)) for r in rel_errors)
    half = max(half, max(np.percentile(r, 90) for r in rel_errors))
    half *= 1.5
    ax.set_ylim(-half, half)

    # Compact one-column legend INSIDE the axes at the top-right corner.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="black", lw=1.6, label="median"),
        Line2D([0], [0], marker="o", color="white", markeredgecolor="black",
               markeredgewidth=1.0, markersize=5, lw=0, label="mean"),
        Line2D([0], [0], color="black", lw=0.7, label=r"P$_{10}$--P$_{90}$"),
    ]
    ax.legend(
        handles=handles, loc="upper right",
        frameon=True, fontsize=7, ncol=1,
        handlelength=1.4, handletextpad=0.5,
        borderpad=0.4, labelspacing=0.4, framealpha=0.92,
    )
    fig.tight_layout()
    savefig(fig, out / "fig_error_distribution.png")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _split_by_sample(df: pd.DataFrame, fracs: dict, seed: int):
    ids = df["sample_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(fracs["train"] * n))
    n_val = int(round(fracs["val"] * n))
    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:n_train + n_val])
    test_ids = set(ids[n_train + n_val:])
    return (df[df["sample_id"].isin(train_ids)].reset_index(drop=True),
            df[df["sample_id"].isin(val_ids)].reset_index(drop=True),
            df[df["sample_id"].isin(test_ids)].reset_index(drop=True))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Path to a trained surrogate checkpoint.")
    p.add_argument("--data", required=True)
    p.add_argument("--history", required=True)
    p.add_argument("--aashto", required=True,
                   help="aashto_comparison.parquet from validate_aashto.py")
    p.add_argument("--niecai",
                   help="niecai_comparison.parquet from validate_nie_cai.py "
                        "(optional; enables fig_niecai_vs_aashto.png)")
    p.add_argument("--out", default="paper/figures")
    p.add_argument("--mc-samples", type=int, default=50)
    args = p.parse_args()

    apply_paper_style()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("[load] data, checkpoint, history…")
    df = pd.read_parquet(args.data)
    predictor = SurrogatePredictor.load(args.checkpoint)
    print(f"[load] using surrogate checkpoint {args.checkpoint}")
    cfg = predictor.model  # implicit access to config via checkpoint
    # Recover the same train/val/test split as training (seed from training.yaml)
    import yaml
    train_cfg = yaml.safe_load(open(REPO_ROOT / "configs/training.yaml"))
    _train_df, _val_df, test_df = _split_by_sample(
        df, train_cfg["splits"], int(train_cfg["seed"])
    )
    # For high-resolution figures: subsample test set if huge.
    if len(test_df) > 200_000:
        test_df = test_df.sample(n=200_000, random_state=0).reset_index(drop=True)
    print(f"[test rows] {len(test_df):,}  [unique sections] "
          f"{test_df['sample_id'].nunique():,}")

    history_path = Path(args.history)
    if history_path.exists():
        print("[fig] training curves…")
        fig_training_curves(history_path, out)
    else:
        print(f"[fig] training curves skipped — {history_path} not yet written")

    print("[fig] parity plots…")
    fig_parity_plots(predictor, test_df, out)

    print("[fig] AASHTO error…")
    fig_aashto_error(Path(args.aashto), out)

    if args.niecai:
        nc_path = Path(args.niecai)
        if nc_path.exists():
            print("[fig] Nie & Cai vs AASHTO…")
            fig_niecai_vs_aashto(nc_path, out)
        else:
            print(f"[fig] Nie & Cai skipped — {nc_path} not found")

    print("[fig] M-phi curves…")
    fig_moment_curvature(predictor, df, out)

    print("[fig] neutral-axis migration…")
    fig_neutral_axis_migration(predictor, df, Path(args.aashto), out)

    print("[fig] MC-Dropout uncertainty…")
    fig_uncertainty_band(predictor, df, out, n_dropout=args.mc_samples)

    print("[fig] error distribution…")
    fig_error_distribution(predictor, test_df, out)

    print(f"[done] figures in {out}/")


if __name__ == "__main__":
    main()
