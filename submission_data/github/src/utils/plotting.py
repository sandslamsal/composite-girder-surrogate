"""Matplotlib helpers for paper figures.

Figures use a clear, colour-coded palette (matplotlib tab10) so series
are immediately distinguishable on screen. Output is PNG-only — never
write a PDF alongside.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


# Single global rcParams pass so every figure matches.
def apply_paper_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 600,                # journal-print resolution
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.6,
        "lines.linewidth": 1.5,
    })


# Single-column vs double-column figure sizes for Elsevier two-column
# layout. Numbers from Elsevier's author guidelines.
COL_SINGLE_IN = 3.54           # 90 mm
COL_DOUBLE_IN = 7.48           # 190 mm


# Colour-coded series cycle (matplotlib tab10 + distinct markers/styles).
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf"]


def color_cycle() -> list[dict]:
    """A 4-item cycle (linestyle, marker, color). The markers and line
    styles still differ across series so that even on a monochrome
    display the curves remain readable, but the primary distinguisher
    is colour."""
    return [
        dict(linestyle="-",  marker="o", color=COLORS[0], markersize=4),
        dict(linestyle="-",  marker="s", color=COLORS[1], markersize=4),
        dict(linestyle="-",  marker="^", color=COLORS[2], markersize=4),
        dict(linestyle="-",  marker="D", color=COLORS[3], markersize=4),
    ]


# Kept as an alias so legacy call sites still resolve.
grayscale_cycle = color_cycle


def savefig(fig: plt.Figure, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
