"""Visual check of the Fig. 68 digitisation (FHWA-HRT-20-005, p. 80).

Overlays the calibrated trace centres (fig68_traces_raw.csv) and the fixed
moment levels (fig68_levels.csv) back onto the 600-dpi render of PDF page 98,
cropped exactly as in digitise_fig68.py, with calibration grid lines drawn from
the fitted pixel maps (fig68_calibration.json).

Usage:
    python overlay_check.py <page98_600dpi.png> <dir with csv/json> <out.png>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def main(render_png: str, ddir: str, out_png: str) -> None:
    d = Path(ddir)
    cal = json.loads((d / "fig68_calibration.json").read_text())
    x0, y0, x1, y1 = [v * 600 / 72.0 for v in cal["render"]["placement_pt_x0y0x1y1"]]
    im = Image.open(render_png).convert("RGB").crop((int(round(x0)), int(round(y0)),
                                                    int(round(x1)), int(round(y1))))
    a = np.asarray(im)
    xm, ym = cal["x_map_in_per_render_px"], cal["y_map_kipft_per_render_px"]
    col = lambda x: (np.asarray(x) - xm["intercept"]) / xm["slope"]
    row = lambda m: (np.asarray(m) - ym["intercept"]) / ym["slope"]

    tr = pd.read_csv(d / "fig68_traces_raw.csv")
    tr = tr[tr.status == "visible"]
    lv = pd.read_csv(d / "fig68_levels.csv").dropna(subset=["displacement_in"])
    marks = {"1S2": "s", "1S1": "o", "2S1": "^", "3S1": "D", "4S1": "v"}

    fig, axs = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.3, 1]})
    for ax, (xl, ml) in zip(axs, [((-0.2, 10.2), (-60, 2560)), ((-0.03, 1.05), (-20, 1120))]):
        ax.imshow(a)
        for x in np.arange(0, 10.01, 0.1 if ax is axs[1] else 1.0):
            ax.axvline(col(x), color="0.5", lw=0.3, ls=":")
        for m in np.arange(0, 2501, 100 if ax is axs[1] else 500):
            ax.axhline(row(m), color="0.5", lw=0.3, ls=":")
        for sid, g in tr.groupby("series"):
            ax.plot(g.centre_render_col_px, g.render_row_px, "+", ms=3 if ax is axs[0] else 5,
                    mew=0.6, color="k")
        for sid, g in lv.groupby("series"):
            ax.plot(col(g.displacement_in), row(g.moment_kipft), marks[sid], mfc="white",
                    mec="k", mew=0.9, ms=4 if ax is axs[0] else 7, label=f"{sid} levels")
        ax.set_aspect("equal" if ax is axs[0] else "auto")
        ax.set_xlim(col(xl[0]), col(xl[1]))
        ax.set_ylim(row(ml[0]), row(ml[1]))
        ax.set_xticks(col(np.arange(0, xl[1], 1.0 if ax is axs[0] else 0.2)))
        ax.set_xticklabels([f"{v:g}" for v in np.arange(0, xl[1], 1.0 if ax is axs[0] else 0.2)])
        ax.set_yticks(row(np.arange(0, ml[1], 500 if ax is axs[0] else 200)))
        ax.set_yticklabels([f"{v:g}" for v in np.arange(0, ml[1], 500 if ax is axs[0] else 200)])
        ax.set_xlabel("Calibrated displacement (in.)")
        ax.set_ylabel("Calibrated moment (kip-ft)")
    axs[0].set_title("(a) Full figure: + trace centres, open marks fixed levels", fontweight="bold")
    axs[1].set_title("(b) Elastic region (x stretched), 0.1 in. and 100 kip-ft grid", fontweight="bold")
    axs[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)


if __name__ == "__main__":
    main(*sys.argv[1:4])
