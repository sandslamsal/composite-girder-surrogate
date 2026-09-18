"""Digitise Fig. 68 of Provines, Ocel & Zmetra (2019), FHWA-HRT-20-005, p. 80.

Moment-displacement plot of the five large-scale static beams (1S2, 1S1,
2S1, 3S1, 4S1). The figure is embedded in the PDF as a single 576 x 384 px
RGB raster (PDF image xref 289, placed at x = 90.0-521.9 pt,
y = 166.8-454.7 pt on PDF page 98). The page is rendered at 600 dpi
(``pdftoppm -r 600 -f 98 -l 98``) and cropped to that placement rectangle;
one source pixel becomes a 6.248 x 6.247 block of the render. The script
checks that sampling the render at the source-pixel centres reproduces the
embedded raster exactly, calibrates both axes on the render, and extracts
the curves on that source-pixel grid (finer sampling of the render adds no
information and only duplicates cells).

Usage:
    python digitise_fig68.py <page98_600dpi.png> <native_fig68.png> <out_dir>

Outputs (in out_dir):
    fig68_calibration.json  axis lines, tick marks and dashed M_n lines in
                            render pixels, the fitted linear maps, their
                            residuals, and the render/native identity check
    fig68_traces_raw.csv    per source-pixel row and per curve: visible run
                            (render px), occlusion flags, centre estimate,
                            calibrated moment and displacement
    fig68_levels.csv        loading-branch displacement at fixed moments

Nothing is fitted to a model. Processing: colour classification, run
finding, a stroke-width rule where a curve is partly hidden by one drawn on
top of it, and linear pixel-to-value maps.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

DPI = 600
PT = DPI / 72.0
PLACE_PT = (90.0, 166.80001831054688, 521.9000244140625, 454.7300109863281)

# pure series colours from the colour histogram of the embedded raster
SERIES = {
    "1S2": (0, 0, 0),
    "1S1": (0, 112, 192),
    "2S1": (0, 176, 80),
    "3S1": (255, 0, 0),
    "4S1": (112, 48, 160),
}
# drawing order read from the figure (later entries are drawn on top)
DRAW_ORDER = ["1S2", "1S1", "2S1", "3S1", "4S1"]
# moment ceiling for each trace: the 1S2 call-out pointer meets its curve
# near 1,270 kip-ft, so 1S2 is read only below 1,240 kip-ft
M_CEILING = {"1S2": 1240.0, "1S1": 1600.0, "2S1": 1600.0, "3S1": 1600.0, "4S1": 1600.0}

X_TICK_VALUES = list(range(0, 11))                 # in.
Y_TICK_VALUES = [0, 500, 1000, 1500, 2000, 2500]   # kip-ft
DASHED = {"Mn_2_transverse_studs": 2157.0, "Mn_1_transverse_stud": 1831.0}
LEVELS = [100, 150, 200, 250, 300, 350, 400, 450, 460, 500, 550, 600, 650, 700,
          750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300,
          1350, 1400, 1450, 1500]


def crop_render(path: str) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    x0, y0, x1, y1 = [v * PT for v in PLACE_PT]
    return np.asarray(im.crop((int(round(x0)), int(round(y0)),
                               int(round(x1)), int(round(y1))))).astype(float)


def runs(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask_1d)
    if idx.size == 0:
        return []
    out, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i != prev + 1:
            out.append((int(start), int(prev)))
            start = i
        prev = i
    out.append((int(start), int(prev)))
    return out


def calibrate(a: np.ndarray, scale: float) -> dict:
    H, W, _ = a.shape
    blk = a.max(axis=2) < 60
    colsum, rowsum = blk.sum(axis=0), blk.sum(axis=1)
    left = np.flatnonzero(colsum[: W // 3] > 0.6 * colsum[: W // 3].max())
    yaxis_cols = left[left <= left.min() + int(np.ceil(scale)) + 1]
    low = np.flatnonzero(rowsum > 0.6 * rowsum.max())
    xaxis_rows = low[low >= low.max() - int(np.ceil(scale)) - 1]
    top_rows = low[low <= low.min() + int(np.ceil(scale)) + 1]
    right = np.flatnonzero(colsum > 0.6 * colsum.max())
    right_cols = right[right >= right.max() - int(np.ceil(scale)) - 1]
    tick_len = int(round(3 * scale))
    band = blk[xaxis_rows.max() + 1: xaxis_rows.max() + 1 + tick_len, :]
    xt = [(s + e) / 2.0 for s, e in runs(band.all(axis=0)) if (e - s + 1) <= 2 * scale + 2]
    band = blk[:, yaxis_cols.min() - tick_len: yaxis_cols.min()]
    yt = [(s + e) / 2.0 for s, e in runs(band.all(axis=1)) if (e - s + 1) <= 2 * scale + 2]
    frame = set(xaxis_rows.tolist()) | set(top_rows.tolist())
    dashed_rows = [r for r in np.flatnonzero(rowsum > 0.15 * rowsum.max()) if r not in frame]
    dashed = [(s + e) / 2.0 for s, e in runs(np.isin(np.arange(H), dashed_rows))]
    return {"y_axis_line_col": float(yaxis_cols.mean()), "x_axis_line_row": float(xaxis_rows.mean()),
            "top_frame_row": float(top_rows.mean()), "right_frame_col": float(right_cols.mean()),
            "x_tick_cols": xt, "y_tick_rows": yt, "dashed_line_rows": dashed}


def fit_map(pix, val) -> dict:
    p = np.polyfit(pix, val, 1)
    res = np.asarray(val, float) - np.polyval(p, pix)
    return {"slope": float(p[0]), "intercept": float(p[1]),
            "max_abs_residual": float(np.abs(res).max()),
            "pixels": [float(v) for v in pix], "values": [float(v) for v in val]}


def classify(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label every cell with a series index (0..4), -1 white, -2 unclassified.
    A cell is a series colour blended with white by coverage alpha."""
    white = np.array([255.0, 255.0, 255.0])
    names = list(SERIES)
    v = white - cells
    resid = np.full(cells.shape[:2] + (len(names),), np.inf)
    alpha = np.zeros(cells.shape[:2] + (len(names),))
    for k, n in enumerate(names):
        d = white - np.asarray(SERIES[n], float)
        al = (v @ d) / (d @ d)
        rs = np.linalg.norm(v - al[..., None] * d, axis=2)
        ok = al > 0.30
        resid[..., k] = np.where(ok, rs, np.inf)
        alpha[..., k] = al
    lab = np.argmin(resid, axis=2)
    best = np.take_along_axis(resid, lab[..., None], axis=2)[..., 0]
    lab = np.where(best < 45.0, lab, -2)
    lab = np.where(cells.min(axis=2) > 235, -1, lab)
    return lab, alpha


def main(render_png: str, native_png: str, out_dir: str) -> None:
    out = Path(out_dir)
    a = crop_render(render_png)
    nat = np.asarray(Image.open(native_png).convert("RGB")).astype(float)
    nh, nw = nat.shape[:2]
    sx, sy = a.shape[1] / nw, a.shape[0] / nh
    ccol = np.clip(np.round((np.arange(nw) + 0.5) * sx - 0.5).astype(int), 0, a.shape[1] - 1)
    crow = np.clip(np.round((np.arange(nh) + 0.5) * sy - 0.5).astype(int), 0, a.shape[0] - 1)
    cells = a[np.ix_(crow, ccol)]
    identity = int((np.abs(cells - nat).max(axis=2) > 0).sum())

    cal_r, cal_n = calibrate(a, sx), calibrate(nat, 1.0)
    for cal in (cal_r, cal_n):
        assert len(cal["x_tick_cols"]) == len(X_TICK_VALUES), cal["x_tick_cols"]
        assert len(cal["y_tick_rows"]) == len(Y_TICK_VALUES), cal["y_tick_rows"]
    xmap = fit_map(cal_r["x_tick_cols"], X_TICK_VALUES)
    ymap = fit_map(cal_r["y_tick_rows"], Y_TICK_VALUES[::-1])
    xmap_n = fit_map(cal_n["x_tick_cols"], X_TICK_VALUES)
    ymap_n = fit_map(cal_n["y_tick_rows"], Y_TICK_VALUES[::-1])
    dashed_check = {}
    for k, v in DASHED.items():
        pr = (v - ymap["intercept"]) / ymap["slope"]
        near = min(cal_r["dashed_line_rows"], key=lambda r: abs(r - pr))
        dashed_check[k] = {"value_kipft": v, "predicted_row_px": pr, "detected_row_px": near,
                           "moment_error_kipft": ymap["slope"] * (near - pr)}

    def to_x(col_render):
        return xmap["slope"] * col_render + xmap["intercept"]

    def to_m(row_render):
        return ymap["slope"] * row_render + ymap["intercept"]

    lab, alpha = classify(cells)
    names = list(SERIES)
    # search window in source-pixel cells
    axis_c = int(round((cal_r["y_axis_line_col"] + 0.5) / sx - 0.5))
    axis_r = int(round((cal_r["x_axis_line_row"] + 0.5) / sy - 0.5))
    c_lo = axis_c + 1
    c_hi = int(np.ceil(((4.0 - xmap["intercept"]) / xmap["slope"] + 0.5) / sx))

    recs = []
    for sid in names:
        k = names.index(sid)
        upper = [names.index(u) for u in DRAW_ORDER[DRAW_ORDER.index(sid) + 1:]]
        for j in range(axis_r - 1, 0, -1):
            m_row = to_m(crow[j])
            if m_row > M_CEILING[sid]:
                break
            seg = lab[j, c_lo:c_hi]
            own = seg == k
            cover = np.isin(seg, upper)
            # merge own cells separated only by cells of an upper-layer curve
            rs = runs(own)
            merged = []
            for s, e in rs:
                if merged and np.all(cover[merged[-1][1] + 1: s]):
                    merged[-1] = (merged[-1][0], e)
                else:
                    merged.append((s, e))
            rec = {"series": sid, "cell_row": j, "render_row_px": int(crow[j]),
                   "moment_kipft": m_row}
            if not merged:
                rec.update(status="hidden")
                recs.append(rec)
                continue
            s, e = merged[0]
            s_abs, e_abs = s + c_lo, e + c_lo
            left_cov = s > 0 and cover[s - 1]
            right_cov = e + 1 < len(seg) and cover[e + 1]
            wts = np.clip(alpha[j, s_abs:e_abs + 1, k], 0.3, 1.0) * own[s:e + 1]
            cen = float((np.arange(s_abs, e_abs + 1) * wts).sum() / wts.sum())
            rec.update(status="visible", run_first_cell=s_abs, run_last_cell=e_abs,
                       n_cells=e_abs - s_abs + 1, left_edge_covered=bool(left_cov),
                       right_edge_covered=bool(right_cov), centre_cell_weighted=cen)
            recs.append(rec)

    # stroke width rule for partly hidden runs: nominal horizontal run width is
    # the median width of fully exposed runs of the same curve within +-12 rows
    by = {}
    for r in recs:
        by.setdefault(r["series"], []).append(r)
    for sid, lst in by.items():
        lst.sort(key=lambda r: -r["cell_row"])
        exposed = [(r["cell_row"], r["n_cells"]) for r in lst if r["status"] == "visible"
                   and not r["left_edge_covered"] and not r["right_edge_covered"]]
        for r in lst:
            if r["status"] != "visible":
                continue
            near = [n for (jj, n) in exposed if abs(jj - r["cell_row"]) <= 12]
            w = float(np.median(near)) if near else 2.0
            r["nominal_width_cells"] = w
            n = r["n_cells"]
            if n >= w or (not r["left_edge_covered"] and not r["right_edge_covered"]):
                c, rule, unc = r["centre_cell_weighted"], "run centre", 0.5
            elif r["left_edge_covered"] and not r["right_edge_covered"]:
                c, rule, unc = r["run_last_cell"] - (w - 1) / 2.0, "right edge minus half width", 0.75
            elif r["right_edge_covered"] and not r["left_edge_covered"]:
                c, rule, unc = r["run_first_cell"] + (w - 1) / 2.0, "left edge plus half width", 0.75
            else:
                c, rule, unc = r["centre_cell_weighted"], "both edges covered, run centre", 1.0
            col_render = (c + 0.5) * sx - 0.5
            r.update(centre_cell=c, centre_rule=rule, centre_render_col_px=col_render,
                     displacement_in=to_x(col_render),
                     displacement_unc_in=unc * sx * xmap["slope"] + xmap["max_abs_residual"],
                     run_first_render_col_px=(r["run_first_cell"]) * sx,
                     run_last_render_col_px=(r["run_last_cell"] + 1) * sx - 1)

    fields = ["series", "cell_row", "render_row_px", "moment_kipft", "status", "run_first_cell",
              "run_last_cell", "run_first_render_col_px", "run_last_render_col_px", "n_cells",
              "left_edge_covered", "right_edge_covered", "nominal_width_cells", "centre_rule",
              "centre_cell", "centre_render_col_px", "displacement_in", "displacement_unc_in"]
    with open(out / "fig68_traces_raw.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for sid in names:
            for r in by[sid]:
                wr.writerow({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in r.items()})

    # fixed moment levels: interpolate between the two bracketing source rows,
    # only if both are read (no interpolation across hidden rows)
    m_unc = abs(ymap["slope"]) * sy / 2.0 + ymap["max_abs_residual"]
    lev_rows = []
    for sid in names:
        lst = sorted(by[sid], key=lambda r: r["moment_kipft"])
        for M in LEVELS:
            if M > M_CEILING[sid]:
                continue
            lo = [r for r in lst if r["moment_kipft"] <= M]
            hi = [r for r in lst if r["moment_kipft"] > M]
            row = {"series": sid, "moment_kipft": M, "moment_unc_kipft": round(m_unc, 1)}
            if lo and hi and lo[-1].get("displacement_in") is not None \
                    and hi[0].get("displacement_in") is not None:
                r0, r1 = lo[-1], hi[0]
                t = (M - r0["moment_kipft"]) / (r1["moment_kipft"] - r0["moment_kipft"])
                x = r0["displacement_in"] + t * (r1["displacement_in"] - r0["displacement_in"])
                row.update(displacement_in=round(x, 4),
                           displacement_unc_in=round(max(r0["displacement_unc_in"],
                                                         r1["displacement_unc_in"]), 4),
                           rules=f'{r0["centre_rule"]} | {r1["centre_rule"]}',
                           rows_used=f'{r0["cell_row"]},{r1["cell_row"]}')
            else:
                hid = [r["cell_row"] for r in (lo[-1:] + hi[:1]) if r.get("displacement_in") is None]
                row.update(displacement_in=None, displacement_unc_in=None,
                           rules=f"hidden under a curve drawn on top (rows {hid})", rows_used="")
            lev_rows.append(row)
    with open(out / "fig68_levels.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["series", "moment_kipft", "moment_unc_kipft",
                                           "displacement_in", "displacement_unc_in", "rules", "rows_used"])
        wr.writeheader()
        wr.writerows(lev_rows)

    cal = {
        "source": "Provines, Ocel & Zmetra (2019) FHWA-HRT-20-005, Figure 68, report p. 80 (PDF page 98)",
        "render": {"dpi": DPI, "crop_px_size_wh": [a.shape[1], a.shape[0]],
                   "native_px_size_wh": [nw, nh], "render_px_per_native_px_xy": [sx, sy],
                   "placement_pt_x0y0x1y1": PLACE_PT,
                   "cells_differing_between_render_samples_and_embedded_raster": identity},
        "pixel_convention": "render pixel-centre indices in the cropped 600-dpi render; rows increase downward",
        "render_calibration": cal_r, "native_calibration": cal_n,
        "x_map_in_per_render_px": xmap, "y_map_kipft_per_render_px": ymap,
        "x_map_native_check": xmap_n, "y_map_native_check": ymap_n,
        "dashed_line_check": dashed_check,
        "series_colours_rgb": SERIES, "draw_order_bottom_to_top": DRAW_ORDER,
        "moment_ceiling_kipft": M_CEILING,
        "search_window_cells": {"col_first": c_lo, "col_last_exclusive": c_hi, "x_axis_row": axis_r},
        "moment_level_uncertainty_kipft": m_unc,
    }
    with open(out / "fig68_calibration.json", "w") as f:
        json.dump(cal, f, indent=1)
    print("identity mismatches:", identity)
    print(json.dumps({k: cal[k] for k in ("x_map_in_per_render_px", "y_map_kipft_per_render_px",
                                          "dashed_line_check")}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
