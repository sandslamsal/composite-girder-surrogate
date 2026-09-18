"""Digitise the Suwaed (2017) PhD thesis beam-test curves (WRAP 90318).

Source: Suwaed ASH (2017) Development of Novel Demountable Shear Connectors
for Precast Steel-Concrete Composite Bridges, PhD thesis, University of
Warwick.  https://wrap.warwick.ac.uk/90318/1/WRAP_Theses_Suwaed_2017.pdf

  * Fig. 7.3 (printed p. 160 = PDF p. 179): jack load (kN, 100 t load cell,
    self-weight and spreader EXCLUDED) vs mid-span deflection (mm, LVDT S12),
    six load cycles.  Only the cycle-1 loading branch is digitised.
  * Fig. 7.6 (printed p. 164 = PDF p. 183): mid-span moment (kN.m, the
    57 kN.m self-weight + spreader moment INCLUDED, text p. 164) vs mid-span
    deflection; test curve, ETABS and ABS ("ABC") predictions.  Digitised as a
    cross-check only.

Both figures are embedded rasters (856 x 514 and 856 x 511 px at 150 ppi,
`pdfimages -list`), not vector graphics.  The pages were rendered with
`pdftoppm -r 600 -png` (4x nearest-neighbour upsampling of the embedded
raster), which is the input here.  The renders are kept outside the
repository (copyright), under the scratchpad lit_sources/suwaed2020 folder.

Method
  1. Calibration: grey gridlines are located as columns/rows in which most
     pixels are neutral grey (R=G=B, 150-235); the centre of each gridline is
     the mean index of its pixel band.  A straight-line fit through all
     labelled gridlines of each axis gives the scale; the residual of that fit
     is reported.
  2. Curve: "Excel blue" pixels (68,114,196) and their anti-aliased blends
     are weighted by w = clip((B - R) / 128, 0, 1).  For every pixel row
     (load level) the leftmost contiguous blue run inside the plot is taken as
     the cycle-1 loading branch (the unloading branch, cycles 2-6 and the
     'Loading' arrow all lie to its right), and its w-weighted centroid column
     is recorded.  Rows are then converted to load and deflection and the
     deflection at each requested load level is linearly interpolated
     between neighbouring rows.
  3. A column-wise pass (centroid row of the blue run in each column) is
     used as a cross-check on the flatter part of the curve.

Outputs (this folder): fig7_3_calibration.json, fig7_3_cycle1_rows.csv
(raw pixel centroids), fig7_3_cycle1_levels.csv (load levels),
fig7_6_calibration.json, fig7_6_test_rows.csv, fig7_6_levels.csv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SCRATCH = Path(
    "/private/tmp/claude-501/-Users-sandeshlamsal-Desktop-CompositeGirder/"
    "a32c3e10-8884-4339-8200-ea3611516154/scratchpad/lit_sources/suwaed2020"
)
RENDER_73 = SCRATCH / "p179_600-179.png"   # pdftoppm -r 600 -f 179 -l 179
RENDER_76 = SCRATCH / "p183_600-183.png"   # pdftoppm -r 600 -f 183 -l 183


def load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).astype(float)


def grey_mask(img: np.ndarray) -> np.ndarray:
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return (
        (np.abs(r - g) < 8) & (np.abs(g - b) < 8) & (r > 150) & (r < 235)
    )


def blue_weight(img: np.ndarray) -> np.ndarray:
    r, b = img[..., 0], img[..., 2]
    return np.clip((b - r) / 128.0, 0.0, 1.0)


def band_centres(counts: np.ndarray, threshold: float, offset: int) -> list:
    """Centres of contiguous index bands whose count exceeds threshold."""
    idx = np.where(counts > threshold)[0]
    if idx.size == 0:
        return []
    bands, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i != prev + 1:
            bands.append((start, prev))
            start = i
        prev = i
    bands.append((start, prev))
    out = []
    for a, b in bands:
        w = counts[a:b + 1]
        out.append(float(offset + (np.arange(a, b + 1) * w).sum() / w.sum()))
    return out


def fit_axis(pixels: list, values: list) -> dict:
    p = np.asarray(pixels)
    v = np.asarray(values)
    slope, intercept = np.polyfit(p, v, 1)
    resid = v - (slope * p + intercept)
    return {
        "gridline_pixels": [round(x, 2) for x in p.tolist()],
        "gridline_values": v.tolist(),
        "value_per_pixel": slope,
        "intercept": intercept,
        "max_abs_residual_value": float(np.abs(resid).max()),
    }


def row_centroids(w: np.ndarray, r0: int, r1: int, c0: int, c1: int,
                  min_w: float = 0.15) -> list:
    """Leftmost blue run per row: returns (row, centroid_col, run_start,
    run_end, weight_sum)."""
    out = []
    for r in range(r0, r1):
        row = w[r, c0:c1]
        on = np.where(row > min_w)[0]
        if on.size == 0:
            continue
        # leftmost contiguous run
        start = on[0]
        end = start
        for i in on[1:]:
            if i == end + 1:
                end = i
            else:
                break
        # widen by anti-aliased neighbours with any weight
        a = start
        while a > 0 and row[a - 1] > 0.02:
            a -= 1
        b = end
        while b < row.size - 1 and row[b + 1] > 0.02:
            b += 1
        ww = row[a:b + 1]
        cc = (np.arange(a, b + 1) * ww).sum() / ww.sum()
        out.append((r, c0 + cc, c0 + a, c0 + b, float(ww.sum())))
    return out


def digitise_fig73() -> dict:
    img = load(RENDER_73)
    # plot region found from the page render (figure block rows 2216-3992)
    R0, R1, C0, C1 = 2150, 4050, 950, 4300
    sub = img[R0:R1, C0:C1]
    g = grey_mask(sub)
    vcols = band_centres(g.sum(axis=0), 800, C0)
    hrows = band_centres(g.sum(axis=1), 1500, R0)
    # 8 vertical gridlines: 0, -50, ..., -350 mm (axis labels, Fig. 7.3)
    # 5 horizontal gridlines: 1000, 750, 500, 250, 0 kN (top to bottom)
    assert len(vcols) == 8, vcols
    assert len(hrows) == 5, hrows
    xcal = fit_axis(vcols, [0, 50, 100, 150, 200, 250, 300, 350])  # |defl| mm
    ycal = fit_axis(hrows, [1000, 750, 500, 250, 0])                # kN
    calib = {
        "figure": "Suwaed (2017) thesis Fig. 7.3, printed p. 160 (PDF p. 179)",
        "render": "pdftoppm -r 600 -f 179 -l 179 -png (4x the 150 ppi embedded raster)",
        "x_axis": "mid-span deflection magnitude (mm); axis labels are negative (downward)",
        "y_axis": "jack load (kN), self-weight and spreader excluded",
        "x": xcal,
        "y": ycal,
        "px_per_mm_600dpi": 1.0 / xcal["value_per_pixel"],
        "px_per_kN_600dpi": -1.0 / ycal["value_per_pixel"],
        "native_px_per_mm": 0.25 / xcal["value_per_pixel"],
        "native_kN_per_px": -4.0 * ycal["value_per_pixel"],
    }

    w = blue_weight(img)
    x0 = vcols[0]
    # rows from the 0 kN gridline up to ~845 kN; columns from 5 mm left of
    # the 0-mm axis to 160 mm (excludes legend at > 250 mm)
    row_top = int(round((845 - ycal["intercept"]) / ycal["value_per_pixel"]))
    row_bot = int(round((0 - ycal["intercept"]) / ycal["value_per_pixel"]))
    col_l = int(x0 - 5 * calib["px_per_mm_600dpi"])
    col_r = int(x0 + 160 * calib["px_per_mm_600dpi"])
    rows = row_centroids(w, row_top, row_bot + 12, col_l, col_r)

    recs = []
    for r, cc, a, b, ws in rows:
        load_kn = ycal["value_per_pixel"] * r + ycal["intercept"]
        defl = xcal["value_per_pixel"] * cc + xcal["intercept"]
        recs.append({
            "row_px": r, "centroid_col_px": round(cc, 3),
            "run_start_col_px": a, "run_end_col_px": b,
            "blue_weight_sum": round(ws, 3),
            "load_kN": round(load_kn, 3), "deflection_mm": round(defl, 4),
        })
    return {"calib": calib, "rows": recs}


def blend_alpha(img: np.ndarray, target: tuple, tol: float = 0.12) -> np.ndarray:
    """Opacity of `target` colour blended over white, or 0 where the pixel is
    not a consistent target/white blend (e.g. overlaps of two series)."""
    t = np.asarray(target, float)
    use = (255.0 - t) > 40.0
    a = (255.0 - img[..., use]) / (255.0 - t[use])
    alpha = a.mean(axis=-1)
    ok = (a.max(axis=-1) - a.min(axis=-1)) < tol
    # channels where the target is ~white must stay ~white
    if (~use).any():
        ok &= (np.abs(img[..., ~use] - 255.0) < 40.0 * np.clip(alpha, 0, 1)[..., None] + 12).all(axis=-1)
    return np.where(ok & (alpha > 0.1), np.clip(alpha, 0, 1), 0.0)


SERIES_76 = {
    "test": (68, 114, 196),
    "ETABS": (255, 0, 0),
    "ABS_ABC": (0, 176, 80),
}


def digitise_fig76() -> dict:
    img = load(RENDER_76)
    R0, R1, C0, C1 = 1080, 2800, 950, 4300
    sub = img[R0:R1, C0:C1]
    g = grey_mask(sub)
    vcols = band_centres(g.sum(axis=0), 900, C0)
    hrows = band_centres(g.sum(axis=1), 1500, R0)
    assert len(vcols) == 6, vcols
    assert len(hrows) == 9, hrows
    xcal = fit_axis(vcols, [0, 100, 200, 300, 400, 500])          # |defl| mm
    ycal = fit_axis(hrows, [1600, 1400, 1200, 1000, 800, 600, 400, 200, 0])
    calib = {
        "figure": "Suwaed (2017) thesis Fig. 7.6, printed p. 164 (PDF p. 183)",
        "render": "pdftoppm -r 600 -f 183 -l 183 -png (4x the 150 ppi embedded raster)",
        "x_axis": "mid-span deflection magnitude (mm)",
        "y_axis": "mid-span moment (kN.m), 57 kN.m self-weight + spreader moment included (thesis p. 164)",
        "x": xcal, "y": ycal,
        "native_px_per_mm": 0.25 / xcal["value_per_pixel"],
        "native_kNm_per_px": -4.0 * ycal["value_per_pixel"],
    }
    x0 = vcols[0]
    ppm = 1.0 / xcal["value_per_pixel"]
    row_top = int(round((1000 - ycal["intercept"]) / ycal["value_per_pixel"]))
    row_bot = int(round((0 - ycal["intercept"]) / ycal["value_per_pixel"]))
    col_l, col_r = int(x0 - 3 * ppm), int(x0 + 60 * ppm)
    recs = []
    for name, colour in SERIES_76.items():
        w = blend_alpha(img, colour)
        for r, cc, a, b, ws in row_centroids(w, row_top, row_bot + 1, col_l, col_r, min_w=0.3):
            recs.append({
                "series": name, "row_px": r, "centroid_col_px": round(cc, 3),
                "run_start_col_px": a, "run_end_col_px": b,
                "weight_sum": round(ws, 3),
                "moment_kNm": round(ycal["value_per_pixel"] * r + ycal["intercept"], 3),
                "deflection_mm": round(xcal["value_per_pixel"] * cc + xcal["intercept"], 4),
            })
    return {"calib": calib, "rows": recs}


def interp_levels(recs: list, levels: list) -> list:
    """Deflection at given loads by linear interpolation along the row
    centroids (rows sorted by increasing load, monotonic branch)."""
    loads = np.array([r["load_kN"] for r in recs])
    defl = np.array([r["deflection_mm"] for r in recs])
    order = np.argsort(loads)
    loads, defl = loads[order], defl[order]
    out = []
    for p in levels:
        # local 9-row window (about +/- 2.6 kN) median to suppress
        # single-row noise
        k = np.searchsorted(loads, p)
        lo, hi = max(k - 4, 0), min(k + 5, loads.size)
        d_interp = float(np.interp(p, loads, defl))
        d_med = float(np.median(defl[lo:hi]))
        out.append((p, d_interp, d_med, float(defl[lo:hi].std())))
    return out


def column_centroids(w: np.ndarray, r0: int, r1: int, c0: int, c1: int,
                     min_w: float = 0.15) -> list:
    """Centroid row of all blue pixels in each column (single-branch
    columns only)."""
    out = []
    for c in range(c0, c1):
        col = w[r0:r1, c]
        on = np.where(col > min_w)[0]
        if on.size == 0:
            continue
        if on[-1] - on[0] > 80:          # more than one branch: skip
            continue
        a, b = on[0], on[-1]
        ww = col[a:b + 1]
        out.append((c, r0 + (np.arange(a, b + 1) * ww).sum() / ww.sum()))
    return out


M_PER_KN = 1.375      # kN.m of mid-span moment per kN of jack load (P/2 x 2.75 m)
M_SELF = 57.0         # kN.m, self-weight + spreader moment at mid-span (thesis p. 164)
M_MAX_TOTAL = 1348.0  # kN.m, maximum moment incl. self-weight (thesis p. 164)
ELASTIC_LIMIT_TOTAL = 0.50 * M_MAX_TOTAL   # 'linear up to 50-55 % of max moment' (p. 165)

if __name__ == "__main__":
    import csv

    # ---------------- Fig. 7.3 ----------------
    res = digitise_fig73()
    calib, recs = res["calib"], res["rows"]
    loads = np.array([r["load_kN"] for r in recs])
    defl = np.array([r["deflection_mm"] for r in recs])
    order = np.argsort(loads)
    loads, defl = loads[order], defl[order]

    # zero offset: deflection at zero jack load (self-weight + spreader
    # deflection after de-propping; the record's zero is the propped state,
    # thesis pp. 163-164).  Linear fit over 10-100 kN, extrapolated to P = 0.
    m = (loads >= 10) & (loads <= 100)
    b1, b0 = np.polyfit(loads[m], defl[m], 1)
    row0 = float(np.interp(0.0, loads, defl))
    calib["zero_offset"] = {
        "method": "linear fit of row centroids for 10-100 kN, extrapolated to P = 0",
        "delta0_fit_mm": round(float(b0), 3),
        "slope_fit_mm_per_kN": round(float(b1), 5),
        "delta_at_P0_row_mm": round(row0, 3),
        "thesis_text": "about 2 mm at mid-span when the props were removed (p. 164)",
    }
    delta0 = round(float(b0), 2)

    # residual scatter of rows about a smooth cubic, 50-450 kN
    m2 = (loads >= 50) & (loads <= 450)
    cfit = np.polyfit(loads[m2], defl[m2], 3)
    resid = defl[m2] - np.polyval(cfit, loads[m2])
    calib["row_scatter_50_450kN_mm"] = {
        "std": round(float(resid.std()), 4), "max_abs": round(float(np.abs(resid).max()), 4)}

    # column-wise cross-check on the flatter part (35-75 mm columns carry
    # only the cycle-1 loading branch)
    img = load(RENDER_73)
    w = blue_weight(img)
    x0 = calib["x"]["gridline_pixels"][0]
    ppm = calib["px_per_mm_600dpi"]
    rtop = int(round((845 - calib["y"]["intercept"]) / calib["y"]["value_per_pixel"]))
    rbot = int(round((0 - calib["y"]["intercept"]) / calib["y"]["value_per_pixel"]))
    cols = column_centroids(w, rtop, rbot, int(x0 + 36 * ppm), int(x0 + 76 * ppm))
    col_defl = np.array([calib["x"]["value_per_pixel"] * c + calib["x"]["intercept"] for c, _ in cols])
    col_load = np.array([calib["y"]["value_per_pixel"] * r + calib["y"]["intercept"] for _, r in cols])
    with open(HERE / "fig7_3_cycle1_columns_36_76mm.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["col_px", "centroid_row_px", "deflection_mm", "load_kN"])
        for (c, r), d_, l_ in zip(cols, col_defl, col_load):
            wr.writerow([c, round(r, 3), round(d_, 4), round(l_, 3)])

    levels_el = [50, 100, 150, 200, 250, 300, 350, 400, 445]
    levels_beyond = [500, 550, 600, 650, 700, 750, 800]
    out = []
    for p, d_i, d_med, d_std in interp_levels(recs, levels_el + levels_beyond):
        m_tot = M_SELF + M_PER_KN * p
        col_cross = ""
        if col_load.size and col_load.min() <= p <= col_load.max():
            o = np.argsort(col_load)
            col_cross = round(float(np.interp(p, col_load[o], col_defl[o])), 2)
        elastic = m_tot <= ELASTIC_LIMIT_TOTAL
        out.append({
            "jack_load_kN": p,
            "deflection_as_plotted_mm": round(d_i, 2),
            "deflection_window_median_mm": round(d_med, 2),
            "deflection_window_std_mm": round(d_std, 3),
            "deflection_columnwise_mm": col_cross,
            "delta0_mm": delta0,
            "deflection_jack_only_mm": round(d_i - delta0, 2),
            "moment_jack_kNm": round(M_PER_KN * p, 1),
            "moment_total_kNm": round(m_tot, 1),
            "elastic": bool(elastic),
        })
    with open(HERE / "fig7_3_cycle1_levels.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        wr.writeheader()
        wr.writerows(out)
    with open(HERE / "fig7_3_cycle1_rows.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
        wr.writeheader()
        wr.writerows(recs)
    (HERE / "fig7_3_calibration.json").write_text(json.dumps(calib, indent=2))

    # ---------------- Fig. 7.6 ----------------
    r76 = digitise_fig76()
    (HERE / "fig7_6_calibration.json").write_text(json.dumps(r76["calib"], indent=2))
    with open(HERE / "fig7_6_rows.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(r76["rows"][0].keys()))
        wr.writeheader()
        wr.writerows(r76["rows"])
    lv76 = []
    for name in SERIES_76:
        rr = [r for r in r76["rows"] if r["series"] == name]
        mm = np.array([r["moment_kNm"] for r in rr])
        dd = np.array([r["deflection_mm"] for r in rr])
        o = np.argsort(mm)
        mm, dd = mm[o], dd[o]
        # secant/through-origin slope over 100-600 kN.m (kN.m per mm)
        sel = (mm >= 100) & (mm <= 600)
        k0 = float((mm[sel] * dd[sel]).sum() / (dd[sel] ** 2).sum()) if sel.any() else float("nan")
        for M in [57, 100, 200, 300, 340, 400, 500, 600, 700, 800, 860, 1000]:
            if mm.size and mm.min() - 5 <= M <= mm.max() + 5:
                lv76.append({"series": name, "moment_kNm": M,
                             "deflection_mm": round(float(np.interp(M, mm, dd)), 2),
                             "through_origin_slope_100_600_kNm_per_mm": round(k0, 2),
                             "digitised_moment_range_kNm": f"{mm.min():.0f}-{mm.max():.0f}"})
    with open(HERE / "fig7_6_levels.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(lv76[0].keys()))
        wr.writeheader()
        wr.writerows(lv76)

    print(json.dumps(calib, indent=1))
    for o_ in out:
        print(o_)
    print(json.dumps(r76["calib"], indent=1))
    for o_ in lv76:
        print(o_)
