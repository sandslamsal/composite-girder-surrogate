"""Digitise midspan load-deflection test points from Viest, Siess, Appleton
and Newmark (1952), University of Illinois Engineering Experiment Station
Bulletin 405, "Full-scale tests of channel shear connectors and composite
T-beams".

Figures handled
---------------
* Fig. 43 (printed p. 102, PDF p. 106), "Load-Deflection Curves, Concentrated
  Load at Midspan": the four "At Midspan" sub-panels (B24W, B24S, B21S,
  B21W). Grid 10 kips per horizontal line (0 at the bottom frame), 0.2 in per
  vertical line (0 at the left frame; labels 0, 0.4, 0.8).
* Fig. 53 (printed p. 123, PDF p. 127), "Effect of Bond on Degree of
  Interaction", right panel "Deflection at Midspan", beam B21W. Grid 5 kips
  per horizontal line (labels 0-40 every 10), 0.2 in per vertical line
  (labels 0, 0.2, 0.4, 0.6). Open markers = test after breaking bond (the
  state of all later tests); filled markers = test before breaking bond
  (recorded for context only).

Source scan and resolution
--------------------------
IDEALS PDF (https://www.ideals.illinois.edu/items/4841, md5
bc4b997648d5e27dab5b3947f88e1ce0). The page images embedded in the PDF are
150 ppi greyscale JPEG (pdfimages -list), so the 600 dpi renders
(pdftoppm -r 600) used here are 4x interpolations: one scan pixel = 4 render
pixels. The renders are NOT kept in the repository (copyright); they live in
the session scratchpad under lit_sources/viest1952/ (r600-106.png,
r600-127.png).

Method
------
1. Grid lines: dark-pixel (grey < 140) column and row fractions inside each
   sub-panel box; runs above a threshold are merged and their weighted
   centres taken. The first line (left frame, bottom frame) is the axis zero;
   every detected line is assigned the nearest lattice value using the median
   line spacing. Lines at labelled values are flagged. Pixel -> value uses a
   piecewise-linear interpolation through the detected lines (this removes the
   hand-drafting irregularity of individual grid cells, up to +/-5 render px);
   a global least-squares line is also reported and its difference from the
   piecewise map at each point is written to the CSV.
2. Markers: normalised correlation of the dark mask with an annulus
   (r = 8-13 render px) gives a ring score; a candidate needs ring score
   >= 0.75, an almost white disc of r = 5.5 px at the centre (dark fraction
   < 0.06), and a closed white interior component of marker size
   (100-400 px^2, bounding-box fill >= 0.5, aspect 0.7-1.4). Non-maximum
   suppression within 40 px. Detections inside documented exclusion boxes
   (panel titles, legend) are dropped. Filled markers (Fig. 53 only) are dark
   blobs surviving a morphological opening with a disc of radius 10 px.
3. Output: calibration.json (grid lines, lattice assignment, residuals),
   markers_raw.csv (every accepted marker with pixel and data coordinates),
   and overlay_check.png (crops with the detected grid lines and the accepted
   markers annotated).

Units: kips and inches, as printed in the source.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy import signal

HERE = Path(__file__).resolve().parent
SCRATCH = Path(
    "/private/tmp/claude-501/-Users-sandeshlamsal-Desktop-CompositeGirder/"
    "a32c3e10-8884-4339-8200-ea3611516154/scratchpad/lit_sources/viest1952"
)

# ----------------------------------------------------------------------------
# panel definitions (render pixel coordinates of the 600 dpi page renders)
# box = (x0, y0, x1, y1) enclosing the sub-panel frame; grid lines are searched
# inside it. marker_box may extend beyond it. exclude = boxes whose detections
# are text (panel title letters, legend symbol), identified on the render.
# ----------------------------------------------------------------------------
PANELS = [
    dict(fig="fig43", beam="B24W", png="r600-106.png", pdf_page=106, printed_page=102,
         box=(556, 1676, 1232, 3322), x_step=0.2, y_step=10.0,
         x_labels=[0.0, 0.4, 0.8], y_labels=[0, 20, 40, 60, 80, 100],
         exclude=[(560, 1690, 900, 1800)], marker="open"),
    dict(fig="fig43", beam="B24S", png="r600-106.png", pdf_page=106, printed_page=102,
         box=(1836, 1676, 2512, 3322), x_step=0.2, y_step=10.0,
         x_labels=[0.0, 0.4, 0.8], y_labels=[0, 20, 40, 60, 80, 100],
         exclude=[(1840, 1690, 2180, 1800)], marker="open"),
    dict(fig="fig43", beam="B21S", png="r600-106.png", pdf_page=106, printed_page=102,
         box=(560, 3500, 1242, 4866), x_step=0.2, y_step=10.0,
         x_labels=[0.0, 0.4, 0.8], y_labels=[0, 20, 40, 60, 80],
         exclude=[(564, 3520, 900, 3640)], marker="open"),
    dict(fig="fig43", beam="B21W", png="r600-106.png", pdf_page=106, printed_page=102,
         box=(1844, 3500, 2520, 4866), x_step=0.2, y_step=10.0,
         x_labels=[0.0, 0.4, 0.8], y_labels=[0, 20, 40, 60, 80],
         exclude=[(1848, 3520, 2190, 3640), (1990, 3960, 2520, 4090)], marker="open"),
    dict(fig="fig53", beam="B21W", png="r600-127.png", pdf_page=127, printed_page=123,
         box=(2016, 926, 2934, 2078), x_step=0.2, y_step=5.0,
         x_labels=[0.0, 0.2, 0.4, 0.6], y_labels=[0, 10, 20, 30, 40],
         exclude=[(2030, 940, 2460, 1150)], marker="open+filled"),
]


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")).astype(float)


def detect_lines(frac: np.ndarray, thr: float, offset: int) -> list[float]:
    idx = np.where(frac > thr)[0]
    if idx.size == 0:
        return []
    groups = np.split(idx, np.where(np.diff(idx) > 3)[0] + 1)
    out = []
    for g in groups:
        lo, hi = max(g[0] - 2, 0), min(g[-1] + 2, frac.size - 1)
        w = frac[lo:hi + 1]
        xs = np.arange(lo, hi + 1)
        out.append(float((w * xs).sum() / w.sum()) + offset)
    return out


def calibrate_axis(lines: list[float], step: float, labels: list[float],
                   increasing_px: bool) -> dict:
    """Assign lattice values to detected grid lines. The zero line is the
    first line (left frame for x, bottom frame for y)."""
    lines = sorted(lines) if increasing_px else sorted(lines, reverse=True)
    zero = lines[0]
    d = np.abs(np.diff(lines))
    spacing = float(np.median(d[(d > 0.5 * np.median(d))]))
    assigned = []
    for p in lines:
        k = abs(p - zero) / spacing
        kr = int(round(k))
        if abs(k - kr) < 0.2 and not any(a["index"] == kr for a in assigned):
            assigned.append({"px": round(p, 2), "index": kr, "value": round(kr * step, 6),
                             "labelled": any(abs(kr * step - v) < 1e-9 for v in labels)})
    px = np.array([a["px"] for a in assigned])
    val = np.array([a["value"] for a in assigned])
    a1, a0 = np.polyfit(px, val, 1)
    resid = val - (a1 * px + a0)
    return {"lines": assigned, "median_spacing_px": spacing,
            "global_fit": {"value_per_px": a1, "intercept": a0,
                           "rms_residual": float(np.sqrt(np.mean(resid ** 2))),
                           "max_abs_residual": float(np.max(np.abs(resid)))},
            "n_labelled_lines_used": int(sum(a["labelled"] for a in assigned))}


def px_to_value(cal: dict, p: float) -> tuple[float, float]:
    px = np.array([a["px"] for a in cal["lines"]])
    val = np.array([a["value"] for a in cal["lines"]])
    order = np.argsort(px)
    px, val = px[order], val[order]
    g = cal["global_fit"]["value_per_px"] * p + cal["global_fit"]["intercept"]
    if px[0] <= p <= px[-1]:
        return float(np.interp(p, px, val)), float(g)
    return float(g), float(g)


def ring_markers(gray, box, exclude):
    x0, y0, x1, y1 = box
    sub = gray[y0:y1, x0:x1]
    dark = (sub < 140).astype(float)
    R = 15
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    r = np.hypot(xx, yy)
    ann = ((r >= 8.0) & (r <= 13.0)).astype(float)
    disk = (r <= 5.5).astype(float)
    rs = signal.fftconvolve(dark, ann[::-1, ::-1], mode="same") / ann.sum()
    ins = signal.fftconvolve(dark, disk[::-1, ::-1], mode="same") / disk.sum()
    score = np.where(ins < 0.06, rs, 0.0)
    mx = ndi.maximum_filter(score, size=31)
    peaks = np.argwhere((score == mx) & (score >= 0.75))
    lab, _ = ndi.label(sub > 170)
    cands = []
    for py, px in peaks:
        L = lab[py, px]
        if L == 0:
            continue
        m = lab == L
        area = int(m.sum())
        if not (100 <= area <= 400):
            continue
        ys, xs = np.nonzero(m)
        h, w = ys.ptp() + 1, xs.ptp() + 1
        if not (0.7 <= h / w <= 1.4) or area / (h * w) < 0.5:
            continue
        cy, cx = ys.mean(), xs.mean()
        X, Y = float(cx + x0), float(cy + y0)
        if any(ex0 <= X <= ex1 and ey0 <= Y <= ey1 for ex0, ey0, ex1, ey1 in exclude):
            continue
        cands.append({"px_x": X, "px_y": Y, "ring_score": float(rs[py, px]),
                      "interior_area_px": area, "marker": "open"})
    cands.sort(key=lambda c: -c["ring_score"])
    kept = []
    for c in cands:
        if all(np.hypot(c["px_x"] - k["px_x"], c["px_y"] - k["px_y"]) > 40 for k in kept):
            kept.append(c)
    return kept


def filled_markers(gray, box, exclude, radius=10, min_area=250, max_area=520):
    x0, y0, x1, y1 = box
    sub = gray[y0:y1, x0:x1] < 140
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    opened = ndi.binary_opening(sub, structure=(xx ** 2 + yy ** 2) <= radius ** 2)
    lab, n = ndi.label(opened)
    out = []
    for i in range(1, n + 1):
        m = lab == i
        area = int(m.sum())
        if not (min_area <= area <= max_area):
            continue
        cy, cx = ndi.center_of_mass(m)
        X, Y = float(cx + x0), float(cy + y0)
        if any(ex0 <= X <= ex1 and ey0 <= Y <= ey1 for ex0, ey0, ex1, ey1 in exclude):
            continue
        out.append({"px_x": X, "px_y": Y, "opened_area_px": area, "marker": "filled"})
    return out


def main():
    grays = {}
    calib_out, rows, crops = {}, [], []
    for P in PANELS:
        if P["png"] not in grays:
            grays[P["png"]] = load_gray(SCRATCH / P["png"])
        g = grays[P["png"]]
        x0, y0, x1, y1 = P["box"]
        sub = g[y0:y1, x0:x1] < 140
        # use only the inner 90 % to avoid the neighbouring frame lines
        cols = detect_lines(sub[int(0.05 * sub.shape[0]):int(0.95 * sub.shape[0])].mean(0), 0.45, x0)
        rws = detect_lines(sub[:, int(0.05 * sub.shape[1]):int(0.95 * sub.shape[1])].mean(1), 0.45, y0)
        xcal = calibrate_axis(cols, P["x_step"], P["x_labels"], increasing_px=True)
        ycal = calibrate_axis(rws, P["y_step"], P["y_labels"], increasing_px=False)
        key = f"{P['fig']}_{P['beam']}"
        calib_out[key] = {k: P[k] for k in ("fig", "beam", "png", "pdf_page", "printed_page", "box", "exclude")}
        calib_out[key].update({"x_axis_deflection_in": xcal, "y_axis_load_kip": ycal})
        mk = ring_markers(g, P["box"], P["exclude"])
        if "filled" in P["marker"]:
            mk += filled_markers(g, P["box"], P["exclude"])
        for m in mk:
            d_pw, d_gl = px_to_value(xcal, m["px_x"])
            p_pw, p_gl = px_to_value(ycal, m["px_y"])
            m.update({"deflection_in": d_pw, "load_kip": p_pw,
                      "deflection_in_globalfit": d_gl, "load_kip_globalfit": p_gl})
        mk.sort(key=lambda m: (m["marker"], m["load_kip"]))
        for m in mk:
            rows.append({"figure": P["fig"], "pdf_page": P["pdf_page"], "printed_page": P["printed_page"],
                         "beam": P["beam"], "marker": m["marker"],
                         "px_x": round(m["px_x"], 1), "px_y": round(m["px_y"], 1),
                         "load_kip": round(m["load_kip"], 2), "deflection_in": round(m["deflection_in"], 4),
                         "load_kip_globalfit": round(m["load_kip_globalfit"], 2),
                         "deflection_in_globalfit": round(m["deflection_in_globalfit"], 4)})
        calib_out[key]["n_markers"] = len(mk)
        # overlay crop
        pad = 30
        cx0, cy0, cx1, cy1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
        img = Image.fromarray(g[cy0:cy1, cx0:cx1].astype(np.uint8)).convert("RGB")
        dr = ImageDraw.Draw(img)
        for a in xcal["lines"]:
            X = a["px"] - cx0
            dr.line([(X, 0), (X, img.height)], fill=(0, 90, 255) if a["labelled"] else (120, 180, 255), width=2)
            dr.text((X + 3, img.height - 22), f"{a['value']:g}", fill=(0, 60, 200))
        for a in ycal["lines"]:
            Y = a["px"] - cy0
            dr.line([(0, Y), (img.width, Y)], fill=(0, 90, 255) if a["labelled"] else (120, 180, 255), width=2)
            dr.text((3, Y - 14), f"{a['value']:g}", fill=(0, 60, 200))
        for m in mk:
            X, Y = m["px_x"] - cx0, m["px_y"] - cy0
            col = (230, 0, 0) if m["marker"] == "open" else (0, 150, 0)
            dr.ellipse([X - 18, Y - 18, X + 18, Y + 18], outline=col, width=3)
            dr.text((X + 20, Y - 6), f"{m['load_kip']:.1f} k, {m['deflection_in']:.3f} in", fill=col)
        dr.text((40, 40), f"{P['fig']} {P['beam']} (PDF p. {P['pdf_page']})", fill=(200, 0, 120))
        crops.append(img)
    # montage
    scale = 0.5
    ims = [c.resize((int(c.width * scale), int(c.height * scale))) for c in crops]
    W = sum(i.width for i in ims[:3]) + 40
    H = max(i.height for i in ims[:3]) + max(i.height for i in ims[3:]) + 30
    mont = Image.new("RGB", (W, H), "white")
    x = 10
    for i in ims[:3]:
        mont.paste(i, (x, 10)); x += i.width + 10
    x = 10
    y = max(i.height for i in ims[:3]) + 20
    for i in ims[3:]:
        mont.paste(i, (x, y)); x += i.width + 10
    mont.save(HERE / "overlay_check.png")
    with open(HERE / "calibration.json", "w") as f:
        json.dump(calib_out, f, indent=1)
    with open(HERE / "markers_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    for key, c in calib_out.items():
        xa, ya = c["x_axis_deflection_in"], c["y_axis_load_kip"]
        print(key, "x lines", [(a["px"], a["value"]) for a in xa["lines"]], "rms", round(xa["global_fit"]["rms_residual"], 4),
              "| y lines", [(a["px"], a["value"]) for a in ya["lines"]], "rms", round(ya["global_fit"]["rms_residual"], 3))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
