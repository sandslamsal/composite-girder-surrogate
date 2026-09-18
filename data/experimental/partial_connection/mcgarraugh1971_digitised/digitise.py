"""Digitise the load-deflection and push-out curves of McGarraugh & Baldwin (1971),
"Lightweight concrete-on-steel composite beams", AISC Engineering Journal 8(3) 90-98.

Figures handled (journal page / PDF page of the AISC EJ archive scan):
  Fig. 5  end reaction vs mid-span deflection, B2 (top) and B4 (bottom)  p. 92 / 3
  Fig. 9  end reaction vs mid-span deflection, B6 (after sustained load) p. 94 / 5
  Fig. 8  time vs mid-span deflection, B6 (t = 0 instantaneous point)   p. 94 / 5
  Fig. 7  push-out load per stud vs slip, companion of B2 and of B4      p. 93 / 4

The page renders are NOT kept in the repository (copyright). Recreate them with
    pdftoppm -r 600 -png -gray -f 3 -l 5 mcgarraugh1971.pdf p600
from https://ej.aisc.org/index.php/engj/article/download/166/165 and pass the folder
as argv[1]. The scan embedded in the PDF is a 600 ppi 1-bit CCITT image, so the
600 dpi render is the native scan pixel for pixel.

Calibration: every labelled grid line (and the frame lines, which carry the end
labels) is located as the centre of its ink run at 100-400 positions along the line
and fitted as a straight line x = c0 + c1*y (vertical) or y = c0 + c1*x
(horizontal); the printed grid is slightly sheared and unevenly spaced (up to ~4 %
between neighbouring intervals). A point is mapped by linear interpolation between
the two neighbouring fitted grid lines evaluated at the point ('local', adopted).
A global affine map fitted through all grid intersections is also reported
('affine'); local - affine is quoted as calibration uncertainty.

Markers (filled circles ~33 px diameter; grid and curve lines 5-12 px): binary
opening with a disk of radius R_OPEN removes lines and text; the surviving blobs are
accepted if their area is 0.65-1.6 x the median single-marker area, and the centre
is refined as the centroid of the binary erosion of the ink by a disk of radius
R_ERODE (the set of positions where the whole marker core fits in ink), which is
insensitive to lines touching the marker. Outputs are written next to this script.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/private/tmp/claude-501/-Users-sandeshlamsal-Desktop-CompositeGirder/"
    "a32c3e10-8884-4339-8200-ea3611516154/scratchpad/lit_sources/mcgarraugh1971")
OVERLAY_DIR = SRC  # overlays contain the figure, so they stay out of the repo
OUT = Path(__file__).resolve().parent

R_OPEN = 12
R_ERODE = 12

# Nominal pixel positions of grid lines (found by column/row ink sums, refined by
# fit_line) and the data value each carries. Exclusion boxes (x0, y0, x1, y1) blank
# legends and labels before marker detection.
PANELS = {
    "fig5_B2": dict(page="p600-3.png", figure="Fig. 5, top panel (B2)", journal_page=92, pdf_page=3,
                    x_label="midspan deflection (in)", y_label="end reaction (kips)",
                    vx={0: 731, 1: 967, 2: 1210, 3: 1455, 4: 1700, 5: 1947, 6: 2191, 7: 2436},
                    hy={0: 4282, 10: 4039, 20: 3788, 30: 3544, 40: 3300},
                    labelled_x=[0, 2, 4, 6], labelled_y=[0, 10, 20, 30, 40],
                    exclude=[(1850, 3880, 2440, 4290), (735, 3300, 1340, 3500), (1150, 3900, 1900, 4030)],
                    refs={"rigid_connection": ((0.10, 0.575), (12.0, 29.5)),
                          "no_connection": ((0.10, 1.70), (2.0, 21.5))}),
    "fig5_B4": dict(page="p600-3.png", figure="Fig. 5, bottom panel (B4)", journal_page=92, pdf_page=3,
                    x_label="midspan deflection (in)", y_label="end reaction (kips)",
                    vx={0: 688, 1: 923, 2: 1167, 3: 1408, 4: 1655, 5: 1899, 6: 2142, 7: 2378},
                    hy={0: 5639, 10: 5402, 20: 5146, 30: 4899, 40: 4655},
                    labelled_x=[0, 2, 4, 6], labelled_y=[0, 10, 20, 30, 40],
                    exclude=[(1800, 5260, 2380, 5645), (692, 4660, 1260, 4880), (1060, 5230, 1600, 5330)],
                    refs={"rigid_connection": ((0.10, 0.59), (12.0, 29.5)),
                          "no_connection": ((0.10, 1.37), (2.0, 20.5))}),
    "fig9_B6": dict(page="p600-5.png", figure="Fig. 9 (B6 after sustained load)", journal_page=94, pdf_page=5,
                    x_label="midspan deflection (in)", y_label="end reaction (kips)",
                    vx={0: 707, 1: 951, 2: 1193, 3: 1436, 4: 1680, 5: 1925, 6: 2166, 7: 2400},
                    hy={0: 5672, 10: 5429, 20: 5185, 30: 4941, 40: 4702},
                    labelled_x=[0, 2, 4, 6], labelled_y=[0, 10, 20, 30, 40],
                    exclude=[(1800, 5250, 2405, 5680), (712, 4705, 1330, 4930), (1060, 5260, 1660, 5360)],
                    refs={"rigid_connection": ((0.10, 0.62), (12.0, 29.5)),
                          "no_connection": ((0.10, 1.45), (2.0, 20.5))}),
    "fig8_B6": dict(page="p600-5.png", figure="Fig. 8 (B6 time-deflection)", journal_page=94, pdf_page=5,
                    x_label="time (days)", y_label="midspan deflection (in)",
                    vx={0: 747, 30: 993, 60: 1236, 90: 1479, 120: 1726, 150: 1971, 180: 2216, 210: 2459},
                    hy={0.0: 1737, 0.25: 1488, 0.5: 1246, 0.75: 1002, 1.0: 756, 1.25: 514},
                    labelled_x=[0, 60, 120, 180], labelled_y=[0.0, 0.5, 1.0],
                    exclude=[(1700, 1150, 2460, 1745)], refs={}),
}

PUSHOUT = {
    "fig7_B2": dict(page="p600-4.png", figure="Fig. 7, top panel (B2)", journal_page=93, pdf_page=4,
                    vx={0.0: 2952, 0.05: 3196, 0.1: 3438, 0.15: 3688, 0.2: 3933, 0.25: 4178, 0.3: 4420, 0.35: 4671},
                    hy={0: 4253, 10: 4006, 20: 3758, 30: 3507, 40: 3251},
                    labelled_x=[0.0, 0.1, 0.2, 0.3], labelled_y=[0, 10, 20, 30, 40]),
    "fig7_B4": dict(page="p600-4.png", figure="Fig. 7, bottom panel (B4)", journal_page=93, pdf_page=4,
                    vx={0.0: 2921, 0.05: 3165, 0.1: 3410, 0.15: 3658, 0.2: 3905, 0.25: 4151, 0.3: 4395, 0.35: 4634},
                    hy={0: 5636, 10: 5394, 20: 5149, 30: 4904, 40: 4650},
                    labelled_x=[0.0, 0.1, 0.2, 0.3], labelled_y=[0, 10, 20, 30, 40]),
}

_ink_cache = {}


def load_ink(page):
    if page not in _ink_cache:
        _ink_cache[page] = np.array(Image.open(SRC / page).convert("L")) < 128
    return _ink_cache[page]


def runs_1d(a):
    d = np.diff(np.concatenate([[0], a.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def fit_line(ink, nominal, along, vertical, half=30, tmin=3, tmax=16):
    pts = []
    for t in along:
        prof = ink[t, nominal - half:nominal + half + 1] if vertical else ink[nominal - half:nominal + half + 1, t]
        best = None
        for s, e in runs_1d(prof):
            if tmin <= e - s + 1 <= tmax:
                c = nominal - half + (s + e) / 2.0
                if best is None or abs(c - nominal) < abs(best - nominal):
                    best = c
        if best is not None:
            pts.append((t, best))
    pts = np.array(pts, float)
    keep = np.ones(len(pts), bool)
    for _ in range(6):
        A = np.c_[np.ones(keep.sum()), pts[keep, 0]]
        coef, *_ = np.linalg.lstsq(A, pts[keep, 1], rcond=None)
        res = pts[:, 1] - (coef[0] + coef[1] * pts[:, 0])
        mad = np.median(np.abs(res[keep])) + 0.5
        keep = np.abs(res) < 3.0 * mad
    return dict(c0=float(coef[0]), c1=float(coef[1]), n_used=int(keep.sum()), n_samples=int(len(pts)),
                rms_px=float(np.sqrt(np.mean(res[keep] ** 2))))


def calibrate(ink, P):
    xs = sorted(P["vx"].items())
    ys = sorted(P["hy"].items())
    y_lo = min(v for _, v in ys); y_hi = max(v for _, v in ys)
    x_lo = min(v for _, v in xs); x_hi = max(v for _, v in xs)
    V = []
    for val, x in xs:
        f = fit_line(ink, x, range(y_lo + 20, y_hi - 20, 3), vertical=True)
        f.update(value=float(val), nominal_px=x, labelled=val in P["labelled_x"])
        V.append(f)
    H = []
    for val, y in ys:
        f = fit_line(ink, y, range(x_lo + 20, x_hi - 20, 3), vertical=False)
        f.update(value=float(val), nominal_px=y, labelled=val in P["labelled_y"])
        H.append(f)
    inter = []
    for v in V:
        for h in H:
            a, b, c, d = v["c0"], v["c1"], h["c0"], h["c1"]
            yy = (c + d * a) / (1 - d * b)
            inter.append((a + b * yy, yy, v["value"], h["value"]))
    inter = np.array(inter)
    A = np.c_[np.ones(len(inter)), inter[:, 0], inter[:, 1]]
    cu, *_ = np.linalg.lstsq(A, inter[:, 2], rcond=None)
    cv, *_ = np.linalg.lstsq(A, inter[:, 3], rcond=None)
    ru = inter[:, 2] - A @ cu
    rv = inter[:, 3] - A @ cv
    return dict(vertical_lines=V, horizontal_lines=H, affine_x=cu.tolist(), affine_y=cv.tolist(),
                affine_resid_x_max=float(np.abs(ru).max()), affine_resid_y_max=float(np.abs(rv).max()),
                affine_resid_x_rms=float(np.sqrt(np.mean(ru ** 2))), affine_resid_y_rms=float(np.sqrt(np.mean(rv ** 2))))


def _interp_extrap(p, P_pos, vals):
    """Piecewise-linear map from pixel position to value; linear extrapolation
    with the end interval outside the grid."""
    order = np.argsort(P_pos)
    P_pos = np.asarray(P_pos)[order]; vals = np.asarray(vals)[order]
    if p < P_pos[0]:
        return float(vals[0] + (p - P_pos[0]) * (vals[1] - vals[0]) / (P_pos[1] - P_pos[0]))
    if p > P_pos[-1]:
        return float(vals[-1] + (p - P_pos[-1]) * (vals[-1] - vals[-2]) / (P_pos[-1] - P_pos[-2]))
    return float(np.interp(p, P_pos, vals))


def to_data(cal, px, py):
    V, H = cal["vertical_lines"], cal["horizontal_lines"]
    X = [v["c0"] + v["c1"] * py for v in V]
    Y = [h["c0"] + h["c1"] * px for h in H]
    u = _interp_extrap(px, X, [v["value"] for v in V])
    w = _interp_extrap(py, Y, [h["value"] for h in H])
    cu, cv = cal["affine_x"], cal["affine_y"]
    return u, w, cu[0] + cu[1] * px + cu[2] * py, cv[0] + cv[1] * px + cv[2] * py


def to_pixel(cal, u, w):
    """Inverse of the local map (fixed-point iteration)."""
    V, H = cal["vertical_lines"], cal["horizontal_lines"]
    vv = [v["value"] for v in V]
    hv = [h["value"] for h in H]
    px = _interp_extrap(u, vv, [v["c0"] for v in V])
    py = _interp_extrap(w, hv, [h["c0"] for h in H])
    for _ in range(10):
        X = [v["c0"] + v["c1"] * py for v in V]
        Y = [h["c0"] + h["c1"] * px for h in H]
        px = _interp_extrap(u, vv, X)
        py = _interp_extrap(w, hv, Y)
    return float(px), float(py)


def disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def panel_box(cal, pad_l=6, pad_r=40, pad_t=40, pad_b=30):
    V, H = cal["vertical_lines"], cal["horizontal_lines"]
    xs = [v["c0"] + v["c1"] * h["c0"] for v in V for h in H]
    ys = [h["c0"] + h["c1"] * v["c0"] for v in V for h in H]
    return int(min(xs)) + pad_l, int(min(ys)) - pad_t, int(max(xs)) + pad_r, int(max(ys)) + pad_b


def find_markers(ink, P, cal):
    x0, y0, x1, y1 = panel_box(cal)
    sub = ink[y0:y1, x0:x1].copy()
    for (ax0, ay0, ax1, ay1) in P["exclude"]:
        sub[max(ay0 - y0, 0):max(ay1 - y0, 0), max(ax0 - x0, 0):max(ax1 - x0, 0)] = False
    opened = ndi.binary_opening(sub, structure=disk(R_OPEN))
    eroded = ndi.binary_erosion(sub, structure=disk(R_ERODE))
    lab, n = ndi.label(opened)
    blobs = []
    for i in range(1, n + 1):
        m = lab == i
        ys, xs = np.nonzero(m)
        blobs.append(dict(area=int(m.sum()), cx=xs.mean(), cy=ys.mean(), w=int(np.ptp(xs)) + 1,
                          h=int(np.ptp(ys)) + 1, mask=m))
    areas = np.array([b["area"] for b in blobs])
    single = float(np.median(areas[(areas > 400) & (areas < 1600)]))
    out, rejected = [], []
    for b in blobs:
        rec = dict(px_open=round(b["cx"] + x0, 1), py_open=round(b["cy"] + y0, 1), area_px=b["area"],
                   blob_w=b["w"], blob_h=b["h"])
        if not (0.65 * single <= b["area"] <= 1.6 * single):
            rec["reject_reason"] = "area %.2f x single marker" % (b["area"] / single)
            rejected.append(rec)
            continue
        core = eroded & ndi.binary_dilation(b["mask"], structure=disk(3))
        ey, ex = np.nonzero(core)
        if len(ex):
            rec.update(px=round(ex.mean() + x0, 1), py=round(ey.mean() + y0, 1), core_px=int(len(ex)))
        else:
            rec.update(px=rec["px_open"], py=rec["py_open"], core_px=0)
        out.append(rec)
    return sorted(out, key=lambda r: (r["px"], -r["py"])), rejected, single


def fit_reference_line(ink, cal, markers, u_range, w_range, band=14):
    """Fit a straight reference line (dashed 'Rigid Connection'/'No Connection')
    from rows between loads w_range, searching near the guess segment from the
    origin to (u_range[1], w_range[1])."""
    x0p, y0p = to_pixel(cal, 0.0, 0.0)
    x1p, y1p = to_pixel(cal, u_range[1], w_range[1])
    mask = ink.copy()
    for m in markers:
        yy, xx = int(m["py"]), int(m["px"])
        mask[yy - 24:yy + 25, xx - 24:xx + 25] = False
    for v in cal["vertical_lines"]:
        pass
    ya = int(to_pixel(cal, 0.0, w_range[0])[1]); yb = int(y1p)
    pts = []
    for it in range(3):
        pts = []
        for y in range(yb, ya, 2):
            xg = x0p + (y - y0p) * (x1p - x0p) / (y1p - y0p)
            lo = int(xg - band)
            prof = mask[y, lo:int(xg + band) + 1]
            best = None
            for s, e in runs_1d(prof):
                if 3 <= e - s + 1 <= 28:
                    c = lo + (s + e) / 2.0
                    if best is None or abs(c - xg) < abs(best - xg):
                        best = c
            if best is not None:
                # skip grid-line crossings
                on_grid = any(abs(best - (v["c0"] + v["c1"] * y)) < 9 for v in cal["vertical_lines"])
                on_hgrid = any(abs(y - (h["c0"] + h["c1"] * best)) < 9 for h in cal["horizontal_lines"])
                if not on_grid and not on_hgrid:
                    pts.append((best, y))
        P = np.array(pts)
        coef = np.polyfit(P[:, 1], P[:, 0], 1)
        res = P[:, 0] - np.polyval(coef, P[:, 1])
        keep = np.abs(res) < max(3.0, 3 * np.median(np.abs(res)))
        coef = np.polyfit(P[keep, 1], P[keep, 0], 1)
        # re-guess segment from fit
        x0p = np.polyval(coef, y0p); x1p = np.polyval(coef, y1p)
        band = 8
    data = np.array([to_data(cal, x, y)[:2] for x, y in P[keep]])
    k_origin = float((data[:, 0] * data[:, 1]).sum() / (data[:, 0] ** 2).sum())
    kfree = np.polyfit(data[:, 0], data[:, 1], 1)
    return dict(n_points=int(keep.sum()), slope_through_origin=k_origin,
                slope_free=float(kfree[0]), intercept_free=float(kfree[1]),
                pixel_fit_x_of_y=[float(c) for c in coef],
                u_at_line_origin=float(to_data(cal, np.polyval(coef, y0p), y0p)[0]),
                sample_points=[[round(a, 4), round(b, 3)] for a, b in data[::max(1, len(data) // 12)]])


def trace_pushout(ink, cal, loads, slips):
    """Trace the solid push-out curve: by rows at the listed loads (steep part) and by
    columns at the listed slips (flat part). The nearest run of plausible thickness to
    a running guess is taken; markers ('+', triangles, dots) are removed first by an
    opening (large blobs) and by rejecting runs thicker than the line."""
    blobs = ndi.binary_opening(ink, structure=disk(8))
    blobs = ndi.binary_dilation(blobs, structure=disk(4))
    m = ink & ~blobs
    res = []
    for w in loads:
        _, y = to_pixel(cal, 0.0, w)
        y = int(round(y))
        x_lo, _ = to_pixel(cal, 0.0, w)
        x_hi, _ = to_pixel(cal, 0.12, w)
        cands = []
        for yy in (y - 1, y, y + 1):
            prof = m[yy, int(x_lo) + 8:int(x_hi)]
            for s, e in runs_1d(prof):
                L = e - s + 1
                if 5 <= L <= 60:
                    cands.append((int(x_lo) + 8 + (s + e) / 2.0, int(L)))
        res.append(dict(mode="row", load_kip=w, candidates_px=[[round(float(c), 1), L] for c, L in cands], y_px=int(y)))
    for sl in slips:
        x, _ = to_pixel(cal, sl, 20.0)
        x = int(round(x))
        cands = []
        _, y_top = to_pixel(cal, sl, 30.0)
        _, y_bot = to_pixel(cal, sl, 12.0)
        for xx in (x - 1, x, x + 1):
            prof = m[int(y_top):int(y_bot), xx]
            for s, e in runs_1d(prof):
                L = e - s + 1
                if 5 <= L <= 20:
                    cands.append((int(y_top) + (s + e) / 2.0, int(L)))
        res.append(dict(mode="col", slip_in=sl, candidates_px=[[round(float(c), 1), L] for c, L in cands], x_px=int(x)))
    return res


def overlay(page, cal, markers, name, extra_pts=(), refs=None):
    im = Image.open(SRC / page).convert("RGB")
    x0, y0, x1, y1 = panel_box(cal, 60, 60, 60, 60)
    im = im.crop((x0, y0, x1, y1))
    d = ImageDraw.Draw(im)
    for v in cal["vertical_lines"]:
        d.line([(v["c0"] + v["c1"] * y0 - x0, 0), (v["c0"] + v["c1"] * y1 - x0, y1 - y0)], fill=(0, 160, 255), width=1)
    for h in cal["horizontal_lines"]:
        d.line([(0, h["c0"] + h["c1"] * x0 - y0), (x1 - x0, h["c0"] + h["c1"] * x1 - y0)], fill=(0, 160, 255), width=1)
    for mk in markers:
        cx, cy = mk["px"] - x0, mk["py"] - y0
        d.ellipse([cx - 20, cy - 20, cx + 20, cy + 20], outline=(255, 0, 0), width=3)
        d.line([(cx - 6, cy), (cx + 6, cy)], fill=(255, 0, 0), width=2)
        d.line([(cx, cy - 6), (cx, cy + 6)], fill=(255, 0, 0), width=2)
    for rname, r in (refs or {}).items():
        if "pixel_fit_x_of_y" not in r:
            continue
        b1, b0 = r["pixel_fit_x_of_y"]
        _, ytop = to_pixel(cal, 0.0, 32.0)
        _, ybot = to_pixel(cal, 0.0, 0.0)
        d.line([(b1 * ytop + b0 - x0, ytop - y0), (b1 * ybot + b0 - x0, ybot - y0)], fill=(0, 200, 0), width=2)
    for (px, py, col) in extra_pts:
        d.ellipse([px - x0 - 6, py - y0 - 6, px - x0 + 6, py - y0 + 6], fill=col)
    im.save(OVERLAY_DIR / f"overlay_{name}.png")


def main():
    summary = {}
    for name, P in PANELS.items():
        ink = load_ink(P["page"])
        cal = calibrate(ink, P)
        markers, rejected, single = find_markers(ink, P, cal)
        rows = []
        for mk in markers:
            u, w, ua, wa = to_data(cal, mk["px"], mk["py"])
            uo, wo, _, _ = to_data(cal, mk["px_open"], mk["py_open"])
            rows.append(dict(px=mk["px"], py=mk["py"], px_open=mk["px_open"], py_open=mk["py_open"],
                             area_px=mk["area_px"], core_px=mk["core_px"],
                             x_local=round(u, 4), y_local=round(w, 3),
                             x_affine=round(ua, 4), y_affine=round(wa, 3),
                             x_open_centroid=round(uo, 4), y_open_centroid=round(wo, 3)))
        with open(OUT / f"{name}_markers.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader(); wr.writerows(rows)
        refs = {}
        for rname, (u_rng, w_rng) in P["refs"].items():
            try:
                refs[rname] = fit_reference_line(ink, cal, markers, u_rng, w_rng)
            except Exception as exc:  # pragma: no cover
                import traceback; traceback.print_exc()
                refs[rname] = dict(error=str(exc))
        meta = dict(panel=name, figure=P["figure"], journal_page=P["journal_page"], pdf_page=P["pdf_page"],
                    render="pdftoppm -r 600 -png -gray (native 600 ppi CCITT scan)",
                    x_axis=P["x_label"], y_axis=P["y_label"], labelled_x=P["labelled_x"], labelled_y=P["labelled_y"],
                    R_OPEN=R_OPEN, R_ERODE=R_ERODE, single_marker_area_px=single,
                    rejected_blobs=rejected, reference_lines=refs, **cal)
        with open(OUT / f"{name}_calibration.json", "w") as fh:
            json.dump(meta, fh, indent=1)
        overlay(P["page"], cal, markers, name, refs=refs)
        summary[name] = dict(n=len(rows), refs={k: (round(v.get("slope_through_origin", float("nan")), 2),
                                                    round(v.get("slope_free", float("nan")), 2),
                                                    round(v.get("intercept_free", float("nan")), 2),
                                                    round(v.get("u_at_line_origin", float("nan")), 4),
                                                    v.get("n_points")) for k, v in refs.items()},
                             affine_resid=(round(cal["affine_resid_x_max"], 4), round(cal["affine_resid_y_max"], 3)))
        print(name, summary[name])
        for r in rows:
            print("   %8.1f %8.1f  x=%.4f y=%.3f | affine %.4f %.3f | open %.4f %.3f | area %d core %d" % (
                r["px"], r["py"], r["x_local"], r["y_local"], r["x_affine"], r["y_affine"],
                r["x_open_centroid"], r["y_open_centroid"], r["area_px"], r["core_px"]))
        if rejected:
            print("   rejected:", rejected)
    for name, P in PUSHOUT.items():
        ink = load_ink(P["page"])
        cal = calibrate(ink, P)
        tr = trace_pushout(ink, cal, loads=[2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5],
                           slips=[0.05, 0.075, 0.1, 0.15, 0.2, 0.25])
        meta = dict(panel=name, figure=P["figure"], journal_page=P["journal_page"], pdf_page=P["pdf_page"],
                    x_axis="slip (in)", y_axis="load per stud (kips)", trace=tr, **cal)
        with open(OUT / f"{name}_calibration.json", "w") as fh:
            json.dump(meta, fh, indent=1)
        print(name)
        for t in tr:
            print("  ", t["mode"], t.get("load_kip", t.get("slip_in")), t["candidates_px"])


if __name__ == "__main__":
    main()
