"""Digitise first-loading (virgin) load versus centreline deflection curves of
the Lehigh composite beams B3-B12 (Slutter and Driscoll 1963 programme).

Sources (open access, Lehigh Preserve Fritz Laboratory Reports, copies via the
Wayback Machine; PDFs and page renders kept in the session scratchpad, NOT in
the repository):
  * Culver, Zarzeczny and Driscoll (June 1960), "Tests of composite beams for
    buildings", Progress Report 1, Fritz Lab. Report 279.2 (B1-B6):
    Fig. 9 (PDF p. 79) B3-S1, Fig. 12 (p. 82) B4-S1, Fig. 15 (p. 85) B5-S1,
    Fig. 18 (p. 88) B6-S1.
  * Culver, Zarzeczny and Driscoll (Jan. 1961), Progress Report 2, Fritz Lab.
    Report 279.6 (B7-B13): Fig. 10 (p. 73) B7-S1, Fig. 12 (p. 75) B8-S1,
    Fig. 15 (p. 78) B9-S1, Fig. 18 (p. 81) B10, Fig. 19 (p. 82) B11,
    Fig. 20 (p. 83) B12.

Scans are 300 ppi bilevel JBIG2 (pdfimages -list); pages were rendered at
600 dpi (pdftoppm -r 600), i.e. 2x.

Method (routines in lehigh_plot_core.py):
  1. The plot frame is found from rows/columns with > 35 % dark pixels; the
     outer edges of the bottom and left frame strokes are fitted as straight
     lines (the scans are slightly rotated) and the origin is their
     intersection (both axes start at zero on the frame in these plots).
  2. Tick marks rising from the bottom frame and running from the left frame
     are located programmatically; tick index k = round(distance / median
     spacing); pixel/value scale from a least-squares fit of distance on k
     with the value per tick given in FIGS (checked against the printed
     labels on the overlay).
  3. Open-circle markers are found by ring-template correlation (dark
     annulus, white centre).
  4. Branch labelling: in these tests the load was raised in increments to
     about Pp/1.85, cycled 10 times between 0 and that load, and then
     increased (279.2 p. 15, 279.6 p. 10). Markers with load <= the cycle
     load + 1.5 kip are grouped by load level (1.5 kip tolerance); within a
     level, the marker with the smallest deflection is the first-loading
     (virgin) reading and the others are post-cycle readings. Zero-load
     markers with deflection > 0.01 in are residual sets. Markers above the
     cycle load are post-cycle loading (context only).
Outputs: calibration.json, markers_raw.csv, overlay_check.png (zoom on the
first-loading region, calibrated grid drawn), overlay_full.png (wider view).
Units: kips and inches, as printed.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import lehigh_plot_core as core

Image.MAX_IMAGE_PIXELS = None
HERE = Path(__file__).resolve().parent
R600 = Path("/private/tmp/claude-501/-Users-sandeshlamsal-Desktop-CompositeGirder/"
            "a32c3e10-8884-4339-8200-ea3611516154/scratchpad/lit_sources/slutter1963/r600")

# dx, dy = value per tick (in, kip); cycle = load cycled 10 times (kips, read
# from the figure annotation); roi_defl = max deflection searched for markers;
# rad = marker template radii (outer, inner, void) in render px (279.6 B7-B9
# plots use smaller circles)
FIGS = [
    dict(key="B3-S1", report="279.2", fig="9", page=79, png="r279_2_p-079.png", dx=0.2, dy=10.0, cycle=40.0, roi_defl=1.3),
    dict(key="B4-S1", report="279.2", fig="12", page=82, png="r279_2_p-082.png", dx=0.2, dy=10.0, cycle=30.0, roi_defl=1.3),
    dict(key="B5-S1", report="279.2", fig="15", page=85, png="r279_2_p-085.png", dx=0.2, dy=10.0, cycle=40.0, roi_defl=1.3),
    dict(key="B6-S1", report="279.2", fig="18", page=88, png="r279_2_p-088.png", dx=0.2, dy=10.0, cycle=40.0, roi_defl=1.3),
    dict(key="B7-S1", report="279.6", fig="10", page=73, png="r279_6_p-073.png", dx=0.2, dy=10.0, cycle=40.0, roi_defl=1.3, rad=(25.0, 15.0, 10.0)),
    dict(key="B8-S1", report="279.6", fig="12", page=75, png="r279_6_p-075.png", dx=0.2, dy=10.0, cycle=40.0, roi_defl=1.3, rad=(25.0, 15.0, 10.0)),
    dict(key="B9-S1", report="279.6", fig="15", page=78, png="r279_6_p-078.png", dx=0.2, dy=10.0, cycle=40.0, roi_defl=1.3, rad=(25.0, 15.0, 10.0)),
    dict(key="B10", report="279.6", fig="18", page=81, png="r279_6_p-081.png", dx=1.0, dy=20.0, cycle=60.0, roi_defl=1.3),
    dict(key="B11", report="279.6", fig="19", page=82, png="r279_6_p-082.png", dx=1.0, dy=20.0, cycle=60.0, roi_defl=1.3),
    dict(key="B12", report="279.6", fig="20", page=83, png="r279_6_p-083.png", dx=1.0, dy=20.0, cycle=60.0, roi_defl=1.3),
]

R_OUT, R_IN, R_VOID = 31.0, 20.0, 15.0
# approximate tick spacing (render px) per value-per-tick, used only to reject
# spurious tick candidates (text, data lines touching the frame)
S0 = {0.2: 405.0, 10.0: 455.0, 1.0: 1012.0, 20.0: 608.0}


def lattice_filter(cands, origin, s0, tol=0.12):
    """Keep tick candidates lying on a lattice of spacing ~s0 from the origin;
    one per lattice index (the longest tick)."""
    best = {}
    for c in cands:
        u = abs(c[0] - origin)
        k = int(round(u / s0))
        if k == 0 or abs(u / s0 - k) > tol:
            continue
        if k not in best or c[1] > best[k][1]:
            best[k] = c
    return [best[k] for k in sorted(best)]


def dedupe(rings, sep=25.0):
    out = []
    for r in sorted(rings, key=lambda r: -r["ring"]):
        if all(np.hypot(r["x"] - o["x"], r["y"] - o["y"]) > sep for o in out):
            out.append(r)
    return out


def groups(ix, gap=5):
    if ix.size == 0:
        return []
    g = np.split(ix, np.where(np.diff(ix) > gap)[0] + 1)
    return [(int(x[0]), int(x[-1])) for x in g]


def process(F, diag=False):
    b = core.load_binary(R600 / F["png"])
    H, W = b.shape
    rows = groups(np.where(b.mean(1) > 0.35)[0])
    cols = groups(np.where(b.mean(0) > 0.35)[0])
    xleft = min(c for c in cols if c[1] < W * 0.3)
    xright = max(c for c in cols if c[0] > W * 0.7)
    # vertical extent of the left frame stroke = frame top and bottom
    colv = b[:, (xleft[0] + xleft[1]) // 2]
    runs = core._runs_1d(colv)
    s_, L_ = max(runs, key=lambda r: r[1])
    ybot = min((r for r in rows if r[0] > H * 0.5), key=lambda r: abs((r[0] + r[1]) / 2 - (s_ + L_)))
    ytop = min((r for r in rows if r[1] < H * 0.5), key=lambda r: abs((r[0] + r[1]) / 2 - s_))
    yg = (ybot[0] + ybot[1]) // 2
    xg = (xleft[0] + xleft[1]) // 2
    bottom = core.fit_bottom_axis(b, yg, xleft[1] + 60, xright[0] - 60, band=18)
    left = core.fit_left_axis(b, xg, ytop[1] + 60, ybot[0] - 60, band=18)
    xt = core.find_ticks_bottom(b, bottom, xleft[1] + 40, xright[0] - 40, bottom[2])
    yt = core.find_ticks_left(b, left, ytop[1] + 40, ybot[0] - 40, left[2])
    xt = lattice_filter(xt, (xleft[0] + xleft[1]) / 2, S0[F["dx"]])
    yt = lattice_filter(yt, (ybot[0] + ybot[1]) / 2, S0[F["dy"]])
    cal = core.Calib(bottom, left, [t[0] for t in xt], [t[0] for t in yt], F["dx"], F["dy"])
    # marker ROI: from the origin to roi_defl, full load height
    px_per_in = cal.sx / F["dx"]
    x2 = int(min(cal.O[0] + F["roi_defl"] * px_per_in, xright[0] - 30))
    roi = (int(cal.O[0]) - 40, ytop[1] + 30, x2, int(cal.O[1]) + 40)
    ro, ri, rv = F.get("rad", (R_OUT, R_IN, R_VOID))
    rings = dedupe(core.detect_rings(b, roi, ro, ri, rv, thr=0.75, void_max=0.10, min_sep=int(ro)), sep=1.3 * ro)
    pts = []
    for r in rings:
        d, P = cal.to_data(r["x"], r["y"])
        pts.append(dict(px_x=r["x"], px_y=r["y"], ring=r["ring"], void=r["void"],
                        deflection_in=d, load_kip=P))
    pts.sort(key=lambda p: (p["load_kip"], p["deflection_in"]))
    # branch labelling
    lv = F["cycle"] + 1.5
    for p in pts:
        p["branch"] = "post_cycle_loading" if p["load_kip"] > lv else None
    below = [p for p in pts if p["branch"] is None]
    below.sort(key=lambda p: p["load_kip"])
    levels = []
    for p in below:
        if levels and abs(p["load_kip"] - levels[-1][-1]["load_kip"]) <= 1.5:
            levels[-1].append(p)
        else:
            levels.append([p])
    for L in levels:
        L.sort(key=lambda p: p["deflection_in"])
        for i, p in enumerate(L):
            if abs(p["load_kip"]) < 1.0:
                p["branch"] = "origin" if p["deflection_in"] < 0.01 else "residual_after_unloading"
            else:
                p["branch"] = "first_loading" if i == 0 else "post_cycle_same_load"
    return dict(cal=cal, pts=pts, roi=roi, xt=xt, yt=yt, frame=(xleft, xright, ytop, ybot), b=b)


def overlay(F, res, scale=0.22):
    b, cal = res["b"], res["cal"]
    x0, y0, x1, y1 = res["roi"]
    x0 -= 250; y1 += 260; x1 += 150; y0 -= 60
    img = Image.fromarray(np.where(b[y0:y1, x0:x1], 0, 255).astype(np.uint8)).convert("RGB")
    dr = ImageDraw.Draw(img)
    for (xc, _, _) in res["xt"]:
        k = round((cal._uv(np.array([xc, cal.yb(xc)]))[0] - cal.ax) / cal.sx)
        X = xc - x0
        dr.line([(X, img.height - 250), (X, img.height - 140)], fill=(0, 90, 255), width=8)
        dr.text((X - 20, img.height - 130), f"{k * F['dx']:.1f}", fill=(0, 60, 220))
    for (yc, _, _) in res["yt"]:
        k = round((cal._uv(np.array([cal.xl(yc), yc]))[1] - cal.ay) / cal.sy)
        Y = yc - y0
        dr.line([(60, Y), (200, Y)], fill=(0, 90, 255), width=8)
        dr.text((5, Y - 10), f"{k * F['dy']:.0f}", fill=(0, 60, 220))
    colours = {"first_loading": (230, 0, 0), "origin": (230, 0, 0), "post_cycle_same_load": (0, 160, 0),
               "residual_after_unloading": (0, 160, 0), "post_cycle_loading": (255, 140, 0)}
    for p in res["pts"]:
        X, Y = p["px_x"] - x0, p["px_y"] - y0
        c = colours[p["branch"]]
        dr.ellipse([X - 40, Y - 40, X + 40, Y + 40], outline=c, width=10)
    img = img.resize((int(img.width * scale), int(img.height * scale)))
    d2 = ImageDraw.Draw(img)
    d2.text((60, 5), f"{F['key']} ({F['report']} Fig. {F['fig']}, PDF p. {F['page']})", fill=(200, 0, 120))
    for p in res["pts"]:
        if p["branch"] in ("first_loading",):
            X, Y = (p["px_x"] - x0) * scale, (p["px_y"] - y0) * scale
            d2.text((X + 10, Y - 4), f"{p['load_kip']:.1f}/{p['deflection_in']:.3f}", fill=(230, 0, 0))
    return img


def overlay_zoom(F, res, dmax=0.6, target_h=700):
    """Crop of the first-loading region with calibrated tick positions and
    every accepted marker (red = first loading, green = post-cycle/residual,
    orange = above the cycle load), labelled load/deflection."""
    b, cal = res["b"], res["cal"]
    pmax = F["cycle"] + 12.0
    ex, ey = cal.ex, cal.ey
    def px(dv, pv):
        u = cal.ax + dv / F["dx"] * cal.sx
        v = cal.ay + pv / F["dy"] * cal.sy
        return cal.O + u * ex + v * ey
    x0, y1 = px(-0.06, -4.0); x1, y0 = px(dmax, pmax)
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    img = Image.fromarray(np.where(b[y0:y1, x0:x1], 0, 255).astype(np.uint8)).convert("RGB")
    dr = ImageDraw.Draw(img)
    sc = target_h / img.height
    lw = max(2, int(3 / sc))
    k = 0
    while True:
        dv = k * F["dx"] / (5 if F["dx"] == 1.0 else 1)
        if dv > dmax:
            break
        X = px(dv, 0.0)[0] - x0
        dr.line([(X, 0), (X, img.height)], fill=(150, 190, 255), width=lw)
        k += 1
    k = 0
    while True:
        pv = k * F["dy"] / (2 if F["dy"] == 20.0 else 1)
        if pv > pmax:
            break
        Y = px(0.0, pv)[1] - y0
        dr.line([(0, Y), (img.width, Y)], fill=(150, 190, 255), width=lw)
        k += 1
    colours = {"first_loading": (230, 0, 0), "origin": (230, 0, 0), "post_cycle_same_load": (0, 160, 0),
               "residual_after_unloading": (0, 160, 0), "post_cycle_loading": (255, 140, 0)}
    r = F.get("rad", (R_OUT,))[0] + 8
    for p in res["pts"]:
        X, Y = p["px_x"] - x0, p["px_y"] - y0
        if 0 <= X < img.width and 0 <= Y < img.height:
            dr.ellipse([X - r, Y - r, X + r, Y + r], outline=colours[p["branch"]], width=lw)
    img = img.resize((int(img.width * sc), int(img.height * sc)))
    d2 = ImageDraw.Draw(img)
    d2.text((5, 5), f"{F['key']} ({F['report']} Fig. {F['fig']}, PDF p. {F['page']}); grid = calibrated "
                    f"{F['dx'] / (5 if F['dx'] == 1.0 else 1):g} in / {F['dy'] / (2 if F['dy'] == 20.0 else 1):g} kip", fill=(200, 0, 120))
    for p in res["pts"]:
        X, Y = (p["px_x"] - x0) * sc, (p["px_y"] - y0) * sc
        if 0 <= X < img.width and 0 <= Y < img.height:
            d2.text((X + 12, Y - 5), f"{p['load_kip']:.1f}k {p['deflection_in']:.3f}", fill=colours[p["branch"]])
    return img


# push-out load-slip curves: dx = 0.02 in per tick on the slip axis (labels
# 1.0, 2.0 x 10^-1 in); dy = kips per tick on the "load per connector" axis
POFIGS = [
    dict(key="P1", report="279.2", fig="28", page=100, png="r279_2_p-100.png", dx=0.02, dy=2.0, s0y=410.0,
         kaxis=0, fleft=1, connector="1/2 in L-studs, 2.25 in high (same material as B3, B4, B6)", excl=(0.205, 0.40, -5, 99)),
    dict(key="P2", report="279.2", fig="29", page=101, png="r279_2_p-101.png", dx=0.02, dy=10.0, s0y=605.0,
         kaxis=0, fleft=0, connector="3U4.1 channels, 4 in long (as B5)", excl=(0.19, 0.40, -5, 99)),
    dict(key="P3", report="279.2", fig="30", page=102, png="r279_2_p-102.png", dx=0.02, dy=2.0, s0y=182.0,
         kaxis=0, fleft=1, connector="3/4 in headed studs", excl=(0.225, 0.40, -5, 99)),
    dict(key="P4", report="279.2", fig="31", page=103, png="r279_2_p-103.png", dx=0.02, dy=1.0, s0y=203.0,
         kaxis=0, fleft=1, connector="1/2 in L-studs, 2.25 in high (same material as B3, B4, B6)", excl=(0.20, 0.40, -5, 99)),
    dict(key="P8", report="279.6", fig="37", page=102, png="r279_6_p-102.png", dx=0.02, dy=2.0, s0y=410.0,
         kaxis=0, fleft=2, connector="1/2 in headed studs, 3 in high (as B8)", rad=(25.0, 15.0, 10.0),
         excl=(0.165, 0.262, -5, 10.5)),
]


def process_pushout(F):
    b = core.load_binary(R600 / F["png"])
    H, W = b.shape
    rows = groups(np.where(b.mean(1) > 0.35)[0])
    cols = groups(np.where(b.mean(0) > 0.35)[0])
    kax, fle, fri = cols[F["kaxis"]], cols[F["fleft"]], cols[-1]
    ybot, ytop = rows[-1], rows[0]
    bottom = core.fit_bottom_axis(b, (ybot[0] + ybot[1]) // 2, fle[1] + 60, fri[0] - 60, band=18)
    left = core.fit_left_axis(b, (fle[0] + fle[1]) // 2, ytop[1] + 60, ybot[0] - 60, band=18)
    kleft = core.fit_left_axis(b, (kax[0] + kax[1]) // 2, ytop[1] + 60, ybot[0] - 60, band=18)
    xt = core.find_ticks_bottom(b, bottom, fle[1] + 40, fri[0] - 40, bottom[2], min_len=20)
    yt = core.find_ticks_left(b, kleft, ytop[0] - 400, ybot[0] - 20, kleft[2], min_len=20)
    xt = lattice_filter(xt, (fle[0] + fle[1]) / 2, 303.0)
    yt = lattice_filter(yt, (ybot[0] + ybot[1]) / 2, F["s0y"])
    cal = core.Calib(bottom, left, [t[0] for t in xt], [t[0] for t in yt], F["dx"], F["dy"])
    roi = (int(cal.O[0]) - 40, ytop[0] - 300, fri[0] - 30, int(cal.O[1]) + 40)
    ro, ri, rv = F.get("rad", (R_OUT, R_IN, R_VOID))
    rings = dedupe(core.detect_rings(b, roi, ro, ri, rv, thr=0.75, void_max=0.10, min_sep=int(ro)), sep=1.3 * ro)
    pts = []
    for r in rings:
        dsl, P = cal.to_data(r["x"], r["y"])
        e = F["excl"]  # (slip_min, slip_max, load_min, load_max) of the specimen sketch/label inset
        if e[0] <= dsl <= e[1] and e[2] <= P <= e[3]:
            continue
        pts.append(dict(px_x=r["x"], px_y=r["y"], ring=r["ring"], void=r["void"], slip_in=dsl, load_kip=P))
    # upper envelope (backbone): sorted by slip, keep a marker if its load
    # exceeds every load at smaller slip by more than 0.15 kip
    pts.sort(key=lambda p: (p["slip_in"], -p["load_kip"]))
    runmax = -1e9
    for p in pts:
        p["envelope"] = p["load_kip"] > runmax + 0.15
        runmax = max(runmax, p["load_kip"])
    return dict(cal=cal, pts=pts, roi=roi, xt=xt, yt=yt, b=b)


def overlay_pushout(F, res, smax=None, target_h=650):
    b, cal = res["b"], res["cal"]
    pmax = max(p["load_kip"] for p in res["pts"]) * 1.1
    smax = smax or max(p["slip_in"] for p in res["pts"]) * 1.05
    def px(dv, pv):
        u = cal.ax + dv / F["dx"] * cal.sx
        v = cal.ay + pv / F["dy"] * cal.sy
        return cal.O + u * cal.ex + v * cal.ey
    x0, y1 = px(-0.01, -pmax * 0.05); x1, y0 = px(smax, pmax)
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    img = Image.fromarray(np.where(b[y0:y1, x0:x1], 0, 255).astype(np.uint8)).convert("RGB")
    dr = ImageDraw.Draw(img)
    sc = target_h / img.height
    lw = max(2, int(3 / sc))
    for k in range(0, int(smax / 0.02) + 1):
        X = px(k * 0.02, 0.0)[0] - x0
        dr.line([(X, 0), (X, img.height)], fill=(150, 190, 255), width=lw)
    k = 0
    while k * F["dy"] <= pmax:
        Y = px(0.0, k * F["dy"])[1] - y0
        dr.line([(0, Y), (img.width, Y)], fill=(150, 190, 255), width=lw)
        k += 1
    r = F.get("rad", (R_OUT,))[0] + 8
    for p in res["pts"]:
        X, Y = p["px_x"] - x0, p["px_y"] - y0
        c = (230, 0, 0) if p["envelope"] else (0, 160, 0)
        dr.ellipse([X - r, Y - r, X + r, Y + r], outline=c, width=lw)
    img = img.resize((int(img.width * sc), int(img.height * sc)))
    d2 = ImageDraw.Draw(img)
    d2.text((5, 5), f"{F['key']} ({F['report']} Fig. {F['fig']}, PDF p. {F['page']}); grid 0.02 in / {F['dy']:g} kip",
            fill=(200, 0, 120))
    for p in res["pts"]:
        X, Y = (p["px_x"] - x0) * sc, (p["px_y"] - y0) * sc
        if p["envelope"] and 0 <= X < img.width:
            d2.text((X + 10, Y - 12), f"{p['load_kip']:.1f}k {p['slip_in']:.4f}", fill=(230, 0, 0))
    return img


def main(diag=False):
    calib, rows, crops, zooms = {}, [], [], []
    for F in FIGS:
        res = process(F, diag)
        cal = res["cal"]
        calib[F["key"]] = dict(report=F["report"], figure=F["fig"], pdf_page=F["page"], render=F["png"],
                               value_per_tick_x_in=F["dx"], value_per_tick_y_kip=F["dy"],
                               cycle_load_kip=F["cycle"], marker_roi_px=res["roi"],
                               n_x_ticks=len(res["xt"]), n_y_ticks=len(res["yt"]),
                               px_per_in=cal.sx / F["dx"], px_per_kip=cal.sy / F["dy"],
                               frame_corner_in_data_coords=dict(zip(("deflection_in", "load_kip"), cal.to_data(*cal.O))),
                               **cal.summary())
        for p in res["pts"]:
            rows.append(dict(beam_test=F["key"], report=F["report"], figure=F["fig"], pdf_page=F["page"],
                             px_x=round(p["px_x"], 1), px_y=round(p["px_y"], 1),
                             ring_score=round(p["ring"], 3), void_score=round(p["void"], 3),
                             load_kip=round(p["load_kip"], 2), deflection_in=round(p["deflection_in"], 4),
                             branch=p["branch"]))
        crops.append(overlay(F, res))
        zooms.append(overlay_zoom(F, res))
        print(F["key"], "px/in %.1f px/kip %.2f  tick rms px x %.2f y %.2f  nticks %d %d" % (
            cal.sx / F["dx"], cal.sy / F["dy"], cal.rms_x_px, cal.rms_y_px, len(res["xt"]), len(res["yt"])))
        for p in res["pts"]:
            print("   %6.2f k %6.3f in  %-26s ring %.2f void %.2f" % (p["load_kip"], p["deflection_in"], p["branch"], p["ring"], p["void"]))
    # montage 5 x 2
    w = max(c.width for c in crops); h = max(c.height for c in crops)
    mont = Image.new("RGB", (5 * w, 2 * h), "white")
    for i, c in enumerate(crops):
        mont.paste(c, ((i % 5) * w, (i // 5) * h))
    mont.save(HERE / "overlay_full.png")
    w = max(c.width for c in zooms); h = max(c.height for c in zooms)
    mont = Image.new("RGB", (5 * w, 2 * h), "white")
    for i, c in enumerate(zooms):
        mont.paste(c, ((i % 5) * w, (i // 5) * h))
    mont.save(HERE / "overlay_check.png")
    porows, pozooms = [], []
    for F in POFIGS:
        res = process_pushout(F)
        cal = res["cal"]
        calib["pushout_" + F["key"]] = dict(report=F["report"], figure=F["fig"], pdf_page=F["page"], render=F["png"],
                                            connector=F["connector"], value_per_tick_x_in=F["dx"],
                                            value_per_tick_y_kip=F["dy"], n_x_ticks=len(res["xt"]),
                                            n_y_ticks=len(res["yt"]), px_per_in=cal.sx / F["dx"],
                                            px_per_kip=cal.sy / F["dy"], **cal.summary())
        print(F["key"], "px/in %.0f px/kip %.1f rms x %.2f y %.2f nticks %d %d" % (
            cal.sx / F["dx"], cal.sy / F["dy"], cal.rms_x_px, cal.rms_y_px, len(res["xt"]), len(res["yt"])))
        for p in res["pts"]:
            porows.append(dict(specimen=F["key"], report=F["report"], figure=F["fig"], pdf_page=F["page"],
                               px_x=round(p["px_x"], 1), px_y=round(p["px_y"], 1), ring_score=round(p["ring"], 3),
                               load_per_connector_kip=round(p["load_kip"], 2), avg_slip_in=round(p["slip_in"], 5),
                               envelope=p["envelope"]))
            if p["envelope"]:
                print("   %6.2f k %7.4f in" % (p["load_kip"], p["slip_in"]))
        pozooms.append(overlay_pushout(F, res))
    w = max(c.width for c in pozooms); h = max(c.height for c in pozooms)
    mont = Image.new("RGB", (3 * w, 2 * h), "white")
    for i, c in enumerate(pozooms):
        mont.paste(c, ((i % 3) * w, (i // 3) * h))
    mont.save(HERE / "overlay_check_pushout.png")
    with open(HERE / "pushout_markers_raw.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(porows[0].keys()))
        wr.writeheader(); wr.writerows(porows)
    with open(HERE / "calibration.json", "w") as f:
        json.dump(calib, f, indent=1)
    with open(HERE / "markers_raw.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)


if __name__ == "__main__":
    main()
