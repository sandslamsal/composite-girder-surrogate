"""Trace the solid push-out curves of McGarraugh & Baldwin (1971) Fig. 7 (B2 top, B4 bottom),
journal p. 93 (PDF page 4), using the grid calibration of digitise.py.

Method: ink of the 600 dpi render; fitted vertical grid lines are blanked (+/-8 px) for the column
scan (horizontal lines kept, bare grid-line runs rejected, points within 8 px of a horizontal
line tagged col_on_gridline), and all grid lines for the row scan; the curve
(line ~11 px thick) is followed by continuity from its right-hand end: column scan leftwards
(max jump 10 px; runs > 18 px are markers and skipped) until 8 consecutive wide runs at slip < 0.08 in mark the steep branch, then row scan downwards
every 4 px (max jump 12 px), skipping rows where a beam-data marker merges with the curve
(run > 30 px; points more than 0.5 kip below the running maximum load at smaller slip are
dropped as marker edges; the search window then widens by 6 px per skipped row and the curve must move
left going down). Points hidden under markers are therefore absent, not interpolated. Output: fig7_<beam>_pushout_curve.csv (slip in, load per stud kips, mode, pixel).
Overlay written to the scratch source folder (contains the figure).
"""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import digitise as dg  # noqa: E402

OUT = Path(__file__).resolve().parent


def trace(name, P, x_end, guess_end_load):
    ink = dg.load_ink(P["page"])
    cal = dg.calibrate(ink, P)
    m = ink.copy()
    H, W = m.shape
    yy, xx = np.mgrid[0:H, 0:W]
    x0p, y0p = dg.to_pixel(cal, 0.0, 0.0)
    x1p, y1p = dg.to_pixel(cal, 0.35, 40.0)
    sl = (slice(int(y1p) - 20, int(y0p) + 20), slice(int(x0p) - 20, int(x1p) + 20))
    raw = m[sl].copy()
    oy, ox = sl[0].start, sl[1].start
    Y, X = np.mgrid[0:raw.shape[0], 0:raw.shape[1]]
    vmask = np.zeros_like(raw)
    for v in cal["vertical_lines"]:
        vmask |= np.abs(X - (v["c0"] + v["c1"] * (Y + oy) - ox)) <= 8
    hmask = np.zeros_like(raw)
    for h in cal["horizontal_lines"]:
        hmask |= np.abs(Y - (h["c0"] + h["c1"] * (X + ox) - oy)) <= 8
    col_img = raw & ~vmask      # column scan: vertical lines removed, horizontal lines kept
    row_img = raw & ~hmask & ~vmask
    sub = raw

    def hgrid_dist(col, yc):
        return min(abs(yc - (h["c0"] + h["c1"] * (col + ox) - oy)) for h in cal["horizontal_lines"])

    pts = []
    _, prev = dg.to_pixel(cal, x_end, guess_end_load)
    prev -= oy
    last = None
    wide = 0
    for s in np.arange(x_end, 0.0, -0.0025):
        xp, _ = dg.to_pixel(cal, s, 15.0)
        col = int(round(xp)) - ox
        runs = [((a + b) / 2.0, b - a + 1) for a, b in dg.runs_1d(col_img[:, col]) if 5 <= b - a + 1 <= 40]
        runs = [r for r in runs if abs(r[0] - prev) <= 12]
        # a bare grid line (thin run centred on a fitted horizontal line) is not the curve
        cur = [r for r in runs if not (hgrid_dist(col, r[0]) < 4 and r[1] <= 10)]
        if not cur:
            continue
        best = min(cur, key=lambda r: abs(r[0] - prev))
        if best[1] > 18:
            wide += 1
            if wide >= 8 and s < 0.08:
                break
            continue
        wide = 0
        prev = best[0]
        u, w, _, _ = dg.to_data(cal, col + ox, best[0] + oy)
        tag = "col_on_gridline" if hgrid_dist(col, best[0]) < 8 else "col"
        pts.append((u, w, tag, col + ox, best[0] + oy))
        last = (col, best[0])
    prevx = last[0]
    _, y_zero = dg.to_pixel(cal, 0.0, 1.0)
    skipped = 0
    for row in range(int(last[1]) + 1, int(y_zero) - oy, 4):
        win = 12 + 6 * skipped    # widen the window after rows hidden by a marker
        runs = [((a + b) / 2.0, b - a + 1) for a, b in dg.runs_1d(row_img[row, :]) if 5 <= b - a + 1 <= 60]
        runs = [r for r in runs if abs(r[0] - prevx) <= win and r[0] <= prevx + 4]
        if not runs:
            skipped += 1
            continue
        best = min(runs, key=lambda r: abs(r[0] - prevx))
        if best[1] > 30:
            skipped += 1
            continue
        skipped = 0
        prevx = best[0]
        u, w, _, _ = dg.to_data(cal, best[0] + ox, row + oy)
        pts.append((u, w, "row", best[0] + ox, row + oy))
    pts.sort(key=lambda p: p[0])
    # drop marker-edge points that break monotonicity (load > 0.5 kip below the running maximum)
    kept, runmax = [], -1.0
    for p in pts:
        if p[1] < runmax - 0.5:
            continue
        kept.append(p)
        runmax = max(runmax, p[1])
    pts = kept
    with open(OUT / f"{name}_pushout_curve.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["slip_in", "load_per_stud_kip", "mode", "px", "py"])
        for p in pts:
            wr.writerow([round(p[0], 4), round(p[1], 3), p[2], round(p[3], 1), round(p[4], 1)])
    im = Image.open(dg.SRC / P["page"]).convert("RGB").crop((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop))
    d = ImageDraw.Draw(im)
    for p in pts:
        d.ellipse([p[3] - ox - 4, p[4] - oy - 4, p[3] - ox + 4, p[4] - oy + 4], fill=(255, 0, 0))
    im.save(dg.SRC / f"overlay_{name}_pushout.png")
    return pts


if __name__ == "__main__":
    for name, xe, we in (("fig7_B2", 0.2775, 23.6), ("fig7_B4", 0.3075, 21.2)):
        pts = trace(name, dg.PUSHOUT[name], xe, we)
        print(name, len(pts))
        for p in pts[::4]:
            print("   %.4f %.2f %s" % p[:3])
