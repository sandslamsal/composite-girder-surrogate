"""Digitise Culver and Coston (1959/1961), Fig. 4, beam B3 (Slutter and
Driscoll's BIII): load P versus mid-span deflection for tests B3-T1, B3-T2
and B3-T3.

Source: Fritz Engineering Laboratory Report 279.1 (April 1959), "Tests of
composite beams with stud shear connectors", C. Culver and R. Coston,
Lehigh Preserve Fritz Laboratory Reports paper 1802 (published as
Proc. ASCE J. Struct. Div. 87(ST2), 1961, Reprint No. 174).
Fig. 4 is on report page 21 = PDF page 24 of the Lehigh Preserve scan.

The scan is a 300 dpi bilevel JBIG2 image; the page is rendered here at
600 dpi (pdftoppm -r 600 -gray), so one native scan pixel = 2 px.

Method (fully programmatic, no hand clicking):
  1. axes: the y-axis is the column (left third of the panel) and the
     x-axis the row (lower half) with the largest dark-pixel counts;
  2. ticks: short dark runs beside each axis; every tick is assigned a
     value from its index (0.2 in per x tick, 10 kip per y tick), the
     labelled ticks (x: 0 and 1.0 in; y: 0, 20, 40 kip, and 50 kip on
     B3-T3) are checked against those indices, and a least-squares line
     through the axis origin and all ticks gives the calibration, whose
     residuals are reported;
  3. markers: the plotted points are small open circles. Their enclosed
     white interiors are found as connected white components not
     touching the crop border, with 1-70 px area and bounding box
     <= 12 px, surrounded by an unbroken dark ring at radius 5-7.5 px
     (letters and digits have larger holes and are further excluded by
     annotation boxes; slivers between two nearly parallel lines fail the
     ring test). The centroid of the interior is the marker centre. On
     B3-T2 and B3-T3 the circles are drawn smaller and their interiors
     are nearly closed (1-36 px), which lowers the centring precision;
  4. the thin theoretical line "Th" is sampled row by row in a window
     where it is isolated, and a line through the origin is fitted to
     it; its slope is compared with the programme's own theoretical
     deflection (report Appendix D, p. 29) as an independent check of
     the calibration.

Run:  python digitise_fig4.py <path to culver_coston_1961_reprint174.pdf>
Writes fig4_b3_markers.csv, fig4_b3_calibration.json next to this file.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
PAGE = 24          # PDF page of Fig. 4
DPI = 600

# Crop boxes (left, top, right, bottom) in 600-dpi page pixels, and
# annotation boxes (panel pixels) whose holes are text, not markers.
PANELS = {
    "B3-T1": dict(box=(600, 3350, 2500, 4700),
                  y_labels={0: 0.0, 2: 20.0, 4: 40.0},
                  x_labels={0: 0.0, 5: 1.0},
                  exclude=[(0, 0, 175, 1300), (180, 380, 360, 520),
                           (1170, 100, 1900, 240), (1450, 1040, 1900, 1300),
                           (380, 700, 420, 745)],
                  th_window=(38.0, 48.0)),
    "B3-T2": dict(box=(2750, 3350, 4200, 4700),
                  y_labels={0: 0.0, 2: 20.0, 4: 40.0},
                  x_labels={0: 0.0, 5: 1.0},
                  exclude=[(0, 0, 207, 1350), (210, 450, 330, 560),
                           (1150, 150, 1450, 260), (1300, 980, 1450, 1100),
                           (280, 1090, 620, 1180)],
                  th_window=(44.0, 52.0)),
    "B3-T3": dict(box=(4750, 3250, 6200, 4700),
                  y_labels={0: 0.0, 2: 20.0, 4: 40.0, 5: 50.0},
                  x_labels={0: 0.0, 5: 1.0},
                  exclude=[(0, 0, 167, 1400), (170, 380, 330, 500),
                           (990, 50, 1450, 200), (1000, 1080, 1450, 1250),
                           (100, 1150, 520, 1260)],
                  th_window=(48.0, 60.0)),
}
X_TICK_IN = 0.2
Y_TICK_KIP = 10.0


def render(pdf: Path, out_dir: Path) -> Path:
    stem = out_dir / "fig4_600"
    subprocess.run(["pdftoppm", "-r", str(DPI), "-f", str(PAGE), "-l", str(PAGE),
                    "-gray", "-png", str(pdf), str(stem)], check=True)
    return next(out_dir.glob("fig4_600*.png"))


def groups(idx: np.ndarray, gap: int = 2) -> list[float]:
    if idx.size == 0:
        return []
    parts = np.split(idx, np.where(np.diff(idx) > gap)[0] + 1)
    return [float(p.mean()) for p in parts if p.size]


def ring_fraction(dark: np.ndarray, x: float, y: float, r0: float, r1: float) -> float:
    rad = int(np.ceil(r1)) + 1
    yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    rr = np.hypot(xx, yy)
    xi, yi = int(round(x)), int(round(y))
    sub = dark[yi - rad: yi + rad + 1, xi - rad: xi + rad + 1]
    return float(sub[(rr >= r0) & (rr <= r1)].mean())


def calibrate(ticks_px: list[float], origin_px: float, step: float,
              sign: int, labels: dict[int, float]) -> dict:
    """Assign each tick an index from the origin, check the labelled
    ones, and fit value = a + b * px by least squares."""
    spacing = np.median(np.diff(sorted(ticks_px)))
    best: dict[int, tuple[float, float]] = {}
    for t in ticks_px:
        k = int(round(sign * (t - origin_px) / spacing))
        err = abs(sign * (t - origin_px) - k * spacing)
        if k >= 1 and err < 0.12 * spacing and (k not in best or err < best[k][1]):
            best[k] = (t, err)
    pairs = [(origin_px, 0.0)] + [(best[k][0], k * step) for k in sorted(best)]
    px = np.array([p for p, _ in pairs])
    val = np.array([v for _, v in pairs])
    b, a = np.polyfit(px, val, 1)
    resid = val - (a + b * px)
    for k, v in labels.items():
        assert any(abs(vv - v) < 1e-9 for vv in val), f"labelled tick {v} not found"
        assert abs(k * step - v) < 1e-9
    return dict(a=float(a), b=float(b), spacing_px=float(spacing),
                ticks=[dict(px=float(p), value=float(v), resid=float(r))
                       for p, v, r in zip(px, val, resid)],
                max_abs_resid=float(np.abs(resid).max()))


def digitise_panel(page: np.ndarray, name: str, cfg: dict) -> tuple[dict, list[dict]]:
    l, t, r, btm = cfg["box"]
    im = page[t:btm, l:r]
    dark = im < 128
    h, w = dark.shape

    colsum = dark[:, : w // 3].sum(axis=0)
    xa = float(np.where(colsum > 0.8 * colsum.max())[0].mean())
    rowsum = dark[h // 2:, :].sum(axis=1)
    ya = float((np.where(rowsum > 0.8 * rowsum.max())[0] + h // 2).mean())
    xa_i, ya_i = int(round(xa)), int(round(ya))

    above = dark[ya_i - 22: ya_i - 6, :].sum(axis=0)
    below = dark[ya_i + 6: ya_i + 22, :].sum(axis=0)
    xt = [g for g in groups(np.where((above >= 10) | (below >= 10))[0]) if g > xa + 20]
    left = dark[:, xa_i - 26: xa_i - 6].sum(axis=1)
    yt = [g for g in groups(np.where(left >= 10)[0]) if g < ya - 20]

    cx = calibrate(xt, xa, X_TICK_IN, +1, cfg["x_labels"])
    cy = calibrate(yt, ya, Y_TICK_KIP, -1, cfg["y_labels"])

    lab, _ = ndi.label(~dark)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    markers = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None or i in border:
            continue
        mask = lab[sl] == i
        area = int(mask.sum())
        hh, ww = mask.shape
        if not (1 <= area <= 70 and max(hh, ww) <= 12):
            continue
        yy, xx = np.argwhere(mask).mean(axis=0)
        px, py = xx + sl[1].start, yy + sl[0].start
        # open-circle test: an unbroken dark ring just outside the interior
        # (thin slivers between two nearly parallel lines fail it)
        if ring_fraction(dark, px, py, 5.0, 6.5) < 0.95 or ring_fraction(dark, px, py, 6.0, 7.5) < 0.80:
            continue
        if any(x0 <= px <= x1 and y0 <= py <= y1 for x0, y0, x1, y1 in cfg["exclude"]):
            continue
        if px < xa - 2 or py > ya + 12:
            continue
        markers.append(dict(test=name, px=round(float(px), 2), py=round(float(py), 2),
                            hole_area_px=area,
                            deflection_in=round(cx["a"] + cx["b"] * px, 4),
                            load_kip=round(cy["a"] + cy["b"] * py, 3)))

    # theoretical line "Th": isolated above the measured curve
    p_lo, p_hi = cfg["th_window"]
    rows = [int(round((p - cy["a"]) / cy["b"])) for p in np.linspace(p_lo, p_hi, 25)]
    pts = []
    for rr in rows:
        if not (0 <= rr < h):
            continue
        p_here = cy["a"] + cy["b"] * rr
        cols = np.where(dark[rr, xa_i + 30:])[0] + xa_i + 30
        runs = groups(cols, gap=1)
        if not runs:
            continue
        # the Th line is the leftmost dark run in these rows
        x_th = runs[0]
        pts.append((cx["a"] + cx["b"] * x_th, p_here))
    th = None
    if len(pts) >= 5:
        d = np.array([p[0] for p in pts]); pp = np.array([p[1] for p in pts])
        keep = np.ones(d.size, bool)
        for _ in range(3):   # drop specks (> 1 kip off the fitted line), refit
            k = float((d[keep] * pp[keep]).sum() / (d[keep] ** 2).sum())
            keep = np.abs(pp - k * d) < 1.0
        k = float((d[keep] * pp[keep]).sum() / (d[keep] ** 2).sum())
        th = dict(slope_kip_per_in=k, n_rows=int(keep.sum()), n_rows_dropped=int((~keep).sum()),
                  rms_resid_kip=float(np.sqrt(np.mean((pp[keep] - k * d[keep]) ** 2))))

    calib = dict(test=name, crop_box_page_px=cfg["box"], y_axis_px=xa, x_axis_px=ya,
                 x_calibration=cx, y_calibration=cy, theory_line=th,
                 n_markers=len(markers))
    return calib, markers


def main() -> None:
    pdf = Path(sys.argv[1])
    with tempfile.TemporaryDirectory() as td:
        png = render(pdf, Path(td))
        page = np.array(Image.open(png).convert("L"))
    calibs, rows = [], []
    for name, cfg in PANELS.items():
        c, m = digitise_panel(page, name, cfg)
        calibs.append(c)
        rows.extend(sorted(m, key=lambda d: (d["load_kip"], d["deflection_in"])))
    (HERE / "fig4_b3_calibration.json").write_text(json.dumps(
        dict(source="Culver & Coston, Fritz Lab Report 279.1 (1959), Fig. 4, report p. 21, PDF p. 24",
             render=f"pdftoppm -r {DPI} -gray, page {PAGE}", panels=calibs), indent=2))
    import csv
    with open(HERE / "fig4_b3_markers.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    for c in calibs:
        print(c["test"], "x resid max", round(c["x_calibration"]["max_abs_resid"], 4),
              "y resid max", round(c["y_calibration"]["max_abs_resid"], 3),
              "Th", c["theory_line"], "markers", c["n_markers"])
    for r_ in rows:
        print(r_)


if __name__ == "__main__":
    main()
