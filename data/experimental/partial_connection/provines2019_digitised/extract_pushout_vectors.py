"""Extract the push-out load-slip curves of FHWA-HRT-20-005 (Provines, Ocel &
Zmetra 2019) from the vector paths of Figures 92-95 (report pp. 103, 105,
107, 108; PDF pages 121, 123, 125, 126).

The four figures are vector drawings (no raster), so the curve vertices are
read directly from the PDF content stream with PyMuPDF. Axes are calibrated
on the tick-mark paths (load 0-300 kips in 50-kip steps, slip 0-0.50 in. in
0.10-in. steps). Each curve is identified by matching its maximum load and
the slip at maximum load to Tables 15-18 of the report.

Each push-out specimen carries four 7/8-in. studs (two per flange, report
p. 23), so the per-stud force is the plotted load divided by four.

Usage:
    python extract_pushout_vectors.py <FHWA-HRT-20-005.pdf> <out_dir>
Outputs:
    pushout_curves_vector.csv   every vertex: figure, series index, colour,
                                pdf x/y (pt), slip (in.), total load (kips)
    pushout_summary.json        calibration, identification and the
                                per-stud strength and secant stiffness values
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from scipy.optimize import linear_sum_assignment

FIGS = {
    "Fig92_CIP_longitudinal": {"pdf_page": 121, "report_page": 103, "table": "Table 15 (p. 104)",
                               "table_rows": {"PO-S1-L3D-CIP": (183.1, 0.38), "PO-S2-L3D-CIP": (185.4, 0.28),
                                              "PO-S1-L4D-CIP": (279.5, 0.41), "PO-S2-L4D-CIP": (205.5, 0.32),
                                              "PO-S1-L5D-CIP": (237.6, 0.41), "PO-S2-L5D-CIP": (259.2, 0.31),
                                              "PO-S1-L6D-CIP": (241.7, 0.30), "PO-S2-L6D-CIP": (208.7, 0.29)}},
    "Fig93_PC_longitudinal": {"pdf_page": 123, "report_page": 105, "table": "Table 16 (p. 106)",
                              "table_rows": {"PO-S1-L3D-PC": (193.3, 0.25), "PO-S2-L3D-PC": (191.0, 0.28),
                                             "PO-S1-L4D-PC": (229.7, 0.43), "PO-S2-L4D-PC": (201.5, 0.28),
                                             "PO-S1-L5D-PC": (197.9, 0.30), "PO-S2-L5D-PC": (197.7, 0.26),
                                             "PO-S1-L6D-PC": (195.5, 0.35), "PO-S2-L6D-PC": (194.4, 0.24)}},
    "Fig94_CIP_transverse": {"pdf_page": 125, "report_page": 107, "table": "Table 17 (p. 107)",
                             "table_rows": {"PO-S1-T3D-CIP": (219.6, 0.35), "PO-S2-T3D-CIP": (238.2, 0.27),
                                            "PO-S1-T4D-CIP": (240.0, 0.32), "PO-S2-T4D-CIP": (219.2, 0.31)}},
    "Fig95_PC_transverse": {"pdf_page": 126, "report_page": 108, "table": "Table 18 (p. 109)",
                            "table_rows": {"PO-S1-T3D-PC": (215.4, 0.27), "PO-S2-T3D-PC": (191.3, 0.28),
                                           "PO-S1-T4D-PC": (194.9, 0.31), "PO-S2-T4D-PC": (213.6, 0.33)}},
}
N_STUDS = 4


def _pts(item):
    return [p for p in item[1:] if isinstance(p, fitz.Point)]


def analyse_page(page):
    d = page.get_drawings()
    # axis: vertical line with 7 tick segments to its left, horizontal with 6 ticks below
    yt = xt = None
    for g in d:
        its = g["items"]
        if g.get("color") != (0.0, 0.0, 0.0) or not all(i[0] == "l" for i in its):
            continue
        segs = [_pts(i) for i in its]
        if len(segs) == 7 and all(abs(s[0].y - s[1].y) < 1e-3 for s in segs):
            yt = sorted(s[0].y for s in segs)            # top (300) to bottom (0)
        if len(segs) == 6 and all(abs(s[0].x - s[1].x) < 1e-3 for s in segs):
            xt = sorted(s[0].x for s in segs)            # 0 to 0.5
    yvals = [300, 250, 200, 150, 100, 50, 0]
    xvals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    xfit_idx = [i for i in range(6)]
    spac = np.diff(xt)
    if spac[-1] < 0.99 * np.median(spac):
        # last tick drawn short of the frame (Fig. 93): calibrate on the first five
        xfit_idx = [0, 1, 2, 3, 4]
    px = np.polyfit([xt[i] for i in xfit_idx], [xvals[i] for i in xfit_idx], 1)
    py = np.polyfit(yt, yvals, 1)
    cal = {"x_ticks_pt": xt, "y_ticks_pt": yt, "x_fit_tick_indices": xfit_idx,
           "slip_in_per_pt": px[0], "slip_intercept": px[1],
           "load_kip_per_pt": py[0], "load_intercept": py[1],
           "x_fit_max_residual_in": float(np.abs(np.polyval(px, [xt[i] for i in xfit_idx])
                                                 - [xvals[i] for i in xfit_idx]).max()),
           "y_fit_max_residual_kip": float(np.abs(np.polyval(py, yt) - yvals).max())}
    curves = []
    for gi, g in enumerate(d):
        its = g["items"]
        if g.get("width") is None or g["width"] < 1.2 or g.get("dashes") not in (None, "[] 0"):
            continue
        lines = [_pts(i) for i in its if i[0] == "l"]
        if len(lines) < 200:
            continue
        xy = [(lines[0][0].x, lines[0][0].y)] + [(s[1].x, s[1].y) for s in lines]
        xy = np.asarray(xy)
        slip = np.polyval(px, xy[:, 0])
        load = np.polyval(py, xy[:, 1])
        curves.append({"drawing_index": gi, "colour_rgb": g["color"], "pt": xy,
                       "slip_in": slip, "load_kip": load})
    return cal, curves


def secant(slip, load, frac, qmax):
    """Per-stud secant stiffness at the first crossing of frac * Qmax (total load)."""
    i = int(np.argmax(load >= frac * qmax))
    if i == 0:
        return None, None
    # linear interpolation between vertices i-1, i
    l0, l1, s0, s1 = load[i - 1], load[i], slip[i - 1], slip[i]
    s = s0 + (frac * qmax - l0) * (s1 - s0) / (l1 - l0) if l1 != l0 else s1
    return (frac * qmax / N_STUDS) / s if s > 0 else None, s


def main(pdf, out_dir):
    out = Path(out_dir)
    doc = fitz.open(pdf)
    summary, rows = {}, []
    for fig, meta in FIGS.items():
        cal, curves = analyse_page(doc[meta["pdf_page"] - 1])
        ident = {}
        names = list(meta["table_rows"])
        peaks = []
        for c in curves:
            imax = int(np.argmax(c["load_kip"]))
            peaks.append((float(c["load_kip"][imax]), float(c["slip_in"][imax])))
        # one-to-one assignment of curves to table rows (max load, slip at max load)
        cost = np.array([[abs(meta["table_rows"][n][0] - q) + 50 * abs(meta["table_rows"][n][1] - s)
                          for n in names] for q, s in peaks])
        ri, ci = linear_sum_assignment(cost)
        assign = {int(r): names[int(c)] for r, c in zip(ri, ci)}
        for k, c in enumerate(curves):
            qmax, smax = peaks[k]
            best = assign[k]
            tq, ts = meta["table_rows"][best]
            rec = {"drawing_index": c["drawing_index"], "colour_rgb": c["colour_rgb"],
                   "n_vertices": len(c["slip_in"]),
                   "max_load_curve_kip": qmax, "slip_at_max_curve_in": smax,
                   "table_max_load_kip": tq, "table_slip_at_max_in": ts,
                   "Qmax_per_stud_curve_kip": qmax / N_STUDS,
                   "Qmax_per_stud_table_kip": tq / N_STUDS,
                   "assignment_cost": float(cost[k, names.index(best)])}
            for frac in (0.25, 0.4, 0.5):
                k_s, s_at = secant(c["slip_in"], c["load_kip"], frac, tq)
                rec[f"secant_k_per_stud_at_{int(frac*100)}pct_Qmax_kip_per_in"] = k_s
                rec[f"slip_at_{int(frac*100)}pct_Qmax_in"] = s_at
            # slip at the first vertex whose load exceeds 5 kips (initial offset indicator)
            i5 = int(np.argmax(c["load_kip"] > 5.0))
            rec["slip_at_first_load_above_5kip_in"] = float(c["slip_in"][i5])
            ident[best] = rec
            for (xp, yp), s, l in zip(c["pt"], c["slip_in"], c["load_kip"]):
                rows.append([fig, best, c["drawing_index"], round(xp, 3), round(yp, 3),
                             round(float(s), 5), round(float(l), 3)])
        summary[fig] = {"report_page": meta["report_page"], "pdf_page": meta["pdf_page"],
                        "table": meta["table"], "calibration": cal, "curves": ident}
    with open(out / "pushout_curves_vector.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["figure", "specimen", "drawing_index", "pdf_x_pt", "pdf_y_pt",
                    "slip_in", "total_load_kip"])
        w.writerows(rows)
    with open(out / "pushout_summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=float)
    for fig, s in summary.items():
        print(fig, {k: round(v, 5) if isinstance(v, float) else v for k, v in s["calibration"].items()
                    if k.endswith("residual_in") or k.endswith("residual_kip")})
        for name, r in sorted(s["curves"].items()):
            print(f"  {name:15s} Qmax curve {r['max_load_curve_kip']:6.1f} table {r['table_max_load_kip']:6.1f}"
                  f" | s@max {r['slip_at_max_curve_in']:.3f} table {r['table_slip_at_max_in']:.2f}"
                  f" | k25 {r['secant_k_per_stud_at_25pct_Qmax_kip_per_in']}"
                  f" k40 {r['secant_k_per_stud_at_40pct_Qmax_kip_per_in']}"
                  f" k50 {r['secant_k_per_stud_at_50pct_Qmax_kip_per_in']}"
                  f" s0 {r['slip_at_first_load_above_5kip_in']:.4f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
