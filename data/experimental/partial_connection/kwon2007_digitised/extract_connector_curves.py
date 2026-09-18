"""Digitise the single-connector direct-shear load-slip curves of Kwon (2008)
dissertation Figs. 3.15 (DBLNB), 3.16 (HTFGB), 3.17 (HASAA), printed pp. 57-58
(PDF pp. 79-80). These are the 7/8-in. connectors from the same rod/bolt lots
as the beam connectors (thesis Table 3.1 vs Sec. 4.2.2.3).

Each curve is a vector path (a thick stroke converted to a filled outline).
Each series path is redrawn alone on a blank page with PyMuPDF, rendered at
1200 dpi, and for every pixel row the centre of the left-most filled run gives
the slip on the loading branch at that load. Axes are calibrated on the
vector tick marks and their numeric labels (least squares, all ticks).
Output: connector_loadslip_raw.csv (every pixel row) and
connector_loadslip_levels.csv (slip at fixed load levels, secant stiffness).
"""
import csv, json, os, sys
import numpy as np
import fitz

HERE = os.path.dirname(os.path.abspath(__file__))
DPI = 1200.0
Z = DPI / 72.0

CHARTS = [
    dict(fig='Fig. 3.15', pdf_page=79, printed_page=57, region=(140, 410, 480, 640),
         xticks_idx=106, yticks_idx=104,
         series={107: 'DBLNB-05ST', 108: 'DBLNB-06ST', 109: 'DBLNB-07ST'}),
    dict(fig='Fig. 3.16', pdf_page=80, printed_page=58, region=(140, 105, 480, 330),
         xticks_idx=8, yticks_idx=6,
         series={9: 'HTFGB-05ST', 10: 'HTFGB-06ST'}),
    dict(fig='Fig. 3.17', pdf_page=80, printed_page=58, region=(140, 405, 480, 625),
         xticks_idx=25, yticks_idx=23,
         series={26: 'HASAA-05ST', 27: 'HASAA-06ST', 28: 'HASAA-07ST'}),
]


def tick_centres(d, axis):
    out = []
    for it in d['items']:
        r = it[1]
        if axis == 'x' and r.height > 3 and r.width < 1.5:
            out.append(0.5 * (r.x0 + r.x1))
        if axis == 'y' and r.width > 3 and r.height < 1.5:
            out.append(0.5 * (r.y0 + r.y1))
    return sorted(out)


def labels(page, region):
    x0, y0, x1, y1 = region
    res = []
    for b in page.get_text('dict')['blocks']:
        for l in b.get('lines', []):
            t = ''.join(s['text'] for s in l['spans']).strip()
            bb = l['bbox']
            if x0 <= bb[0] <= x1 and y0 <= bb[1] <= y1:
                try:
                    res.append((float(t), bb))
                except ValueError:
                    pass
    return res


def fit(ticks, labs, axis):
    if axis == 'x':
        lab = sorted(labs, key=lambda t: t[1][0])
        cen = [0.5 * (b[0] + b[2]) for _, b in lab]
    else:
        lab = sorted(labs, key=lambda t: -t[1][1])
        cen = [0.5 * (b[1] + b[3]) for _, b in lab]
    pairs = []
    for (v, _), c in zip(lab, cen):
        j = int(np.argmin([abs(t - c) for t in ticks]))
        pairs.append((ticks[j], v))
    A = np.array([[1.0, p[0]] for p in pairs]); y = np.array([p[1] for p in pairs])
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    return dict(a=float(a), b=float(b), ticks_pt=[p[0] for p in pairs], values=[p[1] for p in pairs],
                residual_max_abs=float(np.max(np.abs(y - (a + b * A[:, 1])))))


def render_path(src_page, d, region):
    doc = fitz.open(); pg = doc.new_page(width=src_page.rect.width, height=src_page.rect.height)
    sh = pg.new_shape()
    for it in d['items']:
        if it[0] == 'l':
            sh.draw_line(it[1], it[2])
        elif it[0] == 'c':
            sh.draw_bezier(it[1], it[2], it[3], it[4])
        elif it[0] == 're':
            sh.draw_rect(it[1])
    sh.finish(fill=(0, 0, 0), color=None, even_odd=bool(d.get('even_odd', False)), closePath=False)
    sh.commit()
    clip = fitz.Rect(*region)
    pix = pg.get_pixmap(matrix=fitz.Matrix(Z, Z), clip=clip, colorspace=fitz.csGRAY)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    return img < 128, clip


def main(thesis_pdf):
    doc = fitz.open(thesis_pdf)
    raw_rows, lev_rows, meta = [], [], {}
    for C in CHARTS:
        page = doc[C['pdf_page'] - 1]
        dr = page.get_drawings()
        xt = tick_centres(dr[C['xticks_idx']], 'x'); yt = tick_centres(dr[C['yticks_idx']], 'y')
        labs = labels(page, (C['region'][0], C['region'][1], C['region'][2], C['region'][3] + 20))
        xl = [(v, bb) for v, bb in labs if bb[1] > max(yt)]
        yl = [(v, bb) for v, bb in labs if bb[2] < min(xt)]
        cx = fit(xt, xl, 'x'); cy = fit(yt, yl, 'y')
        meta[C['fig']] = dict(pdf_page=C['pdf_page'], printed_page=C['printed_page'],
                              slip_axis_in=cx, load_axis_kips=cy, render_dpi=DPI)
        for idx, name in C['series'].items():
            mask, clip = render_path(page, dr[idx], C['region'])
            prev = None
            for row in range(mask.shape[0]):
                cols = np.flatnonzero(mask[row])
                if cols.size == 0:
                    continue
                # left-most contiguous run
                brk = np.flatnonzero(np.diff(cols) > 1)
                run = cols[:brk[0] + 1] if brk.size else cols
                xc_pt = clip.x0 + (0.5 * (run[0] + run[-1]) + 0.5) / Z
                yc_pt = clip.y0 + (row + 0.5) / Z
                slip = cx['a'] + cx['b'] * xc_pt
                load = cy['a'] + cy['b'] * yc_pt
                raw_rows.append(dict(figure=C['fig'], specimen=name, row_px1200=row, x_pt=round(xc_pt, 4),
                                     y_pt=round(yc_pt, 4), run_width_px=int(run[-1] - run[0] + 1),
                                     slip_in=round(slip, 5), load_kips=round(load, 3)))
            pts = [(r['load_kips'], r['slip_in'], r['run_width_px']) for r in raw_rows if r['specimen'] == name]
            pts.sort()
            loads = np.array([p[0] for p in pts]); slips = np.array([p[1] for p in pts])
            s0 = float(np.median(slips[loads < loads.min() + 0.6]))   # slip at the curve's lowest drawn point
            for L in [1, 2, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 25, 30]:
                sel = np.abs(loads - L) < 0.15
                if not sel.any():
                    continue
                # loading branch: smallest slip at this load level
                s = float(np.min(slips[sel]))
                lev_rows.append(dict(figure=C['fig'], specimen=name, load_kips=L, slip_in=round(s, 5),
                                     slip_rel_start_in=round(s - s0, 5),
                                     secant_stiffness_kip_per_in=round(L / (s - s0), 0) if s - s0 > 1e-4 else ''))
        print(C['fig'], 'x res %.4f in, y res %.3f kip' % (cx['residual_max_abs'], cy['residual_max_abs']))
    with open(os.path.join(HERE, 'connector_loadslip_raw.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(raw_rows[0].keys())); w.writeheader(); w.writerows(raw_rows)
    with open(os.path.join(HERE, 'connector_loadslip_levels.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(lev_rows[0].keys())); w.writeheader(); w.writerows(lev_rows)
    json.dump(meta, open(os.path.join(HERE, 'connector_calibration.json'), 'w'), indent=1)
    for r in lev_rows:
        print(r)


if __name__ == '__main__':
    main(sys.argv[1])
