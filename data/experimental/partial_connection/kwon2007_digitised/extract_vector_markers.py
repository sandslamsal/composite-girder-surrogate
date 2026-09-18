"""Digitise the Kwon (2007) / Kwon (2008) load-deflection charts.

The charts in both PDFs are vector graphics (Excel charts printed through a
PostScript driver), so marker and polyline coordinates are read directly from
the PDF drawing operators with PyMuPDF instead of being traced on a raster.
Every coordinate is also expressed in pixels of a 600-dpi render
(px = pt * 600 / 72; origin top-left of the page), which is what
``pdftoppm -r 600`` produces, and ``verify_raster.py`` checks the marker
centres against that render.

Axis calibration: tick marks (vector) are matched to the numeric tick labels
(PDF text); a least-squares line value = a + b * coordinate is fitted per axis
over all labelled ticks (7 per axis on the dissertation charts, 6 on the
report charts) and the residuals are recorded.

Source PDFs (not in the repository; kept in the session scratchpad):
  report  : Kwon et al. (2007) TxDOT 0-4124-1, library.ctr.utexas.edu/ctr-publications/0-4124-1.pdf
  thesis  : Kwon (2008) PhD dissertation, UT Austin, repositories.lib.utexas.edu (kwond74300.pdf)
Run with the ops_x86 python:  python extract_vector_markers.py <report.pdf> <thesis.pdf>
"""
import csv, json, sys, os
import numpy as np
import fitz

OUT = os.path.dirname(os.path.abspath(__file__))
DPI = 600.0
S = DPI / 72.0


def rgb(c):
    return tuple(round(v, 3) for v in c) if c else None


def numeric_spans(page, region):
    x0, y0, x1, y1 = region
    out = []
    for b in page.get_text('dict')['blocks']:
        for l in b.get('lines', []):
            t = ''.join(s['text'] for s in l['spans']).strip()
            bb = l['bbox']
            if x0 <= bb[0] <= x1 and y0 <= bb[1] <= y1:
                try:
                    out.append((float(t), bb))
                except ValueError:
                    pass
    return out


def text_spans(page, region):
    x0, y0, x1, y1 = region
    out = []
    for b in page.get_text('dict')['blocks']:
        for l in b.get('lines', []):
            t = ''.join(s['text'] for s in l['spans']).strip()
            bb = l['bbox']
            if t and x0 <= bb[0] <= x1 and y0 <= bb[1] <= y1:
                out.append((t, bb))
    return out


def ticks_from_drawings(page, region, style):
    """Return (x_tick_coords, y_tick_coords) in pt."""
    x0, y0, x1, y1 = region
    xs, ys = [], []
    for d in page.get_drawings():
        r = d['rect']
        if not (x0 - 5 <= r.x0 <= x1 + 5 and y0 - 5 <= r.y0 <= y1 + 5):
            continue
        col = rgb(d.get('color')) if d['type'] == 's' else rgb(d.get('fill'))
        if col != (0.0, 0.0, 0.0):
            continue
        for it in d['items']:
            if style == 'report' and it[0] == 'l':
                p, q = it[1], it[2]
                if abs(p.y - q.y) < 0.01 and 4.0 < abs(p.x - q.x) < 7.0:   # y-axis tick
                    ys.append(p.y)
                if abs(p.x - q.x) < 0.01 and 4.0 < abs(p.y - q.y) < 7.0:   # x-axis tick
                    xs.append(p.x)
            if style == 'thesis' and it[0] == 're':
                rr = it[1]
                if rr.width > 3.0 and rr.height < 1.0:     # y tick (short horizontal bar)
                    ys.append(0.5 * (rr.y0 + rr.y1))
                if rr.height > 3.0 and rr.width < 1.0:     # x tick (short vertical bar)
                    xs.append(0.5 * (rr.x0 + rr.x1))
    return sorted(set(round(v, 3) for v in xs)), sorted(set(round(v, 3) for v in ys))


def calibrate(ticks, labels, axis):
    """Match sorted ticks to sorted label centres and fit value = a + b*coord."""
    if axis == 'x':
        lab = sorted(labels, key=lambda t: 0.5 * (t[1][0] + t[1][2]))
        centres = [0.5 * (b[0] + b[2]) for _, b in lab]
    else:
        lab = sorted(labels, key=lambda t: -0.5 * (t[1][1] + t[1][3]))
        centres = [0.5 * (b[1] + b[3]) for _, b in lab]
    vals = [v for v, _ in lab]
    tk = sorted(ticks, reverse=(axis == 'y'))
    # each label is matched to the nearest tick
    pairs = []
    for v, c in zip(vals, centres):
        j = int(np.argmin([abs(t - c) for t in tk]))
        pairs.append((tk[j], v, c))
    coord = np.array([p[0] for p in pairs]); val = np.array([p[1] for p in pairs])
    A = np.vstack([np.ones_like(coord), coord]).T
    (a, b), *_ = np.linalg.lstsq(A, val, rcond=None)
    res = val - (a + b * coord)
    return dict(a=float(a), b=float(b), n_ticks=len(pairs),
                ticks_pt=[float(p[0]) for p in pairs], tick_values=[float(p[1]) for p in pairs],
                ticks_px600=[float(p[0] * S) for p in pairs],
                label_centres_pt=[float(p[2]) for p in pairs],
                residual_max_abs=float(np.max(np.abs(res))), units_per_pt=float(b),
                units_per_px600=float(b / S))


def markers(page, region, fill, kinds, dtype='f'):
    """Marker centres = bounding-box centres of the filled marker shapes of one
    series (fill colour + path signature). A compound path made only of
    rectangles (one 're' item per marker) is split into its rectangles."""
    x0, y0, x1, y1 = region
    out = []
    for idx, d in enumerate(page.get_drawings()):
        c = rgb(d.get('fill')) if 'f' in d['type'] else rgb(d.get('color'))
        k = ''.join(it[0] for it in d['items'])
        if c != fill or d['type'] != dtype:
            continue
        if k == kinds:
            rects = [d['rect']]
        elif kinds == 're' and set(k) == {'r', 'e'}:
            rects = [it[1] for it in d['items']]
        else:
            continue
        for r in rects:
            if r.width > 12 or r.height > 12:
                continue
            cx, cy = 0.5 * (r.x0 + r.x1), 0.5 * (r.y0 + r.y1)
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                out.append(dict(draw_index=idx, cx_pt=cx, cy_pt=cy, w_pt=r.width, h_pt=r.height))
    return out


def polyline(page, region, color, min_items=8):
    x0, y0, x1, y1 = region
    res = []
    for idx, d in enumerate(page.get_drawings()):
        if d['type'] != 's' or rgb(d.get('color')) != color or len(d['items']) < min_items:
            continue
        r = d['rect']
        if not (x0 <= r.x0 and r.x1 <= x1 + 3 and y0 <= r.y0 and r.y1 <= y1 + 3):
            continue
        pts = []
        for it in d['items']:
            if it[0] == 'l':
                for p in (it[1], it[2]):
                    if not pts or (abs(pts[-1][0] - p.x) > 1e-6 or abs(pts[-1][1] - p.y) > 1e-6):
                        pts.append((p.x, p.y))
        res.append(dict(draw_index=idx, vertices=pts))
    return res


def exclude_legend(marks, legend_boxes):
    keep = []
    for m in marks:
        inside = any(bx0 <= m['cx_pt'] <= bx1 and by0 <= m['cy_pt'] <= by1 for bx0, by0, bx1, by1 in legend_boxes)
        m['legend'] = inside
        keep.append(m)
    return keep


FIGS = [
    # key, doc, page(1-based), printed page, figure, chart region (pt), style, legend text names,
    # series: (name, fill rgb, kinds, drawing type)
    dict(key='report_fig5-14', doc='report', page=105, printed=91, figure='Fig. 5.14',
         region=(130, 175, 500, 420), style='report',
         legend=['DBLNB-30B', 'HASAA-30B', 'NON-00B'],
         series=[('DBLNB-30BS_center', (1.0, 0.398, 0.0), 're', 'f'),
                 ('HASAA-30BS_center', (0.602, 0.199, 0.0), 'lll', 'fs'),
                 ('NON-00BS_center', (0.0, 0.0, 0.5), 'llll', 'f')],
         lines=[('DBLNB-30BS_center', (1.0, 0.398, 0.0)), ('HASAA-30BS_center', (0.602, 0.199, 0.0)),
                ('NON-00BS_center', (0.0, 0.0, 0.5))]),
    dict(key='report_fig5-20', doc='report', page=111, printed=97, figure='Fig. 5.20',
         region=(130, 80, 500, 330), style='report',
         legend=['DBLNB-30BS', 'HASAA-30BS', 'NON-00BS'],
         series=[('DBLNB-30BS_center', (1.0, 0.398, 0.0), 're', 'f'),
                 ('HASAA-30BS_center', (0.602, 0.199, 0.0), 'lll', 'fs'),
                 ('NON-00BS_center', (0.0, 0.0, 0.5), 'llll', 'f')],
         lines=[('DBLNB-30BS_center', (1.0, 0.398, 0.0)), ('HASAA-30BS_center', (0.602, 0.199, 0.0)),
                ('NON-00BS_center', (0.0, 0.0, 0.5)), ('theory_noncomposite_solid', (0.0, 0.199, 0.0))]),
]
for pg, pr, fig, spec in [(124, 102, 'Fig. 4.16', 'NON-00BS'), (125, 103, 'Fig. 4.18', 'DBLNB-30BS'),
                          (126, 104, 'Fig. 4.20', 'HASAA-30BS'), (127, 105, 'Fig. 4.22', 'HTFGB-30BS'),
                          (172, 150, 'Fig. 4.61', 'HASAA-30BS1')]:
    FIGS.append(dict(key='thesis_fig%s' % fig.split()[1].replace('.', '-'), doc='thesis', page=pg, printed=pr,
                     figure=fig, region=(130, 430, 480, 670) if pg == 172 else (130, 90, 480, 330), style='thesis',
                     legend=['Center', 'South quarter', 'North quarter'], specimen=spec, series='auto'))

FIGS.append(dict(key='thesis_fig4-24', doc='thesis', page=128, printed=106, figure='Fig. 4.24',
                 region=(130, 90, 480, 300), style='thesis',
                 legend=['NON-00BS', 'DBLNB-30BS', 'HTFGB-30BS', 'HASAA-30BS'], series='auto'))
FIGS.append(dict(key='thesis_fig4-63', doc='thesis', page=174, printed=152, figure='Fig. 4.63',
                 region=(130, 190, 480, 430), style='thesis',
                 legend=['HASAA-30BS', 'HASAA-30BS1', 'NON-00BS'], series='auto'))


def auto_series(page, region, legend_names):
    """Identify series by the legend: the marker drawn immediately left of each
    legend label gives that series' fill colour and marker shape."""
    x0, y0, x1, y1 = region
    spans = {t: bb for t, bb in text_spans(page, region)}
    groups = {}
    for idx, d in enumerate(page.get_drawings()):
        if 'f' not in d['type']:
            continue
        r = d['rect']
        if r.width > 12 or r.height > 12 or r.width < 2:
            continue
        k = ''.join(it[0] for it in d['items'])
        c = rgb(d.get('fill'))
        if c in [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.502, 0.502, 0.502)]:
            continue
        cx, cy = 0.5 * (r.x0 + r.x1), 0.5 * (r.y0 + r.y1)
        if not (x0 <= cx <= x1 and y0 <= cy <= y1):
            continue
        groups.setdefault((c, k, d['type']), []).append((cx, cy))
    series = []
    for name in legend_names:
        bb = spans.get(name)
        if bb is None:
            raise RuntimeError('legend label %s not found' % name)
        ly = 0.5 * (bb[1] + bb[3]); lx = bb[0]
        best = None
        for key, pts in groups.items():
            for cx, cy in pts:
                if abs(cy - ly) < 4.0 and lx - 30 < cx < lx:
                    # prefer the simplest (core) shape: fewest drawing items
                    score = len(key[1])
                    if best is None or score < best[0]:
                        best = (score, key)
        if best is None:
            raise RuntimeError('no legend marker for %s' % name)
        series.append((name, best[1][0], best[1][1], best[1][2]))
    return series


def main(report_pdf, thesis_pdf):
    docs = dict(report=fitz.open(report_pdf), thesis=fitz.open(thesis_pdf))
    calib_all = {}
    for F in FIGS:
        page = docs[F['doc']][F['page'] - 1]
        region = F['region']
        xt, yt = ticks_from_drawings(page, region, F['style'])
        nums = numeric_spans(page, (region[0] - 40, region[1] - 15, region[2] + 20, region[3] + 40))
        # y labels: left of the plot, x labels: below the plot
        xmin_tick = min(xt)
        ymin_tick, ymax_tick = min(yt), max(yt)
        ylabels = [(v, bb) for v, bb in nums if xmin_tick - 25 < bb[2] < xmin_tick
                   and ymin_tick - 10 < 0.5 * (bb[1] + bb[3]) < ymax_tick + 10]
        xlabels = [(v, bb) for v, bb in nums if ymax_tick < bb[1] < ymax_tick + 20 and bb[2] > xmin_tick - 12]
        cx = calibrate(xt, xlabels, 'x')
        cy = calibrate(yt, ylabels, 'y')
        spans = {t: bb for t, bb in text_spans(page, region)}
        legend_boxes = []
        for name in F['legend']:
            bb = spans[name]
            legend_boxes.append((bb[0] - 32, bb[1] - 2, bb[2] + 2, bb[3] + 2))
        series = F['series'] if F['series'] != 'auto' else auto_series(page, region, F['legend'])
        if F.get('specimen'):
            series = [('%s_%s' % (F['specimen'], n.replace(' ', '_').lower()), c, k, t) for n, c, k, t in series]
        rows = []
        for name, fill, kinds, dtype in series:
            ms = exclude_legend(markers(page, region, fill, kinds, dtype), legend_boxes)
            seq = 0
            for m in ms:
                if m['legend']:
                    continue
                seq += 1
                rows.append(dict(figure_key=F['key'], series=name, seq=seq, source='markers',
                                 draw_index=m['draw_index'],
                                 x_pt=round(m['cx_pt'], 3), y_pt=round(m['cy_pt'], 3),
                                 x_px600=round(m['cx_pt'] * S, 1), y_px600=round(m['cy_pt'] * S, 1),
                                 marker_w_pt=round(m['w_pt'], 2), marker_h_pt=round(m['h_pt'], 2),
                                 deflection_in=round(cx['a'] + cx['b'] * m['cx_pt'], 4),
                                 load_kips=round(cy['a'] + cy['b'] * m['cy_pt'], 3)))
        for name, color in F.get('lines', []):
            for pl in polyline(page, region, color):
                for j, (px, py) in enumerate(pl['vertices']):
                    rows.append(dict(figure_key=F['key'], series=name, seq=j + 1, source='polyline_vertex',
                                     draw_index=pl['draw_index'], x_pt=round(px, 3), y_pt=round(py, 3),
                                     x_px600=round(px * S, 1), y_px600=round(py * S, 1),
                                     marker_w_pt='', marker_h_pt='',
                                     deflection_in=round(cx['a'] + cx['b'] * px, 4),
                                     load_kips=round(cy['a'] + cy['b'] * py, 3)))
        # dashed theory line of Fig. 5.20 (composite, drawn as short filled dashes)
        if F['key'] == 'report_fig5-20':
            k = 0
            for idx, d in enumerate(page.get_drawings()):
                if 'f' in d['type'] and rgb(d.get('fill')) == (0.5, 0.0, 0.0):
                    r = d['rect']
                    k += 1
                    for tag, (px, py) in (('dash_start', (r.x0, r.y1)), ('dash_end', (r.x1, r.y0))):
                        rows.append(dict(figure_key=F['key'], series='theory_composite_dashed_' + tag, seq=k,
                                         source='dash_bbox_corner', draw_index=idx,
                                         x_pt=round(px, 3), y_pt=round(py, 3),
                                         x_px600=round(px * S, 1), y_px600=round(py * S, 1),
                                         marker_w_pt=round(r.width, 2), marker_h_pt=round(r.height, 2),
                                         deflection_in=round(cx['a'] + cx['b'] * px, 4),
                                         load_kips=round(cy['a'] + cy['b'] * py, 3)))
        fn = os.path.join(OUT, 'raw_%s.csv' % F['key'])
        with open(fn, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        calib_all[F['key']] = dict(document=F['doc'], pdf_page=F['page'], printed_page=F['printed'],
                                   figure=F['figure'], chart_region_pt=region,
                                   series_style=[dict(name=n, fill_rgb=c, path_kinds=k, draw_type=t) for n, c, k, t in series],
                                   x_axis_deflection_in=cx, y_axis_load_kips=cy,
                                   n_rows=len(rows))
        print(F['key'], 'x res %.4f in' % cx['residual_max_abs'], 'y res %.3f kip' % cy['residual_max_abs'],
              {n: sum(1 for r in rows if r['series'] == n) for n in set(r['series'] for r in rows)})
    with open(os.path.join(OUT, 'calibration.json'), 'w') as fh:
        json.dump(dict(dpi_for_pixel_coordinates=DPI, pixel_origin='top-left of the PDF page',
                       method='vector drawing operators read with PyMuPDF; linear least-squares fit of labelled tick marks',
                       figures=calib_all), fh, indent=1)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
