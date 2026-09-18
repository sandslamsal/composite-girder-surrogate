"""Check the vector-extracted marker centres against 600-dpi renders
(pdftoppm -r 600) of the same PDF pages, and write overlay crops.

For every marker row of raw_<figure>.csv the 600-dpi pixel at
(x_px600, y_px600) and its 3 px neighbourhood are compared with the series
fill colour; a hit is a pixel within RGB distance 60 (0-255 scale). Markers of
another series drawn on top can hide a marker, so the hit rate is reported
per series rather than required to be 100 %.
Usage: python verify_raster.py <render_dir> <overlay_out_dir>
"""
import csv, json, os, sys
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
RENDER = {'report_fig5-14': 'report_p105_fig5-14-105.png', 'report_fig5-20': 'report_p111_fig5-20-111.png',
          'thesis_fig4-16': 'diss_p124-124.png', 'thesis_fig4-18': 'diss_p125-125.png',
          'thesis_fig4-20': 'diss_p126-126.png', 'thesis_fig4-22': 'diss_p127-127.png',
          'thesis_fig4-61': 'diss_p172-172.png', 'thesis_fig4-24': 'diss_p128-128.png',
          'thesis_fig4-63': 'diss_p174-174.png'}


def main(render_dir, out_dir):
    cal = json.load(open(os.path.join(HERE, 'calibration.json')))['figures']
    summary = {}
    for key, png in RENDER.items():
        img = np.asarray(Image.open(os.path.join(render_dir, png)).convert('RGB')).astype(float)
        styles = {s['name']: np.array(s['fill_rgb']) * 255.0 for s in cal[key]['series_style']}
        rows = list(csv.DictReader(open(os.path.join(HERE, 'raw_%s.csv' % key))))
        res = {}
        ov = Image.open(os.path.join(render_dir, png)).convert('RGB')
        dr = ImageDraw.Draw(ov)
        for r in rows:
            if r['source'] != 'markers':
                continue
            name = r['series']
            col = styles.get(name) if name in styles else None
            if col is None:
                # thesis per-specimen figures prefix the specimen name
                for n, c in styles.items():
                    if name.endswith(n.split('_', 1)[-1]) or n in name:
                        col = c
                if col is None:
                    continue
            x, y = float(r['x_px600']), float(r['y_px600'])
            xi, yi = int(round(x)), int(round(y))
            win = img[yi - 3:yi + 4, xi - 3:xi + 4].reshape(-1, 3)
            hit = bool(np.min(np.linalg.norm(win - col, axis=1)) < 60.0)
            res.setdefault(name, [0, 0])
            res[name][0] += hit; res[name][1] += 1
            dr.ellipse([x - 4, y - 4, x + 4, y + 4], outline=(255, 0, 255), width=2)
        # axis calibration marks: labelled ticks
        for ax in ('x_axis_deflection_in', 'y_axis_load_kips'):
            for t in cal[key][ax]['ticks_px600']:
                if ax.startswith('x'):
                    dr.line([t, 0, t, 60], fill=(0, 200, 255), width=3)
                else:
                    dr.line([0, t, 60, t], fill=(0, 200, 255), width=3)
        x0, y0, x1, y1 = [v * 600 / 72 for v in cal[key]['chart_region_pt']]
        ov.crop((int(x0), int(y0), int(x1), int(y1))).save(os.path.join(out_dir, 'overlay_%s.png' % key))
        summary[key] = {n: '%d/%d' % tuple(v) for n, v in res.items()}
        print(key, summary[key])
    json.dump(summary, open(os.path.join(HERE, 'raster_check.json'), 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
