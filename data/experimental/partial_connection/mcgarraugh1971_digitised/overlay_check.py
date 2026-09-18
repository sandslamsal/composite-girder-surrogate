"""Verification overlay for the Fig. 5 digitisation (B2, B4) of McGarraugh & Baldwin (1971).

The digitised values in fig5_<beam>_markers.csv (x_local, y_local) are mapped BACK to pixels
with the inverse of the grid calibration (digitise.to_pixel) and drawn on the 600 dpi render:
red cross = re-projected digitised value, cyan = fitted grid lines at their labelled values,
green = rows used as elastic points in mcgarraugh1971.json. Left: whole panel (half scale);
right: elastic region 0-1.1 in, 0-26 kips at native resolution. Also prints the round-trip
pixel residual and the per-marker spread between centre estimators and calibrations.
Output: overlay_check.png next to this script.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import digitise as dg  # noqa: E402

OUT = Path(__file__).resolve().parent
ELASTIC_MAX_KIP = {"fig5_B2": 14.5, "fig5_B4": 16.5}


def panel(name):
    P = dg.PANELS[name]
    ink = dg.load_ink(P["page"])
    cal = dg.calibrate(ink, P)
    df = pd.read_csv(OUT / f"{name}_markers.csv")
    page = Image.open(dg.SRC / P["page"]).convert("RGB")
    d = ImageDraw.Draw(page)
    for v in cal["vertical_lines"]:
        d.line([(v["c0"] + v["c1"] * 3000, 3000), (v["c0"] + v["c1"] * 6000, 6000)], fill=(0, 190, 255), width=2)
    for h in cal["horizontal_lines"]:
        d.line([(0, h["c0"]), (3000, h["c0"] + h["c1"] * 3000)], fill=(0, 190, 255), width=2)
    res = []
    for _, r in df.iterrows():
        px, py = dg.to_pixel(cal, r.x_local, r.y_local)
        res.append(np.hypot(px - r.px, py - r.py))
        col = (0, 170, 0) if r.y_local <= ELASTIC_MAX_KIP[name] else (230, 0, 0)
        d.line([(px - 22, py), (px + 22, py)], fill=col, width=4)
        d.line([(px, py - 22), (px, py + 22)], fill=col, width=4)
    x0, y0 = dg.to_pixel(cal, 0.0, 40.0)
    x1, y1 = dg.to_pixel(cal, 7.0, 0.0)
    full = page.crop((int(x0) - 40, int(y0) - 40, int(x1) + 40, int(y1) + 40))
    full = full.resize((full.size[0] // 2, full.size[1] // 2))
    zx0, zy0 = dg.to_pixel(cal, -0.05, 26.0)
    zx1, zy1 = dg.to_pixel(cal, 1.1, -0.8)
    zoom = page.crop((int(zx0), int(zy0), int(zx1), int(zy1)))
    spread = dict(
        n=len(df), roundtrip_px_max=float(np.max(res)),
        open_vs_core_in=float(np.max(np.abs(df.x_open_centroid - df.x_local))),
        open_vs_core_kip=float(np.max(np.abs(df.y_open_centroid - df.y_local))),
        affine_vs_local_in_elastic=float(np.max(np.abs((df.x_affine - df.x_local)[df.y_local <= ELASTIC_MAX_KIP[name]]))),
        affine_vs_local_kip_elastic=float(np.max(np.abs((df.y_affine - df.y_local)[df.y_local <= ELASTIC_MAX_KIP[name]]))),
    )
    return full, zoom, spread


def main():
    rows = [panel(n) for n in ("fig5_B2", "fig5_B4")]
    W = max(f.size[0] for f, _, _ in rows) + max(z.size[0] for _, z, _ in rows) + 30
    H = sum(max(f.size[1], z.size[1]) + 60 for f, z, _ in rows)
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    y = 0
    for (full, zoom, sp), name in zip(rows, ("B2", "B4")):
        d.text((10, y + 10), f"{name}: red/green cross = digitised value re-projected through the calibration "
                             f"(green = elastic points); cyan = fitted grid lines. Right: 0-1.1 in, 0-26 kips, native 600 dpi",
               fill=(0, 0, 0))
        canvas.paste(full, (0, y + 50))
        canvas.paste(zoom, (full.size[0] + 30, y + 50))
        y += max(full.size[1], zoom.size[1]) + 60
        print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in sp.items()})
    canvas.save(OUT / "overlay_check.png")


if __name__ == "__main__":
    main()
