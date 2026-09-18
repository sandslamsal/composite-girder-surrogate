"""Core routines for digitising the Lehigh 279-series load-deflection plots.

Pages are rendered at 600 dpi (pdftoppm -r 600) from 300-dpi bilevel scans.
Calibration: the bottom (load = 0) and left (deflection = 0) frame lines are
fitted as straight lines (the scans are slightly rotated), the tick marks on
both axes are located programmatically, and a linear map from tick index to
axis value is fitted by least squares. Open-circle data markers are located
by normalised ring-template correlation (dark ring, white interior).
"""
import json
import numpy as np
from PIL import Image
from scipy import ndimage, signal

DARK = 128


def load_binary(path):
    g = np.array(Image.open(path).convert("L"))
    return g < DARK


def _runs_1d(v):
    """Return list of (start, length) for True runs of a 1D bool array."""
    v = np.asarray(v, dtype=np.int8)
    d = np.diff(np.concatenate(([0], v, [0])))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return list(zip(starts, ends - starts))


def fit_bottom_axis(b, y_guess, x1, x2, band=80, step=4):
    """Fit the outer (lower) edge of the bottom frame line: y = c0 + c1 x.
    Returns (c0, c1, thickness)."""
    xs, ys, th = [], [], []
    for x in range(x1, x2, step):
        col = b[y_guess - band:y_guess + band, x]
        rs = _runs_1d(col)
        if not rs:
            continue
        # lowest run (outer edge of frame line is its bottom)
        s, L = rs[-1]
        if L > 60:
            continue
        xs.append(x)
        ys.append(y_guess - band + s + L - 1)
        th.append(L)
    xs, ys = np.array(xs), np.array(ys, float)
    for _ in range(3):
        c1, c0 = np.polyfit(xs, ys, 1)
        r = ys - (c0 + c1 * xs)
        keep = np.abs(r) < max(3.0, 2.5 * np.std(r))
        xs, ys = xs[keep], ys[keep]
        th = np.array(th)[keep] if len(th) == len(keep) else th
    c1, c0 = np.polyfit(xs, ys, 1)
    return c0, c1, float(np.median(th))


def fit_left_axis(b, x_guess, y1, y2, band=80, step=4):
    """Fit the outer (left) edge of the left frame line: x = c0 + c1 y."""
    ys, xs, th = [], [], []
    for y in range(y1, y2, step):
        row = b[y, x_guess - band:x_guess + band]
        rs = _runs_1d(row)
        if not rs:
            continue
        s, L = rs[0]
        if L > 60:
            continue
        ys.append(y)
        xs.append(x_guess - band + s)
        th.append(L)
    ys, xs, th = np.array(ys), np.array(xs, float), np.array(th)
    for _ in range(3):
        c1, c0 = np.polyfit(ys, xs, 1)
        r = xs - (c0 + c1 * ys)
        keep = np.abs(r) < max(3.0, 2.5 * np.std(r))
        ys, xs, th = ys[keep], xs[keep], th[keep]
    c1, c0 = np.polyfit(ys, xs, 1)
    return c0, c1, float(np.median(th))


def find_ticks_bottom(b, axis, x1, x2, thick, min_len=25, max_len=140,
                      exclude=()):
    """Tick marks rising from the inner edge of the bottom frame line.
    Returns x-centres of tick columns (clustered)."""
    c0, c1, _ = axis
    lens = np.zeros(x2 - x1)
    for i, x in enumerate(range(x1, x2)):
        y_in = int(round(c0 + c1 * x - thick - 2))  # just above inner edge
        L = 0
        while y_in - L > 0 and b[y_in - L, x]:
            L += 1
        lens[i] = L
    cand = (lens >= min_len) & (lens <= max_len)
    for (e1, e2) in exclude:
        cand[max(0, e1 - x1):max(0, e2 - x1)] = False
    lab, n = ndimage.label(cand)
    centres = []
    for k in range(1, n + 1):
        idx = np.where(lab == k)[0]
        if 3 <= len(idx) <= 30:
            centres.append((x1 + idx.mean(), float(np.median(lens[idx])), len(idx)))
    return centres


def find_ticks_left(b, axis, y1, y2, thick, min_len=25, max_len=140,
                    exclude=()):
    c0, c1, _ = axis
    lens = np.zeros(y2 - y1)
    for i, y in enumerate(range(y1, y2)):
        x_in = int(round(c0 + c1 * y + thick + 2))
        L = 0
        while x_in + L < b.shape[1] and b[y, x_in + L]:
            L += 1
        lens[i] = L
    cand = (lens >= min_len) & (lens <= max_len)
    for (e1, e2) in exclude:
        cand[max(0, e1 - y1):max(0, e2 - y1)] = False
    lab, n = ndimage.label(cand)
    centres = []
    for k in range(1, n + 1):
        idx = np.where(lab == k)[0]
        if 3 <= len(idx) <= 30:
            centres.append((y1 + idx.mean(), float(np.median(lens[idx])), len(idx)))
    return centres


def ring_kernel(r_out, r_in, r_void):
    R = int(np.ceil(r_out)) + 2
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    rr = np.hypot(xx, yy)
    ring = ((rr <= r_out) & (rr >= r_in)).astype(float)
    void = (rr <= r_void).astype(float)
    return ring / ring.sum(), void / void.sum()


def detect_rings(b, roi, r_out, r_in, r_void, thr=0.55, void_max=0.25,
                 min_sep=None):
    """Detect open-circle markers inside roi=(x1,y1,x2,y2).
    Score = mean darkness on ring annulus; require the interior (void) to be
    mostly white. Returns list of dict(x, y, ring, void)."""
    x1, y1, x2, y2 = roi
    sub = b[y1:y2, x1:x2].astype(float)
    kr, kv = ring_kernel(r_out, r_in, r_void)
    ring = signal.fftconvolve(sub, kr[::-1, ::-1], mode="same")
    void = signal.fftconvolve(sub, kv[::-1, ::-1], mode="same")
    score = ring - void
    sep = int(min_sep or r_out)
    mx = ndimage.maximum_filter(score, size=2 * sep + 1)
    pk = (score == mx) & (ring >= thr) & (void <= void_max)
    ys, xs = np.where(pk)
    out = []
    for y, x in zip(ys, xs):
        # sub-pixel refinement: centroid of score in 5x5 neighbourhood
        yy0, yy1 = max(0, y - 2), min(score.shape[0], y + 3)
        xx0, xx1 = max(0, x - 2), min(score.shape[1], x + 3)
        w = score[yy0:yy1, xx0:xx1]
        w = w - w.min()
        gy, gx = np.mgrid[yy0:yy1, xx0:xx1]
        cy = (w * gy).sum() / w.sum() if w.sum() > 0 else y
        cx = (w * gx).sum() / w.sum() if w.sum() > 0 else x
        out.append(dict(x=float(cx + x1), y=float(cy + y1),
                        ring=float(ring[y, x]), void=float(void[y, x])))
    return out


class Calib:
    """Affine map pixel -> (deflection, load) built from axis lines and ticks."""

    def __init__(self, bottom, left, xticks, yticks, dx_per_tick, dy_per_tick):
        self.bottom, self.left = bottom, left
        c0b, c1b, tb = bottom
        c0l, c1l, tl = left
        # centre lines of the frame strokes
        self.yb = lambda x: c0b + c1b * x - tb / 2.0
        self.xl = lambda y: c0l + c1l * y + tl / 2.0
        # origin = intersection
        y = c0b - tb / 2.0
        for _ in range(20):
            x = self.xl(y)
            y = self.yb(x)
        self.O = np.array([x, y])
        ex = np.array([1.0, c1b]); ex /= np.linalg.norm(ex)
        ey = np.array([c1l, 1.0]); ey /= np.linalg.norm(ey)
        ey = -ey  # up
        self.ex, self.ey = ex, ey
        M = np.column_stack([ex, ey])
        self.Minv = np.linalg.inv(M)
        # tick parameters along axes
        ux = np.array([self._uv(np.array([t, self.yb(t)]))[0] for t in xticks])
        vy = np.array([self._uv(np.array([self.xl(t), t]))[1] for t in yticks])
        self.ux, self.vy = ux, vy
        sx = np.median(np.diff(np.sort(ux)))
        sy = np.median(np.diff(np.sort(vy)))
        kx = np.round(ux / sx)
        ky = np.round(vy / sy)
        # least-squares: u = a + k*s   (a should be ~0 if the frame is the zero)
        Ax = np.column_stack([np.ones_like(kx), kx])
        (ax, sx2), resx, _, _ = np.linalg.lstsq(Ax, ux, rcond=None)
        Ay = np.column_stack([np.ones_like(ky), ky])
        (ay, sy2), resy, _, _ = np.linalg.lstsq(Ay, vy, rcond=None)
        self.kx, self.ky = kx, ky
        self.ax, self.sx, self.ay, self.sy = ax, sx2, ay, sy2
        self.dx_per_tick, self.dy_per_tick = dx_per_tick, dy_per_tick
        self.rms_x_px = float(np.sqrt(np.mean((ux - (ax + kx * sx2)) ** 2)))
        self.rms_y_px = float(np.sqrt(np.mean((vy - (ay + ky * sy2)) ** 2)))

    def _uv(self, p):
        return self.Minv @ (p - self.O)

    def to_data(self, x, y):
        u, v = self._uv(np.array([x, y], float))
        d = (u - self.ax) / self.sx * self.dx_per_tick
        P = (v - self.ay) / self.sy * self.dy_per_tick
        return float(d), float(P)

    def summary(self):
        return dict(origin_px=[float(self.O[0]), float(self.O[1])],
                    bottom_axis_outer_edge=dict(c0=self.bottom[0], c1=self.bottom[1],
                                                stroke_px=self.bottom[2]),
                    left_axis_outer_edge=dict(c0=self.left[0], c1=self.left[1],
                                              stroke_px=self.left[2]),
                    x_ticks_u_px=[float(v) for v in self.ux],
                    x_tick_index=[int(v) for v in self.kx],
                    y_ticks_v_px=[float(v) for v in self.vy],
                    y_tick_index=[int(v) for v in self.ky],
                    x_offset_px=float(self.ax), x_px_per_tick=float(self.sx),
                    y_offset_px=float(self.ay), y_px_per_tick=float(self.sy),
                    x_value_per_tick=self.dx_per_tick, y_value_per_tick=self.dy_per_tick,
                    x_tick_fit_rms_px=self.rms_x_px, y_tick_fit_rms_px=self.rms_y_px)
