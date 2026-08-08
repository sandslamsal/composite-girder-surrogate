# -*- coding: utf-8 -*-
"""Shared publication style for the composite-girder result figures.

Rules (unchanged in spirit from the tensegrity paper-2 style, retuned for
this manuscript):

* figures are designed at their FINAL printed width, so fonts render at
  true size.  This manuscript is Elsevier ``cas-sc`` **single column**
  with ``\\textwidth = 6.48 in``; ``FIG_W = 6.5`` is therefore the width
  of a figure that is included at ``width=\\linewidth`` (scale 1.0, so
  :func:`audit` judges the true printed size) and ``HALF_W = 3.15`` is a
  side-by-side panel.  A figure included at ``0.8\\linewidth`` must be
  drawn 0.8 as wide, not shrunk after the fact.
* Helvetica/Arial with matching sans math; 9.5 pt base type; 1.0 pt axes;
  outward ticks; bold is used only for panel headings (letter and title
  together), never in running annotation.
* Okabe-Ito palette with a FIXED entity-to-colour assignment across every
  figure (see ``ENTITY`` below), and line styles + markers as the
  grayscale-safe secondary encoding.  An entity keeps the same colour,
  dash pattern and marker in every figure of the paper.
* no legend or label is ever placed over data (:func:`place_legend`,
  :func:`inside_label`, :func:`headroom`).
* scientific numbers are typeset as ``a x 10^b`` (:func:`sci`), never
  ``1e-01``; paired panels share scales.
* the four ``eta_c`` bins are labelled identically everywhere
  (:func:`eta_bin_axis`), in the order 25-50, 50-70, 70-90, 90-100 %.
* the words 'truth' and 'ground truth' never appear in a figure; the
  reference is named 'OpenSeesPy fiber section'.

Every rule that can be expressed as an rcParam is set once in
:func:`apply`; scripts should not repeat axes/tick/legend/font settings
locally.  The size constants (``FS_*``) below are the only sanctioned
font sizes.

Entity registry
---------------
Colours are assigned so that entities which ever appear *together* are
far apart in hue, and every such group is also separated by dash pattern
and marker so the figure survives grayscale printing.

Predictors (they share panels in the Nie & Cai bar chart, the M-phi
overlays, the neutral-axis migration figure and the beam-level check)::

    opensees   black    solid          o    the reference response
    aashto     vermilion dotted        D    AASHTO transformed section
    niecai     sky blue  dash-dot      ^    Nie & Cai analytical
    beam       magenta   long dash     v    beam-level, discrete connectors
    surrogate  green     dashed        s    the proposed neural surrogate

Load regimes (they share the R_EI design chart and the deviation charts)::

    service    blue      solid         o    M/Mp <= 0.4
    extended   amber     dashed        s    M/Mp <= 0.6

Deck reinforcement levels::

    rho0       black     solid         o    rho_l = 0, primary database
    rho07      magenta   long dash     D    rho_l = 0.7 %, companion database

Prediction targets (the two non-trivial outputs)::

    y_na       sky blue  solid         o    neutral-axis depth
    curvature  amber     dashed        s    curvature

Data splits (the training-curve figure)::

    train      amber     solid         -    training loss
    heldout    blue      dotted        -    held-out / validation loss

Colours are re-used across groups that never share an axes: sky blue is
Nie & Cai *and* the neutral-axis target; amber is the extended-elastic
regime, the curvature target *and* the training loss; magenta is the
beam-level model *and* rho_l = 0.7 %; black is the OpenSeesPy reference
*and* rho_l = 0.  None of those pairs can appear in one panel.

When two registries *are* crossed in one panel - e.g. the AASHTO
deviation shown for both regimes at both reinforcement levels - colour
carries the regime and the reinforcement level carries the dash pattern
(``rho0`` solid, ``rho07`` long dash), which is exactly the linestyle
those two entities own.

Mathtext and the size floor
---------------------------
Matplotlib renders a subscript at 0.7 of its parent size, so ``$\\rho_\\ell$``
in an 8.5 pt legend prints at 5.95 pt - below the 6.5 pt floor, and
:func:`audit` will say so.  Every entity therefore carries two labels: a
mathtext-free ``label`` that is legal at any sanctioned size, and a
``label_math`` used only when the type is at least ``MATH_MIN_FS``
(= 6.5 / 0.7 = 9.29 pt).  :func:`entity_label` picks the right one for a
given font size; :func:`style` returns the plot kwargs.
"""

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Okabe-Ito
BLACK = '#000000'
ORANGE = '#FFB300'
SKY = '#00A2FF'
GREEN = '#00C853'
BLUE = '#2962FF'
VERM = '#FF4E11'
PURPLE = '#E8318A'
GRAY = '#9E9E9E'

CYCLE = [BLACK, VERM, SKY, GREEN, PURPLE, BLUE, ORANGE, GRAY]

# printed widths (in).  cas-sc single column: \textwidth = 6.4803 in, so a
# figure drawn 6.5 in wide and included at \linewidth is printed at scale
# 1.0 and every glyph lands at its declared size.
FIG_W = 6.5
HALF_W = 3.15

# the only sanctioned type sizes (pt)
FS_BASE = 9.5       # rcParams font.size
FS_PANEL = 10.5     # bold panel letter
FS_TITLE = 10.0     # axes title
FS_LABEL = 9.5      # axis label, panel title text
FS_TICK = 8.5       # tick labels
FS_LEGEND = 8.5     # legend entries and direct end labels
FS_ANNOT = 7.5      # in-axes annotations pointing at data
FS_SMALL = 6.8      # dense in-axes notes; the floor
MIN_FONT = 6.5      # nothing may be smaller than this

# smallest declared size at which a single mathtext subscript still prints
# at or above MIN_FONT (matplotlib renders script level at 0.7 of parent)
MATH_MIN_FS = MIN_FONT / 0.7


# ------------------------------------------------------------ entity registry
# One fixed identity per recurring entity of this paper: colour, dash
# pattern and marker, used in EVERY figure.  See the module docstring for
# the rationale of the colour assignment.
ENTITY = {
    # ---- predictors / models
    'opensees': dict(
        color=BLACK, ls='-', marker='o',
        label='OpenSeesPy fiber section',
        label_math='OpenSeesPy fiber section'),
    'aashto': dict(
        color=VERM, ls=(0, (1.4, 1.3)), marker='D',
        label='AASHTO transformed section',
        label_math='AASHTO transformed section'),
    'niecai': dict(
        color=SKY, ls='-.', marker='^',
        label='Nie & Cai analytical',
        label_math='Nie & Cai analytical'),
    'beam': dict(
        color=PURPLE, ls=(0, (5, 2)), marker='v',
        label='beam-level, discrete connectors',
        label_math='beam-level, discrete connectors'),
    'surrogate': dict(
        color=GREEN, ls='--', marker='s',
        label='proposed surrogate',
        label_math='proposed surrogate'),

    # ---- load regimes
    'service': dict(
        color=BLUE, ls='-', marker='o',
        label='service load, M/Mp ≤ 0.4',
        label_math=r'service load ($M/M_p \leq 0.4$)'),
    'extended': dict(
        color=ORANGE, ls=(0, (4, 2)), marker='s',
        label='extended elastic, M/Mp ≤ 0.6',
        label_math=r'extended elastic ($M/M_p \leq 0.6$)'),

    # ---- deck longitudinal reinforcement
    'rho0': dict(
        color=BLACK, ls='-', marker='o',
        label='no deck reinforcement',
        label_math=r'$\rho_\ell = 0$ (no deck rebar)'),
    'rho07': dict(
        color=PURPLE, ls=(0, (5, 1.8)), marker='D',
        label='deck reinforcement 0.7 %',
        label_math=r'$\rho_\ell = 0.7\,\%$'),

    # ---- prediction targets
    'y_na': dict(
        color=SKY, ls='-', marker='o',
        label='neutral-axis depth',
        label_math=r'neutral-axis depth $y_{na}$'),
    'curvature': dict(
        color=ORANGE, ls='--', marker='s',
        label='curvature',
        label_math=r'curvature $\varphi$'),

    # ---- data splits
    'train': dict(
        color=ORANGE, ls='-', marker='',
        label='train', label_math='train'),
    'heldout': dict(
        color=BLUE, ls=(0, (1, 1.1)), marker='',
        label='held out', label_math='held out'),
}

# canonical ordering inside each group, for legends and bar groups
PREDICTORS = ('opensees', 'aashto', 'niecai', 'beam', 'surrogate')
REGIMES = ('service', 'extended')
REINFORCEMENT = ('rho0', 'rho07')
TARGETS = ('y_na', 'curvature')
SPLITS = ('train', 'heldout')

# column name in the comparison parquets -> registry key
TARGET_KEY = {
    'neutral_axis_in': 'y_na',
    'curvature_1_per_in': 'curvature',
}


def entity_label(key, fontsize=FS_LEGEND, math=None):
    """Label for `key`, mathtext only when it prints at or above the floor.

    `math=True/False` forces the choice; the default decides from
    `fontsize` (a subscript renders at 0.7 of its parent).
    """
    e = ENTITY[key]
    if math is None:
        math = fontsize >= MATH_MIN_FS - 1e-9
    return e['label_math'] if math else e['label']


def style(key, fontsize=FS_LEGEND, math=None, label=True, **over):
    """Plot kwargs for a registered entity: colour, dash, marker, label.

    ``ax.plot(x, y, **FS.style('surrogate'))`` draws the surrogate the
    same way in every figure.  Pass ``label=False`` for a repeat of a
    series that is already in the legend, and any keyword to override
    (e.g. ``lw=1.4``, ``marker=''``, ``markevery=8``).
    """
    e = ENTITY[key]
    kw = dict(color=e['color'], ls=e['ls'], marker=e['marker'])
    if label:
        kw['label'] = entity_label(key, fontsize=fontsize, math=math)
    kw.update(over)
    return kw


def color(key):
    """Just the registered colour of `key` (for patches, text, spines)."""
    return ENTITY[key]['color']


def handle(key, fontsize=FS_LEGEND, math=None, **over):
    """A proxy Line2D for `key`, for hand-built and shared legends."""
    from matplotlib.lines import Line2D
    return Line2D([], [], **style(key, fontsize=fontsize, math=math, **over))


# ------------------------------------------------------------- eta_c bins
# The degree-of-composite-action bins as they are stored in the comparison
# parquets, and the single sanctioned way of labelling them.
ETA_BINS = ('25-50%', '50-70%', '70-90%', '90-100%')
ETA_BIN_LABELS = ('25–50', '50–70', '70–90', '90–100')
ETA_AXIS_LABEL = r'degree of composite action $\eta_c$ (%)'
ETA_AXIS_LABEL_PLAIN = 'degree of composite action (%)'

# ordered ramp low -> high composite action, plus a grayscale-safe
# dash/marker for each bin when the bins are drawn as curves
ETA_BIN_STYLE = {
    '25-50%':  dict(color=VERM,   ls='-',            marker='o'),
    '50-70%':  dict(color=ORANGE, ls=(0, (4, 2)),    marker='s'),
    '70-90%':  dict(color=GREEN,  ls='-.',           marker='^'),
    '90-100%': dict(color=BLUE,   ls=(0, (1.4, 1.3)), marker='D'),
}


def eta_bin_style(b, label=True, **over):
    """Plot kwargs for one eta_c bin, keyed by the parquet bin string."""
    kw = dict(ETA_BIN_STYLE[b])
    if label:
        kw['label'] = ETA_BIN_LABELS[ETA_BINS.index(b)] + ' %'
    kw.update(over)
    return kw


def eta_bin_axis(ax, axis='x', positions=None, label=True, math=True):
    """Label a categorical eta_c-bin axis identically in every figure.

    `positions` defaults to 0..3 (matplotlib bar/plot convention); pass
    ``range(1, 5)`` for boxplot/violinplot, which are 1-indexed.  Returns
    the positions used.
    """
    pos = list(range(len(ETA_BINS))) if positions is None else list(positions)
    a = ax.xaxis if axis == 'x' else ax.yaxis
    a.set_ticks(pos)
    a.set_ticklabels(list(ETA_BIN_LABELS))
    if label:
        txt = ETA_AXIS_LABEL if math else ETA_AXIS_LABEL_PLAIN
        (ax.set_xlabel if axis == 'x' else ax.set_ylabel)(txt)
    return pos


RC = {
    # ---- type
    'font.size': FS_BASE,
    'font.family': 'sans-serif',
    # Arial leads, not Helvetica. On macOS both Helvetica and Helvetica
    # Neue ship as .ttc collections from which matplotlib cannot extract
    # the bold face: findfont returns the SAME file for weight='normal'
    # and weight='bold', so every fontweight='bold' request rendered at
    # regular weight and the bold panel letters this module documents
    # were silently not bold (measured: identical ink, ratio 1.00).
    # Arial resolves to separate Arial.ttf / Arial Bold.ttf (ratio 1.30).
    'font.sans-serif': ['Arial', 'Helvetica Neue', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'stixsans',
    'text.usetex': False,
    'axes.unicode_minus': True,
    'pdf.fonttype': 42, 'ps.fonttype': 42,

    # ---- axes frame
    'axes.titlesize': FS_TITLE, 'axes.titleweight': 'normal',
    'axes.titlepad': 5.0,
    'axes.labelsize': FS_LABEL,
    'axes.labelweight': 'normal',
    'axes.linewidth': 1.0,
    'axes.edgecolor': 'black',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': False, 'axes.axisbelow': True,
    'axes.xmargin': 0.05, 'axes.ymargin': 0.05,
    'axes.formatter.use_mathtext': True,
    'axes.prop_cycle': plt.cycler(color=CYCLE),

    # ---- ticks
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.width': 1.0, 'ytick.major.width': 1.0,
    'xtick.minor.width': 0.7, 'ytick.minor.width': 0.7,
    'xtick.major.size': 3.0, 'ytick.major.size': 3.0,
    'xtick.minor.size': 1.7, 'ytick.minor.size': 1.7,
    'xtick.labelsize': FS_TICK, 'ytick.labelsize': FS_TICK,
    'xtick.color': 'black', 'ytick.color': 'black',
    'xtick.top': False, 'ytick.right': False,

    # ---- legend
    'legend.frameon': False, 'legend.fontsize': FS_LEGEND,
    'legend.title_fontsize': FS_LEGEND,
    'legend.handlelength': 2.2, 'legend.handleheight': 0.7,
    'legend.handletextpad': 0.55,
    'legend.columnspacing': 1.4, 'legend.labelspacing': 0.32,
    'legend.borderpad': 0.3, 'legend.borderaxespad': 0.0,
    'legend.markerscale': 1.0, 'legend.numpoints': 1, 'legend.scatterpoints': 1,
    'legend.framealpha': 0.92, 'legend.facecolor': 'white',
    'legend.edgecolor': '0.8', 'legend.fancybox': False,

    # ---- marks
    'lines.linewidth': 2.2,
    'lines.markersize': 5.5,
    'patch.linewidth': 0.8, 'hatch.linewidth': 0.6,
    'errorbar.capsize': 2.0,
    'grid.color': '0.88', 'grid.linewidth': 0.6, 'grid.linestyle': '-',
    'grid.alpha': 1.0,

    # ---- canvas and output
    'figure.figsize': (FIG_W, 2.65),
    'figure.facecolor': 'white', 'figure.edgecolor': 'none',
    'figure.dpi': 200, 'figure.constrained_layout.use': False,
    'savefig.dpi': 600, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    'savefig.facecolor': 'white', 'savefig.edgecolor': 'none',
    'savefig.transparent': False,
}


def apply(**overrides):
    """Install the paper rcParams (optionally with per-figure overrides)."""
    plt.rcParams.update(RC)
    if overrides:
        plt.rcParams.update(overrides)


def panel(ax, letter, title='', dy=1.03):
    """Panel heading outside the axes: letter AND title both bold.

    The whole heading is bold, not just the letter.  That is the author's
    house rule for this manuscript and it is applied identically in the
    TikZ schematics, so every panel heading in the paper reads the same
    way.  (Note that ``fontweight='bold'`` only became effective once the
    sans stack was reordered to lead with Arial: macOS ships Helvetica as
    a .ttc collection from which matplotlib cannot extract a bold face,
    so bold requests silently rendered at regular weight.)
    """
    ax.text(-0.005, dy, letter, transform=ax.transAxes, fontsize=FS_PANEL,
            fontweight='bold', va='bottom', ha='left')
    if title:
        ax.annotate(title, xy=(-0.005, dy), xycoords='axes fraction',
                    xytext=(16, 1.2), textcoords='offset points',
                    fontsize=FS_LABEL, fontweight='bold',
                    va='bottom', ha='left', color='0.15')


def sci(v):
    """Typeset a value as $a\\times10^{b}$ (or a plain decimal near 1)."""
    if v == 0:
        return '0'
    import math
    b = int(math.floor(math.log10(abs(v))))
    if -2 <= b <= 2:
        return f'{v:.2g}'
    a = v / 10 ** b
    return rf'${a:.1f}{{\times}}10^{{{b}}}$'


def end_label(ax, x, y, text, color, dy=0.0, fontsize=FS_LEGEND):
    """Direct label at the right end of a line (never bold)."""
    ax.annotate(text, xy=(x, y), xytext=(5, dy), textcoords='offset points',
                color=color, fontsize=fontsize, va='center', ha='left',
                annotation_clip=False)


# ---------------------------------------------------------------- geometry


def _renderer(fig):
    try:
        return fig.canvas.get_renderer()
    except AttributeError:
        fig.canvas.draw()
        return fig.canvas.get_renderer()


def _to_axes_frac(ax, pts):
    """Data coordinates -> axes fraction, dropping non-finite rows."""
    pts = np.asarray(pts, float)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.empty((0, 2))
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] == 0:
        return np.empty((0, 2))
    f = (ax.transData + ax.transAxes.inverted()).transform(pts)
    return f[np.isfinite(f).all(axis=1)]


def occupancy(ax):
    """Every plotted vertex of `ax`, expressed in axes fraction."""
    chunks = []
    for ln in ax.lines:
        try:
            d = ln.get_xydata()
        except Exception:
            continue
        if d is not None and len(d):
            chunks.append(np.asarray(d, float))
    for col in ax.collections:
        try:
            o = np.asarray(col.get_offsets(), float)
        except Exception:
            continue
        if o.ndim == 2 and len(o):
            chunks.append(o)
    for p in ax.patches:
        try:
            chunks.append(np.asarray(p.get_bbox().corners(), float))
        except Exception:
            continue
    if not chunks:
        return np.empty((0, 2))
    f = _to_axes_frac(ax, np.vstack(chunks))
    if f.shape[0] == 0:
        return f
    keep = (f[:, 0] > -0.02) & (f[:, 0] < 1.02) & \
           (f[:, 1] > -0.02) & (f[:, 1] < 1.02)
    return f[keep]


_ANCHOR = {'left': 0.0, 'right': 1.0, 'lower': 0.0, 'upper': 1.0,
           'center': 0.5}

LEGEND_LOCS = ['upper right', 'upper left', 'lower right', 'lower left',
               'center right', 'center left', 'upper center', 'lower center']


def loc_box(loc, size=(0.38, 0.32), pad=0.015):
    """(x0, y0, w, h) in axes fraction of the region a legend `loc` fills."""
    w, h = size
    v, _, u = loc.partition(' ')
    ya, xa = _ANCHOR.get(v, 0.5), _ANCHOR.get(u or 'center', 0.5)
    x0 = pad if xa == 0.0 else (1.0 - pad - w if xa == 1.0 else 0.5 - 0.5 * w)
    y0 = pad if ya == 0.0 else (1.0 - pad - h if ya == 1.0 else 0.5 - 0.5 * h)
    return max(x0, 0.0), max(y0, 0.0), w, h


def legend_loc(ax, size=(0.38, 0.32), pad=0.015, candidates=None):
    """Loc string for the emptiest region of `ax` (ties break by order)."""
    try:
        pts = occupancy(ax)
    except Exception:
        return 'best'
    order = list(candidates or LEGEND_LOCS)
    if pts.shape[0] == 0:
        return order[0]
    best, best_n = order[0], None
    for loc in order:
        x0, y0, w, h = loc_box(loc, size=size, pad=pad)
        inside = ((pts[:, 0] >= x0) & (pts[:, 0] <= x0 + w) &
                  (pts[:, 1] >= y0) & (pts[:, 1] <= y0 + h))
        n = int(inside.sum())
        if best_n is None or n < best_n:
            best, best_n = loc, n
        if best_n == 0:
            break
    return best


def place_legend(ax, *args, size=(0.38, 0.32), **kw):
    """ax.legend() dropped into the emptiest region instead of over data."""
    if 'loc' not in kw and 'bbox_to_anchor' not in kw:
        kw['loc'] = legend_loc(ax, size=size)
    return ax.legend(*args, **kw)


def keep_inside(artist, ax=None, pad=0.012):
    """Nudge a Text/Annotation so its rendered box stays inside the axes."""
    ax = ax if ax is not None else getattr(artist, 'axes', None)
    if ax is None:
        return artist
    try:
        r = _renderer(ax.figure)
        bb = artist.get_window_extent(renderer=r)
        inv = ax.transAxes.inverted()
        (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
    except Exception:
        return artist
    dx = dy = 0.0
    if x1 - x0 < 1.0 - 2 * pad:
        if x0 < pad:
            dx = pad - x0
        elif x1 > 1.0 - pad:
            dx = (1.0 - pad) - x1
    if y1 - y0 < 1.0 - 2 * pad:
        if y0 < pad:
            dy = pad - y0
        elif y1 > 1.0 - pad:
            dy = (1.0 - pad) - y1
    if dx == 0.0 and dy == 0.0:
        return artist
    try:
        p0 = ax.transAxes.transform((0.0, 0.0))
        p1 = ax.transAxes.transform((dx, dy))
        tr = artist.get_transform()
        here = tr.transform(artist.get_position())
        artist.set_position(tr.inverted().transform(here + (p1 - p0)))
    except Exception:
        pass
    return artist


def inside_label(ax, x, y, text, pad=0.012, **kw):
    """Text at (x, y) in data coordinates, guaranteed to stay in the axes."""
    kw.setdefault('fontsize', FS_ANNOT)
    kw.setdefault('clip_on', False)
    t = ax.text(x, y, text, **kw)
    return keep_inside(t, ax, pad=pad)


def headroom(ax, top=0.14, bottom=0.0):
    """Grow the y range by a fraction of its span to clear annotations."""
    lo, hi = ax.get_ylim()
    span = hi - lo
    ax.set_ylim(lo - bottom * span, hi + top * span)


def _mathtext_floor(text, size):
    """Smallest size mathtext will actually render inside this string.

    Matplotlib renders a script level at 0.7 of its parent and a
    scriptscript level at 0.5, so a subscript on a 6.8 pt label is about
    4.8 pt even though Text.get_fontsize() still reports 6.8. Auditing the
    declared size alone therefore misses every subscript in the figure,
    which is how figures were passing the floor while printing at 4.76 pt.
    """
    import re as _re
    smallest = size
    for seg in _re.findall(r'\$[^$]*\$', text):
        body = seg[1:-1]
        depth = 0
        for m in _re.finditer(r'[_^]', body):
            j = m.end()
            # a braced group can itself carry another level
            nested = body[j:j + 1] == '{' and _re.search(r'[_^]', body[j:j + 40])
            depth = max(depth, 2 if nested else 1)
        if depth:
            smallest = min(smallest, size * (0.5 if depth >= 2 else 0.7))
    return smallest


def audit(fig, min_fontsize=MIN_FONT, verbose=True, scale=1.0):
    """List type below the size floor and text that overlaps other text.
    Returns the problem strings; call it just before save().

    scale is the factor the figure is reduced by when included, i.e.
    (inclusion width) / (figure width). Type is judged at printed size,
    since that is the size a reader sees.
    """
    probs = []
    try:
        fig.canvas.draw()
        r = _renderer(fig)
    except Exception:
        return probs
    for i, ax in enumerate(fig.axes):
        boxes = []
        items = list(ax.texts) + [ax.xaxis.label, ax.yaxis.label, ax.title]
        items += list(ax.get_xticklabels()) + list(ax.get_yticklabels())
        lg = ax.get_legend()
        if lg is not None:
            items += list(lg.get_texts())
        for t in items:
            s = (t.get_text() or '').strip()
            if not s or not t.get_visible():
                continue
            eff = _mathtext_floor(s, t.get_fontsize()) * scale
            if eff < min_fontsize - 1e-6:
                probs.append(f'ax{i}: {eff:.2f} pt printed < '
                             f'{min_fontsize} pt on "{s[:28]}"'
                             + ('' if scale == 1.0 else f' (scale {scale:.3f})'))
            try:
                bb = t.get_window_extent(renderer=r)
            except Exception:
                continue
            if bb.width > 0 and bb.height > 0:
                boxes.append((s, bb))
        for a in range(len(boxes)):
            for b in range(a + 1, len(boxes)):
                ba, bb_ = boxes[a][1], boxes[b][1]
                if ba.overlaps(bb_):
                    ov = (min(ba.x1, bb_.x1) - max(ba.x0, bb_.x0)) * \
                         (min(ba.y1, bb_.y1) - max(ba.y0, bb_.y0))
                    if ov > 0.12 * min(ba.width * ba.height,
                                       bb_.width * bb_.height):
                        probs.append(f'ax{i}: "{boxes[a][0][:20]}" overlaps '
                                     f'"{boxes[b][0][:20]}"')
    if verbose and probs:
        print('  [audit] ' + '\n  [audit] '.join(probs))
    return probs


def bbox_artists(fig, margin=0.5):
    """Artists the tight bbox must contain. Axes.get_tightbbox collapses the
    width of an x label (and the height of a y label) to one pixel, so a long
    label on an outer panel is cropped; listing them explicitly prevents it.
    Anything reporting an extent further than `margin` inches outside the
    figure is dropped (3D axes mis-report their tick label positions)."""
    extra = []
    try:
        extra += list(fig.get_default_bbox_extra_artists())
        for ax in fig.axes:
            extra += list(ax.get_default_bbox_extra_artists())
            if getattr(ax, 'name', '') == '3d':
                continue
            for axis, lim, k in ((ax.xaxis, ax.get_xlim(), 0),
                                 (ax.yaxis, ax.get_ylim(), 1)):
                if not axis.get_visible():
                    continue
                if axis.label.get_text():
                    extra.append(axis.label)
                # tick labels off the ends of a log/date axis still exist as
                # artists but are never drawn; including them inflates the
                # crop, so keep only the ticks inside the view interval
                lo, hi = min(lim), max(lim)
                pad = 1e-9 + 1e-6 * abs(hi - lo)
                for t in axis.get_ticklabels():
                    try:
                        v = t.get_position()[k]
                    except Exception:
                        v = None
                    if v is None or (lo - pad) <= v <= (hi + pad):
                        extra.append(t)
                extra.append(axis.get_offset_text())
        extra = [a for a in extra if a.get_visible() and a.get_in_layout()]
        fig.canvas.draw()
        r = _renderer(fig)
        m = margin * fig.dpi
        x0, y0, x1, y1 = -m, -m, fig.bbox.width + m, fig.bbox.height + m
        keep = []
        for a in extra:
            try:
                bb = a.get_tightbbox(r)
            except Exception:
                bb = None
            if bb is None or (bb.x0 >= x0 and bb.x1 <= x1
                              and bb.y0 >= y0 and bb.y1 <= y1):
                keep.append(a)
    except Exception:
        return None
    return keep or None


def printed_scale(fig, include_width=None, linewidth_in=6.4803):
    """Scale that \\includegraphics applies to the cropped file.

    `include_width` is the fraction of \\linewidth the figure is included
    at (1.0 for width=\\linewidth, 0.8 for width=0.8\\linewidth). Pass the
    result to audit(scale=...) so type is judged at printed size.
    """
    frac = 1.0 if include_width is None else float(include_width)
    fig.canvas.draw()
    crop = fig.get_tightbbox(_renderer(fig), bbox_extra_artists=bbox_artists(fig))
    w_in = crop.width + 2 * plt.rcParams['savefig.pad_inches']
    return frac * linewidth_in / w_in


def save(fig, path_png, check=False, normalise_width=True, target_w=None):
    """Write PNG (600 dpi) and a matching vector PDF for the manuscript.

    With ``normalise_width`` the tight crop is padded symmetrically out to
    ``target_w`` inches (default FIG_W). Every figure then leaves this
    function at the SAME printed width, so a manuscript that includes them
    all at \\linewidth applies the same scale factor to each and the type
    renders at its designed size everywhere. Without this, the tight crop
    lands wherever the content happens to end (measured spread 5.97 to
    6.59 in across this paper's figures), and \\includegraphics silently
    rescales each figure differently, up to 8.5 percent, which reads as
    inconsistent font sizes between figures. Figures already wider than
    the target are left alone; content is never cropped.
    """
    if check:
        audit(fig)
    kw = {}
    extra = bbox_artists(fig)
    if extra and plt.rcParams['savefig.bbox'] == 'tight':
        kw['bbox_extra_artists'] = extra
    if normalise_width and plt.rcParams['savefig.bbox'] == 'tight':
        from matplotlib.transforms import Bbox
        tw = FIG_W if target_w is None else target_w
        try:
            bb = fig.get_tightbbox(_renderer(fig), bbox_extra_artists=extra)
            bb = bb.padded(plt.rcParams.get('savefig.pad_inches', 0.02))
            if bb.width < tw:
                dx = 0.5 * (tw - bb.width)
                bb = Bbox.from_extents(bb.x0 - dx, bb.y0, bb.x1 + dx, bb.y1)
            kw = {'bbox_inches': bb}   # extras already inside this bbox
        except Exception:
            pass                        # fall back to the plain tight crop
    fig.savefig(path_png, **kw)
    fig.savefig(str(path_png).rsplit('.', 1)[0] + '.pdf', **kw)
