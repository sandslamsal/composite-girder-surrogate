#!/usr/bin/env python
"""R_EI(eta_c) stiffness-reduction design chart (Discussion, sec:design-rec).

Four series: the service-load (M/Mp <= 0.4) and extended-elastic
(M/Mp <= 0.6) regimes, each at deck reinforcement rho_l = 0 and 0.7 %.

ONE grid for the three blocks
.............................
The sheet carries three stacked blocks: a value table, the panel, and a
note.  They are laid out on a single grid measured in inches on the
sheet, with ONE left edge (GRID_L) and ONE right edge (GRID_R):

    table    top rule, head rule and bottom rule run GRID_L to GRID_R;
             the stub (regime headings and line samples) starts at
             GRID_L; the value columns stay centred on the bin mid-point
             rules of the panel;
    panel    the axes box IS GRID_L to GRID_R, so the left spine stands
             on the same edge as the rules and the last data column ends
             on the same right edge;
    note     wraps between GRID_L and GRID_R, its upright label flush on
             GRID_L.

Only the y tick labels and the y axis label sit outside that edge, in
the margin the grid reserves for them, which is what an axis label is
supposed to do.  Before this the block and the note were flush at
0.07 in while the left spine stood at 0.74 in, so the table and the note
overhung the panel by two thirds of an inch on the left and by nothing
on the right; that lopsided overhang is what read as asymmetric.

Because the stub now has to live inside the same left margin as the
panel, it is only about 0.75 in wide, so the reinforcement level is
named the way a table names a stub: a stub head, "deck rebar", over two
short entries, "none" and "0.7 %".  The words that used to sit in every
row now sit once, in the head.

Encoding, and why it is doubled up
..................................
The chart has to survive greyscale printing, so no distinction rests on
colour alone:

    regime         colour AND line weight   blue thin / amber thick
    reinforcement  dash AND marker shape    solid + circle / dashed + diamond

Amber (#FFB300) converts to 71 % grey, so an amber line of the same
weight as the blue one all but disappears next to it once desaturated,
and amber *text* on white is illegible.  The extended-elastic curves
therefore carry the extra weight that compensates their pale value, and
every piece of annotation is set in ink dark enough to read desaturated
(the amber ink is a darkened tint of the registry amber, used for type
only; the lines keep the registry colour).

The key IS the value table, and it sits above the panel
.......................................................
The key and the tabulated values are one block, full width, above the
plot, in the idiom the other figures of this manuscript use (a full
width key over the panels, never a column in the right margin).  The
block is set as a table is set: a top rule, a spanner head naming the
tabulated quantity, a column-head row giving the eta_c mid-point each
column belongs to, a head rule, then the four series rows grouped two by
two under their regime, and a bottom rule that also serves as the lid of
the panel.  Each series row carries, left to right: its line sample
drawn in the series' actual colour, weight, dash and marker; its stub
entry; and its four tabulated values, each printed in the column of the
bin mid-point it belongs to, so a printed value sits directly above the
vertical rule and the marker that it names.

Three rules, not four.  A cmidrule under the spanner head used to run a
tenth of an inch above the head rule, two thin rules close enough to
read as one thick smudge; the spanner head sits directly over the column
heads and needs no rule to claim them.  The three that remain each do
work: the top rule opens the block on the grid, the head rule separates
head from body, the bottom rule closes the block and lids the panel.

Two layers per series on the panel:
  * the continuous bin-mean curve of R_EI = 1 - Delta = 1 + phi_error/100,
    capped at 1.00 (AASHTO stiffness is never amplified);
  * a marker at the mid-point of each of the four eta_c bins of
    Table tab:design-correction, i.e. the number the engineer reads.
The markers are computed from the same parquets as the table and are
asserted to reproduce the published values.  The circle is drawn small
and the diamond large so the two shapes stay separable at print size,
both keep a dark rim so they survive desaturation, and a small
horizontal dodge keeps the coincident service pair from stacking into
one glyph.

The grid on the panel is ranked, and each of its two sets has one job.
Vertical rules stand only at the four bin mid-points, under the four
columns of printed values: they carry the column identity, tying a
marker to the number printed above it, and they are the darker set.
Horizontal rules stand only at the major y ticks: they serve reading a
value BETWEEN markers, which is the one thing the table above cannot do,
so they are regular and as pale as a rule can be and still read.  What
they replaced was one dotted leader per tabulated value, eleven of them,
running from the y axis out to each marker; the leaders duplicated the
table, since every value they pointed at is printed in the column above
its own marker, and at 0.92, 0.93 and 0.94 three of them ran a hundredth
of a unit apart and read as a single grey smudge.  The 1.00 cap keeps a
dash of its own, darker than either grid set, because it is a rule of
the method and not a reading aid.

Annotation discipline
.....................
Nothing stands loose on the panel.  Symbols keyed to footnotes were
tried and removed: floating in whitespace they attach to nothing a
reader can see.  Everything a reader needs in order to USE the chart is
a drawn glyph or a printed value; everything that qualifies the values
is one consolidated note under the figure, set as AASHTO sets a note
under a table: an upright "Note:" label, then a single italic block of
short imperative sentences.  It is held to TWO lines, so that the light
block at the foot does not compete with the ruled block at the head.

Run:
    /opt/anaconda3/envs/ops_x86/bin/python scripts/figs/make_fig_rei_curve.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from src.utils import figstyle as FS

AASHTO = {
    'rho0': REPO_ROOT / 'reports/aashto_full/aashto_comparison.parquet',
    'rho07': REPO_ROOT / 'reports/aashto_full_rebar007/aashto_comparison.parquet',
}
REGIME_CUT = {'service': 0.4, 'extended': 0.6}
OUT = REPO_ROOT / 'paper/revision_1/submission/sources/figures/fig_rei_curve.png'

# Table tab:design-correction, bins 25-50 / 50-70 / 70-90 / 90-100 %
PUBLISHED = {
    ('service', 'rho0'): [0.83, 0.92, 0.99, 1.00],
    ('service', 'rho07'): [0.83, 0.93, 1.00, 1.00],
    ('extended', 'rho0'): [0.50, 0.65, 0.76, 0.84],
    ('extended', 'rho07'): [0.59, 0.76, 0.87, 0.94],
}
BIN_EDGES = [0.25, 0.50, 0.70, 0.90, 1.00]
BIN_MIDS = [0.5 * (BIN_EDGES[j] + BIN_EDGES[j + 1]) for j in range(4)]

N_CURVE_BINS = 10          # continuous curve resolution over eta in [0.25, 1]

# -- greyscale-safe encoding (see the module docstring)
# Line weight is the second, colour-free carrier of the load regime; the
# lighter amber is drawn heavier so the two regimes have comparable ink
# once desaturated.  Within a regime the reinforcement level keeps the
# figstyle dash and marker, and rho07 is drawn thinner and on top of
# rho0 so it stays visible where the two series coincide (they do, to
# within 0.01, over the whole service-load curve).
LW_REGIME = {'service': 1.35, 'extended': 1.95}
LW_RHO = {'rho0': 1.0, 'rho07': 0.72}

# Marker size per reinforcement level.  A diamond of the same nominal
# size as a circle carries half its area and reads as a small blob, so
# the diamond is drawn larger: the two shapes then differ in outline AND
# in area, which is what makes them separable at 6 pt on paper and after
# desaturation.  Both are a size down from the first draft, where the
# diamonds sat on the curve like beads and hid the dash under them.
MS_RHO = {'rho0': 4.0, 'rho07': 5.2}
MEC = '0.18'               # dark rim, so a marker survives greyscale
MEW = 0.7

# Type ink.  The registry amber is a line colour: as small type on white
# it prints at 71 % grey and fails desaturated, so annotation uses a
# darkened amber of the same hue.  Everything else is near-black or a
# grey light enough to stay subordinate.
INK = {'service': FS.color('service'), 'extended': '#9A6A00'}
INK_NOTE = '0.35'          # the note under the figure
INK_HEAD = '0.30'          # spanner head, column heads and stub head
RULE_TOP = '0.42'          # top rule of the block, at the sheet edge
RULE_BOT = '0.32'          # bottom rule, which is also the lid of the panel
RULE_HEAD = '0.60'         # head rule, under the column heads
# The two grid sets are ranked by BOTH tone and weight, so the ranking
# survives a press that flattens light greys.  The horizontal set is the
# paper's standard grid tone (figstyle grid.color) drawn a shade finer;
# the vertical set, which carries the columns, is darker and heavier.
GRID_COL = '0.80'          # bin mid-point rules: they carry the columns
GRID_ROW = '0.88'          # horizontal grid at the major y ticks, subordinate
BAND = '0.945'             # the held-flat band
CAP_INK = '0.40'           # the 1.00 cap rule

MARK_DODGE = 0.009         # split the two coincident rho markers

# -- the key/value block: one row per series, in the order the curves
# stack on the panel (service above extended, no rebar before 0.7 %).
# The stub entries are short because the stub column is narrow (see the
# module docstring); STUB_HEAD carries the words they drop.
SERIES = (
    ('service', 'rho0', 'none'),
    ('service', 'rho07', '0.7 %'),
    ('extended', 'rho0', 'none'),
    ('extended', 'rho07', '0.7 %'),
)
GROUP = {'service': 'service load, M/Mp ≤ 0.4',
         'extended': 'extended elastic, M/Mp ≤ 0.6'}
# Spanner head over the four value columns, and the head of the stub
# column under it.  The stub head sits ON the grid's left edge, with the
# regime headings and the line samples, and the two short entries it
# governs are indented under it; that is the whole hierarchy of the stub,
# and it costs no rule.  Mathtext is banned in the block (a subscript at
# these sizes prints under the 6.5 pt floor), so the spanner names the
# quantity in the same words the y label uses.
SPAN_HEAD = 'tabulated stiffness reduction factor at bin mid-point'
STUB_HEAD = 'deck rebar'

# =====================================================================
# THE GRID.  Every horizontal position in this figure resolves to one of
# these two edges or to a data coordinate inside them.  Inches on a
# FS.FIG_W sheet, measured from its left edge.
# =====================================================================
GRID_L = 0.66              # table rules, table stub, left spine, note
GRID_R = 6.40              # table rules, right end of the axes, note wrap

# The sheet is deep enough that the panel keeps roughly two thirds of it.
# The block above is five rows and three rules and will always be heavy;
# the answer is not to squeeze the block until it is illegible but to
# give the curves the height they need to out-weigh it.
FIG_H = 5.45
AX_L, AX_W = GRID_L, GRID_R - GRID_L
AX_BOT = 0.95              # panel floor: x label and the note go below it

# -- vertical rhythm of the block, from the top of the sheet down
TOP_PAD = 0.055            # sheet top to the top rule
SPAN_DY = 0.135            # top rule to the spanner head baseline
COLS_DY = 0.152            # spanner head baseline to the column-head baseline
HEAD_RULE_DY = 0.078       # column-head baseline to the head rule
FIRST_HEAD_DY = 0.148      # head rule to the first regime heading
HEAD_DY = 0.168            # regime heading to its first row
ROW_DY = 0.150             # between the two rows of a regime
GROUP_DY = 0.205           # last row of a regime to the next heading
BLOCK_PAD = 0.112          # last row of the block to its bottom rule
# The bottom rule sits close enough to the panel that the four column
# rules read as running on out of the table and down through the plot,
# which is the path from a printed value to its own marker.
RULE_GAP = 0.030           # bottom rule to the top of the panel

# -- the stub, inside the grid's left margin.  The line sample has to
# be long enough that the DASH is legible with the marker sitting on it:
# a short sample with a centred diamond shows two stubs and reads solid,
# which loses the encoding that carries the reinforcement level.  The
# marker therefore sits at the near third and leaves a clear run of the
# dash pattern to its right.
KEY_X0, KEY_X1 = GRID_L, GRID_L + 0.33    # line sample of a key row
KEY_MARK = GRID_L + 0.098                 # marker on the sample, off centre
STUB_X = GRID_L + 0.40                    # stub entries, indented under the head
KEY_SAMPLE_DY = 0.026                     # sample above the row baseline

NOTE_Y0 = 0.075            # baseline of the LAST note line before the hang
NOTE_DY = 0.136            # between note lines
NOTE_GAP = 0.175           # x label to the top of the note block
NOTE_LABEL_PAD = 0.030     # "Note:" to the first italic word
NOTE_LINES = 2             # the note is held to two lines

# -- the single note under the figure, in the style AASHTO sets a note
# under a table: an upright label, then one italic block of short
# imperative sentences.  Mathtext is banned here (a subscript at FS_SMALL
# prints at 4.8 pt, under the floor), so the note names quantities in
# words; the y label already carries R_EI.  Three facts, two lines: the
# cap, the held-flat band and what it costs, and how to interpolate and
# what that costs.  Every number quoted is asserted against the databases
# below or is the fine-binned check reported in the text.
NOTE_LABEL = 'Note:'
NOTE_BODY = (
    'Cap the factor at 1.00; AASHTO stiffness is never amplified. In the '
    'shaded band the value is held flat and is unconservative, 0.83 against '
    'a finely binned 0.775 at service load. Interpolate linearly between '
    'markers, to within 0.005 (service) and 0.027 (extended elastic).'
)


def _half_up(x, n):
    """Round half away from zero, as the manuscript tables do."""
    f = 10.0 ** n
    x = np.asarray(x, float)
    return np.sign(x) * np.floor(np.abs(x) * f + 0.5) / f


def curves():
    """Continuous R_EI(eta_c) and the four binned table values per series."""
    edges = np.linspace(0.25, 1.0, N_CURVE_BINS + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    out = {}
    for rho, path in AASHTO.items():
        df = pd.read_parquet(path)
        df = df.assign(rei=1.0 + df['phi_error_pct'] / 100.0)
        for reg, cut in REGIME_CUT.items():
            s = df[df['moment_ratio'] <= cut]
            eta = s['composite_action'].to_numpy()
            rei = s['rei'].to_numpy()
            k = np.clip(np.digitize(eta, edges) - 1, 0, N_CURVE_BINS - 1)
            mean = np.array([rei[k == j].mean() if (k == j).any() else np.nan
                             for j in range(N_CURVE_BINS)])
            # Table tab:design-correction is R_EI = 1 - Delta with Delta the
            # bin-mean AASHTO stiffness deviation as PUBLISHED in
            # tab:aashto, i.e. rounded to 0.1 %; reproduce that chain
            # exactly (half-up rounding) so chart and table cannot drift.
            delta = np.array([-s.loc[s['eta_bin'] == b, 'phi_error_pct'].mean()
                              for b in FS.ETA_BINS])
            delta1 = _half_up(delta, 1)
            binned = np.minimum(_half_up(1.0 - delta1 / 100.0, 2), 1.0)
            out[(reg, rho)] = dict(x=centres,
                                   y=np.minimum(mean, 1.0),
                                   delta=delta1,
                                   bin=binned)
    return out


def _line_kw(reg, rho):
    """Exactly how a series is drawn on the panel, key sample included."""
    return dict(color=FS.color(reg), ls=FS.ENTITY[rho]['ls'],
                lw=LW_REGIME[reg] * LW_RHO[rho])


def _marker_kw(reg, rho):
    return dict(ls='none', marker=FS.ENTITY[rho]['marker'], ms=MS_RHO[rho],
                mfc=FS.color(reg), mec=MEC, mew=MEW)


def _ftext(fig, x_in, y_in, s, **kw):
    """Text placed in INCHES from the bottom left of the sheet."""
    return fig.text(x_in / FS.FIG_W, y_in / FIG_H, s, **kw)


def _fline(fig, xs_in, ys_in, **kw):
    """Line placed in INCHES from the bottom left of the sheet."""
    ln = Line2D([x / FS.FIG_W for x in xs_in], [y / FIG_H for y in ys_in],
                transform=fig.transFigure, **kw)
    fig.add_artist(ln)
    return ln


def _wrap(fig, text, fontsize, width_in, first_width_in=None, **tkw):
    """Greedy wrap of `text`, measured on the real renderer.

    `first_width_in` gives the first line a shorter measure, which is how
    the note leaves room for its upright "Note:" label on line one and
    still runs full width afterwards.
    """
    r = FS._renderer(fig)
    lines, cur = [], ''
    for word in text.split():
        trial = f'{cur} {word}'.strip()
        probe = fig.text(0.0, 0.0, trial, fontsize=fontsize, **tkw)
        w = probe.get_window_extent(renderer=r).width / fig.dpi
        probe.remove()
        limit = width_in if lines or first_width_in is None else first_width_in
        if w > limit and cur:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def _width(fig, s, **tkw):
    """Rendered width of `s` in inches, without leaving the probe behind."""
    r = FS._renderer(fig)
    probe = fig.text(0.0, 0.0, s, **tkw)
    w = probe.get_window_extent(renderer=r).width / fig.dpi
    probe.remove()
    return w


def _report(fig, texts, floor=None, gap=0.0, left=GRID_L, right=GRID_R):
    """Geometry check: nothing off the grid, into `floor`, or overlapping.

    `left` and `right` are the grid edges every block honours, with a
    small tolerance for the optical overhang of a glyph.  Returns a list
    of complaints so the build fails loudly rather than shipping a note
    that collides with the x label or a stub that runs into the value
    columns.
    """
    fig.canvas.draw()
    r = FS._renderer(fig)
    out = []
    boxes = []
    for a in texts:
        bb = a.get_window_extent(renderer=r)
        label = (a.get_text() or '')[:26]
        if bb.x1 / fig.dpi > right + 0.02:
            out.append(f'"{label}" ends at {bb.x1 / fig.dpi:.2f} in '
                       f'> grid right {right:.2f} in')
        if bb.x0 / fig.dpi < left - 0.02:
            out.append(f'"{label}" starts at {bb.x0 / fig.dpi:.2f} in '
                       f'< grid left {left:.2f} in')
        if floor is not None:
            fb = floor.get_window_extent(renderer=r)
            if bb.y1 > fb.y0 - gap * fig.dpi:
                out.append(f'"{label}" clears the x label by '
                           f'{(fb.y0 - bb.y1) / fig.dpi:.3f} in')
        boxes.append((label, bb))
    for a in range(len(boxes)):
        for b in range(a + 1, len(boxes)):
            ba, bc = boxes[a][1], boxes[b][1]
            if ba.overlaps(bc):
                out.append(f'"{boxes[a][0][:18]}" overlaps '
                           f'"{boxes[b][0][:18]}"')
    return out


def build() -> None:
    FS.apply()
    data = curves()

    # -- the plotted bin values must be the published table values
    for key, d in data.items():
        want = np.array(PUBLISHED[key])
        got = d['bin']
        assert np.allclose(got, want, atol=1e-9), (key, got, want)
        print(f'[check] {key[0]:9s} {key[1]:6s} Delta {d["delta"]} '
              f'-> R_EI {got} (published {want})')

    # -- rows of the block, from the top of the sheet down: top rule,
    # spanner head, column heads, head rule, then heading, two series
    # rows, heading, two series rows, bottom rule, panel.
    y_top_rule = FIG_H - TOP_PAD
    y_span = y_top_rule - SPAN_DY
    y_cols = y_span - COLS_DY
    y_head_rule = y_cols - HEAD_RULE_DY

    y_head = {}
    y_row = {}
    y = y_head_rule
    for reg in ('service', 'extended'):
        y -= FIRST_HEAD_DY if reg == 'service' else GROUP_DY
        y_head[reg] = y
        y -= HEAD_DY
        y_row[(reg, 'rho0')] = y
        y -= ROW_DY
        y_row[(reg, 'rho07')] = y
    y_rule = y - BLOCK_PAD
    ax_top = y_rule - RULE_GAP

    fig = plt.figure(figsize=(FS.FIG_W, FIG_H))
    ax = fig.add_axes((AX_L / FS.FIG_W, AX_BOT / FIG_H, AX_W / FS.FIG_W,
                       (ax_top - AX_BOT) / FIG_H))
    ax.set_axisbelow(True)

    x_lo, x_hi = 0.25, 1.00
    y_lo, y_hi = 0.40, 1.045

    def x_in(eta):
        """Data eta_c -> inches from the left edge of the sheet."""
        return AX_L + AX_W * (eta - x_lo) / (x_hi - x_lo)

    # -- the tabulated values, and where their markers land.  rho0 and
    # rho07 coincide over most of the service curve, so their markers are
    # dodged either side of the bin mid-point rather than stacked.
    marks = {}
    for (reg, rho), d in data.items():
        sgn = -1.0 if rho == 'rho0' else 1.0
        marks[(reg, rho)] = (np.array(BIN_MIDS) + sgn * MARK_DODGE, d['bin'])

    # -- the held-flat band, topped out at the cap so its edges align
    # with the cap rule and with the bounded left spine.  Its right edge
    # is the first bin mid-point, i.e. the first column rule, so the band
    # needs no edge of its own and no symbol: the note names it.
    ax.add_patch(Rectangle((x_lo, y_lo), BIN_MIDS[0] - x_lo, 1.0 - y_lo,
                           facecolor=BAND, edgecolor='none', zorder=0))

    # -- micro-grid, ranked in two levels and in two jobs.
    #
    # Vertical, at the four bin mid-points: these carry the column
    # identity, tying each marker to the value printed above it, so they
    # are the darker set and they are the only vertical rules drawn.
    #
    # Horizontal, at the major y ticks only: these serve interpolation
    # BETWEEN markers, which is the one thing the table above cannot do,
    # so they are regular, evenly spaced and as pale as a rule can be
    # while still reading.  They replace the eleven dotted read-off
    # leaders that used to run from the axis to every marker.  Those
    # leaders duplicated the table (every value they pointed at is
    # printed in the column above the marker) and, worse, at 0.92, 0.93
    # and 0.94 three of them ran a hundredth apart and read as one grey
    # smudge instead of three guides.  Eleven irregular dotted lines were
    # the densest ink on the panel and the least informative.
    for m in BIN_MIDS:
        ax.plot([m, m], [y_lo, y_hi], color=GRID_COL, lw=0.7, zorder=0.5,
                solid_capstyle='butt')
    for yv in np.arange(0.50, 0.901, 0.10):   # 0.40 is the floor, 1.00 the cap
        ax.plot([x_lo, x_hi], [yv, yv], color=GRID_ROW, lw=0.5, zorder=0.35,
                solid_capstyle='butt')

    # the cap is a rule of the method, not a read-off guide, so it is
    # dashed where the leaders are dotted: at 0.99 the two run within
    # a millimetre of each other and must not read as the same thing.
    ax.axhline(1.0, color=CAP_INK, ls=(0, (3.6, 2.2)), lw=0.8, zorder=1)

    # rho0 first, rho07 (thinner, long dash) on top of it, so the two
    # coincident service curves both stay legible
    for reg, rho, _ in SERIES:
        d = data[(reg, rho)]
        ax.plot(d['x'], d['y'], zorder=4, **_line_kw(reg, rho))
        # Tabulated values as MARKERS at the bin mid-points, not as treads
        # spanning each bin. A tread asserts a step rule ("read the bin
        # value"); the continuous curve asserts interpolation. Drawing both
        # made the figure contradict itself, and the two disagreed by up to
        # 0.09 in R_EI at the left edge. Interpolating between these markers
        # reproduces the curve to within 0.005 (service) and 0.027
        # (extended elastic), i.e. four to seven times closer in RMS than
        # the step rule, so the marker-plus-curve form is the honest one.
        xs, ys = marks[(reg, rho)]
        ax.plot(xs, ys, zorder=5, **_marker_kw(reg, rho))

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xticks(BIN_EDGES)
    ax.set_xticklabels([f'{v:.2f}' for v in BIN_EDGES])
    ax.set_xticks(BIN_MIDS, minor=True)
    ax.set_yticks(np.arange(0.40, 1.001, 0.10))
    # two decimals on both axes: the chart is read against a table of
    # two-decimal values, and 0.4 next to 0.83 reads as a different
    # kind of number
    ax.set_yticklabels([f'{v:.2f}' for v in np.arange(0.40, 1.001, 0.10)])
    ax.spines['left'].set_bounds(y_lo, 1.00)
    ax.set_xlabel(r'degree of composite action  $\eta_c = \Sigma Q_n / C_f$',
                  labelpad=3.0)
    # the y label is centred on the DATA range, not on the axes box: the
    # box carries a little headroom over the 1.00 cap so the markers there
    # are not clipped, and the default 0.5 would float the label above the
    # middle of the plotted data.  labelpad is set explicitly because the
    # grid reserves the left margin and the label has to sit in it.
    ax.set_ylabel(r'stiffness reduction factor  $R_{EI}$',
                  y=(0.5 * (y_lo + 1.00) - y_lo) / (y_hi - y_lo),
                  va='bottom', labelpad=2.5)

    # -- the key AND the tabulated values, as one full-width block above
    # the panel, set as a table: three rules, a spanner head over the
    # value columns, a column-head row giving the eta_c mid-points, then
    # the four series rows, each value standing in the column of the bin
    # mid-point rule it belongs to.  All three rules run the full grid.
    _fline(fig, [GRID_L, GRID_R], [y_top_rule] * 2,
           color=RULE_TOP, lw=0.7, solid_capstyle='butt')
    _fline(fig, [GRID_L, GRID_R], [y_head_rule] * 2,
           color=RULE_HEAD, lw=0.6, solid_capstyle='butt')
    _fline(fig, [GRID_L, GRID_R], [y_rule] * 2,
           color=RULE_BOT, lw=0.9, solid_capstyle='butt')

    stub = [_ftext(fig, GRID_L, y_cols, STUB_HEAD, ha='left',
                   va='baseline', fontsize=FS.FS_ANNOT, color=INK_HEAD)]
    block = [_ftext(fig, 0.5 * (x_in(BIN_MIDS[0]) + x_in(BIN_MIDS[-1])),
                    y_span, SPAN_HEAD, ha='center', va='baseline',
                    fontsize=FS.FS_LEGEND, color=INK_HEAD)] + stub
    for m in BIN_MIDS:
        block.append(_ftext(fig, x_in(m), y_cols, f'{m:.3f}', ha='center',
                            va='baseline', fontsize=FS.FS_ANNOT,
                            color=INK_HEAD))
    for reg in ('service', 'extended'):
        block.append(_ftext(fig, GRID_L, y_head[reg], GROUP[reg],
                            ha='left', va='baseline',
                            fontsize=FS.FS_LEGEND, color=INK[reg]))
    for reg, rho, name in SERIES:
        yr = y_row[(reg, rho)]
        _fline(fig, [KEY_X0, KEY_X1], [yr + KEY_SAMPLE_DY] * 2,
               solid_capstyle='butt', **_line_kw(reg, rho))
        _fline(fig, [KEY_MARK], [yr + KEY_SAMPLE_DY], **_marker_kw(reg, rho))
        entry = _ftext(fig, STUB_X, yr, name, ha='left', va='baseline',
                       fontsize=FS.FS_ANNOT, color=INK[reg])
        stub.append(entry)
        block.append(entry)
        vals = data[(reg, rho)]['bin']
        for j in range(4):
            # the tabulated values are the payload of the figure and are
            # set a size up from the stub entries that qualify them
            block.append(_ftext(fig, x_in(BIN_MIDS[j]), yr, f'{vals[j]:.2f}',
                                ha='center', va='baseline',
                                fontsize=FS.FS_LEGEND, color=INK[reg]))

    # -- the note, in the figure but outside the axes, set as a table
    # note is set: small, grey, flush on the grid's left edge, wrapped to
    # its right edge, an upright label and one italic block in one voice.
    full_w = GRID_R - GRID_L
    lab_w = _width(fig, NOTE_LABEL, fontsize=FS.FS_SMALL) + NOTE_LABEL_PAD
    lines = _wrap(fig, NOTE_BODY, FS.FS_SMALL, full_w,
                  first_width_in=full_w - lab_w, style='italic')
    notes = []
    for j, s in enumerate(lines):
        yb = NOTE_Y0 + (len(lines) - 1 - j) * NOTE_DY
        notes.append(_ftext(fig, GRID_L + (lab_w if j == 0 else 0.0), yb, s,
                            ha='left', va='baseline', fontsize=FS.FS_SMALL,
                            color=INK_NOTE, style='italic'))
    label = _ftext(fig, GRID_L, notes[0].get_position()[1] * FIG_H,
                   NOTE_LABEL, ha='left', va='baseline',
                   fontsize=FS.FS_SMALL, color=INK_NOTE)
    notes.append(label)
    # hang the block a fixed distance under the x label, whatever the wrap
    # returns, and let the tight crop take the slack off the bottom
    fig.canvas.draw()
    rr = FS._renderer(fig)
    drop = ((ax.xaxis.label.get_window_extent(renderer=rr).y0
             - NOTE_GAP * fig.dpi
             - notes[0].get_window_extent(renderer=rr).y1) / fig.dpi) / FIG_H
    for t in notes:
        t.set_y(t.get_position()[1] + drop)

    bad = (_report(fig, notes, floor=ax.xaxis.label, gap=0.09)
           + _report(fig, block))
    if len(lines) != NOTE_LINES:
        bad.append(f'note wraps to {len(lines)} lines, not {NOTE_LINES}')
    # the stub must not run into the first value column
    r = FS._renderer(fig)
    stub_r = max(t.get_window_extent(renderer=r).x1 for t in stub) / fig.dpi
    col0_l = min(t.get_window_extent(renderer=r).x0
                 for t in block if abs(t.get_position()[0] * FS.FIG_W
                                       - x_in(BIN_MIDS[0])) < 1e-6) / fig.dpi
    if stub_r > col0_l - 0.04:
        bad.append(f'stub ends at {stub_r:.2f} in, first column starts at '
                   f'{col0_l:.2f} in')

    lab = ax.xaxis.label.get_window_extent(renderer=r)
    top = max(n.get_window_extent(renderer=r).y1 for n in notes)
    wid = max(n.get_window_extent(renderer=r).x1 for n in notes) / fig.dpi
    print(f'[grid] left {GRID_L:.2f} in, right {GRID_R:.2f} in, '
          f'sheet {FS.FIG_W:.2f} x {FIG_H:.2f} in')
    print(f'[fit] note {len(lines)} lines, top {(lab.y0 - top) / fig.dpi:.3f} '
          f'in below the x label, widest line ends at {wid:.2f} in')
    print(f'[fit] stub {GRID_L:.2f} to {stub_r:.2f} in, first column from '
          f'{col0_l:.2f} in')
    print(f'[fit] panel {AX_W:.2f} x {ax_top - AX_BOT:.2f} in, '
          f'block {FIG_H - ax_top:.2f} in')
    print('[fit] clear' if not bad else '[fit] ' + '; '.join(bad))
    probs = FS.audit(fig)
    print('[audit] clean' if not probs else f'[audit] {len(probs)} problem(s)')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    FS.save(fig, OUT)
    print(f'[save] {OUT}')


if __name__ == '__main__':
    build()
