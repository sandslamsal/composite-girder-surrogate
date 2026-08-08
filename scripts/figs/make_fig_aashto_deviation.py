#!/usr/bin/env python
"""Consolidated AASHTO transformed-section deviation figure (revision 1).

Replaces the three separate figures of the submitted manuscript
(``fig_aashto_error``, ``fig_deviation_vs_moment``,
``fig_neutral_axis_migration``) with one three-panel figure:

    (a) stiffness over-prediction Delta by eta_c bin, for both load
        regimes and both deck-reinforcement levels -- the graphical form
        of Table tab:aashto (bars = bin mean, ticks = bin median);
    (b) Delta as a continuous function of the moment ratio M/M_p,
        stratified by eta_c bin, with the two reporting conventions
        (M/M_p = 0.4 and 0.6) marked;
    (c) neutral-axis migration with curvature for one representative
        section, against the fixed AASHTO elastic neutral axis.

Sign convention (identical to the manuscript):
``Delta = (phi_OS - phi_AASHTO) / phi_OS = -phi_error_pct``.

Panel (c) datum.  ``neutral_axis_in`` in the dataset is referenced to the
fiber-area centroid of the section (OpenSeesPy >= 3.4 auto-centres fiber
sections), so the per-section centroid depth from
``scripts/make_figures._fiber_centroid_depth_in`` is added to convert to
the deck-top datum on which the AASHTO elastic neutral axis is computed.
The corrected panel is self-validating: the fiber neutral axis must
coincide with the AASHTO elastic line at low curvature.

Usage::

    python scripts/figs/make_fig_aashto_deviation.py [--cache PATH]

``--cache`` stores/reuses the panel-(c) arrays (whose construction reads
the 3.9M-row dataset and runs the surrogate) so the layout can be
iterated quickly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from src.utils import figstyle as FS

AASHTO_RHO0 = ROOT / 'reports/aashto_full/aashto_comparison.parquet'
AASHTO_RHO07 = ROOT / 'reports/aashto_full_rebar007/aashto_comparison.parquet'
FULL_DATA = ROOT / 'data/raw/full_50k.parquet'
CHECKPOINT = ROOT / 'weights/best.pt'
OUT = ROOT / 'paper/revision_1/submission/sources/figures/fig_aashto_deviation.png'

REGIME_CUT = {'service': 0.4, 'extended': 0.6}
CONDITIONS = (('service', 'rho0'), ('service', 'rho07'),
              ('extended', 'rho0'), ('extended', 'rho07'))


# --------------------------------------------------------------- panels a, b
def _delta(path: Path) -> pd.DataFrame:
    """Comparison table with the manuscript's Delta sign convention."""
    df = pd.read_parquet(path, columns=['moment_ratio', 'eta_bin',
                                        'phi_error_pct'])
    return df.assign(delta=-df['phi_error_pct'])


def panel_a_stats() -> dict:
    """Bin mean/median of Delta for both regimes x both reinforcement levels."""
    frames = {'rho0': _delta(AASHTO_RHO0), 'rho07': _delta(AASHTO_RHO07)}
    stats = {}
    for regime, rho in CONDITIONS:
        df = frames[rho]
        sub = df[df['moment_ratio'] <= REGIME_CUT[regime]]
        g = sub.groupby('eta_bin')['delta']
        stats[(regime, rho)] = dict(
            mean=np.array([g.mean()[b] for b in FS.ETA_BINS]),
            median=np.array([g.median()[b] for b in FS.ETA_BINS]),
            n=int(len(sub)),
        )
    return stats


def panel_b_curves(n_bins: int = 26) -> tuple:
    """Mean Delta versus moment ratio, per eta_c bin (rho_l = 0 database)."""
    df = _delta(AASHTO_RHO0)
    edges = np.linspace(float(df['moment_ratio'].min()), 0.6, n_bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    curves = {}
    for b in FS.ETA_BINS:
        sub = df[df['eta_bin'] == b]
        mr = sub['moment_ratio'].to_numpy()
        d = sub['delta'].to_numpy()
        idx = np.clip(np.digitize(mr, edges) - 1, 0, len(centres) - 1)
        curves[b] = np.array([d[idx == k].mean() if (idx == k).any() else np.nan
                              for k in range(len(centres))])
    return centres, curves


# --------------------------------------------------- panel (a) bar identity
def bar_kw(regime: str, rho: str) -> dict:
    """The one definition of a panel-(a) bar style.

    Colour carries the load regime (the registry colour of ``regime``);
    fill carries the deck reinforcement level: ``rho0`` solid, ``rho07``
    white with hatching drawn in the same regime colour.  The key uses
    this function too, so a swatch can never drift from its bars.
    """
    col = FS.color(regime)
    if rho == 'rho0':
        return dict(facecolor=col, edgecolor=col, linewidth=0.8)
    return dict(facecolor='white', edgecolor=col, linewidth=0.9,
                hatch='////')


# The key strings, in one place.  Each is a name for a mark, never a
# sentence about it: the key already draws the black tick and the grey
# number beside its label, so spelling the mark out in brackets would
# only describe what the reader is looking at.
KEY_DIM = {'row': 'load regime', 'col': 'deck longitudinal reinforcement'}
KEY_MEDIAN = 'bin median'
KEY_MEAN = 'bin mean'
KEY_MEAN_TOKEN = '12.3'      # a value that appears nowhere in the panel


def key_col_label(rho: str, fontsize: float) -> str:
    """Column head of the key: the registry label of a rebar level.

    The dimension is named above the two columns, so the plain-language
    gloss the registry carries on ``rho0`` ('no deck rebar') would only
    repeat it and would make the two heads wildly different in width.
    The companion database's ratio is never restated here: the
    ``rho07`` head is the registry label verbatim.
    """
    lab = FS.entity_label(rho, fontsize)
    return r'$\rho_\ell = 0$' if rho == 'rho0' else lab


def _widths(fig, items: dict) -> dict:
    """Rendered width (in) of each {name: (text, fontsize)} entry."""
    probes = {k: fig.text(0.0, 0.0, s, fontsize=fs)
              for k, (s, fs) in items.items()}
    fig.canvas.draw()
    r = FS._renderer(fig)
    out = {}
    for k, t in probes.items():
        out[k] = t.get_window_extent(renderer=r).width / fig.dpi
        t.remove()
    return out


def crossing_key(fig, ax, bar_w_in: float, height_in: float = 0.80,
                 gap_in: float = 0.06) -> float:
    """Draw the two-by-two key of panel (a) above `ax`; return its height.

    The two encodings are crossed, not two independent lists, so the key
    is a table: load regime down the rows, deck reinforcement across the
    columns, and in every cell the *actual* bar style of that
    combination, drawn at the true bar width by :func:`bar_kw`.  Every
    bar in the panel therefore has exactly one swatch, and neither
    encoding can be read as a category of its own.  A second block keys
    the two marks that are drawn on top of a bar: the median tick and the
    printed bin mean.

    Column positions are measured from the rendered text, so the table
    cannot collide if the font substitutes differently on another
    machine.
    """
    h_ax_in = ax.get_position().height * fig.get_figheight()
    w_ax_in = ax.get_position().width * fig.get_figwidth()
    kax = ax.inset_axes([0.0, 1.0 + gap_in / h_ax_in, 1.0,
                         height_in / h_ax_in], transform=ax.transAxes)
    kax.set_xlim(0.0, w_ax_in)
    kax.set_ylim(0.0, height_in)
    kax.set_axis_off()
    kax.patch.set_visible(False)

    L, S, D = FS.FS_LEGEND, FS.FS_LABEL, FS.FS_ANNOT
    wd = _widths(fig, {
        'service': (FS.entity_label('service', L), L),
        'extended': (FS.entity_label('extended', L), L),
        'rho0': (key_col_label('rho0', S), S),
        'rho07': (key_col_label('rho07', S), S),
        'dim_row': (KEY_DIM['row'], D), 'dim_col': (KEY_DIM['col'], D),
        'median': (KEY_MEDIAN, L), 'mean': (KEY_MEAN, L)})

    sw, sh = bar_w_in, 0.17           # swatch = a bar, at its true width
    pad = 0.24                        # gap between table columns
    x_lab = max(wd['service'], wd['extended'])          # row-label column
    w_c = {k: max(wd[k], sw) for k in ('rho0', 'rho07')}
    x_col = {'rho0': x_lab + pad + 0.5 * w_c['rho0']}
    x_col['rho07'] = (x_col['rho0'] + 0.5 * w_c['rho0'] + pad
                      + 0.5 * w_c['rho07'])
    # the column dimension name is centred on the two columns; shift the
    # whole block right if that would run into the row dimension name
    x_mid = 0.5 * (x_col['rho0'] + x_col['rho07'])
    push = max(0.0, (x_lab + 0.20 + 0.5 * wd['dim_col']) - x_mid)
    x_col = {k: v + push for k, v in x_col.items()}
    x_mid += push

    # second block: the two marks drawn ON a bar, after a light rule
    x_sep = x_col['rho07'] + 0.5 * w_c['rho07'] + 0.46
    x_tok = x_sep + 0.30 + 0.5 * sw
    x_txt = x_tok + 0.5 * sw + 0.16
    used = x_txt + max(wd['median'], wd['mean'])

    # centre the whole key on the panel it explains
    dx = max(0.0, 0.5 * (w_ax_in - used))
    x_lab += dx
    x_mid += dx
    x_col = {k: v + dx for k, v in x_col.items()}
    x_sep, x_tok, x_txt = x_sep + dx, x_tok + dx, x_txt + dx

    y_dim, y_lvl = 0.72, 0.55
    y_row = {'service': 0.33, 'extended': 0.11}

    kax.text(x_lab, y_dim, KEY_DIM['row'], ha='right', va='center',
             fontsize=D, color='0.35')
    kax.text(x_mid, y_dim, KEY_DIM['col'], ha='center', va='center',
             fontsize=D, color='0.35')
    for rho in ('rho0', 'rho07'):
        kax.text(x_col[rho], y_lvl, key_col_label(rho, S), ha='center',
                 va='center', fontsize=S)
    for regime in ('service', 'extended'):
        kax.text(x_lab, y_row[regime], FS.entity_label(regime, L),
                 ha='right', va='center', fontsize=L)
        for rho in ('rho0', 'rho07'):
            kax.add_patch(Rectangle((x_col[rho] - 0.5 * sw,
                                     y_row[regime] - 0.5 * sh), sw, sh,
                                    **bar_kw(regime, rho)))

    kax.plot([x_sep, x_sep], [0.02, 0.64], color='0.85', lw=0.8,
             solid_capstyle='butt')
    kax.plot([x_tok - 0.5 * sw, x_tok + 0.5 * sw], [y_row['service']] * 2,
             color='black', lw=1.3, solid_capstyle='butt')
    kax.text(x_txt, y_row['service'], KEY_MEDIAN, ha='left', va='center',
             fontsize=L)
    # the token is drawn exactly as the numbers on the bars are drawn
    kax.text(x_tok, y_row['extended'], KEY_MEAN_TOKEN, ha='center',
             va='center', fontsize=FS.FS_SMALL, color='0.25')
    kax.text(x_txt, y_row['extended'], KEY_MEAN, ha='left', va='center',
             fontsize=L)

    if used > w_ax_in + 1e-6:
        print(f'  [key] {used:.2f} in wide, axes is {w_ax_in:.2f} in')
    return height_in


# ------------------------------------------------------------------ panel c
def panel_c_data(cache: Path | None = None) -> dict:
    """Neutral-axis migration of one representative section, deck-top datum."""
    if cache is not None and cache.exists():
        return {k: v for k, v in np.load(cache).items()}

    from make_figures import (_fiber_centroid_depth_in,
                              _pick_representative_sections)
    from src.models.inference import SurrogatePredictor

    df = pd.read_parquet(FULL_DATA)
    sid = _pick_representative_sections(df, n=4)[1]
    sub = df[df['sample_id'] == sid].sort_values('step_index')
    del df

    # dataset NA is referenced to the fiber-area centroid; shift to deck top
    y_c = _fiber_centroid_depth_in(sub.iloc[0])
    pred = SurrogatePredictor.load(CHECKPOINT).predict(sub)

    aashto = pd.read_parquet(AASHTO_RHO0)
    row = aashto[aashto['sample_id'] == sid].head(1)

    out = dict(
        sid=np.array(sid), eta=np.array(float(sub['composite_action'].iloc[0])),
        centroid=np.array(y_c),
        y_na_aashto=np.array(float(row['y_na_aashto_in'].iloc[0])),
        phi_true=sub['curvature_1_per_in'].to_numpy(),
        y_true=sub['neutral_axis_in'].to_numpy() + y_c,
        phi_pred=pred['curvature_1_per_in'].to_numpy(),
        y_pred=pred['neutral_axis_in'].to_numpy() + y_c,
        mr=sub['moment_ratio'].to_numpy(),
        total_depth=np.array(float(sub['total_depth_in'].iloc[0])),
        deck_t=np.array(float(sub['deck_thickness_in'].iloc[0])),
    )
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **out)
    return out


# ------------------------------------------------------------------- figure
def build(cache: Path | None = None) -> None:
    FS.apply()
    stats = panel_a_stats()
    centres, curves = panel_b_curves()
    c = panel_c_data(cache)

    fig = plt.figure(figsize=(FS.FIG_W, 5.30))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 0.92],
                  hspace=0.48, wspace=0.30,
                  left=0.085, right=0.985, top=0.800, bottom=0.078)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    # ---------------------------------------------------------- (a) bin bars
    x = np.arange(len(FS.ETA_BINS), dtype=float)
    w = 0.19
    offs = np.array([-1.5, -0.5, 0.5, 1.5]) * (w + 0.015)
    for (regime, rho), dx in zip(CONDITIONS, offs):
        s = stats[(regime, rho)]
        ax_a.bar(x + dx, s['mean'], width=w, zorder=2,
                 **bar_kw(regime, rho))
        # median as a black tick across the bar
        for xi, m in zip(x + dx, s['median']):
            ax_a.plot([xi - 0.5 * w, xi + 0.5 * w], [m, m], color='black',
                      lw=1.3, solid_capstyle='butt', zorder=4)
        # bin mean printed above (below, when negative) the bar
        for xi, m in zip(x + dx, s['mean']):
            ax_a.annotate(f'{m:.1f}', xy=(xi, m),
                          xytext=(0, 2.5 if m >= 0 else -3.0),
                          textcoords='offset points', ha='center',
                          va='bottom' if m >= 0 else 'top',
                          fontsize=FS.FS_SMALL, color='0.25')

    ax_a.axhline(0.0, color='black', lw=0.8, zorder=1)
    ax_a.set_xlim(-0.55, len(FS.ETA_BINS) - 0.45)
    ax_a.set_ylim(-9.5, 60)
    ax_a.set_yticks([0, 20, 40, 60])
    # what a bar below the zero rule means is a sentence, not a label for
    # a mark, and the caption states it; it is not repeated on the panel
    FS.eta_bin_axis(ax_a, math=True)
    ax_a.set_ylabel('stiffness over-prediction\n' r'$\Delta$ (%)')
    ax_a.spines['bottom'].set_visible(False)
    ax_a.tick_params(axis='x', length=0, pad=3)

    # the key: a two-by-two table of the four real bar styles
    pos = ax_a.get_position()
    bar_w_in = w / (ax_a.get_xlim()[1] - ax_a.get_xlim()[0]) * \
        pos.width * fig.get_figwidth()
    key_h = crossing_key(fig, ax_a, bar_w_in)
    dy_a = 1.0 + (key_h + 0.16) / (pos.height * fig.get_figheight())

    # ------------------------------------------------- (b) deviation vs M/Mp
    for b in FS.ETA_BINS:
        ax_b.plot(centres, curves[b], **FS.eta_bin_style(b, lw=1.8, marker=''))
    ax_b.axhline(0.0, color='0.6', lw=0.7, ls=':')
    ax_b.set_xlim(0.04, 0.665)
    ax_b.set_ylim(-6, 97)
    ax_b.set_yticks([0, 20, 40, 60, 80])
    for cut, txt in ((0.4, 'service'), (0.6, 'extended')):
        ax_b.axvline(cut, color='0.55', lw=0.8, ls=(0, (3, 2.2)), zorder=1)
        ax_b.annotate(f'{txt}\nM/Mp = {cut}', xy=(cut, 1.0),
                      xycoords=('data', 'axes fraction'),
                      xytext=(-3, -1), textcoords='offset points',
                      ha='right', va='top', fontsize=FS.FS_SMALL,
                      color='0.35', linespacing=1.15)
    ax_b.set_xlabel(r'moment ratio $M/M_p$')
    ax_b.set_ylabel('stiffness over-prediction\n' r'$\Delta$ (%)')
    ax_b.legend(loc='upper left', bbox_to_anchor=(0.0, 0.86),
                fontsize=FS.FS_TICK, ncol=2, handlelength=1.9,
                columnspacing=1.0, labelspacing=0.25,
                borderaxespad=0.3, alignment='left',
                title='degree of composite action',
                title_fontproperties={'size': FS.FS_SMALL})
    ax_b.get_legend().get_title().set_color('0.3')

    # -------------------------------------------------------- (c) NA migration
    phi_t = c['phi_true'] * 1e3
    phi_p = c['phi_pred'] * 1e3
    order = np.argsort(phi_p)
    ax_c.plot(phi_t, c['y_true'], **FS.style('opensees', lw=1.9, markersize=3.6,
                                             markevery=10))
    ax_c.plot(phi_p[order], c['y_pred'][order],
              **FS.style('surrogate', lw=1.9, markersize=3.6, markevery=10))
    ax_c.axhline(float(c['y_na_aashto']), lw=1.9,
                 **{k: v for k, v in FS.style('aashto').items()
                    if k in ('color', 'ls', 'label')})
    ax_c.set_xlim(0, float(phi_t.max()) * 1.02)
    lo = min(float(c['y_na_aashto']), float(c['y_true'].min()))
    ax_c.set_ylim(float(c['y_true'].max()) + 0.4, lo - 1.9)   # inverted: depth down
    ax_c.set_xlabel(r'curvature $\varphi \times 10^{3}$ (1/in)')
    ax_c.set_ylabel('neutral axis below\ndeck top (in)')
    ax_c.legend(loc='center right', handlelength=1.9, borderaxespad=0.4)
    # The coincidence of the fiber and AASHTO neutral axes at low
    # curvature is read straight off the plot and is quantified in the
    # text (to within 0.06 in), so it is not also annotated here.  The
    # self-validation print below still checks it on every run.

    # panel letters and short titles
    FS.panel(ax_a, '(a)', 'bin-wise deviation', dy=dy_a)
    FS.panel(ax_b, '(b)', 'growth with load level', dy=1.03)
    FS.panel(ax_c, '(c)', 'neutral-axis migration', dy=1.03)

    # judged at printed size: the figure is included at width=\linewidth
    scale = FS.printed_scale(fig, 1.0)
    probs = FS.audit(fig, scale=scale)
    print(f'[audit] {"clean" if not probs else f"{len(probs)} problem(s)"} '
          f'(printed scale {scale:.3f})')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    FS.save(fig, OUT)
    print(f'[save] {OUT}')

    # self-validation printed for the record
    lo = c['y_true'][:3].mean()
    print(f"[check] sid {int(c['sid'])}  eta_c {float(c['eta']):.3f}  "
          f"centroid {float(c['centroid']):.3f} in")
    print(f"[check] fiber NA at low curvature {lo:.2f} in vs AASHTO elastic "
          f"{float(c['y_na_aashto']):.2f} in  "
          f"(diff {lo - float(c['y_na_aashto']):+.2f} in); "
          f"migrates to {c['y_true'].max():.2f} in")
    for k, s in stats.items():
        print(f"[check] {k[0]:8s} {k[1]:5s} n={s['n']:7d} "
              f"mean={np.round(s['mean'], 1)} median={np.round(s['median'], 1)}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, default=None)
    build(p.parse_args().cache)
