#!/usr/bin/env python
"""Surrogate-prediction figure, fig:surrogate-pred (replaces old Figs 14 and 15).

One held-out (test split) section, drawn two ways.

Section      chosen by :func:`pick_joint_median`: among the first 150
             held-out sections whose moment rises monotonically to the last
             step, the pool median of the mean curvature discrepancy
             (surrogate against the OpenSeesPy reference, as a percentage
             of the section's peak curvature) is taken over all 150, and
             the section drawn is the one NEAREST that median among the
             candidates whose MC-Dropout band covers every load step.
Panel (a)    moment-curvature reproduction: OpenSeesPy fiber-section
             reference (black solid, circles) and the deterministic
             surrogate prediction (green dashed, squares).  The note gives
             mean |dphi| as a percentage of peak curvature, with
             dphi = reference curvature minus surrogate curvature.
Panel (b)    the same section in deviation coordinates: every mark is
             plotted as its curvature MINUS the panel (a) surrogate
             curvature, against the reference curvature of the step.  The
             surrogate is therefore the dashed zero line, the black curve
             is dphi of panel (a), and the shaded band is the MC-Dropout
             mean +/- 1.96 sigma over T = 50 dropout-active passes, drawn
             about the MC-Dropout mean (which need not coincide with the
             deterministic prediction).  A step counts as inside the band
             when |reference - MC mean| <= 1.96 sigma, which is exactly the
             reference curve lying inside the shading.

Both panels use the paper's fixed entity identities (figstyle.ENTITY):
``opensees`` = black solid with circles, ``surrogate`` = green dashed with
squares.  One shared legend above the panel row; its glyphs carry the same
markers as the drawn series.

Units: moment in 10^3 kip-in, curvature in 10^-3 1/in (a tick of 1.0 is
1.0e-3 1/in).

Run:
    /opt/anaconda3/envs/ops_x86/bin/python scripts/figs/make_fig_surrogate_prediction.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from src.utils import figstyle as FS
from src.models.inference import SurrogatePredictor

DATA = REPO_ROOT / "data/raw/full_50k.parquet"
CKPT = REPO_ROOT / "weights/best.pt"            # deployed 13-feature model
TRAIN_CFG = REPO_ROOT / "configs/training.yaml"
OUT = REPO_ROOT / "paper/revision_2/submission/sources/figures/fig_surrogate_prediction.pdf"

MC_SAMPLES = 50
SEED = 0
BAND_ALPHA = 0.30      # MC-Dropout band fill (legend patch uses the same)
LW = 1.5
MS = 3.4


# ---------------------------------------------------------------- data ----
def split_by_sample(df: pd.DataFrame):
    """Train/val/test split by sample_id, identical to make_figures.py."""
    cfg = yaml.safe_load(TRAIN_CFG.read_text())
    fr, seed = cfg["splits"], int(cfg["seed"])
    ids = df["sample_id"].unique()
    np.random.default_rng(seed).shuffle(ids)
    n = len(ids)
    n_tr = int(round(fr["train"] * n))
    n_va = int(round(fr["val"] * n))
    return set(ids[:n_tr]), set(ids[n_tr:n_tr + n_va]), set(ids[n_tr + n_va:])


def _monotonic(df: pd.DataFrame) -> dict:
    """sample_id -> True if the moment rises to the final step (no softening)."""
    if not hasattr(_monotonic, "_cache"):
        g = df.groupby("sample_id")["moment_ratio"]
        _monotonic._cache = g.apply(
            lambda x: int(np.argmax(x.to_numpy())) == len(x) - 1).to_dict()
    return _monotonic._cache


def section(df: pd.DataFrame, sid: int) -> pd.DataFrame:
    return df[df["sample_id"] == sid].sort_values("step_index")


# --------------------------------------------------------------- figure ----
def pick_joint_median(df, predictor, cand_ids, n_pool: int = 150):
    """Section nearest the median accuracy whose MC-Dropout band covers it.

    Two criteria.  Panel (a) should be typical, so the target is the pool
    MEDIAN of the mean curvature discrepancy over all ``n_pool`` candidates,
    not its minimum.  Panel (b) is meant to show what the band looks like
    when it works, so the section is taken, among the candidates the band
    covers at every step, as the one nearest that median.  Full coverage is
    NOT typical: the band is under-dispersed, and the coverage statistics
    of the same candidate pool are printed here and reported in
    Section 4.7.
    """
    rows = []
    for sid in cand_ids[:n_pool]:
        sub = df[df["sample_id"] == sid].sort_values("step_index")
        o = predictor.predict(sub)
        torch.manual_seed(SEED)
        u = predictor.predict_with_uncertainty(sub, n_samples=MC_SAMPLES)
        a = sub["curvature_1_per_in"].to_numpy() * 1e3
        b = o["curvature_1_per_in"].to_numpy() * 1e3
        pm = u["curvature_1_per_in_mean"].to_numpy() * 1e3
        bd = 1.96 * u["curvature_1_per_in_std"].to_numpy() * 1e3
        rows.append((int(sid), 100.0 * np.abs(a - b).mean() / a.max(),
                     float((np.abs(a - pm) <= bd).mean())))
    med = float(np.median([r[1] for r in rows]))
    covered = [r for r in rows if r[2] >= 0.999]
    pool = covered or rows
    sid, e, c = min(pool, key=lambda r: abs(r[1] - med))
    cov = np.array([r[2] for r in rows])
    print(f'[pick] sid {sid}: err {e:.2f} % (pool median {med:.2f} %), '
          f'coverage {100*c:.0f} %; {len(covered)}/{len(rows)} fully covered')
    print(f'[calibration] {len(rows)} candidates: coverage median '
          f'{100*np.median(cov):.1f} %, mean {100*cov.mean():.1f} %, '
          f'{len(covered)} covered in full')
    return sid


def build() -> None:
    FS.apply()
    df = pd.read_parquet(DATA)
    _tr, _va, te_ids = split_by_sample(df)
    df_te = df[df["sample_id"].isin(te_ids)].reset_index(drop=True)
    predictor = SurrogatePredictor.load(CKPT)

    mono = _monotonic(df)
    pool = [int(i) for i in df_te["sample_id"].drop_duplicates()
            if mono.get(int(i), False)]
    sid = pick_joint_median(df, predictor, pool)
    sub = section(df, sid)

    fig = plt.figure(figsize=(FS.FIG_W, 3.20))
    gs = fig.add_gridspec(1, 2, wspace=0.30,
                          left=0.085, right=0.988, top=0.800, bottom=0.160)
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    for ax in (axa, axb):
        ax.grid(True, color='0.90', ls=':', lw=0.5, zorder=0)
        ax.set_axisbelow(True)

    # ---- (a) moment-curvature reproduction
    pred = predictor.predict(sub)
    phi_t = sub["curvature_1_per_in"].to_numpy() * 1e3      # 10^-3 1/in
    phi_p = pred["curvature_1_per_in"].to_numpy() * 1e3
    mom = sub["moment_kip_in"].to_numpy() / 1e3             # 10^3 kip-in
    n = len(mom)
    mev = max(1, n // 12)
    # the two curves coincide at this scale, so the marker sets are
    # staggered by half a period: circles never hide under squares
    mev_ref, mev_sur = (mev // 2, mev), (0, mev)
    axa.plot(phi_t, mom, **FS.style('opensees', label=False, lw=LW, ms=MS,
                                    markevery=mev_ref), zorder=3)
    axa.plot(phi_p, mom, **FS.style('surrogate', label=False, lw=LW, ms=MS,
                                    markevery=mev_sur), zorder=4)
    err = 100.0 * np.abs(phi_t - phi_p).mean() / phi_t.max()
    axa.set_xlim(0.0, phi_t.max() * 1.04)
    axa.set_ylim(0.0, mom.max() * 1.06)
    axa.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    axa.set_xlabel(r'curvature $\varphi$ ($10^{-3}$ 1/in)')
    axa.set_ylabel(r'moment $M$ ($10^{3}$ kip-in)')

    d = float(sub['total_depth_in'].iloc[0])
    e = float(sub['composite_action'].iloc[0])
    st = 'W' if sub['section_type'].iloc[0] == 'W' else 'welded plate'
    axa.text(0.955, 0.135, rf'{st}, $d = {d:.0f}$ in, $\eta_c = {e:.2f}$',
             transform=axa.transAxes, ha='right', va='bottom',
             fontsize=FS.FS_LABEL)
    axa.text(0.955, 0.05,
             rf'mean $|\Delta\varphi|$ = {err:.1f} % of peak $\varphi$',
             transform=axa.transAxes, ha='right', va='bottom',
             fontsize=FS.FS_SMALL, color='0.35')

    # ---- (b) same section, every mark measured from the (a) surrogate
    torch.manual_seed(SEED)
    unc = predictor.predict_with_uncertainty(sub, n_samples=MC_SAMPLES)
    phi_m = unc["curvature_1_per_in_mean"].to_numpy() * 1e3
    band = 1.96 * unc["curvature_1_per_in_std"].to_numpy() * 1e3
    x = phi_t                                  # abscissa: reference curvature
    dev = phi_t - phi_p                        # reference minus surrogate
    off = phi_m - phi_p                        # MC mean minus surrogate
    axb.fill_between(x, off - band, off + band, color=FS.color('surrogate'),
                     alpha=BAND_ALPHA, lw=0, zorder=1)
    axb.plot(x, np.zeros_like(x),
             **FS.style('surrogate', label=False, lw=LW, ms=MS,
                        markevery=mev_sur), zorder=3)
    axb.plot(x, dev, **FS.style('opensees', label=False, lw=LW, ms=MS,
                                markevery=mev_ref), zorder=4)
    out = np.abs(phi_t - phi_m) > band         # same test as the pick
    if out.any():
        axb.plot(x[out], dev[out], ls='none', marker='o', ms=MS + 0.4,
                 mfc='white', mec=FS.color('opensees'), mew=0.9, zorder=5)
    lo = float(min(dev.min(), (off - band).min()))
    hi = float(max(dev.max(), (off + band).max()))
    dmax = max(abs(lo), abs(hi))
    axb.set_xlim(0.0, x.max() * 1.04)
    axb.set_ylim(-1.32 * dmax, 1.15 * dmax)   # room for the note below
    axb.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    axb.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10],
                                            symmetric=True))
    axb.set_xlabel(r'curvature $\varphi$ ($10^{-3}$ 1/in)')
    axb.set_ylabel(r'$\Delta\varphi$ from surrogate ($10^{-3}$ 1/in)')
    inside = int((~out).sum())
    cov = 100.0 * inside / n
    axb.text(0.955, 0.035, f'{inside} of {n} steps inside band ({cov:.0f} %)',
             transform=axb.transAxes, ha='right', va='bottom',
             fontsize=FS.FS_SMALL, color='0.35')
    print(f'[coverage] {inside} of {n} steps inside')
    print(f'[offset] |MC mean - surrogate| max {np.abs(off).max():.4f}, '
          f'|dev| max {np.abs(dev).max():.4f}, band max {band.max():.4f} '
          f'(10^-3 1/in)')

    FS.panel(axa, '(a)', 'Moment–curvature reproduction')
    FS.panel(axb, '(b)', 'Epistemic uncertainty')

    handles = [FS.handle('opensees', lw=LW, ms=MS),
               FS.handle('surrogate', lw=LW, ms=MS),
               Patch(facecolor=FS.color('surrogate'), alpha=BAND_ALPHA, lw=0,
                     label=r'MC-Dropout $\pm 1.96\sigma$, 50 passes')]
    if out.any():
        handles.append(Line2D([], [], ls='none', marker='o', ms=MS + 0.4,
                              mfc='white', mec=FS.color('opensees'), mew=0.9,
                              label='step outside band'))
    fig.legend(handles=handles, loc='lower center',
               bbox_to_anchor=(0.53, 0.905), ncol=len(handles),
               frameon=False, fontsize=FS.FS_LEGEND, handlelength=2.6,
               columnspacing=1.8)

    probs = FS.audit(fig)
    print('[audit] clean' if not probs else f'[audit] {len(probs)} problem(s)')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    FS.save(fig, OUT)
    w = fig.get_tightbbox(fig.canvas.get_renderer()).width
    print(f'[width] tight crop {w:.3f} in (target {FS.FIG_W})')
    print(f'[save] {OUT}')


if __name__ == '__main__':
    build()
