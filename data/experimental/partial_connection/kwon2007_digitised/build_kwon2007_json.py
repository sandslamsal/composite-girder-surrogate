"""Assemble data/experimental/partial_connection/kwon2007.json from the raw
vector extractions in this folder, and write the overlay check.

Inputs (this folder):
  raw_<figure>.csv, calibration.json      extract_vector_markers.py (beam charts)
  connector_loadslip_raw.csv              extract_connector_curves.py (direct-shear charts)
Outputs:
  ../kwon2007.json
  measured_center_deflection.csv          every centre-deflection reading of the five beams,
                                          zero-corrected, from the primary charts
  connector_backbones.csv                 cleaned loading-branch backbones per connector test
  overlay_check.png                       (with --render) thesis Fig. 4.20 at 600 dpi with the
                                          extracted markers and calibrated grid overlaid

Run with the ops_x86 python:
  python build_kwon2007_json.py --render <600-dpi render of dissertation PDF p. 126>

Units: kip, inch, ksi (source units; the programme is in US units).
Page references: "PDF p." counts pages of the PDF file; "printed p." is the
number printed on the page. Dissertation: printed = PDF - 22. Report 0-4124-1:
printed = PDF - 14.
"""
import argparse, csv, json, math, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(os.path.dirname(HERE), 'kwon2007.json')

DISS = 'Kwon (2008) dissertation'
REP = 'Kwon et al. (2007) TxDOT report 0-4124-1'


def dp(pdf):
    return '%s, PDF p. %d (printed p. %d)' % (DISS, pdf, pdf - 22)


def rp(pdf):
    return '%s, PDF p. %d (printed p. %d)' % (REP, pdf, pdf - 14)


def r(x, n=4):
    return None if x is None else float(round(float(x), n))


# =====================================================================
# 1. Beam load vs centre-deflection readings
# =====================================================================
PRIMARY = {
    'NON-00BS': ('thesis_fig4-16', 'NON-00BS_center', 'Fig. 4.16', 124),
    'DBLNB-30BS': ('thesis_fig4-18', 'DBLNB-30BS_center', 'Fig. 4.18', 125),
    'HASAA-30BS': ('thesis_fig4-20', 'HASAA-30BS_center', 'Fig. 4.20', 126),
    'HTFGB-30BS': ('thesis_fig4-22', 'HTFGB-30BS_center', 'Fig. 4.22', 127),
    'HASAA-30BS1': ('thesis_fig4-61', 'HASAA-30BS1_center', 'Fig. 4.61', 172),
}
CROSS = {
    'NON-00BS': [('report_fig5-14', 'NON-00BS_center'), ('report_fig5-20', 'NON-00BS_center'),
                 ('thesis_fig4-24', 'NON-00BS'), ('thesis_fig4-63', 'NON-00BS')],
    'DBLNB-30BS': [('report_fig5-14', 'DBLNB-30BS_center'), ('report_fig5-20', 'DBLNB-30BS_center'),
                   ('thesis_fig4-24', 'DBLNB-30BS')],
    'HASAA-30BS': [('report_fig5-14', 'HASAA-30BS_center'), ('report_fig5-20', 'HASAA-30BS_center'),
                   ('thesis_fig4-24', 'HASAA-30BS'), ('thesis_fig4-63', 'HASAA-30BS')],
    'HTFGB-30BS': [('thesis_fig4-24', 'HTFGB-30BS')],
    'HASAA-30BS1': [('thesis_fig4-63', 'HASAA-30BS1')],
}
FIG_REF = {'report_fig5-14': rp(105).replace(REP, REP + ' Fig. 5.14'),
           'report_fig5-20': rp(111).replace(REP, REP + ' Fig. 5.20'),
           'thesis_fig4-24': dp(128).replace(DISS, DISS + ' Fig. 4.24'),
           'thesis_fig4-63': dp(174).replace(DISS, DISS + ' Fig. 4.63')}

# Programme statements used to flag readings (all loads in kips). See the
# per-specimen elastic_limit blocks below for wording and pages.
FLAGS = {
    # elastic: strictly per CONTEXT (below first yield indication and not past the
    # programme's reported linearity limit); steel_elastic: below first yield indication only
    'NON-00BS': dict(elastic_max=99.0, steel_elastic_max=99.0, context_max=131.0,
                     states=[(40.5, 'bond_intact'), (1e9, 'bond_broken')]),
    'DBLNB-30BS': dict(elastic_max=66.0, steel_elastic_max=129.0, context_max=151.0,
                       states=[(66.0, 'friction_and_bond_intact'), (1e9, 'post_friction_release_bearing')]),
    'HASAA-30BS': dict(elastic_max=119.0, steel_elastic_max=119.0, context_max=151.0,
                       states=[(1e9, 'no_release_event_reported')]),
    'HTFGB-30BS': dict(elastic_max=56.0, steel_elastic_max=174.0, context_max=151.0,
                       states=[(56.0, 'friction_locked'), (150.5, 'stick_slip_load_drops'), (1e9, 'bearing')]),
    'HASAA-30BS1': dict(elastic_max=36.0, steel_elastic_max=159.0, context_max=151.0,
                        states=[(36.0, 'friction_intact'), (1e9, 'post_friction_release_bearing')]),
}


def load_series(key, series):
    a = pd.read_csv(os.path.join(HERE, 'raw_%s.csv' % key))
    m = a[(a.series == series) & (a.source == 'markers')].sort_values('seq')
    return m.deflection_in.to_numpy(float), m.load_kips.to_numpy(float), m


def zero_correct(d, p):
    """Shift so the first (zero-load) reading sits at the origin."""
    return d - d[0], p - p[0], float(d[0]), float(p[0])


def dedupe(d, p, idx):
    keep = [0]
    for i in range(1, len(d)):
        j = keep[-1]
        if abs(d[i] - d[j]) < 0.003 and abs(p[i] - p[j]) < 0.6:
            continue
        keep.append(i)
    keep = np.array(keep)
    return d[keep], p[keep], idx[keep]


def cross_value(P, d_ref, dc, pc):
    """Deflection of a cross-check series at load P on the branch nearest d_ref."""
    best = None
    for j in range(len(pc) - 1):
        p0, p1 = pc[j], pc[j + 1]
        if p0 == p1 or not (min(p0, p1) - 1e-9 <= P <= max(p0, p1) + 1e-9):
            continue
        v = dc[j] + (P - p0) * (dc[j + 1] - dc[j]) / (p1 - p0)
        if abs(v - d_ref) > 0.3:
            continue
        if best is None or abs(v - d_ref) < abs(best - d_ref):
            best = v
    return best


def beam_readings():
    cal = json.load(open(os.path.join(HERE, 'calibration.json')))['figures']
    out, csv_rows, stats = {}, [], {}
    for spec, (key, series, fig, pdf) in PRIMARY.items():
        d_pl, p_pl, m = load_series(key, series)
        d, p, d0, p0 = zero_correct(d_pl, p_pl)
        d, p, idx = dedupe(d, p, np.arange(len(d)))
        crosses = {}
        for ck, cs in CROSS[spec]:
            cd, cp, _ = load_series(ck, cs)
            if len(cd) == 0:       # NON-00BS markers on Fig. 4.63 were not extracted
                continue
            cdz, cpz, cd0, cp0 = zero_correct(cd, cp)
            crosses[ck] = (cdz, cpz, cd0, cp0)
        F = FLAGS[spec]
        rows, past_elastic, past_steel, announced = [], False, False, False
        pmax_seen = -1e9
        diffs = {ck: [] for ck in crosses}
        for k, (dk, pk, ik) in enumerate(zip(d, p, idx)):
            drop = pk < pmax_seen - 2.0
            # a drop at unchanged deflection is load relaxation during a hold (not a
            # slip event): that reading alone is excluded; a drop with a deflection
            # jump is an interface slip event and ends the elastic range
            relax = drop and k > 0 and abs(dk - d[k - 1]) < 0.01
            slip_event = drop and not relax
            pmax_seen = max(pmax_seen, pk)
            if pk > F['elastic_max'] or slip_event:
                past_elastic = True
            if pk > F['steel_elastic_max']:
                past_steel = True
            state = next(st for lim, st in F['states'] if (pk <= lim and not (past_elastic and st in ('friction_locked', 'friction_intact', 'friction_and_bond_intact'))))
            cc = {}
            for ck, (cdz, cpz, _, _) in crosses.items():
                v = cross_value(pk, dk, cdz, cpz) if pk > 0.5 else None
                cc[ck] = r(v, 4)
                if v is not None and not past_elastic and not relax and dk > 0.02:
                    diffs[ck].append(v - dk)
            flags = []
            if past_elastic and not announced and k > 0:
                announced = True
                flags.append('first reading past the adopted elastic limit (see elastic_limit.adopted)')
            if relax:
                flags.append('load relaxation reading: load fell by more than 2 kips at unchanged deflection (hold); excluded from stiffness (elastic=false for this reading only)')
            if slip_event:
                flags.append('reading after a load drop of more than 2 kips with a deflection jump (interface slip / friction release)')
            if k > 0 and pk < 7.0:
                flags.append('low-load reading: +/-0.01 in digitisation uncertainty is > 20 % of the deflection')
            row = dict(load=r(pk, 2), load_unit='kip (applied mid-span load, zero at start of test)',
                       deflection=r(dk, 4), deflection_unit='in (mid-span, applied load only; zero-load reading subtracted)',
                       load_kips=r(pk, 2), deflection_in=r(dk, 4),
                       as_plotted=dict(load_kips=r(p_pl[ik], 3), deflection_in=r(d_pl[ik], 4), marker_seq=int(m.seq.iloc[ik])),
                       source='digitised (vector markers), %s %s, centre series; raw row in kwon2007_digitised/raw_%s.csv' % (DISS, fig + ', ' + dp(pdf).split(', ', 1)[1], key),
                       elastic=bool(not past_elastic and not relax), steel_elastic=bool(not past_steel),
                       interface_state=state, crosscheck_deflection_in=cc, flags=flags)
            csv_rows.append(dict(specimen=spec, figure=fig, reading=k, load_kips=r(pk, 3), deflection_in=r(dk, 4),
                                 plotted_load_kips=r(p_pl[ik], 3), plotted_deflection_in=r(d_pl[ik], 4),
                                 elastic=int(not past_elastic and not relax), steel_elastic=int(not past_steel), interface_state=state,
                                 **{'xcheck_' + ck: cc[ck] for ck in cc}))
            if k > 0 and pk <= F['context_max']:     # the zero-load reference reading is not listed
                rows.append(row)
            if pk > F['context_max']:
                break
        # finish the CSV with the rest of the curve (full record)
        for k2 in range(k + 1, len(d)):
            csv_rows.append(dict(specimen=spec, figure=fig, reading=k2, load_kips=r(p[k2], 3), deflection_in=r(d[k2], 4),
                                 plotted_load_kips=r(p_pl[idx[k2]], 3), plotted_deflection_in=r(d_pl[idx[k2]], 4),
                                 elastic=0, steel_elastic=0, interface_state='', **{'xcheck_' + ck: None for ck in crosses}))
        stat = {}
        for ck, v in diffs.items():
            if v:
                v = np.array(v)
                stat[ck] = dict(n=int(len(v)), mean_in=r(v.mean(), 4), rms_in=r(np.sqrt((v ** 2).mean()), 4),
                                max_abs_in=r(np.abs(v).max(), 4))
        out[spec] = dict(rows=rows, zero_offset_as_plotted=dict(deflection_in=r(d0, 4), load_kips=r(p0, 3)),
                         cross_zero_offsets={ck: dict(deflection_in=r(v[2], 4), load_kips=r(v[3], 3)) for ck, v in crosses.items()},
                         crosscheck_stats=stat, n_readings_total=int(len(d)),
                         calibration=dict(x_residual_max_in=r(cal[key]['x_axis_deflection_in']['residual_max_abs'], 4),
                                          y_residual_max_kips=r(cal[key]['y_axis_load_kips']['residual_max_abs'], 3)))
    with open(os.path.join(HERE, 'measured_center_deflection.csv'), 'w', newline='') as fh:
        keys = []
        for row in csv_rows:
            for kk in row:
                if kk not in keys:
                    keys.append(kk)
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(csv_rows)
    return out


def theory_lines_fig520():
    """Programme's own theoretical stiffness lines on report Fig. 5.20."""
    a = pd.read_csv(os.path.join(HERE, 'raw_report_fig5-20.csv'))
    t = a[a.series == 'theory_noncomposite_solid']
    rise = t[t.load_kips < 159.0]
    k_nc = np.polyfit(rise.deflection_in, rise.load_kips, 1)[0]
    st = a[a.series == 'theory_composite_dashed_dash_start'].reset_index(drop=True)
    en = a[a.series == 'theory_composite_dashed_dash_end'].reset_index(drop=True)
    xc = 0.5 * (st.deflection_in + en.deflection_in)
    yc = 0.5 * (st.load_kips + en.load_kips)
    inc = yc < 220.0
    k_c = np.polyfit(xc[inc], yc[inc], 1)[0]
    plateau = float(yc[yc > 228].mean())
    return dict(noncomposite_bare_steel_kips_per_in=r(k_nc, 1), noncomposite_plateau_kips=r(t.load_kips.max(), 1),
                partial_composite_Ieff_kips_per_in=r(k_c, 1), partial_composite_plateau_kips=r(plateau, 1),
                note='Slopes fitted to the vector line (solid, bare steel) and to the centres of the dashes (dashed, AISC I_eff, Eq. 2.25 of the report / Eq. 2.17 of the dissertation) drawn by the authors; theory based on DBLNB-30BS measured properties (dissertation PDF p. 140). Labels on the figure: 160 kips and 232 kips.',
                source=rp(111).replace(REP, REP + ' Fig. 5.20') + '; raw rows in kwon2007_digitised/raw_report_fig5-20.csv')


# =====================================================================
# 2. Direct-shear (single connector) tests
# =====================================================================
CONN = {  # Table 3.2 (PDF p. 79), Table 3.7 (PDF p. 94), Table 3.1 (PDF p. 73)
    'DBLNB-05ST': dict(type='DBLNB', fig='Fig. 3.15', qu=43.77, smax=0.45, p02=35.98, fc=3020, grout=3670, fu=147.0, fv=91.1),
    'DBLNB-06ST': dict(type='DBLNB', fig='Fig. 3.15', qu=39.57, smax=0.40, p02=33.10, fc=3020, grout=3670, fu=147.0, fv=91.1),
    'DBLNB-07ST': dict(type='DBLNB', fig='Fig. 3.15', qu=40.45, smax=0.31, p02=38.69, fc=3020, grout=3670, fu=147.0, fv=91.1),
    'HTFGB-05ST': dict(type='HTFGB', fig='Fig. 3.16', qu=55.33, smax=1.45, p02=23.93, fc=3550, grout=None, fu=148.6, fv=87.7),
    'HTFGB-06ST': dict(type='HTFGB', fig='Fig. 3.16', qu=50.67, smax=1.54, p02=26.45, fc=3550, grout=None, fu=148.6, fv=87.7),
    'HASAA-05ST': dict(type='HASAA', fig='Fig. 3.17', qu=37.07, smax=0.41, p02=35.52, fc=2990, grout=None, fu=147.0, fv=91.1),
    'HASAA-06ST': dict(type='HASAA', fig='Fig. 3.17', qu=34.69, smax=0.42, p02=33.84, fc=2990, grout=None, fu=147.0, fv=91.1),
    'HASAA-07ST': dict(type='HASAA', fig='Fig. 3.17', qu=36.79, smax=0.39, p02=34.21, fc=2990, grout=None, fu=147.0, fv=91.1),
}
FIG_PDF = {'Fig. 3.15': 79, 'Fig. 3.16': 80, 'Fig. 3.17': 80}


def clean_backbone(g, qu):
    L = g.load_kips.to_numpy(float)
    S = g.slip_in.to_numpy(float)
    s0 = float(np.median(S[L < L.min() + 0.6]))
    levels = np.r_[np.arange(0.5, 5.0, 0.5), np.arange(5.0, math.floor(qu - 1.0) + 1e-9, 1.0)]
    lv, sv = [], []
    for x in levels:
        sel = np.abs(L - x) < 0.15
        if not sel.any():
            sel = np.abs(L - x) < 0.4
        if not sel.any():
            continue
        lv.append(float(x))
        sv.append(float(S[sel].min() - s0))
    lv, sv = np.array(lv), np.array(sv)
    # a dash gap can make a row pick a run right of the loading branch: take the
    # reverse cumulative minimum so slip never exceeds a slip at a higher load
    sv = np.minimum.accumulate(sv[::-1])[::-1]
    # peak: highest drawn row (half a stroke above Q_u); slip there
    top = L > L.max() - 0.3
    s_peak = float(np.median(S[top]) - s0)
    pts = []
    for x, s in zip(lv, sv):
        if s <= 2.5e-4:            # below ~2 px at 1200 dpi: not resolvable from zero
            continue
        if pts and s <= pts[-1][0] + 1e-5:
            continue
        pts.append((round(s, 5), round(x, 2)))
    if s_peak > pts[-1][0]:
        pts.append((round(s_peak, 5), qu))
    return pts, s0, s_peak


def slip_at(pts, load):
    s = np.array([p[0] for p in pts]); f = np.array([p[1] for p in pts])
    s = np.r_[0.0, s]; f = np.r_[0.0, f]
    if load > f.max():
        return None
    i = int(np.argmax(f >= load))
    return float(s[i - 1] + (load - f[i - 1]) * (s[i] - s[i - 1]) / (f[i] - f[i - 1]))


def load_at(pts, slip):
    s = np.array([0.0] + [p[0] for p in pts]); f = np.array([0.0] + [p[1] for p in pts])
    return float(np.interp(slip, s, f)) if slip <= s.max() else None


def connector_tests():
    raw = pd.read_csv(os.path.join(HERE, 'connector_loadslip_raw.csv'))
    per, bb_rows = {}, []
    for name, c in CONN.items():
        g = raw[raw.specimen == name]
        pts, s0, s_peak = clean_backbone(g, c['qu'])
        s5, s10, shalf = slip_at(pts, 5.0), slip_at(pts, 10.0), slip_at(pts, 0.5 * c['qu'])
        p02_dig = load_at(pts, 0.2)
        failed = name != 'HTFGB-05ST'
        full = list(pts) + ([(c['smax'], round(0.95 * c['qu'], 2))] if failed and c['smax'] > pts[-1][0] else [])
        per[name] = dict(
            connector_type=c['type'], figure=c['fig'] + ', ' + dp(FIG_PDF[c['fig']]),
            concrete_fc_psi=c['fc'], grout_fc_psi=c['grout'], connector_Fu_ksi=c['fu'], connector_Fv_ksi=c['fv'],
            Qu_kips=c['qu'], s_max_in=c['smax'], load_at_0p2in_slip_kips_table=c['p02'],
            load_at_0p2in_slip_kips_digitised=r(p02_dig, 2),
            slip_at_peak_in_digitised=r(s_peak, 4),
            secant_stiffness_kips_per_in=dict(at_5_kips=r(5.0 / s5, 0) if s5 else None,
                                              at_10_kips=r(10.0 / s10, 0) if s10 else None,
                                              at_half_Qu=r(0.5 * c['qu'] / shalf, 0) if shalf else None,
                                              at_0p2in_slip_table=r(c['p02'] / 0.2, 0)),
            slip_zero_offset_as_plotted_in=r(s0, 5),
            backbone_points_slip_in_force_kips=[[a, b] for a, b in full],
            backbone_note=('loading branch sampled at 0.5-kip steps to 5 kips and 1-kip steps to Q_u - 1, then (slip at the highest drawn point, tabulated Q_u)'
                           + (', then (tabulated s_max, 0.95 Q_u): the text states connector failure at about 0.95 Q_u on the descending branch and takes that slip as s_max (PDF p. 79)' if failed and c['smax'] > pts[-1][0] else
                              '; HTFGB-05ST was not loaded to connector failure (stopped for block cracking; the text gives 45.0 kips at 1.83 in slip, PDF p. 82, whereas Table 3.2 lists s_max = 1.45 in and the plotted curve ends near 1.45 in), so no descending point is given')),
        )
        for a, b in full:
            bb_rows.append(dict(connector_test=name, type=c['type'], slip_in=a, force_kips=b))
    types = {}
    for t in ('DBLNB', 'HTFGB', 'HASAA'):
        names = [n for n, c in CONN.items() if c['type'] == t]
        qus = np.array([CONN[n]['qu'] for n in names])
        top = math.floor(qus.min() - 1.0)
        levels = [x for x in np.r_[np.arange(0.5, 5.0, 0.5), np.arange(5.0, top + 1e-9, 1.0)]]
        mean_pts = []
        for x in levels:
            ss = [slip_at(per[n]['backbone_points_slip_in_force_kips'], x) for n in names]
            if any(s is None for s in ss):
                continue
            s = float(np.mean(ss))
            if s <= 2.5e-4 or (mean_pts and s <= mean_pts[-1][0] + 1e-5):
                continue
            mean_pts.append([round(s, 5), round(float(x), 2)])
        sp = float(np.mean([per[n]['slip_at_peak_in_digitised'] for n in names]))
        mean_pts.append([round(sp, 5), round(float(qus.mean()), 2)])
        failed = [n for n in names if n != 'HTFGB-05ST']
        smax_mean = float(np.mean([CONN[n]['smax'] for n in failed]))
        if smax_mean > sp:
            mean_pts.append([round(smax_mean, 4), round(0.95 * float(qus.mean()), 2)])
        k5 = [per[n]['secant_stiffness_kips_per_in']['at_5_kips'] for n in names]
        k10 = [per[n]['secant_stiffness_kips_per_in']['at_10_kips'] for n in names]
        kh = [per[n]['secant_stiffness_kips_per_in']['at_half_Qu'] for n in names]
        k02 = [per[n]['secant_stiffness_kips_per_in']['at_0p2in_slip_table'] for n in names]
        types[t] = dict(
            tests=names, n_tests=len(names),
            Qu_mean_kips=r(qus.mean(), 2), Qu_range_kips=[float(qus.min()), float(qus.max())],
            s_max_mean_in=r(np.mean([CONN[n]['smax'] for n in names]), 3),
            load_at_0p2in_slip_mean_kips=r(np.mean([CONN[n]['p02'] for n in names]), 2),
            secant_stiffness_kips_per_in=dict(
                at_5_kips_mean=r(np.mean(k5), 0), at_5_kips_range=[min(k5), max(k5)],
                at_10_kips_mean=r(np.mean(k10), 0), at_10_kips_range=[min(k10), max(k10)],
                at_half_Qu_mean=r(np.mean(kh), 0), at_half_Qu_range=[min(kh), max(kh)],
                at_0p2in_slip_mean=r(np.mean(k02), 0)),
            mean_backbone_points_slip_in_force_kips=mean_pts,
            mean_backbone_note='slip averaged over the tests at equal force up to min(Q_u) - 1 kip, then (mean slip at peak, mean Q_u), then (mean s_max of the tests loaded to failure, 0.95 mean Q_u). CAUTION: the first segments (slopes 2,000-20,000 kip/in) rest on slips below 0.002 in, at the resolution limit of the chart; the secant at 10 kips is the lowest reliable stiffness measure (HTFGB: friction, slip below 10 kips unresolved). If the model needs a defined toe, pass k_initial_kip_per_in to connector_curve and report it as an assumption.')
        for a, b in mean_pts:
            bb_rows.append(dict(connector_test='MEAN_' + t, type=t, slip_in=a, force_kips=b))
    with open(os.path.join(HERE, 'connector_backbones.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['connector_test', 'type', 'slip_in', 'force_kips'])
        w.writeheader()
        w.writerows(bb_rows)
    return per, types


# =====================================================================
# 3. Specimens
# =====================================================================
W30 = dict(designation='W30x99 (ASTM A992)', d_in=29.7, bf_in=10.5, tf_in=0.670, tw_in=0.520, A_in2=29.1,
           Ix_in4=3990.0, Sx_in3=269.0, Zx_in3=312.0, k_des_in=1.16, weight_plf=99.0,
           provenance='designation: dissertation PDF p. 103 and Fig. 4.1 (PDF p. 104); dimensions and properties from the AISC Steel Construction Manual (14th ed.) shape table, not printed in the sources; Zx = 312 in3 is also stated in the dissertation (PDF p. 154); overall depth with slab 36.7 in on Fig. 4.13 (PDF p. 121) = 29.7 + 7.0, so no haunch')
SPAN = 456.0
A_SC = 0.80 * math.pi / 4 * 0.875 ** 2


def as_fy(flange_fy, web_fy):
    af = 2 * W30['bf_in'] * W30['tf_in']
    return af * flange_fy + (W30['A_in2'] - af) * web_fy


def eta_block(fc_psi, fy_f, fy_w, conn_type, types):
    ac = 84.0 * 7.0
    cf_steel = as_fy(fy_f, fy_w)
    cf_conc = 0.85 * fc_psi / 1000.0 * ac
    cf = min(cf_steel, cf_conc)
    q_nom = 30.1 if conn_type != 'HTFGB' else 28.9
    fu_meas = 147.0 if conn_type != 'HTFGB' else 148.6
    fv_meas = 91.1 if conn_type != 'HTFGB' else 87.6
    q_eq228 = 0.5 * A_SC * fu_meas
    q_ds = types[conn_type]['Qu_mean_kips']
    cf_nom = min(W30['A_in2'] * 50.0, 0.85 * 3.0 * ac)
    n = 16
    return dict(
        definition='eta = sum(Q_n over one shear span) / C_f, C_f = min(A_s F_y, 0.85 f\'c b t_s) (report Eqs. 2.25-2.26, PDF pp. 36-37; dissertation Sec. 4.2.1.2, PDF p. 106); 16 connectors per shear span; longitudinal rebar not counted in C_f (as the authors did, PDF p. 106)',
        authors_stated=0.30,
        C_f_nominal_kips=r(cf_nom, 0),
        C_f_measured_kips=r(cf, 0), C_f_governed_by='steel' if cf_steel <= cf_conc else 'concrete',
        A_s_F_y_measured_kips=r(cf_steel, 0), concrete_0p85_fc_Ac_kips=r(cf_conc, 0),
        options=[
            dict(label='nominal (authors design basis)', Q_n_kips=q_nom, C_f='nominal (F_y 50 ksi, f\'c 3000 psi)', eta=r(n * q_nom / cf_nom, 3),
                 note='the authors call this 30 %; recomputation with A_s = 29.1 in2 gives the value shown'),
            dict(label='Eq. 2.28 with measured F_u (authors theory line, Fig. 4.34)', Q_n_kips=r(q_eq228, 2), C_f='measured', eta=r(n * q_eq228 / cf, 3)),
            dict(label='mean direct-shear Q_u of this connector type (recommended: strength of the measured backbone used in the model)', Q_n_kips=q_ds, C_f='measured', eta=r(n * q_ds / cf, 3)),
            dict(label='A_sc x measured shear strength F_v', Q_n_kips=r(A_SC * fv_meas, 2), C_f='measured', eta=r(n * A_SC * fv_meas / cf, 3)),
        ],
        recommended=r(n * q_ds / cf, 3))


def design_space(fc_ksi, fy_flange, eta, spec_note=''):
    def item(v, lo, hi, unit, note=None):
        d = dict(value=v, range=[lo, hi], unit=unit, inside=bool(lo <= v <= hi))
        if note:
            d['note'] = note
        return d
    return dict(
        span=item(38.0, 20, 240, 'ft'),
        deck_thickness=item(7.0, 4.5, 12, 'in'),
        girder_spacing=item(7.0, 5, 12, 'ft', 'isolated beam: spacing taken as the slab width 84 in, as for the Chapman beams in the manuscript'),
        fc=item(fc_ksi, 3, 10, 'ksi', 'test-day cylinder strength'),
        eta_c=item(eta, 0.25, 1.00, '-', spec_note or None),
        fy=item(fy_flange, 36, 100, 'ksi', 'measured static flange yield stress (web slightly higher)'),
        steel_depth=item(29.7, 12, 84, 'in'),
        bf_over_ds=item(round(10.5 / 29.7, 3), 0.30, 0.70, '-'),
        tf_over_bf=item(round(0.670 / 10.5, 4), 0.04, 0.12, '-'),
        tw_over_ds=item(round(0.520 / 29.7, 4), 0.015, 0.04, '-'),
        section=dict(value='rolled W', inside=True),
        support=dict(value='simply supported (roller + hinge)', inside=True),
        loading=dict(value='monotonic positive bending, single mid-span point load', inside=True),
        rho_l=dict(value_percent=round(100 * 14 * 0.20 / 588.0, 3), note='two mats of #4 longitudinal bars at 12 in (7 bars per mat, 14 x 0.20 in2 = 2.80 in2) over 84 x 7 in: between the manuscript tables at 0 and 0.7 %'))


def main(render):
    beams = beam_readings()
    conn_per, conn_types = connector_tests()
    theory = theory_lines_fig520()

    common_geometry = dict(
        span_in=SPAN, span_ft=38.0,
        span_provenance='"simply supported beam with a 38ft long span" (%s); "38-ft long, simply supported" (%s); supports under end web stiffeners (Fig. 4.13, %s)' % (rp(93), dp(119), dp(121)),
        supports='roller at one end, hinge at the other; end bracing that does not restrain longitudinal slab movement; web stiffeners at both ends and mid-span (%s)' % dp(119),
        load_position='single concentrated load at mid-span (x = 228 in) through two 100-ton rams and a load cell onto a 10 x 20 x 1 in steel plate on the slab (%s)' % dp(119),
        lateral_bracing='mid-span lateral bracing with Teflon sheets (%s)' % dp(119),
        steel_section=W30,
        slab=dict(type='solid cast-in-place normal-weight concrete (3/4 in river gravel), no decking, no haunch', width_in=84.0, thickness_in=7.0,
                  provenance='Fig. 4.1 (%s), Sec. 4.2.2.1 (%s), Fig. 4.13 (%s)' % (dp(104), dp(115), dp(121))),
        reinforcement=dict(transverse='#5 and #4 Grade 60 at 6 in (top and bottom mats)', longitudinal='#4 Grade 60 at 12 in in each of two mats (7 bars per mat, first bar 6 in from the slab edge, as drawn)',
                           A_long_total_in2=2.80, rho_l_percent=round(100 * 2.80 / 588.0, 3),
                           rho_l_definition='combined top and bottom longitudinal mats / (84 x 7 in), the manuscript definition',
                           bar_depths='not dimensioned; scaled from Fig. 4.1: top longitudinal bars about 2.4 in below the slab top, bottom longitudinal bars about 2.0 in above the slab soffit (the single-connector blocks of the same detail had 1.5 in top and 1 in bottom clear cover, PDF p. 62)',
                           provenance='text %s; Fig. 4.1 (%s); Fig. 4.2 (%s)' % (dp(103), dp(104), dp(105))))

    uniform_rows = [r(14.25 + 28.5 * i, 2) for i in range(16)]
    conc_rows = [12.0 * i for i in range(1, 9)] + [SPAN - 12.0 * i for i in range(8, 0, -1)]

    def uniform_layout(ctype):
        return dict(number=32, per_shear_span=16, rows=16, per_row=2,
                    transverse_arrangement='pairs, one either side of the web (%s)' % rp(120),
                    pitch_in=28.5, row_x_in_from_left_support=uniform_rows,
                    pitch_provenance='"distributed uniformly along the length of the beam" (%s); spacing 28.5 in (%s; %s)' % (dp(131), dp(141), rp(120)),
                    end_distance_assumption='End distance not stated. 16 rows x 28.5 in = 456 in = span, so a uniform 28.5 in pitch symmetric about mid-span puts the first row 14.25 in (half a pitch) from each support; the FE model mesh (Fig. 4.44, %s) and schematic Fig. 4.58(a) (%s) show evenly spaced pairs over the full length. Sensitivity expected small; the model agent may test end distance 0-28.5 in.' % (dp(156), dp(170)),
                    transverse_gauge='not stated in either document')

    def installation(ctype):
        if ctype == 'DBLNB':
            return dict(description='double-nut bolt: 7/8 in ASTM A193 B7 threaded rod, 7.25 in long, 5 in embedment, embedded double nut; 2.5 in cored hole through the slab, 15/16 in hole in the flange; pretensioned to 39 kips (Squirter DTI washers); slab hole grouted with Five Star Highway Patch',
                        provenance='%s to %s' % (dp(108), dp(109)), grout_fc_psi_test_day=7570, grout_provenance=dp(116))
        if ctype == 'HTFGB':
            return dict(description='high-tension friction-grip bolt: 7/8 in ASTM A325 bolt, 7.25 in long, 5 in embedment; 2 in x 2.75 in counterbore and 1 in cored hole through the slab (oversized in the concrete), 15/16 in hole in the flange; pretensioned to 39 kips; slab hole grouted with Five Star Highway Patch',
                        provenance='%s to %s' % (dp(111), dp(112)), grout_fc_psi_test_day=9130, grout_provenance=dp(116))
        return dict(description='adhesive anchor: 7/8 in ASTM A193 B7 threaded rod, 5 in embedment in a drilled hole (7/8 in bit worked larger), 15/16 in hole in the flange; Hilti HIT HY 150 adhesive also filling the annulus in the flange hole; nut torqued to 150 lb-ft (the report says 125 lb-ft)',
                    provenance='%s to %s; %s' % (dp(113), dp(114), rp(99)))

    conn_material = {
        'DBLNB': dict(material='ASTM A193 B7 threaded rod, same lot for all beams', Fu_specified_ksi=125.0, Fu_measured_ksi=147.0, Fv_measured_ksi=91.1, A_sc_in2=r(A_SC, 4),
                      A_sc_note='effective shear area 80 % of the gross 7/8 in area (threads in the shear plane)', provenance=dp(118) + '; ' + dp(106)),
        'HASAA': dict(material='ASTM A193 B7 threaded rod, same lot for all beams', Fu_specified_ksi=125.0, Fu_measured_ksi=147.0, Fv_measured_ksi=91.1, A_sc_in2=r(A_SC, 4),
                      A_sc_note='effective shear area 80 % of the gross 7/8 in area (threads in the shear plane)', provenance=dp(118) + '; ' + dp(106)),
        'HTFGB': dict(material='ASTM A325 bolt, same lot for all beams', Fu_specified_ksi=120.0, Fu_measured_ksi=148.6, Fv_measured_ksi=87.6, A_sc_in2=r(A_SC, 4),
                      A_sc_note='design took threads in the shear plane (28.9 kips, PDF p. 106); in the direct-shear tests the threads were NOT in the shear plane (PDF p. 82), so the direct-shear Q_u (53.0 kips mean) may overstate the beam connector strength; thread position in the beams not stated',
                      provenance=dp(118) + '; ' + dp(106) + '; ' + dp(82)),
    }

    def conn_stiffness_block(ctype):
        t = conn_types[ctype]
        return dict(
            source='single-connector direct-shear tests of the same 7/8 in connectors (Chapter 3; 1 x 6 x 36 in A36 plate on a 7 x 24 x 24 in reinforced block, displacement control, %s); per-test data in programme-level connector_direct_shear_tests' % dp(62),
            Qu_mean_kips=t['Qu_mean_kips'], s_max_mean_in=t['s_max_mean_in'],
            load_at_0p2in_slip_mean_kips=t['load_at_0p2in_slip_mean_kips'],
            secant_stiffness_kips_per_in=t['secant_stiffness_kips_per_in'],
            mean_backbone_points_slip_in_force_kips=t['mean_backbone_points_slip_in_force_kips'],
            provenance='Q_u and s_max: Table 3.2 (%s); load at 0.2 in slip: Table 3.7 (%s; its column headings misprint the test numbers, matched here by Q_u); curves digitised from Figs. 3.15-3.17 (%s, %s)' % (dp(79), dp(94), dp(79), dp(80)),
            caveats={'DBLNB': 'stiff to about 12 kips per connector, then a sudden slip of the rod in the oversized flange hole once friction from the 39 kip pretension is overcome (PDF p. 81); the block-test clamping and hole position differ from the beam',
                     'HTFGB': 'friction transfer to about 11-12 kips per connector (< 0.01 in slip at 11 kips, PDF p. 82), then slip in the oversized concrete and flange holes',
                     'HASAA': 'bearing from low load, no abrupt slip because the adhesive fills the flange-hole annulus (PDF p. 83); lower initial stiffness than DBLNB'}[ctype],
            authors_FE_model_law='the authors\' ABAQUS models used the Ollgaard et al. (1971) curve with Q_u = 0.5 A_sc F_u, not these measured curves (%s)' % dp(154))

    fc = {'NON-00BS': (5190, 6250, 3), 'DBLNB-30BS': (3560, 3680, 7), 'HASAA-30BS': (3500, 3610, 4),
          'HTFGB-30BS': (3850, 4060, 5.5), 'HASAA-30BS1': (2590, 3220, 7)}
    heat = {'NON-00BS': 1, 'DBLNB-30BS': 1, 'HASAA-30BS': 1, 'HTFGB-30BS': 2, 'HASAA-30BS1': 2}
    steel = {1: dict(flange_Fy_ksi=56.9, flange_Fu_ksi=77.4, flange_elong_pct=38, web_Fy_ksi=60.9, web_Fu_ksi=78.6, web_elong_pct=36),
             2: dict(flange_Fy_ksi=55.0, flange_Fu_ksi=75.5, flange_elong_pct=37, web_Fy_ksi=59.3, web_Fu_ksi=77.5, web_elong_pct=36)}
    rebar = {'NON-00BS': (61.6, 103.5, 35), 'DBLNB-30BS': (61.6, 103.5, 35), 'HASAA-30BS': (57.6, 99.2, 37),
             'HTFGB-30BS': (63.1, 102.5, 24), 'HASAA-30BS1': (63.1, 102.5, 24)}

    def materials(spec):
        f28, ftest, slump = fc[spec]
        s = steel[heat[spec]]
        return dict(
            concrete=dict(fc_28day_psi=f28, fc_test_day_psi=ftest, fc_test_day_ksi=ftest / 1000.0, slump_in=slump, specified_fc_psi=3000,
                          cylinders='6 x 12 in, cured beside the specimens', Ec='not measured or reported', unit_weight='normal weight (not reported; ready-mix, 3/4 in river gravel)',
                          provenance='Table 4.1 (%s); Sec. 4.2.2.1 (%s); %s' % (dp(117), dp(115), dp(104))),
            steel_beam=dict(heat=heat[spec], E_s_ksi='not measured; the authors used 29,000 ksi (%s)' % dp(154), **s,
                            yield_definition='static yield stress, mean of three readings on the plateau; two flange and two web coupons per heat',
                            provenance='Table 4.2 (%s); Sec. 4.2.2.2 (%s)' % (dp(118), dp(117))),
            rebar_no4=dict(Fy_ksi=rebar[spec][0], Fu_ksi=rebar[spec][1], elong_pct=rebar[spec][2], specified='Grade 60', provenance='Table 4.3 (%s)' % dp(119)))

    def elastic_limit(spec):
        common_my = ('first yield of the bare steel under test load only would be at P = 4 (F_y S_x - M_DL) / L; M_DL = 124.7 ft-kips = 1496 kip-in acts on the steel alone (unshored, %s)' % dp(154))
        if spec == 'NON-00BS':
            st = ['slight reduction in stiffness at about 40 kips, attributed to breaking of bond between steel and slab and possibly small support movements (%s)' % dp(128),
                  'first slab cracks at 70 kips; at 100 kips whitewash began to flake off the bottom flange "although the load-deflection plot showed no indication of yielding"; significant yielding at 130 kips; stiffness loss and switch to displacement control at 150 kips (%s)' % dp(128),
                  'peak 163.1 kips at about 6.8 in (%s)' % dp(129)]
            adopted = 'elastic = readings below 100 kips (first whitewash flaking). Readings up to 40 kips carry bond (interface_state bond_intact); the bare-steel check should use the stiffness above 40 kips or the increment 40-95 kips.'
        elif spec == 'DBLNB-30BS':
            st = ['"At load levels past about 65 kips, Specimen DBLNB-30BS started losing stiffness", attributed to connector nonlinearity near the supports and friction being overcome with slip of the rods in the oversized flange holes (%s)' % dp(131),
                  'report: load-end slip slope decreases at 60 kips (%s)' % rp(113),
                  'whitewash flaking on bottom flange and web at 130 kips; first slab crack 200 kips; displacement control from 220 kips (%s)' % dp(131),
                  'peak 231 kips; first connector fracture at 4.25 in, more than 10 at 4.5 in (%s to %s)' % (dp(131), dp(132))]
            adopted = 'elastic = readings up to 65 kips (programme linearity limit), except the load-relaxation reading at 46 kips (load fell from 49.9 kips at unchanged deflection). steel_elastic extends to readings below 130 kips (first whitewash flaking); the 65-130 kip range is steel-elastic but past the interface friction release.'
        elif spec == 'HASAA-30BS':
            st = ['no stiffness-change load is reported; report: "did not show a sudden change of slope in the load-slip curves" (%s)' % rp(113),
                  'whitewash flaking on bottom web at 120 kips, bottom flange 190 kips; first slab crack 200 kips; displacement control from 225 kips (%s)' % dp(134),
                  'peak 239 kips; 13 of 16 connectors in the south shear span fractured at 4.75-5.00 in (%s)' % dp(134)]
            adopted = 'elastic = readings below 120 kips (first whitewash flaking). The digitised secant stiffness falls gradually from about 157 kips/in at 10-25 kips to 138 kips/in at 50 kips and 126 kips/in at 100 kips, so there is continuous softening without a reported event; M/M_p <= 0.6 (applied by the model agent) will likely bind before 120 kips.'
        elif spec == 'HTFGB-30BS':
            st = ['"did not lose that initial stiffness until 55 kips loading", attributed to full composite action by friction; slip < 0.001 in at 50 kips (%s to %s)' % (dp(136), dp(137)),
                  'friction first overcome between 55 and 60 kips, load dropped by about 10 kips to 48 kips; similar drops up to 150 kips; thereafter less stiff than DBLNB-30BS and HASAA-30BS (%s)' % dp(137),
                  'first slab crack 150 kips; whitewash flaking on bottom flange and web first at 175 kips; displacement control from 210 kips (%s)' % dp(137),
                  'peak 257 kips at 9.53 in (%s)' % dp(137)]
            adopted = 'elastic = readings up to 55 kips (programme linearity limit; friction-locked interface). Readings after the first load drop are non-monotonic in load (stick-slip) up to 150 kips and are flagged.'
        else:
            st = ['"behaved very much like Specimen HASAA-30BS in the elastic range"; friction due to the anchor torque overcome at 40 kips with a loud noise and a load drop of less than 5 kips (%s)' % dp(173),
                  'whitewash flaking on the bottom flange at 160 kips; slab cracks at 190 kips; displacement control from 210 kips (%s to %s)' % (dp(173), dp(174)),
                  'peak 234.4 kips at 7.75 in; first connector failure at 6.75 in (%s)' % dp(174)]
            adopted = 'elastic = readings up to 35 kips (the last reading before the friction release at 40 kips; the next recorded reading, 38.3 kips, is after the drop with a 0.053 in deflection jump). steel_elastic extends to readings below 160 kips.'
        return dict(programme_statements=st, adopted=adopted, first_yield_note=common_my,
                    reading_interval='readings at 5-kip intervals in the elastic range (NON-00BS at about 2.5 kips as plotted), then at 0.25 in mid-span deflection increments (%s)' % dp(122),
                    plastic_strengths_by_authors='nominal properties: 137 kips non-composite, 203-206 kips at 30 %%, 236 kips full (Fig. 4.4, %s; report Fig. 5.4); measured properties (DBLNB-30BS basis): 160 kips non-composite, 232 kips 30 %% (report Fig. 5.20, %s)' % (dp(107), rp(111)))

    deflection_zero = ('Applied test load only. The steel beams were unshored: formwork hung from the flanges, slab cast, formwork removed after 7 days (%s); '
                       'the connectors were post-installed in the hardened slab, so slab and steel self-weight (M_DL = 124.7 ft-kips) are carried by the steel alone and are not in the record; the authors model it the same way (%s). '
                       'Deflection: two displacement transducers at mid-span (%s; the plotted centre curve is taken to be their mean, not stated). The load cell reads the ram force; the loading plate and load cell weight are not in the load. '
                       'As plotted, the first (zero-load) reading of each dissertation curve sits at about -0.006 to -0.023 in and -1.2 to +0.4 kips, while the report charts start at the origin to within their 0.024 in / 1.06 kip plotting resolution; '
                       'the zero-load reading has therefore been subtracted from every reading (as_plotted keeps the uncorrected values). After the shift the loads fall on the 5-kip reading steps.') % (dp(104), dp(154), dp(122))

    specs = []
    for spec in ['NON-00BS', 'DBLNB-30BS', 'HASAA-30BS', 'HTFGB-30BS', 'HASAA-30BS1']:
        b = beams[spec]
        ctype = spec.split('-')[0]
        mats = materials(spec)
        fy_f = mats['steel_beam']['flange_Fy_ksi']
        entry = dict(id=spec)
        if spec == 'NON-00BS':
            entry.update(
                role='non-composite baseline (eta = 0 reference for the bare-steel / model check)',
                included=False,
                screening=dict(decision='excluded from the R_EI statistics; use as the eta = 0 reference',
                               reasons=['eta = 0 is outside the design space eta_c 0.25-1.00 (no connectors except four welded studs at mid-span for safety, where they develop little or no composite action, %s)' % dp(120),
                                        'every other parameter is inside the design space (same section, span, slab and reinforcement as the composite beams)',
                                        'the authors state its stiffness is essentially that of the bare steel beam (%s)' % dp(140),
                                        'bond between slab and flange gives extra stiffness below about 40 kips (%s)' % dp(128)],
                               design_space_check=design_space(6.25, fy_f, 0.0, 'no shear connection')),
                geometry=common_geometry,
                materials=mats,
                connectors=dict(type='none (non-composite)', note='four welded shear studs on the flange at mid-span "to help maintain safety of the setup"; stud size not stated (%s to %s)' % (dp(119), dp(120)), degree_of_connection=dict(eta=0.0)))
        else:
            eta = eta_block(fc[spec][1], fy_f, mats['steel_beam']['web_Fy_ksi'], ctype, conn_types)
            layout = uniform_layout(ctype)
            reasons = ['all tabulated design-space parameters inside the ranges (see design_space_check); rho_l = 0.48 % lies between the manuscript tables at 0 and 0.7 %',
                       'solid slab, simply supported, monotonic positive bending under a mid-span point load (the database and Chapman validation use their own load patterns; point load noted for the model)',
                       'discrete connectors, but post-installed 7/8 in high-strength rods or bolts in oversized holes with pretension or nut torque, not welded studs: at low load the interface carries shear by friction (and by slab-flange bond, as NON-00BS shows below about 40 kips), so the early stiffness is closer to full interaction than connector flexibility alone would give; connector stiffness and strength are available from the programme\'s own direct-shear tests',
                       'unshored with connectors installed after the slab hardened: the record contains applied load only, acting on the composite section, which is the case the comparison models']
            if spec == 'DBLNB-30BS':
                reasons.append('elastic range limited to 65 kips by the programme (friction release, rod slip in oversized flange holes); 13 readings')
            if spec == 'HASAA-30BS':
                reasons.append('cleanest case: adhesive fills the flange-hole annulus, no slip event reported')
            if spec == 'HTFGB-30BS':
                reasons.append('CAVEAT: friction-grip connection. Up to the 55 kip programme limit the interface is friction-locked (slip < 0.001 in at 50 kips), so the elastic data test a near-full-interaction beam rather than partial interaction through connector deformation, and beyond it the record has repeated load drops to 150 kips. Kept because the specimen is inside the design space; recommend reporting it separately from the bearing-type connectors')
                reasons.append('eta_c depends strongly on the connector strength used: 0.34 with 0.5 A_sc F_u(measured), 0.51 with the direct-shear Q_u whose tests had no threads in the shear plane')
            if spec == 'HASAA-30BS1':
                layout = dict(number=32, per_shear_span=16, rows=16, per_row=2,
                              transverse_arrangement='assumed pairs either side of the web as in the other beams (connectors pass through the flange beside the web; schematic Fig. 4.58(c), %s, shows 8 connector symbols at each end)' % dp(170),
                              pitch_in=12.0, row_x_in_from_left_support=conc_rows,
                              pitch_provenance='"concentrated near the support and installed from the support at a 12 in. spacing. The total number of shear connectors installed in the beam was the same" (%s)' % dp(172),
                              end_distance_assumption='first row taken 12 in from each support (rows at 12-96 in from each support); the text does not give the first-row distance. Alternatives 6-90 in or 0-84 in are possible and should be run as a sensitivity. If the 16 per shear span were single connectors in 16 rows the group would extend to 192 in; the pair reading is adopted.',
                              embedment_note='6 of the 32 connectors had 4 in instead of 5 in embedment because of rebar or chairs; their positions are not given (%s)' % dp(172),
                              transverse_gauge='not stated')
                reasons.append('connectors concentrated near the supports (12 in pitch) instead of uniform: eta_c is unchanged but the R_EI table is built for connection uniform along the span, so this specimen tests the layout sensitivity; the beam model can take the layout directly')
                reasons.append('C_f governed by the concrete (f\'c 3.22 ksi); six connectors with 4 in embedment')
            entry.update(
                role='partially composite beam with post-installed connectors',
                included=True,
                screening=dict(decision='included' + (' (flagged: friction-grip connection)' if spec == 'HTFGB-30BS' else ' (flagged: non-uniform connector layout)' if spec == 'HASAA-30BS1' else ''),
                               reasons=reasons,
                               design_space_check=design_space(fc[spec][1] / 1000.0, fy_f, eta['recommended'], 'recommended value; range over definitions in connectors.degree_of_connection')),
                geometry=common_geometry,
                materials=mats,
                connectors=dict(type=ctype, installation=installation(ctype), embedment_in=5.0, layout=layout,
                                material=conn_material[ctype], stiffness_strength_from_direct_shear_tests=conn_stiffness_block(ctype),
                                degree_of_connection=eta))
        entry['loading'] = dict(pattern='single concentrated load at mid-span on a 10 x 20 in plate', control='load control with readings every 5 kips in the elastic range, then displacement control at 0.25 in increments (%s)' % dp(122),
                                monotonic=True, shoring='unshored during casting; connectors post-installed after the slab hardened',
                                dead_load_moment_on_steel_kip_ft=124.7, dead_load_provenance=dp(154),
                                load_control_switch_kips={'NON-00BS': 150, 'DBLNB-30BS': 220, 'HASAA-30BS': 225, 'HTFGB-30BS': 210, 'HASAA-30BS1': 210}[spec],
                                peak_load_kips={'NON-00BS': 163.1, 'DBLNB-30BS': 231, 'HASAA-30BS': 239, 'HTFGB-30BS': 257, 'HASAA-30BS1': 234.4}[spec])
        entry['deflection_zero_includes'] = deflection_zero
        entry['elastic_limit'] = elastic_limit(spec)
        entry['measured'] = b['rows']
        n_el = sum(1 for x in b['rows'] if x['elastic'] and x['load'] > 0.5)
        entry['digitisation'] = dict(
            primary_source='%s %s (centre series), %s' % (DISS, PRIMARY[spec][2], dp(PRIMARY[spec][3])),
            method='Charts are vector graphics (Excel via PostScript). Marker shapes were read from the PDF drawing operators with PyMuPDF (extract_vector_markers.py); a marker position is its bounding-box centre. Each axis was calibrated by a least-squares line through all labelled tick marks (7 per axis) matched to the numeric tick labels (calibration.json). Coordinates are also given in pixels of a 600-dpi render (pdftoppm -r 600); verify_raster.py and overlay_check.png confirm the positions on that render. The zero-load reading was subtracted (see deflection_zero_includes).',
            n_readings_in_elastic_range=n_el, n_readings_in_file=len(b['rows']), n_readings_full_curve=b['n_readings_total'],
            zero_offset_as_plotted=b['zero_offset_as_plotted'],
            calibration_residuals=b['calibration'],
            crosscheck_sources={k: FIG_REF[k] for k in b['cross_zero_offsets']},
            crosscheck_zero_offsets_as_plotted=b['cross_zero_offsets'],
            crosscheck_stats_elastic_readings=b['crosscheck_stats'],
            crosscheck_note='crosscheck_deflection_in in each reading is the deflection of the same specimen on another chart (zero-load reading subtracted likewise), interpolated at the same load on the nearest branch. The report charts (Figs. 5.14, 5.20) were rasterised by Excel at 96 dpi before printing, so their markers sit on a 0.75 pt grid (0.024 in, 1.06 kips steps); the dissertation charts are on a 0.06 pt grid (0.003 in, 0.12 kips).',
            uncertainty=dict(load_kips=0.5, deflection_in=0.01,
                             basis='dissertation charts: coordinate grid 0.003 in / 0.12 kip; axis calibration residuals up to 0.013 in and 0.5 kip; zero-reading shift reproducible between charts of the same specimen to about 0.005 in; elastic-range agreement with the independent report and overlay charts in crosscheck_stats (rms about 0.01 in). Relative deflection uncertainty is therefore about 0.01 / delta: 3 % at 0.35 in, 1.5 % at 0.7 in.'))
        entry['assumptions'] = [
            'W30x99 dimensions from the AISC Manual (not printed in the sources).',
            'Span 456 in between support centrelines, load at 228 in.',
            'Deflection record contains applied load only (unshored, connectors post-installed); zero-load reading subtracted.',
            'Centre deflection = mean of the two mid-span transducers (the charts give one centre curve; averaging not stated).',
        ] + ([] if spec == 'NON-00BS' else [
            'Connector pairs at 28.5 in pitch with half-pitch end distance (uniform beams) or at 12 in pitch from 12 in (HASAA-30BS1); see connectors.layout.',
            'Connector backbone from the programme\'s direct-shear tests, whose block, clamping and hole tolerances differ from the beams; the beam response at low load also includes interface friction and bond that the direct-shear stiffness may not represent.',
            'E_c not reported: the comparison must use the ACI expression at the test-day f\'c (CONTEXT).'])
        specs.append(entry)

    doc = dict(
        programme='Kwon, Hungerford, Kayir, Schaap, Ju, Klingner, Engelhardt (2007) TxDOT report 0-4124-1; Kwon (2008) PhD dissertation, UT Austin; journal version Kwon, Engelhardt, Klingner (2011)',
        key='kwon2007',
        citations=[
            'Kwon G, Hungerford B, Kayir H, Schaap B, Ju YK, Klingner R, Engelhardt M (2007). Strengthening Existing Non-Composite Steel Bridge Girders Using Post-Installed Shear Connectors. Report FHWA/TX-07/0-4124-1, Center for Transportation Research, The University of Texas at Austin.',
            'Kwon G (2008). Strengthening Existing Steel Bridge Girders by the Use of Post-Installed Shear Connectors. PhD dissertation, The University of Texas at Austin.',
            'Kwon G, Engelhardt MD, Klingner RE (2011). Experimental behavior of bridge beams retrofitted with postinstalled shear connectors. Journal of Bridge Engineering 16(4):536-545. doi:10.1061/(ASCE)BE.1943-5592.0000184 (paywalled, not opened).',
            'Not the beam tests: Kwon G, Engelhardt MD, Klingner RE (2010). Behavior of post-installed shear connectors under static and fatigue loading. Journal of Constructional Steel Research 66(4):532-541 (single-connector tests).'],
        sources=[
            dict(document='report 0-4124-1', url='https://library.ctr.utexas.edu/ctr-publications/0-4124-1.pdf', pages='Ch. 5, PDF pp. 91-120 (printed 77-106); Fig. 5.14 PDF p. 105; Fig. 5.20 PDF p. 111; conclusions PDF p. 120', local_copy='scratchpad lit_sources/kwon2007/kwon2007_report_0-4124-1.pdf (not in the repository)'),
            dict(document='dissertation', url='https://repositories.lib.utexas.edu/items/8f1481bf-7cad-4785-ba31-7903fef5370a (kwond74300.pdf)', pages='Ch. 2 PDF p. 38 (shear connection ratio); Ch. 3 PDF pp. 61-101 (direct-shear tests); Ch. 4 PDF pp. 102-180 (beam tests); Figs. 4.16-4.24 PDF pp. 124-128; Figs. 4.61, 4.63 PDF pp. 172, 174', local_copy='scratchpad lit_sources/kwon2007/kwon2008_dissertation.pdf (not in the repository)')],
        data_files=dict(digitisation_folder='kwon2007_digitised/', measured_curves_csv='kwon2007_digitised/measured_center_deflection.csv',
                        connector_backbones_csv='kwon2007_digitised/connector_backbones.csv', overlay_check='kwon2007_digitised/overlay_check.png',
                        scripts=['extract_vector_markers.py', 'verify_raster.py', 'extract_connector_curves.py', 'build_kwon2007_json.py']),
        baseline_reference_specimen='NON-00BS',
        specimens=specs,
        programme_theory_lines_report_fig5_20=theory,
        programme_stiffness_statement='"the elastic stiffness of the retrofitted beams was about 90 to 100%% greater than the baseline non-composite specimen" and was well predicted by the AISC effective moment of inertia (%s)' % rp(120),
        connector_direct_shear_tests=dict(
            description='Single-connector direct-shear tests (%s): 1 x 6 x 36 in ASTM A36 plate (F_y 48.1 ksi mill) on a 7 x 24 x 24 in block with #4 at 12 in and #5 at 6 in top and bottom, specified f\'c 3000 psi; one 7/8 in connector installed through the plate; monotonic displacement control; slip measured between plate and block (Fig. 3.14, %s)' % (dp(62), dp(77)),
            per_test=conn_per, per_type=conn_types,
            digitisation=dict(method='each curve is a vector path; it was redrawn alone with PyMuPDF, rasterised at 1200 dpi, and for every pixel row the centre of the left-most filled run gives the loading-branch slip (extract_connector_curves.py, raw rows in connector_loadslip_raw.csv, axis calibration in connector_calibration.json). build_kwon2007_json.py samples the loading branch at fixed forces, subtracts the slip at the curve start, applies a reverse cumulative minimum to remove dash-gap picks, and appends the tabulated Q_u and s_max.',
                              verification='digitised load at 0.2 in slip vs Table 3.7 and highest drawn load vs Table 3.2 Q_u are listed per test (load_at_0p2in_slip_kips_table / _digitised); the cleaned backbones give the Table 3.7 load at 0.2 in slip to within +/-0.6 kip (-0.57 to +0.48) for all eight tests; the highest drawn row is 0.4-1.0 kip above Q_u (half the stroke width), so the tabulated Q_u is used for the peak',
                              uncertainty='force +/-0.5 kip; slip +/-0.0005 in near the origin (calibration residual <= 0.0005 in on Figs. 3.15 and 3.17, 0.002 in on Fig. 3.16; stroke 0.003-0.012 in wide). Secant stiffness at 5 kips (slip 0.001-0.004 in) is therefore uncertain by 15-50 %; at 10 kips by about 5-10 % for HASAA and DBLNB; HTFGB slips below 10 kips are unresolvable (< 0.001 in).')))
    with open(OUT_JSON, 'w') as fh:
        json.dump(doc, fh, indent=1)
    print('wrote', OUT_JSON)
    for s in specs:
        el = [x for x in s['measured'] if x['elastic'] and x['load'] > 0.5]
        print(s['id'], 'included', s['included'], 'n_elastic', len(el), 'last elastic', el[-1]['load'], el[-1]['deflection'],
              'xstats', s['digitisation']['crosscheck_stats_elastic_readings'])
    for t, v in conn_types.items():
        print(t, v['Qu_mean_kips'], v['secant_stiffness_kips_per_in'])
    print(theory)
    if render:
        overlay(render)


def overlay(render_png):
    from PIL import Image, ImageDraw, ImageFont
    cal = json.load(open(os.path.join(HERE, 'calibration.json')))['figures']['thesis_fig4-20']
    ax, ay = cal['x_axis_deflection_in'], cal['y_axis_load_kips']
    S = 600.0 / 72.0
    img = Image.open(render_png).convert('RGB')
    a = pd.read_csv(os.path.join(HERE, 'raw_thesis_fig4-20.csv'))
    m = a[(a.series == 'HASAA-30BS_center') & (a.source == 'markers')]
    spec = next(s for s in json.load(open(OUT_JSON))['specimens'] if s['id'] == 'HASAA-30BS')
    el_seq = {x['as_plotted']['marker_seq']: x['elastic'] for x in spec['measured']}

    def px(d, p):
        return ((d - ax['a']) / ax['b'] * S, (p - ay['a']) / ay['b'] * S)

    try:
        font = ImageFont.truetype('/Library/Fonts/Arial.ttf', 40)
        small = ImageFont.truetype('/Library/Fonts/Arial.ttf', 34)
    except Exception:
        font = small = ImageFont.load_default()

    # panel A: whole chart, calibrated grid and all extracted centre markers
    x0, y0, x1, y1 = [v * S for v in cal['chart_region_pt']]
    full = img.crop((int(x0), int(y0), int(x1), int(y1))).copy()
    dr = ImageDraw.Draw(full)
    for dv in [0, 2, 4, 6, 8, 10, 12]:
        X, _ = px(dv, 0)
        dr.line([X - int(x0), 0, X - int(x0), full.height], fill=(0, 170, 255), width=3)
    for pv in [0, 50, 100, 150, 200, 250, 300]:
        _, Y = px(0, pv)
        dr.line([0, Y - int(y0), full.width, Y - int(y0)], fill=(0, 170, 255), width=3)
    for _, rr in m.iterrows():
        X, Y = rr.x_px600 - int(x0), rr.y_px600 - int(y0)
        dr.line([X - 14, Y, X + 14, Y], fill=(255, 0, 200), width=4)
        dr.line([X, Y - 14, X, Y + 14], fill=(255, 0, 200), width=4)
    full = full.resize((int(full.width * 0.6), int(full.height * 0.6)), Image.LANCZOS)

    def zoom(dlo, dhi, plo, phi, dgrid, pgrid, zs=4):
        zx0, zy1 = px(dlo, plo)
        zx1, zy0 = px(dhi, phi)
        ox, oy = int(zx0), int(zy0)
        z = img.crop((ox, oy, int(zx1), int(zy1))).copy()
        z = z.resize((z.width * zs, z.height * zs), Image.NEAREST)
        dz = ImageDraw.Draw(z)
        for dv in dgrid:
            X, _ = px(dv, 0)
            dz.line([(X - ox) * zs, 0, (X - ox) * zs, z.height], fill=(0, 170, 255), width=2)
            dz.text(((X - ox) * zs + 6, 6), '%.1f in' % dv, fill=(0, 120, 200), font=small)
        for pv in pgrid:
            _, Y = px(0, pv)
            dz.line([0, (Y - oy) * zs, z.width, (Y - oy) * zs], fill=(0, 170, 255), width=2)
            dz.text((6, (Y - oy) * zs - 40), '%d kips' % pv, fill=(0, 120, 200), font=small)
        for _, rr in m.iterrows():
            X, Y = (rr.x_px600 - ox) * zs, (rr.y_px600 - oy) * zs
            if not (0 <= X <= z.width and 0 <= Y <= z.height):
                continue
            flag = el_seq.get(int(rr.seq))
            col = (150, 150, 150) if flag is None else (0, 200, 0) if flag else (255, 0, 200)
            dz.line([X - 22, Y, X + 22, Y], fill=col, width=3)
            dz.line([X, Y - 22, X, Y + 22], fill=col, width=3)
            dz.ellipse([X - 8, Y - 8, X + 8, Y + 8], outline=col, width=3)
        dz.rectangle([0, 0, z.width - 1, z.height - 1], outline=(0, 0, 0), width=3)
        return z

    zb = zoom(-0.05, 0.62, -4.0, 86.0, [0.0, 0.2, 0.4, 0.6], [0, 50])
    zc = zoom(1.02, 1.62, 118.0, 196.0, [1.2, 1.4, 1.6], [150])
    top = 170
    W = full.width + zb.width + zc.width + 80
    H = top + max(full.height, zb.height, zc.height) + 20
    canvas = Image.new('RGB', (W, H), 'white')
    canvas.paste(full, (0, top))
    canvas.paste(zb, (full.width + 40, top))
    canvas.paste(zc, (full.width + zb.width + 80, top))
    dc = ImageDraw.Draw(canvas)
    dc.text((20, 15), 'Kwon (2008) Fig. 4.20, HASAA-30BS, 600 dpi. Left: blue = calibrated 2 in / 50 kip lines; crosses = extracted Center markers.', fill='black', font=font)
    dc.text((20, 65), 'Middle: 0-0.6 in, 0-85 kips, 4x (quarter-point markers cover the left half of each Center diamond). Right: 1.0-1.6 in, 118-196 kips, 4x.', fill='black', font=font)
    dc.text((20, 115), 'Green: elastic in kwon2007.json; magenta: beyond; grey: duplicate reading dropped. As-plotted positions (before zero shift).', fill='black', font=font)
    canvas.save(os.path.join(HERE, 'overlay_check.png'))
    print('wrote overlay_check.png', canvas.size)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--render', default=None)
    main(ap.parse_args().render)
