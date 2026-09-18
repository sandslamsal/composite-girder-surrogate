"""Assemble data/experimental/partial_connection/provines2019.json.

Source: Provines, Ocel & Zmetra (2019), FHWA-HRT-20-005. Report page = PDF page - 18.
Inputs (this folder): fig68_levels.csv, fig68_traces_raw.csv (digitise_fig68.py),
pushout_curves_vector.csv (extract_pushout_vectors.py). Everything numeric that is
not read from those files is typed below with its page reference.

Usage:  python build_provines2019_json.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "provines2019.json"

A_SHEAR_IN = 138.0          # load point to bearing centreline (Fig. 32 p. 41; 11.5-ft shear spans p. 40)
SPAN_IN = 348.0
ELASTIC_MAX = 700.0         # kip-ft, linear range of the digitised traces (see elastic_limit)
TAN_WIN = (300.0, 700.0)    # kip-ft, tangent-stiffness window
LEVELS = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700,
          750, 800, 850, 900, 950, 1000, 1050, 1100]

# ---------------------------------------------------------------- Fig. 68
lev = pd.read_csv(HERE / "fig68_levels.csv")
raw = pd.read_csv(HERE / "fig68_traces_raw.csv")
vis = raw[raw.status == "visible"].copy()


def tangent(sid):
    g = vis[(vis.series == sid) & (vis.moment_kipft >= TAN_WIN[0]) & (vis.moment_kipft <= TAN_WIN[1])]
    p, cov = np.polyfit(g.moment_kipft, g.displacement_in, 1, cov=True)
    res = g.displacement_in - np.polyval(p, g.moment_kipft)
    return {"k_tangent_kipft_per_in": round(1.0 / p[0], 0),
            "k_tangent_kip_per_in_per_load_point": round(1.0 / p[0] * 12.0 / A_SHEAR_IN, 1),
            "slope_standard_error_pct": round(100 * np.sqrt(cov[0, 0]) / p[0], 1),
            "intercept_in": round(p[1], 4), "n_rows": int(len(g)),
            "rms_residual_in": round(float(np.sqrt((res ** 2).mean())), 4),
            "window_kipft": list(TAN_WIN),
            "method": "least-squares line displacement = M/k + intercept through every visible source-pixel "
                      "row of the trace in the window (fig68_traces_raw.csv)"}, p


def point(M, x, unc_x, unc_m, rule, src_rows, intercept, extra_flags=()):
    flags = list(extra_flags)
    el = M <= ELASTIC_MAX + 1e-9
    if M < TAN_WIN[0] and el:
        flags.append("below 300 kip-ft: initial branch differs between replicates (stiff toe / record-zero "
                     "ambiguity); secant stiffness here is unreliable")
    if not el:
        flags.append("beyond the linear range (context only)")
    return {"load": round(float(M), 1),
            "load_unit": "kip-ft (applied midspan moment from load cells; dead-load moment excluded)",
            "load_kip_per_load_point": round(float(M) * 12.0 / A_SHEAR_IN, 2),
            "deflection": round(float(x), 4),
            "deflection_unit": "in. (midspan LVDT, zero at start of test)",
            "deflection_offset_corrected_in": round(float(x) - intercept, 4),
            "uncertainty": {"load_kipft": round(float(unc_m), 1), "deflection_in": round(float(unc_x), 3),
                            "relative_deflection": round(float(unc_x) / max(float(x), 1e-6), 3)},
            "source": f"digitised, FHWA-HRT-20-005 Fig. 68, report p. 80 (PDF p. 98); {src_rows}; rule: {rule}",
            "elastic": bool(el),
            "in_tangent_window": bool(TAN_WIN[0] <= M <= TAN_WIN[1]),
            "flags": flags}


def measured_levels(sid, intercept):
    g = lev[(lev.series == sid) & lev.displacement_in.notna() & lev.moment_kipft.isin(LEVELS)]
    return [point(r.moment_kipft, r.displacement_in, r.displacement_unc_in, r.moment_unc_kipft,
                  r.rules, f"fixed level interpolated between source rows {r.rows_used}", intercept)
            for r in g.itertuples()]


def measured_nearest_rows(sid, intercept, tol=12.0):
    """3S1 is mostly hidden under 4S1 (drawn on top): no interpolation across hidden rows;
    take the visible source row nearest each level within +-tol kip-ft, at its own moment."""
    g = vis[vis.series == sid]
    out = []
    for M in LEVELS:
        d = (g.moment_kipft - M).abs()
        if d.min() <= tol:
            r = g.loc[d.idxmin()]
            out.append(point(r.moment_kipft, r.displacement_in, r.displacement_unc_in, 7.9,
                             r.centre_rule, f"nearest visible source row {int(r.cell_row)} to level {M}",
                             intercept, ["load is the source-row moment, not a round level"]))
    return out


# ---------------------------------------------------------------- push-out
po = pd.read_csv(HERE / "pushout_curves_vector.csv")
PO_TABLE = {  # total max load (kips), slip at max (in.); Tables 15-18, report pp. 104, 106, 107, 109
    "PO-S1-L3D-CIP": (183.1, 0.38), "PO-S2-L3D-CIP": (185.4, 0.28), "PO-S1-L4D-CIP": (279.5, 0.41),
    "PO-S2-L4D-CIP": (205.5, 0.32), "PO-S1-L5D-CIP": (237.6, 0.41), "PO-S2-L5D-CIP": (259.2, 0.31),
    "PO-S1-L6D-CIP": (241.7, 0.30), "PO-S2-L6D-CIP": (208.7, 0.29),
    "PO-S1-L3D-PC": (193.3, 0.25), "PO-S2-L3D-PC": (191.0, 0.28), "PO-S1-L4D-PC": (229.7, 0.43),
    "PO-S2-L4D-PC": (201.5, 0.28), "PO-S1-L5D-PC": (197.9, 0.30), "PO-S2-L5D-PC": (197.7, 0.26),
    "PO-S1-L6D-PC": (195.5, 0.35), "PO-S2-L6D-PC": (194.4, 0.24),
    "PO-S1-T3D-CIP": (219.6, 0.35), "PO-S2-T3D-CIP": (238.2, 0.27), "PO-S1-T4D-CIP": (240.0, 0.32),
    "PO-S2-T4D-CIP": (219.2, 0.31),
    "PO-S1-T3D-PC": (215.4, 0.27), "PO-S2-T3D-PC": (191.3, 0.28), "PO-S1-T4D-PC": (194.9, 0.31),
    "PO-S2-T4D-PC": (213.6, 0.33)}
FIGPAGE = {"Fig92_CIP_longitudinal": "Fig. 92, report p. 103", "Fig93_PC_longitudinal": "Fig. 93, report p. 105",
           "Fig94_CIP_transverse": "Fig. 94, report p. 107", "Fig95_PC_transverse": "Fig. 95, report p. 108"}


def ascending(g):
    L, S = g.total_load_kip.to_numpy() / 4.0, g.slip_in.to_numpy()
    ip = int(np.argmax(L))
    return L[: ip + 1], S[: ip + 1], L, S, ip


def slip_at(Lq, Sq, q):
    i = int(np.argmax(Lq >= q))
    if Lq[i] < q:
        return None
    if i == 0:
        return float(Sq[0])
    return float(Sq[i - 1] + (q - Lq[i - 1]) * (Sq[i] - Sq[i - 1]) / (Lq[i] - Lq[i - 1]))


po_rows = {}
for (fig, sid), g in po.groupby(["figure", "specimen"], sort=False):
    La, Sa, L, S, ip = ascending(g)
    qmax = PO_TABLE[sid][0] / 4.0
    m = (La >= 0.05 * qmax) & (La <= 0.40 * qmax)
    p = np.polyfit(Sa[m], La[m], 1)
    sec = {}
    for fr in (0.25, 0.40, 0.50):
        s = slip_at(La, Sa, fr * qmax)
        sec[f"secant_{int(fr * 100)}pct_kip_per_in"] = round(fr * qmax / s, 0) if s and s > 0 else None
    noisy = bool(Sa.min() < -0.002 or p[0] <= 0 or np.any(np.diff(Sa[La <= 0.4 * qmax]) < -0.002))
    po_rows[sid] = {"figure": FIGPAGE[fig], "deck": "PC (precast panel, grouted pocket)" if sid.endswith("PC") else "CIP",
                    "stud_spacing": sid.split("-")[2][:1] + " " + sid.split("-")[2][1:],
                    "Qmax_total_kip_table": PO_TABLE[sid][0], "Qmax_per_stud_kip": round(qmax, 2),
                    "slip_at_Qmax_in_table": PO_TABLE[sid][1],
                    "k_fit_5_40pct_kip_per_in_per_stud": round(float(p[0]), 0), **sec,
                    "noisy_initial_branch": noisy}

PC_L = [k for k in po_rows if k.endswith("PC") and "-L" in k]
PC_L_clean = [k for k in PC_L if not po_rows[k]["noisy_initial_branch"]]
PC_T_clean = [k for k in po_rows if k.endswith("PC") and "-T" in k and not po_rows[k]["noisy_initial_branch"]]
CIP_L_clean = [k for k in po_rows if k.endswith("CIP") and "-L" in k and not po_rows[k]["noisy_initial_branch"]]

QGRID = [2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 25, 30, 35, 40, 45]
backbone = []
for q in QGRID:
    ss = []
    for k in PC_L_clean:
        g = po[po.specimen == k]
        La, Sa, *_ = ascending(g)
        s = slip_at(La, Sa, q)
        if s is not None:
            ss.append(s)
    backbone.append({"Q_per_stud_kip": q, "slip_mean_in": round(float(np.mean(ss)), 4),
                     "slip_min_in": round(float(np.min(ss)), 4), "slip_max_in": round(float(np.max(ss)), 4),
                     "n_curves": len(ss)})


def curve_points(sid):
    g = po[po.specimen == sid]
    La, Sa, L, S, ip = ascending(g)
    pts = []
    for q in list(np.arange(2.5, La.max(), 2.5)):
        s = slip_at(La, Sa, q)
        pts.append({"Q_per_stud_kip": round(float(q), 2), "slip_in": round(s, 4), "branch": "ascending"})
    pts.append({"Q_per_stud_kip": round(float(La.max()), 2), "slip_in": round(float(Sa[-1]), 4), "branch": "peak (curve)"})
    Sp, Lp = S[ip:], L[ip:]
    for s_t in (0.30, 0.35, 0.40, 0.45, 0.50):
        j = np.flatnonzero(Sp >= s_t)
        if j.size:
            pts.append({"Q_per_stud_kip": round(float(Lp[j[0]]), 2), "slip_in": round(float(Sp[j[0]]), 4),
                        "branch": "post-peak"})
    return pts


def mean_of(keys, field):
    v = [po_rows[k][field] for k in keys]
    return round(float(np.mean(v)), 1), round(float(np.std(v, ddof=1)), 1) if len(v) > 1 else None


# ---------------------------------------------------------------- common specimen data
STEEL = {"designation": "W27x84, ASTM A992 Grade 50, rolled (Fig. 12 p. 19; p. 11)",
         "handbook_dimensions": {"d_in": 26.7, "bf_in": 9.96, "tf_in": 0.640, "tw_in": 0.460, "A_in2": 24.7,
                                 "Ix_in4": 2850, "Sx_in3": 213, "Zx_in3": 244, "k_des_in": 1.24,
                                 "source": "AISC Steel Construction Manual (15th ed.) Table 1-1; not tabulated in the "
                                           "report (report quotes only the designation). Ratios: bf/d 0.373, tf/bf 0.064, tw/d 0.0172"},
         "length_in": 360, "bearing_stiffeners": "1-in. A36 plates both sides of web at each bearing (Fig. 12 p. 19)"}
SLAB = {"type": "solid full-depth precast concrete deck panels (no ribs, no steel deck; p. 11, p. 80)",
        "width_in": 48, "thickness_in": 8,
        "panels": "two panels 171 in. (14 ft 3 in.) long placed end to end with an 18-in. grouted transverse closure pour "
                  "at midspan (p. 11; Fig. 7 p. 14; Fig. 105 drawing D4)",
        "haunch": "1-in. grout haunch formed with levelling bolts over the top flange, filled with prebagged non-shrink "
                  "grout together with the pockets and the closure joint (p. 13; Fig. 12 p. 19)",
        "pockets": "full-depth grouted shear pockets 10 in. transverse; length 5 in. (1 stud), 8.5 in. (2), 12 in. (3), "
                   "15.25 in. (4) (Fig. 109 drawing D8, report p. 131)",
        "reinforcement": {"longitudinal": "6 No. 4 bars (mark 402, 15 ft 6 in. long, projecting into the closure pour) "
                                          "in each of the top and bottom mats = 12 No. 4, As = 2.40 in2 "
                                          "(reinforcement schedules of Fig. 105 drawing D4, 12-in. panel, report p. 127, and Fig. 108 drawing D7, "
                                          "48-in. panel, p. 130: identical; 24- and 36-in. panels not checked)",
                          "transverse": "No. 5 (mark 501) in top and bottom mats: at 12 in. in the 12-in. panel (D4); 18 per "
                                        "mat at variable spacing plus 12 short No. 5 pocket trim bars (mark 503) in the "
                                        "48-in. panel (D7)",
                          "clear_cover_in": {"top": 2.5, "bottom": 1.0, "source": "Fig. 105 drawing D4 Section B"},
                          "rho_l_pct": round(100 * 2.40 / (48 * 8), 3),
                          "rho_l_definition": "combined top+bottom longitudinal area / gross deck area 48 x 8 in. "
                                              "(manuscript definition)",
                          "bar_layer_order": "not legible at available resolution (longitudinal vs transverse outer layer); "
                                             "centroid depths therefore approximate"},
        "flange_interface": "top flange greased before placing panels to eliminate bond between flange and grouted "
                            "haunch; horizontal shear carried by the studs only (p. 13)"}
LOADING = {"support": "simply supported on bearings 6 in. from each beam end, span 348 in. (29 ft) c/c; west roller, "
                      "east pin (Fig. 32 p. 41)",
           "pattern": "four-point bending, two equal loads 138 in. (11.5 ft) from the bearings, 72-in. constant-moment "
                      "region (p. 40; Fig. 32 p. 41)",
           "application": "transverse spreader beam at each load point pulled down by PT rods and hollow-core jacks; "
                          "load cells on spherical bearings between spreader and deck; passive lateral bracing (p. 40)",
           "load_to_moment": "M (kip-ft) = P per load point (kips) x 11.5 ft; P = M x 12 / 138",
           "history": "monotonic to failure in one test; loading paused about 30 min for laser-tracker readings at applied "
                      "moments of 460, 860, 1,270, 1,380 and 1,500 kip-ft (pp. 42, 80, 87); pauses show as small moment dips",
           "shoring": "not stated. Panels were set on the steel beam with levelling bolts and grouted in place (p. 13), so "
                      "slab, grout and steel self-weight were carried by the steel section; no shoring is mentioned"}
ZERO = ("Applied load only. Fig. 68 plots applied midspan moment (load cells) against midspan LVDT displacement, both "
        "starting at zero at the start of the test; the dead-load moment is excluded (p. 79), so self-weight deflection "
        "of steel, panels and grout is not in the record. Not stated: whether the spreader beams, load cells and "
        "bearings were in place before the load cells and LVDT were zeroed. The 300-700 kip-ft tangent lines of the four "
        "replicates extrapolate to -0.05 to -0.17 in. at zero moment, i.e. the record from 0 to about 300 kip-ft is "
        "stiffer than the later linear branch and, for 2S1, stiffer than any full-interaction section could be, so the "
        "absolute zero is uncertain by up to about 0.1-0.17 in. (see elastic_limit and assumptions).")
ELASTIC = {"programme_statements": [
               "No linearity limit, first-yield load or strain data are reported for the static beams; yield lines were "
               "visible after testing (Fig. 63 p. 75).",
               "Large-scale fatigue beams of identical construction were cycled 'while still maintaining elastic behavior' "
               "(p. 28) with a maximum actuator load of 41.9 kips per load point (Table 10 p. 71), i.e. an applied "
               "moment of 41.9 x 11.5 = 482 kip-ft.",
               "Calculated Mn (AISC, measured properties, dead-load moment subtracted): 1,831 kip-ft for the one-stud "
               "beams (1S1 1,831; 2S1 1,834; 3S1 1,831; 4S1 1,829) and 2,157 kip-ft for 1S2 (Table 11 p. 81)."],
           "digitised_linearity": "For 1S1, 2S1 and 4S1 the fixed-level displacements depart from the 300-600 kip-ft "
                                  "line by 0.003-0.025 in. at 700 kip-ft (within digitisation uncertainty), by 0.03-0.07 "
                                  "in. at 800 kip-ft and by 0.12-0.22 in. at 1,000-1,100 kip-ft. "
                                  "Elastic flag set for M <= 700 kip-ft.",
           "adopted_limit_kipft": ELASTIC_MAX,
           "note": "M/M_p is to be computed by the modelling step with the database definition; with M_p of order "
                   "1,830 + about 50 kip-ft dead-load moment, 700 kip-ft applied is about M/M_p = 0.37-0.40 "
                   "(service regime)."}

LAYOUT = {
    "1S1": {"studs_per_shear_span": 12, "pattern": "single studs at 12-in. pitch",
            "positions_from_bearing_in_per_half": [3, 15, 27, 39, 51, 63, 75, 87, 99, 111, 123, 135],
            "transverse_per_row": 1, "figure": "Fig. 7 p. 14"},
    "2S1": {"studs_per_shear_span": 12, "pattern": "6 clusters of 2 studs, 24-in. cluster spacing, 3.5-in. (4d) pitch within cluster",
            "cluster_centres_from_bearing_in_per_half": [9, 33, 57, 81, 105, 129],
            "positions_from_bearing_in_per_half": [c + o for c in [9, 33, 57, 81, 105, 129] for o in (-1.75, 1.75)],
            "transverse_per_row": 1, "figure": "Fig. 9 p. 16"},
    "3S1": {"studs_per_shear_span": 12, "pattern": "4 clusters of 3 studs, 36-in. cluster spacing, 3.5-in. pitch",
            "cluster_centres_from_bearing_in_per_half": [15, 51, 87, 123],
            "positions_from_bearing_in_per_half": [c + o for c in [15, 51, 87, 123] for o in (-3.5, 0.0, 3.5)],
            "transverse_per_row": 1, "figure": "Fig. 10 p. 17"},
    "4S1": {"studs_per_shear_span": 12, "pattern": "3 clusters of 4 studs, 48-in. cluster spacing, 3.5-in. pitch",
            "cluster_centres_from_bearing_in_per_half": [21, 69, 117],
            "positions_from_bearing_in_per_half": [c + o for c in [21, 69, 117] for o in (-5.25, -1.75, 1.75, 5.25)],
            "transverse_per_row": 1, "figure": "Fig. 11 p. 18"},
    "1S2": {"studs_per_shear_span": 24, "pattern": "12 rows at 12-in. pitch, 2 studs per row 4-3/8 in. apart transversely",
            "positions_from_bearing_in_per_half": [3, 15, 27, 39, 51, 63, 75, 87, 99, 111, 123, 135],
            "transverse_per_row": 2, "figure": "Fig. 8 p. 15"},
}
FC = {"1S2": (8.0, "8.0, 8.1"), "1S1": (9.1, "8.9, 9.2"), "2S1": (10.0, "10.3, 9.6"), "3S1": (9.2, "10.0, 8.4"),
      "4S1": (8.5, "9.0, 8.0")}
GROUT = {"1S2": 7.6, "1S1": 6.9, "2S1": 7.0, "3S1": 6.7, "4S1": 7.6}
MN = {"1S2": 2157, "1S1": 1831, "2S1": 1834, "3S1": 1831, "4S1": 1829}
MMAX = {"1S2": 2077, "1S1": 1667, "2S1": 1692, "3S1": 1693, "4S1": 1711}
FAIL = {"1S2": "grout crushing at midspan", "1S1": "horizontal shear stud failure in east shear span",
        "2S1": "grout crushing at midspan", "3S1": "grout crushing at midspan", "4S1": "grout crushing at midspan"}

QN_AISC = round(0.601 * 73.5, 1)               # Asc Fu, stud steel governs (p. 79)
CF = round(24.7 * 56.4, 0)                      # As Fy; 0.85 f'c Ac = 0.85 x 8.0-10.0 x 384 = 2,611-3,264 kips


def connectors(sid):
    lay = LAYOUT[sid]
    n = lay["studs_per_shear_span"]
    n_cm = 3 * lay["transverse_per_row"]
    qpo = mean_of(PC_L, "Qmax_per_stud_kip")[0]
    kpo, kpo_sd = mean_of(PC_L_clean, "k_fit_5_40pct_kip_per_in_per_stud")
    return {
        "type": "headed shear studs 7/8 in. diameter x 6 in. height, welded to AASHTO/AWS D1.5, 5-in. embedment into "
                "the deck, at least 3-in. cover, in full-depth grouted pockets (p. 13; Fig. 12 p. 19)",
        "Asc_in2": 0.601,
        "stud_material": {"Fu_ksi": 73.5, "Fy_0.2pct_ksi": 56.4, "static_yield_ksi": 54.3,
                          "source": "Table 21 p. 141 (5 tests, Fu 71.6-74.9); nominal Fu 60 ksi on drawings (Fig. 12)"},
        "layout": {**lay,
                   "constant_moment_region_studs_per_half": {
                       "positions_from_bearing_in": [146, 157, 168], "per_position": lay["transverse_per_row"],
                       "note": "three rows per half between load point (138 in.) and midspan (174 in.), the last one "
                               "in the grouted closure joint; not counted by the programme in the "
                               f"{n} studs per shear span (Figs. 7-11)"},
                   "symmetry": "mirror image about midspan; positions measured from the bearing centreline"},
        "degree_of_connection": {
            "programme_statement": "12 studs per shear span 'corresponds to approximately 38 percent composite action' "
                                   "(p. 13, AISC with measured properties p. 80)" if n == 12 else
                                   "not stated for 1S2 (24 studs per shear span; designed as the trial beam, p. 12)",
            "Qn_AISC_kip": QN_AISC, "Qn_basis": "Asc Fu = 0.601 x 73.5, stud steel governs (p. 79)",
            "Cf_kip": CF, "Cf_basis": "min(As Fy = 24.7 x 56.4 = 1,393; 0.85 f'c b t = 0.85 x f'c x 48 x 8 >= 2,611)",
            "eta_plastic_shear_span_count_AISC_Qn": round(n * QN_AISC / CF, 3),
            "eta_plastic_including_constant_moment_studs_AISC_Qn": round((n + n_cm) * QN_AISC / CF, 3),
            "eta_plastic_shear_span_count_pushout_Q": round(n * qpo / CF, 3),
            "note": "The programme counts only the studs between bearing and load point. The pocket-grout term "
                    "0.5 Asc sqrt(f'c Ec) with grout f'c 6.7-7.6 ksi and Ec 2,300 ksi gives 37.9-40.3 kips, below Asc Fu; "
                    "the programme nevertheless states that stud steel governed for the beams (p. 79), apparently using "
                    "deck-concrete properties; the PC push-outs reached 48-57 kips per stud (Table 16), above both."},
        "pushout_programme_own": {
            "description": "24 small-scale static push-out tests, W10x60 beam 24 in. long, two 24 x 24 x 8 in. slabs, "
                           "2 studs of the same 7/8 x 6 in. type per flange (4 studs total), studs spaced longitudinally "
                           "3d-6d or transversely 3d-4d; 12 CIP (f'c 9.6 ksi) and 12 PC panels with grouted pockets "
                           "(panel f'c 7.5 ksi, grout 7.8 ksi); no haunch drawn; flange greasing not mentioned; "
                           "displacement control 0.0005 in./min; slip by video extensometer (pp. 23-25, 45-47, 103; "
                           "Fig. 16 p. 25; Tables 26-27 pp. 145-146)",
            "relevance": "PC longitudinal series (Fig. 93, Table 16) matches the beams' deck type, grout product and "
                         "3.5-in. (4d) cluster pitch; PO-S1/S2-L4D-PC is the direct match for 2S1-4S1, L6D-PC the "
                         "closest to the isolated studs of 1S1. The authors found no strength trend with spacing >= 3d.",
            "recommended_model_input": {
                "Q_u_per_stud_kip": qpo,
                "Q_u_basis": "mean of all 8 PC longitudinal push-outs, Table 16 max load / 4 (SD "
                             f"{mean_of(PC_L, 'Qmax_per_stud_kip')[1]} kips); 7 excluding the high PO-S1-L4D-PC: "
                             f"{round(float(np.mean([po_rows[k]['Qmax_per_stud_kip'] for k in PC_L if k != 'PO-S1-L4D-PC'])), 1)} kips",
                "k_s_per_stud_kip_per_in": kpo,
                "k_s_basis": f"mean (SD {kpo_sd}) of least-squares load-slip slopes over 5-40 % of Qmax on the ascending "
                             f"branch of the {len(PC_L_clean)} PC longitudinal curves without negative-slip excursions "
                             f"({', '.join(PC_L_clean)}); vector-extracted from Fig. 93",
                "k_s_L4D_pair_kip_per_in": mean_of(["PO-S1-L4D-PC", "PO-S2-L4D-PC"], "k_fit_5_40pct_kip_per_in_per_stud")[0],
                "Q_u_L4D_pair_kip": mean_of(["PO-S1-L4D-PC", "PO-S2-L4D-PC"], "Qmax_per_stud_kip")[0],
                "for_comparison_CIP_longitudinal_k_s_kip_per_in": mean_of(CIP_L_clean, "k_fit_5_40pct_kip_per_in_per_stud")[0],
                "caveats": ["slip resolution in the plotted data is 0.001 in. (vertex quantisation), 5-20 % of the slip "
                            "at 25-40 % Qmax, so individual secant stiffnesses carry +-10-20 %",
                            "push-outs have no 1-in. grout haunch; in the beams the stud shank passes through the haunch, "
                            "so beam connector stiffness is probably lower than the push-out value",
                            "push-out flanges not stated to be greased; any bond would stiffen the initial branch",
                            "whether the plotted slip is one side or the mean of both slabs is not stated (p. 103)"]},
            "PC_longitudinal_mean_backbone_per_stud": {"curves": PC_L_clean, "points": backbone,
                                                      "source": "vector paths of Fig. 93 (report p. 105), "
                                                                "pushout_curves_vector.csv"},
            "curve_points_per_stud": {k: curve_points(k) for k in ("PO-S1-L4D-PC", "PO-S2-L4D-PC")},
            "per_specimen": po_rows}}


def specimen(sid, included, screening, measured, tan, extra_assump=()):
    fc, fc_ind = FC[sid]
    return {
        "id": sid, "included": included, "screening": screening,
        "geometry": {"span_in": SPAN_IN, "span_ft": SPAN_IN / 12, "load_points_from_bearings_in": A_SHEAR_IN,
                     "constant_moment_length_in": 72.0, "steel": STEEL, "slab": SLAB,
                     "girder_spacing_equivalent_ft": 4.0},
        "materials": {
            "steel": {"Fy_ksi": 56.4, "Fy_basis": "overall longitudinal 0.2 % offset average (Table 8 p. 49; Table 19 p. 137)",
                      "Fy_flange_ksi": 54.0, "Fy_web_ksi": 57.4, "static_yield_overall_ksi": 54.8, "Fu_ksi": 72.9,
                      "E_ksi": None, "E_note": "not measured; 29,000 ksi to be assumed",
                      "heat": "all W27x84 from one heat (p. 137)"},
            "concrete": {"fc_ksi": fc, "cylinders_ksi": fc_ind, "design_fc_ksi": 6.0,
                         "source": "Table 22 p. 142 (4 x 8 in. cylinders, >= 6 months old, tested 1-2 days after test start)",
                         "Ec_ksi": None, "Ec_note": "not measured for the large-scale decks; unit weight not reported "
                                                   "(normal-weight bridge deck mix, p. 11)"},
            "grout": {"cube_strength_ksi": GROUT[sid], "design_ksi": 8.0, "source": "Table 23 p. 143 (2-in. cubes)",
                      "E_ksi": 2300, "E_source": "prior FHWA research with the same product, cited in footnotes p. 105 "
                                                 "and p. 108; not measured here"}},
        "connectors": connectors(sid),
        "loading": LOADING,
        "deflection_zero_includes": ZERO,
        "elastic_limit": ELASTIC,
        "capacity": {"Mn_calc_kipft": MN[sid], "M_max_applied_kipft": MMAX[sid], "initial_failure": FAIL[sid],
                     "source": "Table 11 p. 81"},
        "tangent_stiffness": tan,
        "measured": measured,
        "digitisation": {
            "figure": "Fig. 68, report p. 80 (PDF p. 98): midspan moment 0-2,500 kip-ft vs midspan displacement 0-10 in.",
            "method": "embedded 576 x 384 px raster, rendered at 600 dpi and verified cell-for-cell against the embedded "
                      "image; axes calibrated on 11 x-ticks and 6 y-ticks (residual 0.009 in., 3.9 kip-ft); check: dashed "
                      "Mn lines 1,831 and 2,157 kip-ft land within 3.5 kip-ft; curves separated by series colour "
                      "(legend call-outs: 1S2 black, 1S1 blue, 2S1 green, 3S1 red, 4S1 purple; identity confirmed by "
                      "the load drops at about 6 in. for 1S1 and 7 in. for 3S1 described on p. 81); per source-pixel row "
                      "the stroke centre is the alpha-weighted run centre, or edge +- half the local stroke width where "
                      "a later-drawn curve covers one edge",
            "files": ["provines2019_digitised/digitise_fig68.py", "provines2019_digitised/fig68_calibration.json",
                      "provines2019_digitised/fig68_traces_raw.csv", "provines2019_digitised/fig68_levels.csv",
                      "provines2019_digitised/overlay_check.py", "provines2019_digitised/overlay_check.png"],
            "uncertainty": "moment +-7.9 kip-ft; displacement +-0.020 in. (exposed stroke), +-0.025 in. (one edge "
                           "covered), +-0.030 in. (both edges covered); i.e. +-7 % at 500 kip-ft, +-13 % at 300 kip-ft for "
                           "secant values, before the record-zero ambiguity; tangent stiffness 300-700 kip-ft +-5 %"},
        "assumptions": [
            "E_s = 29,000 ksi (not measured).",
            "Concrete unit weight not reported; normal weight assumed.",
            "W27x84 section dimensions from the AISC Manual, not the report.",
            "Stud positions per half from Figs. 7-11 (bearing centreline datum); clusters symmetric about the stated centres.",
            "deflection_offset_corrected_in = deflection - intercept of the 300-700 kip-ft tangent fit (derived, not "
            "measured): it removes the record-zero/initial-toe offset so that the elastic branch passes through the "
            "origin. Use either the raw secant or this corrected value and say which; the tangent stiffness is the "
            "offset-independent measure.",
            *extra_assump]}


SCREEN_INC = ("INCLUDED as very close. Design space: span 29 ft, deck 8 in., f'c {fc} ksi, eta 0.38, fy 56.4 ksi, "
              "d 26.7 in., bf/d 0.373, tf/bf 0.064, tw/d 0.0172, rolled W, simply supported, monotonic positive bending: "
              "all inside Table tab:features. Outside: slab width 48 in. = 4-ft equivalent girder spacing, 1 ft (20 %) "
              "below the 5-ft minimum (same as the Chapman beams already used){fcnote}. Differences from the modelled "
              "cast-in-place solid slab: precast full-depth solid panels with grouted pockets, 1-in. grout haunch, "
              "greased flange, 18-in. grouted closure joint at midspan{clus}. Judged very close because the slab is "
              "solid and continuous in compression, and composite action is carried only by discrete studs (greased "
              "flange, no bond), which is exactly the two-chain discrete-connector mechanism; the grouted-pocket "
              "detail enters through the connector stiffness and strength, which the programme measured on "
              "matching PC push-outs; the haunch is explicit geometry.")

specs = []
for sid in ("1S1", "2S1", "3S1", "4S1"):
    tan, p = tangent(sid)
    meas = measured_nearest_rows(sid, p[1]) if sid == "3S1" else measured_levels(sid, p[1])
    if sid == "4S1":
        meas = [m for m in meas if m["load"] <= 1100]   # guard: the 1,450 kip-ft level in fig68_levels.csv is a false pick on the 1S2 call-out box
    fcnote = "; f'c 10.0 ksi at the upper bound" if sid == "2S1" else ""
    clus = {"1S1": "; studs uniformly spaced at 12 in. (no clustering)",
            "2S1": "; studs clustered in pairs at 24 in.",
            "3S1": "; studs clustered in threes at 36 in. (beyond the AASHTO 24-in. maximum)",
            "4S1": "; studs clustered in fours at 48 in. (beyond the AASHTO 24-in. maximum)"}[sid]
    extra = []
    if sid == "3S1":
        extra.append("3S1 is drawn under 4S1: its elastic-range rows are mostly read from one exposed edge (edge rule, "
                     "+-0.025 in.); levels are the nearest visible rows, not interpolated.")
    if sid == "2S1":
        extra.append("2S1 below 300 kip-ft plots stiffer than a full-interaction section (e.g. 0.047 in. at 200 kip-ft "
                     "against a rough full-interaction value of about 0.10 in.), so its absolute zero is not reliable; "
                     "its tangent stiffness agrees with the other replicates within 4 %.")
    specs.append(specimen(sid, True, SCREEN_INC.format(fc=FC[sid][0], fcnote=fcnote, clus=clus), meas, tan, extra))

# 1S2: screened in, data not recoverable in the linear range
g = vis[(vis.series == "1S2") & (vis.moment_kipft <= 1100)]
meas_1s2 = [point(r.moment_kipft, r.displacement_in, r.displacement_unc_in, 7.9, r.centre_rule,
                  f"visible source row {int(r.cell_row)}", 0.0,
                  ["specimen excluded; values not assessed for linearity or zero"]) for r in g.itertuples()
            if r.moment_kipft < 720 or abs((r.moment_kipft + 25) % 50 - 25) < 4.5]
for m in meas_1s2:
    m["elastic"] = False
    m["in_tangent_window"] = False
    m.pop("deflection_offset_corrected_in")
s12 = specimen("1S2", False,
               "EXCLUDED (data, not screening). Screening alone would admit it like 1S1 (eta 0.76, f'c 8.0 ksi, same "
               "geometry). But 1S2 was drawn first in Fig. 68 and is hidden under the four later curves below 830 "
               "kip-ft: only 12 single-pixel glimpses between 580 and 710 kip-ft (+-0.03 in.) survive, there is no "
               "low-load branch to fix the record zero (which is offset by 0.05-0.17 in. in the other four beams), and "
               "the continuously visible part (830-1,240 kip-ft) is already softening (tangent about 1,200 kip-ft/in. "
               "at 850-1,100 against about 1,400 from the glimpses below). Usable at most as a flagged eta = 0.76 bound; "
               "its visible points are kept below.", meas_1s2, None)
s12.pop("tangent_stiffness")
specs.insert(0, s12)

specs.append({
    "id": "1F1-1F3, 2F1-2F3, 3F1-3F3, 4F1-4F3 (12 fatigue beams)", "included": False,
    "screening": "EXCLUDED: fatigue-cycled (shear stress ranges 10-20 ksi, minimum 8.4 and maximum 25.2-41.9 kips per "
                 "actuator, Table 10 p. 71). No static load-deflection record before cycling is published: results are "
                 "neutral-axis location, slip and uplift versus cycles (pp. 57-71); static laser-tracker readings at the "
                 "maximum fatigue load (p. 38) are slip/uplift profiles only. Construction identical to the static beams.",
    "measured": []})

doc = {
    "programme": "Provines, Ocel & Zmetra (2019), FHWA-HRT-20-005: Strength and fatigue resistance of clustered shear "
                 "stud connectors in composite steel girders",
    "key": "provines2019",
    "citations": ["Provines, J.T., Ocel, J.M., Zmetra, K. (2019). Strength and Fatigue Resistance of Clustered Shear Stud "
                  "Connectors in Composite Steel Girders. Report FHWA-HRT-20-005, Federal Highway Administration, "
                  "Turner-Fairbank Highway Research Center, McLean, VA, November 2019."],
    "sources": {"urls": ["https://rosap.ntl.bts.gov/view/dot/42728/dot_42728_DS1.pdf",
                         "https://www.fhwa.dot.gov/publications/research/infrastructure/structures/bridge/20005/index.cfm"],
                "local_copy": "scratchpad/lit_sources/provines2019/FHWA-HRT-20-005.pdf (not in repository)",
                "page_convention": "report page numbers; PDF page = report page + 18",
                "pages_used": {"test matrix, construction, greasing, haunch": "pp. 11-13",
                               "beam drawings and sections": "Figs. 7-12 pp. 14-19",
                               "push-out design and drawing": "pp. 23-25, Fig. 16",
                               "fatigue setup and loads": "pp. 25-28, 38, Table 10 p. 71",
                               "static loading": "pp. 40-42, Fig. 32",
                               "materials": "Tables 8-9 p. 49; Tables 19, 21-23, 26-27 pp. 137-146",
                               "moment-displacement": "Fig. 68 p. 80, Table 11 p. 81",
                               "push-out results": "Figs. 92-95, Tables 15-18 pp. 103-109",
                               "deck panel drawings": "Figs. 104-109 pp. 126-131"}},
    "screening": ("Solid-slab question: the deck is a solid 8 x 48 in. slab of full-depth precast panels joined at "
                  "midspan by a grouted closure, connected by studs in grouted pockets through a 1-in. grout haunch to a "
                  "greased flange. Judged very close to the modelled cast-in-place solid slab acting through discrete "
                  "connectors: (1) section geometry is a solid rectangle (no ribs or voids), continuous in compression "
                  "across the grouted joint; (2) the greased flange leaves the studs as the only shear path, which is "
                  "the discrete-connector mechanism of the beam model; (3) what the precast/grout detail changes is "
                  "the connector load-slip law, and the programme measured it on PC push-outs with the same stud, grout "
                  "and 4d pitch (about 1,150 kip/in. per stud against about 2,000 for its CIP push-outs), so it is a "
                  "model input, not an unmodelled mechanism; (4) the 1-in. haunch is explicit geometry. Caveat to "
                  "state: the connection is softer than a typical CIP stud of the same size, so these beams test the "
                  "eta-only R_EI at the soft end of connector stiffness. Included: 1S1, 2S1, 3S1, 4S1 (eta 0.38). "
                  "Excluded: 1S2 (data hidden in Fig. 68 in the elastic range), 12 fatigue beams (no pre-cycling "
                  "static record)."),
    "specimens": specs,
    "data_files": ["provines2019_digitised/"]}


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


OUT.write_text(json.dumps(clean(doc), indent=1))
print("wrote", OUT)
for s in specs:
    if s.get("tangent_stiffness"):
        t = s["tangent_stiffness"]
        print(s["id"], "k_tan", t["k_tangent_kipft_per_in"], "se%", t["slope_standard_error_pct"], "int", t["intercept_in"],
              "n", t["n_rows"], "| elastic pts", sum(m["elastic"] for m in s["measured"]),
              "window pts", sum(m["in_tangent_window"] for m in s["measured"]), "total", len(s["measured"]))
print("PC_L clean", PC_L_clean, "PC_T clean", PC_T_clean, "CIP_L clean", CIP_L_clean)
c = specs[1]["connectors"]
print(json.dumps(c["degree_of_connection"], indent=0)[:600])
print(json.dumps(c["pushout_programme_own"]["recommended_model_input"], indent=0))
print(json.dumps(backbone))
