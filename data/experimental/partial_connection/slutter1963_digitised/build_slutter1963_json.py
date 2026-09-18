"""Assemble ../slutter1963.json for the Lehigh composite beam programme
(Slutter and Driscoll 1963, with progress reports 279.2 and 279.6) from the
transcribed report data and the digitised markers (markers_raw.csv,
pushout_markers_raw.csv written by digitise_slutter1963.py).

Printed page = PDF page - 5 for 279.2 and 279.6; PDF page - 7 for 279.15.
Units: kips, in, ksi/psi as in the reports.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "slutter1963.json"

R2 = "Culver, Zarzeczny and Driscoll (1960), Fritz Lab. Report 279.2 (Progress Report 1)"
R6 = "Culver, Zarzeczny and Driscoll (1961), Fritz Lab. Report 279.6 (Progress Report 2)"
R15 = "Slutter and Driscoll (1963), Fritz Lab. Report 279.15"

# 12WF27, handbook: the reports give A = 7.97 in2, d = 11.95 in, I = 204.1 in4
# ("measured values were all very close to the handbook properties",
# 279.2 App. 10.1 p. 45; 279.6 App. 10.1 p. 34); b_f, t_f, t_w from the AISC
# WF table (not in the reports).
W = dict(designation="12WF27", d=11.95, bf=6.497, tf=0.400, tw=0.240, A=7.97, Ix=204.1)
AF = 2 * W["bf"] * W["tf"]
AW = W["A"] - AF

SERIES = {
    1: dict(fc=3600, fc_src="279.2 App. 10.1 p. 45 (average of 18 cylinders at various ages; Table 2 p. 58); 279.15 Table 1 p. 27",
            fy_f=39.0, fy_w=44.0, fy_src="279.2 App. 10.1 p. 45 (static yield of coupons, Table 3 p. 59); all six beams from one rolling, one concrete mix (279.2 p. 10)",
            mesh="6 x 6 in mesh of 1/4 in rods at mid-depth of the slab (279.2 p. 9 and App. p. 45)", mesh_depth=2.0,
            C_authors=324, C_src="279.2 App. 10.3 p. 52: C = 324 k"),
    2: dict(fc=3300, fc_alt=3337, fc_src="279.6 App. 10.1 p. 34 (3300 psi average); 279.15 Table 1 p. 27 gives 3337 psi",
            fy_f=37.4, fy_w=41.9, fy_src="279.6 App. 10.1 p. 34 (coupons, Table 4 p. 50); B7-B9 from one rolling",
            mesh="6 x 6 in mesh of 1/4 in rods 1 in below the top of the slab, plus 5/8 in transverse bars at 6 in (extent illegible in the OCR layer) (279.6 p. 5)", mesh_depth=1.0,
            C_authors=310, C_src="279.6 App. 10.3 p. 41: C = 310 k"),
    3: dict(fc=3600, fc_alt=3595, fc_src="279.6 App. 10.1 p. 34 (3600 psi average); 279.15 Table 1 p. 27 gives 3595 psi",
            fy_f=36.6, fy_w=44.7, fy_src="279.6 App. 10.1 p. 34 (coupons, Table 4 p. 50); B10-B13 from one rolling",
            mesh="6 x 6 in mesh of 1/4 in rods 1 in below the top of the slab, plus 5/8 in transverse bars at 6 in (extent illegible in the OCR layer) (279.6 p. 5)", mesh_depth=1.0,
            C_authors=314, C_src="279.6 App. 10.3 p. 42: Q_p = 15.7 k per connector over 20 connectors, C = 314 k"),
}

PUSHOUT = {
    "P1": dict(report="279.2", table="Table 8 p. 64; Fig. 28 p. 95", connector="1/2 in L-stud, 2.25 in (material as B3, B4, B6)", qu=11.0, fc=3600, failure="shearing of studs, no slab cracks"),
    "P4": dict(report="279.2", table="Table 8 p. 64; Fig. 31 p. 98", connector="1/2 in L-stud, 2.25 in (material as B3, B4, B6)", qu=10.4, fc=3600, failure="shearing of studs, no slab cracks"),
    "P2": dict(report="279.2", table="Table 8 p. 64; Fig. 29 p. 96", connector="3U4.1 channel, 4 in long (as B5)", qu=47.5, fc=3600, failure="separation of slabs and beam stub, large cracks"),
    "P3": dict(report="279.2", table="Table 8 p. 64; Fig. 30 p. 97", connector="3/4 in headed stud (as B9)", qu=21.2, fc=3600, failure="shearing of studs, large cracks"),
    "P5": dict(report="279.2", table="Table 8 p. 64", connector="1/2 in headed stud machined from 1 in bar", qu=12.1, fc=3600, failure="shearing of studs"),
    "P6": dict(report="279.2", table="Table 8 p. 64", connector="1/2 in headed stud machined from 1 in bar", qu=12.1, fc=3600, failure="shearing of studs"),
    "P7": dict(report="279.6", table="Table 11 p. 57", connector="1/2 in L-stud (as B7, B10-B12)", qu=6.75, fc=3063, failure="shearing of studs; the authors consider the test unsuccessful and neglect it (279.6 p. 22)"),
    "P8": dict(report="279.6", table="Table 11 p. 57; Fig. 37 p. 97", connector="1/2 in headed stud, 3 in (as B8)", qu=12.1, fc=3063, failure="shearing of studs, no cracks"),
    "P9": dict(report="279.6", table="Table 11 p. 57", connector="3/4 in headed stud (as B9)", qu=16.0, fc=3063, failure="shearing of studs; considered unsuccessful by the authors (279.6 p. 22)"),
}
Q_LSTUD = (11.0 + 10.4) / 2  # P1, P4


def aisc_stud(d, fc_psi, fu=None):
    Asc = 3.14159265 * d * d / 4
    Ec = 57.0 * fc_psi ** 0.5
    q = 0.5 * Asc * (fc_psi / 1000 * Ec) ** 0.5
    return q if fu is None else min(q, Asc * fu)


def aisc_channel(tf, tw, L, fc_psi):
    Ec = 57.0 * fc_psi ** 0.5
    return 0.3 * (tf + 0.5 * tw) * L * (fc_psi / 1000 * Ec) ** 0.5


def rows_from(pos, per_row):
    return [dict(x_in=x, n=per_row) for x in pos]


def mirror(half):
    return sorted(set(half + [180.0 - x for x in half]))


P75 = [7.5 * k for k in range(25)]
SPECS = {
    "B3-S1": dict(series=1, fig="279.2 Fig. 9 (PDF p. 79, printed p. 74)", stud="1/2 in L-stud, h = 2.25 in, A = 0.196 in2",
                  pos=P75, per_row=2, n_span=22, q_po=Q_LSTUD, q_po_src="mean of P1 (11.0) and P4 (10.4) kips, 279.2 Table 8",
                  q_aisc=aisc_stud(0.5, 3600), eta_sd=round(0.772 * 22 / 18, 3), eta_sd_src="279.15 Table 4 B3-T7 0.772 with 18 studs in its 62 in shear span, scaled to the 22 studs in the 81 in S1 shear span (279.2 App. p. 52 counts 22)",
                  load="two loads P/2 on top of the slab at 81 and 99 in from the supports (2b = 18 in)", M_per_P=40.5, cycle=40, slip_load=50,
                  slip_src="bond held until the symbol S in Fig. 9, at the 50 kip reading (279.2 p. 21 text, OCR 'load of 5806 kips', reads 58.6 or 50.6)",
                  included=True),
    "B4-S1": dict(series=1, fig="279.2 Fig. 12 (PDF p. 82, printed p. 77)", stud="1/2 in L-stud, h = 2.25 in", pos=P75, per_row=2, n_span=22,
                  q_po=Q_LSTUD, q_po_src="mean of P1 and P4", q_aisc=aisc_stud(0.5, 3600), eta_sd=None, eta_sd_src="not computable: C reduced by web holes of unstated size",
                  load="two loads P/2 hung from the steel beam through holes in the web at 81 and 99 in", M_per_P=40.5, cycle=30, slip_load=None,
                  slip_src="S symbol above the 40 kip reading", included=False,
                  reason="EXCLUDED: loads hung from the steel beam through web holes drilled for the loading fixtures (279.2 p. 7 and App. p. 51); hole sizes are not given, so the reduced section (authors' M_p 2770 vs 2862 kip-in) cannot be modelled; otherwise a duplicate of B3-S1."),
    "B5-S1": dict(series=1, fig="279.2 Fig. 15 (PDF p. 85, printed p. 80)", stud="3U4.1 channel (3 in, 4.1 lb), 4 in long, 3/16 in fillet welds at toe and heel; handbook t_f 0.273, t_w 0.170 in",
                  pos=[0, 20, 40, 60, 80, 100, 120, 140, 160, 180], per_row=1, n_span=5, q_po=47.5, q_po_src="P2, 279.2 Table 8",
                  q_aisc=aisc_channel(0.273, 0.170, 4.0, 3600), eta_sd=round(0.437 * 5 / 3, 3), eta_sd_src="279.15 Table 4 B5-T11 0.437 with 3 channels in its 52 in shear span, scaled to 5 channels in 81 in (279.2 App. p. 52: 64.9 k per connector = 324/5)",
                  load="two loads P/2 on top of the slab at 81 and 99 in (2b = 18 in)", M_per_P=40.5, cycle=40, slip_load=40,
                  slip_src="'measured slips were small up to about 40 kips', loud noise then faster slip (279.2 p. 24); S in Fig. 15 at 40 kips", included=True),
    "B6-S1": dict(series=1, fig="279.2 Fig. 18 (PDF p. 88, printed p. 83)", stud="1/2 in L-stud, h = 2.25 in, single row", pos=P75, per_row=1, n_span=11,
                  q_po=Q_LSTUD, q_po_src="mean of P1 and P4", q_aisc=aisc_stud(0.5, 3600), eta_sd=0.473, eta_sd_src="279.15 Table 4 p. 31 (B6-T2 = S1) 0.473; 279.10 Fig. 14 legend 0.473 (legend as read by the verification step)",
                  load="two loads P/2 on top of the slab at 81 and 99 in (2b = 18 in)", M_per_P=40.5, cycle=40, slip_load=40,
                  slip_src="'No slip occurred up to about a load of 40 kips' (279.2 p. 25); S in Fig. 18 at 40 kips", included=True),
    "B7-S1": dict(series=2, fig="279.6 Fig. 10 (PDF p. 73, printed p. 68)", stud="1/2 in L-stud, h = 2.25 in, A = 0.196 in2", pos=P75, per_row=2, n_span=22,
                  q_po=Q_LSTUD, q_po_src="first-series P1/P4 (same stud type; the second-series L-stud push-out P7 was rejected by the authors)",
                  q_aisc=aisc_stud(0.5, 3300), eta_sd=round(0.897 * 22 / 20, 3), eta_sd_src="279.15 Table 4 B7-T4 0.897 with 20 studs in its 72 in shear span, scaled to 22 studs (279.6 App. p. 41 counts 22 for S1)",
                  load="two loads P/2 on top of the slab at 81 and 99 in (2b = 18 in)", M_per_P=40.5, cycle=40, slip_load=20,
                  slip_src="S symbol at 20 kips in Fig. 10 ('load at which slip first occurred')", included=True),
    "B8-S1": dict(series=2, fig="279.6 Fig. 12 (PDF p. 75, printed p. 70)", stud="1/2 in headed stud, h = 3 in (head welded to a straight bar, 279.6 p. 6)", pos=P75, per_row=2, n_span=22,
                  q_po=12.1, q_po_src="P8, 279.6 Table 11", q_aisc=aisc_stud(0.5, 3300), eta_sd=0.988, eta_sd_src="279.10 Fig. 14 legend (B8-S1 0.988, as read by the verification step)",
                  load="two loads P/2 on top of the slab at 81 and 99 in", M_per_P=40.5, cycle=40, slip_load=20, slip_src="S at 20 kips in Fig. 12", included=False,
                  reason="EXCLUDED: only two first-loading readings (20 and 40 kips) are plotted below the cycle load, too few for a stiffness comparison; the same layout is covered by B7-S1."),
    "B9-S1": dict(series=2, fig="279.6 Fig. 15 (PDF p. 78, printed p. 73)", stud="3/4 in headed stud, h = 3 in, A = 0.441 in2", pos=[15.0 * k for k in range(13)], per_row=2, n_span=12,
                  q_po=21.2, q_po_src="P3 (279.2 Table 8; first series, 3/4 in headed); second-series P9 (16.0) rejected by the authors",
                  q_aisc=aisc_stud(0.75, 3300), eta_sd=1.210, eta_sd_src="279.10 Fig. 14 legend (B9-S1 1.210, as read by the verification step)",
                  load="two loads P/2 on top of the slab at 81 and 99 in", M_per_P=40.5, cycle=40, slip_load=10, slip_src="S at 10 kips in Fig. 15", included=True),
    "B10": dict(series=3, fig="279.6 Fig. 18 (PDF p. 81, printed p. 76)", stud="1/2 in L-stud, h = 2.25 in, pairs", pos=mirror([9.0 * k for k in range(10)]), per_row=2, n_span=20,
                q_po=Q_LSTUD, q_po_src="first-series P1/P4 (P7 rejected)", q_aisc=aisc_stud(0.5, 3600), eta_sd=0.888, eta_sd_src="279.15 Table 4 B10-T13",
                load="five equal loads P/5 on top of the slab at 30, 60, 90, 120, 150 in (6 @ 30 in), via distributor beams", M_per_P=27.0, cycle=60, slip_load=40,
                slip_src="S at 40 kips in Fig. 18", included=True),
    "B11": dict(series=3, fig="279.6 Fig. 19 (PDF p. 82, printed p. 77)", stud="1/2 in L-stud, h = 2.25 in, pairs", pos=mirror([9.0 * k for k in range(10)]), per_row=2, n_span=20,
                q_po=Q_LSTUD, q_po_src="first-series P1/P4 (P7 rejected)", q_aisc=aisc_stud(0.5, 3600), eta_sd=0.888, eta_sd_src="279.15 Table 4 B11-T13",
                load="five equal loads P/5 at 30, 60, 90, 120, 150 in", M_per_P=27.0, cycle=60, slip_load=30, slip_src="S at 30 kips in Fig. 19", included=True),
    "B12": dict(series=3, fig="279.6 Fig. 20 (PDF p. 83, printed p. 78)", stud="1/2 in L-stud, h = 2.25 in, pairs, spacing per the shear diagram",
                pos=mirror([0, 5, 10, 15, 20, 25, 37.5, 45, 52.5, 75]), per_row=2, n_span=20,
                q_po=Q_LSTUD, q_po_src="first-series P1/P4 (P7 rejected)", q_aisc=aisc_stud(0.5, 3600), eta_sd=0.888, eta_sd_src="279.15 Table 4 B12-T13",
                load="five equal loads P/5 at 30, 60, 90, 120, 150 in", M_per_P=27.0, cycle=60, slip_load=30, slip_src="S at 30 kips in Fig. 20", included=True),
}


def build():
    rows = list(csv.DictReader(open(HERE / "markers_raw.csv")))
    porows = list(csv.DictReader(open(HERE / "pushout_markers_raw.csv")))
    envelopes = {}
    for r in porows:
        if r["envelope"] == "True":
            envelopes.setdefault(r["specimen"], []).append(dict(slip_in=float(r["avg_slip_in"]), load_per_connector_kip=float(r["load_per_connector_kip"])))
    specimens = []
    for sid, S in SPECS.items():
        ser = SERIES[S["series"]]
        AsFy = AF * ser["fy_f"] + AW * ser["fy_w"]
        conc = 0.85 * ser["fc"] / 1000 * 48 * 4
        Cf = min(AsFy, conc)
        eta_po = S["n_span"] * S["q_po"] / Cf
        eta_aisc = S["n_span"] * S["q_aisc"] / Cf
        long_area = 8 * 0.0491
        rho = long_area / (48 * 4)
        Mp_auth = {1: 2862, 2: 2790, 3: 2840}[S["series"]]
        high = sid in ("B10", "B11", "B12")
        unc = dict(load_kip=0.3, deflection_in=0.015 if sid == "B10" else 0.01) if high else dict(load_kip=0.2, deflection_in=0.005)
        measured, context = [], []
        for r in rows:
            if r["beam_test"] != sid:
                continue
            P, d, br = float(r["load_kip"]), float(r["deflection_in"]), r["branch"]
            item = dict(load=round(P, 2), load_unit="kip (total applied jack load P)", deflection=round(d, 4),
                        deflection_unit="in (centreline, applied load only)", moment_max_applied_kip_in=round(P * S["M_per_P"], 0),
                        m_over_mp_authors=round(P * S["M_per_P"] / Mp_auth, 3), uncertainty=unc,
                        source=f"digitised, {S['fig']}; raw marker in slutter1963_digitised/markers_raw.csv (px={r['px_x']}, py={r['px_y']})",
                        branch=br)
            if br == "first_loading" and P <= S["cycle"] + 1.5:
                flags = []
                if S["slip_load"] is not None and P <= S["slip_load"] + 0.5:
                    flags.append(f"at or below the first-slip load {S['slip_load']} kips: natural bond may carry part of the shear")
                if P < 12:
                    flags.append(f"low-load point: +/-{unc['deflection_in']} in is {unc['deflection_in'] / max(d, 1e-3) * 100:.0f} % of the reading")
                item.update(elastic=True, flags=flags)
                measured.append(item)
            elif br in ("post_cycle_loading", "post_cycle_same_load") and P <= S["cycle"] + 15 and d <= 0.8:
                item.update(elastic=False, flags=["after 10 cycles to the cycle load; includes any residual set; context only"]
                            + (["M/M_p (authors' M_p) > 0.6"] if P * S["M_per_P"] / Mp_auth > 0.6 else []))
                context.append(item)
        measured.sort(key=lambda m: m["load"])
        context.sort(key=lambda m: m["load"])
        n_el = len(measured)
        oor = ["span 15 ft: 5 ft (25 %) below 20 ft", "slab 4.0 in: 0.5 in (11 %) below 4.5 in",
               "slab width 4 ft as girder spacing: 1 ft (20 %) below 5 ft (identical to the Chapman beams already used)",
               "steel depth 11.95 in: 0.05 in (0.4 %) below 12 in"]
        if eta_po < 0.25:
            oor.append("eta below 0.25")
        if S["series"] == 1 and sid == "B5-S1":
            oor.append("channel connectors (not studs)")
        if high:
            oor.append("five-point loading (not one of the module's three patterns; close to UDL, the authors call it an approximation of uniform load)")
        oor.append("natural bond not prevented: first-slip loads marked in the figures")
        screening = (("INCLUDED, very close (lab scale, flagged). " if S["included"] else S["reason"] + " ")
                     + "Solid 48 x 4 in slab on 12WF27, simply supported 15 ft, virgin first loading; b_f/d 0.544, t_f/b_f 0.062, t_w/d 0.020, "
                     f"f'c {ser['fc'] / 1000:.2f} ksi, F_y flange {ser['fy_f']} / web {ser['fy_w']} ksi in range. Outside: " + "; ".join(oor) + ".")
        spec = dict(
            id=sid, included=S["included"], screening=screening,
            spec_summary=dict(
                id=sid, included=S["included"],
                reason=(("partial connection" if eta_po < 0.95 else "near-full reference") + " virgin test; very close to the design space at lab scale (span, deck and spacing below range, as the Chapman beams)")
                if S["included"] else S["reason"],
                eta_source=(f"Slutter-Driscoll convention: {S['eta_sd']} ({S['eta_sd_src']}); this file with programme push-out strength: {S['n_span']} x {S['q_po']:.2f} kips ({S['q_po_src']}) / C_f {Cf:.1f} kips = {eta_po:.3f}; "
                            f"with AISC Q_n {S['q_aisc']:.2f} kips: {eta_aisc:.3f}"),
                eta_plastic=round(eta_po, 3) if S["included"] or sid == "B8-S1" else round(eta_po, 3),
                span_ft=15.0, steel="12WF27 rolled (d 11.95, b_f 6.497, t_f 0.400, t_w 0.240 in)",
                slab=f"solid 48 x 4 in, 6x6-1/4 in mesh, rho_l {rho * 100:.2f} %",
                fc_ksi=ser["fc"] / 1000, fy_ksi=ser["fy_f"],
                connectors=f"{S['stud']}; {S['n_span']} in each shear span; positions (in from a support) {S['pos']} with {S['per_row']} per position",
                loading=S["load"], n_elastic_points=n_el if S["included"] else n_el,
                data_form="digitised figure (load vs centreline deflection, kips/in)",
                out_of_range="; ".join(oor)),
            geometry=dict(
                span_in=180.0, span_ft=15.0, beam_overhang_in="3 in beyond each support centreline (279.2 Fig. 1 p. 66; 279.6 Fig. 2 p. 60)",
                supports="rocker and hinge (279.6 Fig. 5 p. 63)",
                load_positions=S["load"],
                steel_section=dict(**W, A_flanges_in2=round(AF, 3), A_web_incl_fillets_in2=round(AW, 3),
                                   source="A, d, I from 279.2/279.6 App. 10.1 (handbook values, measured dimensions 'very close'); b_f, t_f, t_w from the AISC WF table"),
                slab=dict(width_in=48.0, thickness_in=4.0, haunch_in=0.0, source="279.2 p. 9 and Fig. 1; 279.6 p. 5 and Figs. 1-2",
                          reinforcement=ser["mesh"], longitudinal_area_in2=round(long_area, 3), longitudinal_depth_from_top_in=ser["mesh_depth"],
                          rho_l=round(rho, 4), rho_note="8 longitudinal 1/4 in rods at 6 in over 48 in assumed; 279.6 p. 18 states the longitudinal reinforcement was 0.2 % of the slab area")),
            materials=dict(
                concrete=dict(fc_psi=ser["fc"], fc_alternative_psi=ser.get("fc_alt"), source=ser["fc_src"], Ec="not measured in the reports; authors used n = 10 (279.2/279.6 App. 10.1)"),
                steel_beam=dict(flange_fy_ksi=ser["fy_f"], web_fy_ksi=ser["fy_w"], E_ksi_authors=30000, source=ser["fy_src"]),
                studs="1/2 in stud bar stock (second series): yield 58.4-59.4, ultimate 66.9-67.7 ksi; 3/4 in: yield 61.5-62.5, ultimate 75.4-76.2 ksi (279.6 Table 5 p. 51, as transcribed by the verification step; not re-read here)"),
            connectors=dict(
                type=S["stud"], positions_from_left_support_in=S["pos"], per_position=S["per_row"],
                connectors_in_shear_span=S["n_span"],
                layout_source=("279.6 Fig. 2 p. 60 (dimensioned: first row on the support centreline)" if high else
                               "pitch from 279.2 Fig. 1 p. 66 / 279.15 Table 1 p. 27 (end distance not dimensioned); first row on the support centreline assumed as dimensioned for B10-B12, which reproduces the authors' counts (22 studs or 5 channels in the 81 in S1 shear span, 279.2 App. p. 52 and 279.6 App. p. 41; 18 studs in 62 in for B3-T7 and 3 channels in 52 in for B5-T11, 279.15 Table 4)"),
                strength=dict(pushout_kip=round(S["q_po"], 2), pushout_source=S["q_po_src"], aisc_Qn_kip=round(S["q_aisc"], 2),
                              aisc_note="0.5 A_sc sqrt(f'c E_c) with E_c = 57000 sqrt(f'c) (channels: 0.3 (t_f + 0.5 t_w) L_c sqrt(f'c E_c)); cap A_sc F_u not governing",
                              slutter_driscoll_note="279.15 Eq. 10 (studs, H/d > 4.1) with f'c at test, capped at a 70 ksi tensile strength; the formula is not legible in the OCR layer; Table 4 values imply about 13.9 kips per 1/2 in stud",
                              beam_vs_pushout="279.2 Table 9 p. 65: connector force at beam failure / push-out ultimate = 0.97-1.53 (1/2 in L-studs), 1.86 (channel); 279.6 Table 12 p. 58: 1.39 (B8/P8)"),
                push_out_programme={k: v for k, v in PUSHOUT.items()},
                push_out_envelopes_digitised={k: v for k, v in envelopes.items()},
                push_out_digitisation="upper envelope (backbone) of the digitised load-slip markers, average slip; markers lying on the frame at zero slip (bond stage) and small markers below 15 kips in P2 were not detected; see pushout_markers_raw.csv and overlay_check_pushout.png",
                degree_of_connection=dict(C_f_kip=round(Cf, 1), AsFy_kip=round(AsFy, 1), concrete_085fcbt_kip=round(conc, 1), C_authors_kip=ser["C_authors"], C_authors_source=ser["C_src"],
                                          eta_pushout=round(eta_po, 3), eta_aisc=round(eta_aisc, 3), eta_slutter_driscoll=S["eta_sd"],
                                          shear_span="support to the nearer load point (81 in) for S1; support to midspan (90 in) for the five-point loading (279.6 App. p. 42)")),
            loading=dict(
                pattern=S["load"],
                module_pattern=("two symmetric point loads at a = 81 in" if not high else "not directly supported: five equal point loads at L/6 (use explicit point loads or report the UDL approximation)"),
                sequence=f"increments to about P_p/1.85, cycled 10 times between 0 and {S['cycle']} kips (annotation in the figure), then increments to yield and beyond (279.2 p. 15; 279.6 p. 10); S1 was the first test of the beam (virgin)",
                moment_per_unit_P_kip_in=S["M_per_P"],
                first_slip=dict(load_kip=S["slip_load"], source=S["slip_src"])),
            deflection_zero_includes="applied jack load only: curves start at (0, 0) at the start of the first test; self-weight of beam and slab (about 77 kip-in at midspan per the verification notes: difference between 279.15 Table 2 and 279.6 Tables 8-9 maximum moments) and distributor-beam weight are not in the record (not stated explicitly; shoring during casting not stated)",
            elastic_limit=dict(
                cycle_load_kip=S["cycle"], rule="first-loading readings up to the cycle load (about P_p/1.85, the authors' working-load estimate) are flagged elastic; M/M_p at these loads is <= 0.57 with the authors' M_p",
                yield_moment_authors=("M_y = f_y I / c with I = 587.7 in4, c = 11.60 in (279.2 App. 10.3 p. 48 with c from App. 10.1 p. 46; 279.6 App. 10.1 p. 35); "
                                      f"M_y = {ser['fy_f'] * 587.7 / 11.60:.0f} kip-in, P_y = {ser['fy_f'] * 587.7 / 11.60 / S['M_per_P']:.1f} kips (computed here)"),
                plastic_moment_authors_kip_in=Mp_auth,
                mill_scale_flaking=("first mill-scale flaking (symbol M) at about 44-45 kips in Figs. 10 and 12" if S["series"] == 2 and not high else "not marked in the first-loading range")),
            authors_theory=dict(full_interaction_deflection=("279.2 App. 10.4 p. 53 and 279.6 App. 10.4 p. 45: P = 40 kips, 2b = 18 in: bending 0.272 (0.271) + shear 0.052 = 0.324 (0.323) in, with I = 587.7 in4, E = 30,000 ksi, A_w = 2.68 in2"
                                                             if not high else "279.6 App. 10.4 p. 45: P = 60 kips, five-point load: bending 0.309 + shear 0.052 = 0.356 in")),
            measured=measured,
            context_measured=context,
            digitisation=dict(method="600 dpi render of the 300 ppi bilevel scan; frame lines fitted, tick marks detected and fitted on a lattice (0.2 in and 10 kips per tick for 279.2 and 279.6 B7-B9; 1.0 in and 20 kips for B10-B12); open markers by ring-template correlation; first-loading branch = smallest deflection at each load level up to the cycle load; see digitise_slutter1963.py",
                              files=["slutter1963_digitised/digitise_slutter1963.py", "slutter1963_digitised/lehigh_plot_core.py", "slutter1963_digitised/calibration.json",
                                     "slutter1963_digitised/markers_raw.csv", "slutter1963_digitised/overlay_check.png", "slutter1963_digitised/overlay_full.png"],
                              uncertainty=("+/-0.01 in (B10 +/-0.015 in: frame corner at -0.015 in in the tick calibration) and +/-0.3 kip; 1012 px/in, 30.4 px/kip, tick-fit rms <= 3.7 px" if high else
                                           "+/-0.005 in and +/-0.2 kip; about 2015-2030 px/in and 45.5 px/kip, tick-fit rms <= 8.7 px (0.004 in) and <= 2 px, frame corner within 0.007 in of zero")),
            assumptions=[
                "Handbook 12WF27 plate dimensions; web area A - 2 b_f t_f (includes fillets); C_f = min(0.85 f'c b t, A_f F_yf + A_w F_yw), which reproduces the authors' C (324, 310, 314 kips).",
                "Connector strength for eta and the model: the programme's own push-out ultimates (L-studs: P1/P4 mean 10.7 kips, also used for the second series because P7 was rejected by the authors); push-out backbones digitised for stiffness.",
                "Connector positions: first row on each support centreline (dimensioned for B10-B12, assumed for the others); rows between the two S1 load points (82.5, 90, 97.5 in) lie in the constant-moment zone.",
                "Deflection record excludes self-weight and loading-beam weight (curves start at zero); not stated explicitly.",
            ],
        )
        if not S["included"]:
            spec["spec_summary"]["reason"] = S["reason"]
        if sid == "B4-S1":
            spec["spec_summary"]["eta_source"] += " (C_f of the unreduced section; the web holes reduce C_f and raise eta)"
        specimens.append(spec)

    excluded_groups = [
        dict(id="B1-S1, B2-S1", included=False, reason="no shear connectors (bond and friction only; slab separated); eta = 0, below the 0.25 lower bound (279.2 pp. 19-20)"),
        dict(id="retests B3-S2/S3 (T4, T7), B4-S2/S4 (T4, T8), B5-S2/S5 (T4, T11), B7-S2 (T4), B8-S2/S4 (T4, T9), B9-S3/S5 (T5, T10)", included=False,
             reason="second and final tests of beams already loaded to incipient slab crushing (residual deflections up to 1.2 in or more, cracked slab, yielded steel): not monotonic loading of an undamaged beam. These are the tests to which 279.15 Table 4 sum(q_u)/C = 0.437-0.897 refers (except B6, B10-B12)."),
        dict(id="B13", included=False, reason="two-span continuous beam (negative bending over the support)"),
        dict(id="BI, BII, BIII (Culver and Coston 1961)", included=False, reason="separate programme (culver1961 key); not part of this file"),
    ]

    data = dict(
        programme="Lehigh University composite beams for buildings (AISC project 279): Slutter and Driscoll (1963) with progress reports 279.2 (B1-B6) and 279.6 (B7-B13)",
        key="slutter1963", accessible=True, included=True,
        citations=[
            "Slutter, R.G. and Driscoll, G.C. Jr. (1963). The flexural strength of steel and concrete composite beams. Fritz Engineering Laboratory Report 279.15, Lehigh University. Journal version: J. Struct. Div. ASCE 91(ST2), 71-99 (1965).",
            "Culver, C., Zarzeczny, P.J. and Driscoll, G.C. Jr. (1960). Tests of composite beams for buildings (Composite design for buildings, Progress Report 1). Fritz Engineering Laboratory Report 279.2, Lehigh University.",
            "Culver, C., Zarzeczny, P.J. and Driscoll, G.C. Jr. (1961). Tests of composite beams for buildings (Progress Report 2). Fritz Engineering Laboratory Report 279.6, Lehigh University.",
            "Slutter, R.G. and Driscoll, G.C. Jr. (1962). Test results and design recommendations for composite beams (Progress Report 3). Fritz Engineering Laboratory Report 279.10, Lehigh University (Fig. 14 legend eta values).",
        ],
        sources=[
            dict(report="279.15", url="https://web.archive.org/web/20200318193301id_/https://preserve.lehigh.edu/cgi/viewcontent.cgi?article=2805&context=engr-civil-environmental-fritz-lab-reports",
                 pages="Table 1 p. 27, Table 2 p. 28, Table 3a p. 29, Table 4 p. 31, Eqs. 9-14 pp. 10-12 (printed = PDF - 7)", md5="f9fdb3d1dcd9d2774bac06ceda53eb15"),
            dict(report="279.2", url="Lehigh Preserve Fritz Lab reports (Wayback copy of the bepress PDF; local scratchpad lit_sources/slutter1963/r279_2.pdf, 107 pp)",
                 pages="description pp. 6-16, results pp. 18-31, App. 10 pp. 45-56, Tables 8-9 pp. 64-65, Fig. 1 p. 66, beam Figs. 9-18 pp. 74-83, push-out Figs. 28-33 pp. 95-100 (printed = PDF - 5)"),
            dict(report="279.6", url="https://web.archive.org/web/2020*/preserve.lehigh.edu/cgi/viewcontent.cgi?article=2810* (bepress article 2810; local scratchpad lit_sources/slutter1963/r279_6.pdf, 104 pp, md5 77560200e1f3dd40340bae687950417e)",
                 pages="description pp. 2-12, results pp. 15-23, App. 10 pp. 34-46, Tables 3-5 pp. 49-51, Tables 11-12 pp. 57-58, Figs. 1-2 pp. 59-60, beam Figs. 10-20 pp. 68-78, push-out Fig. 37 p. 97 (printed = PDF - 5)"),
            dict(report="279.10", url="bepress article 2802 via the Wayback Machine (local scratchpad r279_10.pdf)", pages="Fig. 14 legend (PDF p. 66)"),
        ],
        screening=("Solid 48 x 4 in slab on 12WF27, 15 ft simple span, studs or channels: every section ratio, f'c and F_y is inside the design space; "
                   "span (15 ft, -25 %), deck (4 in, -11 %) and slab width/girder spacing (4 ft, -20 %) are below it. The Chapman beams already accepted as very close "
                   "are outside by span (-10 %) and spacing (-20 %); these beams have the same lab scale and proportions (span/total depth 11.3 vs 12.0) with a larger span "
                   "deficit and a thinner deck, so they are included as a flagged lab-scale group (the author may prefer to show them alongside Chapman rather than with the bridge-scale tests). "
                   "Only first (virgin) loadings are used; 279.15 Table 4 values (0.437-0.897) belong mostly to retests of damaged beams. With the programme's own push-out strengths "
                   "the virgin tests span eta = 0.36 (B6) to 0.82 (B9); natural bond was not prevented, and several first-loading records are wholly below the first-slip load."),
        specimens=specimens,
        excluded_other=excluded_groups,
        data_files=dict(digitisation_script="slutter1963_digitised/digitise_slutter1963.py", core="slutter1963_digitised/lehigh_plot_core.py",
                        builder="slutter1963_digitised/build_slutter1963_json.py", raw_markers="slutter1963_digitised/markers_raw.csv",
                        pushout_raw_markers="slutter1963_digitised/pushout_markers_raw.csv", calibration="slutter1963_digitised/calibration.json",
                        overlay_check="slutter1963_digitised/overlay_check.png", overlay_full="slutter1963_digitised/overlay_full.png",
                        overlay_check_pushout="slutter1963_digitised/overlay_check_pushout.png"),
        notes=[
            "Authors' full-interaction deflections versus digitised first loading: S1 at 40 kips 0.324 in (theory, bending + shear) versus 0.289 (B3), 0.318 (B5), 0.369 (B6), 0.358 (B7), 0.385 (B9); five-point loading at 60 kips 0.356 in versus 0.353 (B10), 0.413 (B11), 0.378 (B12).",
            "279.10 p. 24 states that the B6 (0.473), B8 (0.988) and B9 (1.21) load-deflection curves coincide up to M/M_u = 0.59; with bond intact up to the first-slip load, little elastic stiffness reduction should be expected in the first-loading data.",
            "B6-S1, B3-S1 and B5-S1 first-loading readings all lie at or below the first-slip load (bond); B7-S1 (slip at 20 kips), B9-S1 (10 kips), B10 (40 kips), B11 and B12 (30 kips) have readings above first slip.",
            "Stud strength basis matters for eta: push-out (10.7 kips per 1/2 in L-stud) and AISC Q_n (10.3-10.9 kips) agree, while the Slutter-Driscoll formula implies about 13.9 kips; eta_plastic here uses push-out.",
        ],
    )
    OUT.write_text(json.dumps(data, indent=1))
    for sp in specimens:
        ss = sp["spec_summary"]
        print(ss["id"], ss["included"], ss["eta_plastic"], ss["n_elastic_points"], [(m["load"], m["deflection"]) for m in sp["measured"]])


if __name__ == "__main__":
    build()
