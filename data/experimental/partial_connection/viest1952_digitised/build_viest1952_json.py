"""Assemble ../viest1952.json from the transcribed bulletin data and the
digitised markers (markers_raw.csv written by digitise_viest1952.py).

Every number carries its provenance (Bulletin 405 printed page; PDF page =
printed page + 4). Derived quantities (plate areas, C_f, eta) are computed
here so that the arithmetic is auditable. Units: source units (kips, in,
psi/ksi) with explicit conversions.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "viest1952.json"
B405 = "Viest et al. (1952), Univ. Illinois Eng. Exp. Sta. Bulletin 405"

# ---------------------------------------------------------------- handbook
# Rolled WF shapes: dimensions not given in the bulletin (it tabulates only
# A and I, Table 18, p. 94). Values from the AISC Manual WF tables (same
# rolled shapes as today's W21x68 and W24x76); A and I agree with Table 18.
SHAPES = {
    "21WF68": dict(d=21.13, bf=8.270, tf=0.685, tw=0.430, A=20.02, Ix=1478.3),
    "24WF76": dict(d=23.91, bf=8.990, tf=0.680, tw=0.440, A=22.37, Ix=2096.4),
}

# ---------------------------------------------------------------- per beam
BEAMS = {
    "B24W": dict(shape="24WF76", t_slab=6.25, fc=5500, Ec=4.16e6, fy_f=35.8, fu_f=61.3, fy_w=38.7, fu_w=64.4,
                 Eb=30.7e3, shored=False, top_long="3/8 in bars at 12 in", top_long_area=6 * 0.11,
                 bot_trans="1/2 in bars at 6 in", ch_width=6.0, ch_fy=(44.4, 47.0), ch_fu=(67.25, 67.15),
                 positions="9 + 18k in, k = 0..24 (25 channels; first 9 in from each support centreline, 18 in pitch, one on the midspan line)",
                 pos_list=[9 + 18 * k for k in range(25)],
                 table18=dict(A_b=22.37, I_b=2096, E_b_1e6psi=30.7, A_slab_incl_bars=469.6, I_slab=1529, E_c_1e6psi=4.16, n=7.38,
                              I_composite=5943, NA_above_steel_centroid=10.97, k_over_s_1e6psi=0.374, weight_lb_ft=567),
                 table25=dict(one_over_C=39, k_1e6_lb_in=6.73, spacing="18 in"),
                 t19_defl=102, t19_incomplete=104, t19_none=248, first_yield_theory=64.6, first_yield_meas=79,
                 ult_test="111 (test stopped, not failed; estimated 120)", ult_theory=116.7,
                 bond="white lead on top flange to prevent bond (Sec. 31, p. 82)",
                 pushout=dict(ref="4C3C9 (6-in-wide C4x5.4, f'c 5340 psi)", q003=20.7, q006=33.7, q020=60.5, qu=116.1,
                              note="nearest Group I f'c below 5500 psi; 4C3C10 (5740 psi) is flagged by the authors as of questionable accuracy (21.8/32.1/60.1 kips at 0.003/0.006/0.020 in, ult 112.1)"),
                 tests="Table 17 (p. 92): midspan flexural tests to 40 and 60 kips, midspan yield test to 100 kips, capacity test 111 kips"),
    "B24S": dict(shape="24WF76", t_slab=6.17, fc=5620, Ec=4.15e6, fy_f=35.2, fu_f=59.2, fy_w=37.9, fu_w=61.2,
                 Eb=30.6e3, shored=True, top_long="3/8 in bars at 12 in", top_long_area=6 * 0.11,
                 bot_trans="5/8 in bars at 6 in", ch_width=6.0, ch_fy=(44.4, 47.0), ch_fu=(67.25, 67.15),
                 positions="per half span from each support centreline: 6, then 6 spaces at 12 in, 5 at 15 in, 4 at 18 in to midspan (31 channels, one on the midspan line)",
                 pos_list=None,
                 table18=dict(A_b=22.37, I_b=2096, E_b_1e6psi=30.6, A_slab_incl_bars=463.8, I_slab=1471, E_c_1e6psi=4.15, n=7.38,
                              I_composite=5943, NA_above_steel_centroid=10.97, k_over_s_1e6psi=0.513, weight_lb_ft=552),
                 table25=dict(one_over_C=54, k_1e6_lb_in="7.49 (6.15 for s_min)", spacing="12/15/18 in, average 14.6 in, equivalent 13.1 in"),
                 t19_defl=102, t19_incomplete=103, t19_none=249, first_yield_theory=69.5, first_yield_meas=78,
                 ult_test="115 (test stopped, not failed; estimated 130)", ult_theory=114.9,
                 bond="white lead on top flange to prevent bond (Sec. 31, p. 82)",
                 pushout=dict(ref="4C3C9 (6-in-wide C4x5.4, f'c 5340 psi)", q003=20.7, q006=33.7, q020=60.5, qu=116.1,
                              note="nearest reliable Group I specimen to f'c 5620 psi"),
                 tests="Table 17 (p. 92): midspan tests to 60 and 65 kips, midspan yield test to 115 kips (stopped)"),
    "B21S": dict(shape="21WF68", t_slab=6.25, fc=6480, Ec=4.58e6, fy_f=35.1, fu_f=58.3, fy_w=41.8, fu_w=60.4,
                 Eb=29.6e3, shored=True, top_long="3/8 in bars at 12 in", top_long_area=6 * 0.11,
                 bot_trans="5/8 in bars at 6 in", ch_width=6.0, ch_fy=(42.2, 42.7), ch_fu=(60.5, 63.0),
                 positions="as B24S (31 channels, one on the midspan line)", pos_list=None,
                 table18=dict(A_b=20.02, I_b=1478, E_b_1e6psi=29.6, A_slab_incl_bars=467.1, I_slab=1521, E_c_1e6psi=4.58, n=6.46,
                              I_composite=4652, NA_above_steel_centroid=10.72, k_over_s_1e6psi=0.366, weight_lb_ft=565),
                 table25=dict(one_over_C=44, k_1e6_lb_in="5.35 (4.29 for s_min)", spacing="12/15/18 in, average 14.6 in, equivalent 13.4 in"),
                 t19_defl=101, t19_incomplete=101, t19_none=262, first_yield_theory=56.6, first_yield_meas=62,
                 ult_test="102 (failed)", ult_theory=100.6,
                 bond="white lead on top flange to prevent bond (Sec. 31, p. 82)",
                 pushout=dict(ref="4C3C11 (6-in-wide C4x5.4, f'c 6320 psi, white lead instead of grease)", q003=None, q006=30.0, q020=65.8, qu=118.2,
                              note="nearest Group I f'c; no reading at 0.003 in"),
                 tests="Table 17 (p. 92): midspan tests to 50, 50 and 75 kips, capacity test 102 kips"),
    "B21W": dict(shape="21WF68", t_slab=6.11, fc=5580, Ec=4.45e6, fy_f=34.3, fu_f=57.9, fy_w=41.4, fu_w=61.8,
                 Eb=29.4e3, shored=False, top_long="1/2 in bars at 12 in (1/2 in used instead of the design 3/8 in, Sec. 31, p. 82)",
                 top_long_area=6 * 0.196, bot_trans="5/8 in bars at 6 in", ch_width=4.0, ch_fy=(38.9, 39.7), ch_fu=(54.5, 56.6),
                 positions="9 + 36k in, k = 0..12 (13 channels; first 9 in from each support centreline, 36 in pitch, one on the midspan line)",
                 pos_list=[9 + 36 * k for k in range(13)],
                 table18=dict(A_b=20.02, I_b=1478, E_b_1e6psi=29.4, A_slab_incl_bars=461.1, I_slab=1434, E_c_1e6psi=4.45, n=6.61,
                              I_composite=4580, NA_above_steel_centroid=10.58, k_over_s_1e6psi=0.0907, weight_lb_ft=551),
                 table25=dict(one_over_C=11, k_1e6_lb_in=3.28, spacing="36 in"),
                 t19_defl=113, t19_incomplete=115, t19_none=262, first_yield_theory=47.9, first_yield_meas=47,
                 ult_test="79 (failed: butt welds of longitudinal slab bars fractured, all connectors yielded, west-half connectors later broke)",
                 ult_theory=96.8,
                 bond="no bond prevention; natural bond broken before the reported tests by eleven repetitions of 40 kips at midspan (Sec. 35, p. 93; Sec. 44, p. 122)",
                 pushout=dict(ref="4C3W2 (4-in-wide C4x5.4, the B21W connector, f'c 4430 psi, natural bond NOT broken)", q003=23.4, q006=29.6, q020=46.0, qu=81.9,
                              note="failure by the connector welds (WCA, S+T); the only push-out of the 4-in channel; alternative: 4C3C9 scaled by width 4/6 (13.8/22.5/40.3 kips, ult 77.4)"),
                 tests="Table 17 (p. 92): 1A midspan 40 kips (bond present), 11 repetitions of 40 kips, 1B midspan 40, 2 off-midspan (section 5E) 30, 3 midspan 40, 4 off-midspan (5W) 30, 5A midspan 50, 5B midspan 50, 6 capacity 79 kips"),
}


def plate_areas(s):
    Af = 2 * s["bf"] * s["tf"]
    return Af, s["A"] - Af


def c_f(beam):
    b = BEAMS[beam]
    s = SHAPES[b["shape"]]
    Af, Aw = plate_areas(s)
    AsFy = Af * b["fy_f"] + Aw * b["fy_w"]
    conc = 0.85 * b["fc"] / 1000 * 72.0 * b["t_slab"]
    return dict(A_flanges_in2=round(Af, 3), A_web_incl_fillets_in2=round(Aw, 3), AsFy_kip=round(AsFy, 1),
                concrete_085fcAc_kip=round(conc, 1), C_f_kip=round(min(AsFy, conc), 1))


def half_span_count(beam):
    return {"B24W": 12, "B24S": 15, "B21S": 15, "B21W": 6}[beam]


def markers():
    rows = list(csv.DictReader(open(HERE / "markers_raw.csv")))
    return rows


def build():
    rows = markers()
    specimens = []
    for beam, b in BEAMS.items():
        s = SHAPES[b["shape"]]
        C = c_f(beam)
        n_half = half_span_count(beam)
        qu = b["pushout"]["qu"]
        eta_po = n_half * qu / C["C_f_kip"]
        # AISC channel formula (as the database would use when no test data):
        # Qn = 0.3 (tf + 0.5 tw) Lc sqrt(f'c Ec), C4x5.4 tf = 0.296, tw = 0.184 in
        Ec_aci = 57.0 * (b["fc"]) ** 0.5  # ksi (57000 sqrt(psi) psi)
        qn_aisc = 0.3 * (0.296 + 0.5 * 0.184) * b["ch_width"] * (b["fc"] / 1000 * Ec_aci) ** 0.5
        eta_aisc = n_half * qn_aisc / C["C_f_kip"]
        long_bot_area = 12 * 0.196
        rho = (b["top_long_area"] + long_bot_area) / (72.0 * b["t_slab"])
        sd_eta = {"B24W": 1.41, "B24S": 1.59, "B21S": 1.95, "B21W": 0.50}[beam]

        measured, context = [], []
        for r in rows:
            if r["beam"] != beam:
                continue
            P, d = float(r["load_kip"]), float(r["deflection_in"])
            src_fig = "Fig. 43, left sub-panel 'At Midspan' (printed p. 102, PDF p. 106)" if r["figure"] == "fig43" else \
                "Fig. 53, right panel 'Deflection at Midspan' (printed p. 123, PDF p. 127)"
            unc = dict(load_kip=0.3, deflection_in=0.006) if r["figure"] == "fig43" else dict(load_kip=0.2, deflection_in=0.004)
            item = dict(load=round(P, 2), load_unit="kip (single concentrated load at midspan, applied load only)",
                        deflection=round(d, 4), deflection_unit="in (midspan, bottom flange, from readings taken just before the first increment)",
                        moment_midspan_applied_kip_in=round(P * 450.0 / 4.0, 0), uncertainty=unc,
                        source=f"digitised, {B405} {src_fig}; raw marker in viest1952_digitised/markers_raw.csv (px={r['px_x']}, py={r['px_y']})")
            if r["figure"] == "fig53" and r["marker"] == "filled":
                item.update(series="Fig. 53 test BEFORE breaking bond (test 1A)", elastic=True,
                            flags=["natural bond intact: not the modelled condition; context only"])
                context.append(item)
                continue
            series = "Fig. 43 midspan test (5A or 5B of Table 17, the only midspan tests reaching 50 kips before the capacity test; the figure does not say which)" \
                if r["figure"] == "fig43" and beam == "B21W" else \
                ("Fig. 53 test AFTER breaking bond (test 1B of Table 17, to 40 kips)" if r["figure"] == "fig53" else
                 "Fig. 43 midspan test (test not identified in the figure; plotted maxima are consistent with the midspan yield test of Table 17)")
            fy_meas = b["first_yield_meas"]
            elastic = P < fy_meas - 0.5
            flags = []
            if not elastic:
                flags.append(f"at or above the measured first-yield load {fy_meas} kips (Table 21, p. 106)")
            if P < 15 and r["figure"] == "fig43":
                flags.append("low-load point: +/-0.006 in is about 3 % of the reading")
            item.update(series=series, elastic=elastic, flags=flags)
            measured.append(item)
        measured.sort(key=lambda m: (m["series"], m["load"]))
        n_el = sum(m["elastic"] for m in measured)
        is_partial = beam == "B21W"
        oor = []
        if b["fy_f"] < 36.0:
            oor.append(f"flange F_y {b['fy_f']} ksi below 36 ksi by {36.0 - b['fy_f']:.1f} ksi ({(36.0 - b['fy_f']) / 36 * 100:.0f} %); web F_y {b['fy_w']} ksi in range; area-weighted {C['AsFy_kip'] / s['A']:.1f} ksi")
        if eta_po > 1.0:
            oor.append(f"eta_plastic {eta_po:.2f} > 1.00 (over-connected; reference for the full-composite end, R_EI read at eta = 1)")
        oor.append("channel connectors (not studs): representable as discrete connectors with the bulletin's push-out load-slip data")
        if not b["shored"]:
            oor.append("unshored: slab dead load carried by the steel section alone before the tests; deflection record is applied load only")
        screening = (
            ("INCLUDED (partial connection). " if is_partial else "INCLUDED as a near-full reference (eta >= 1 by every strength basis). ")
            + f"Solid 72 x {b['t_slab']} in cast-in-place slab, simply supported 37.5 ft span, {b['shape']} ({s['d']} in deep), f'c {b['fc'] / 1000:.2f} ksi, "
            f"b_f/d {s['bf'] / s['d']:.3f}, t_f/b_f {s['tf'] / s['bf']:.4f}, t_w/d {s['tw'] / s['d']:.4f}, girder spacing (slab width) 6 ft, "
            f"monotonic positive bending with a midspan point load in the elastic tests: all inside the design space except: " + "; ".join(oor) + "."
        )
        spec = dict(
            id=beam, included=True, screening=screening,
            spec_summary=dict(
                id=beam, included=True,
                reason=("partial connection by the plastic definition with the programme's own push-out strength; in range except flange F_y and connector type" if is_partial
                        else "near-full reference (over-connected); in range except flange F_y and connector type"),
                eta_source=(f"Slutter and Driscoll (1963) Table 4: sum(q_u)/C = {sd_eta} (q_u from their channel formula Eq. 12); "
                            f"this file: {n_half} channels per half span x {qu} kips (Bulletin 405 push-out {b['pushout']['ref'].split(' (')[0]}, Table 5, p. 26-27) / C_f {C['C_f_kip']} kips = {eta_po:.3f}; "
                            f"AISC channel Q_n {qn_aisc:.1f} kips gives {eta_aisc:.3f}"),
                eta_plastic=round(eta_po, 3), span_ft=37.5, steel=f"{b['shape']} rolled (d {s['d']}, b_f {s['bf']}, t_f {s['tf']}, t_w {s['tw']} in)",
                slab=f"solid 72 x {b['t_slab']} in, rho_l {rho * 100:.2f} % (longitudinal, both layers)",
                fc_ksi=b["fc"] / 1000, fy_ksi=b["fy_f"],
                connectors=f"C4x5.4 channels {b['ch_width']:.0f} in long, {n_half} per half span (+1 on the midspan line); {b['positions']}",
                loading="single concentrated load at midspan (14 x 14 in plate), simply supported 450 in span",
                n_elastic_points=n_el,
                data_form="digitised figure (Fig. 43" + (" and Fig. 53" if beam == "B21W" else "") + ") + tabulated deflection ratio (Table 19)",
                out_of_range="; ".join(oor)),
            geometry=dict(
                span_in=450.0, span_ft=37.5,
                support_positions="centre to centre of bearings: 6 in diameter half-cylinder at one end, 2.5 in roller at the other, 12 x 12 x 2 in bearing plates, 6 x 9 x 1 in sole plates (Sec. 31, p. 82; Sec. 34, p. 91; Fig. 32, p. 80)",
                overhang_beyond_supports_in="4 (slab and beam end to support centreline, Fig. 32 elevation)",
                load_positions="single load at midspan through a 14 x 14 x 2 in steel plate grouted with gypsum plaster (Sec. 34, p. 91)",
                steel_section=dict(designation=b["shape"], source="rolled WF (ASTM A7-46); dimensions from the AISC Manual WF table (not given in the bulletin); A and I agree with Table 18 (p. 94)",
                                   **{k: v for k, v in s.items()}, A_flanges_in2=C["A_flanges_in2"], A_web_incl_fillets_in2=C["A_web_incl_fillets_in2"],
                                   stiffeners="two 4.5 x 0.5 in plates each side of the web over each support (Sec. 31, pp. 81-82)"),
                slab=dict(width_in=72.0, thickness_in=b["t_slab"], thickness_source="actual average, Fig. 32 (p. 80); design 6 in (Slutter and Driscoll 1963 Table 1 transposes B24W and B21W)",
                          haunch_in=0.0, haunch_source="slab cast on plywood forms flush with the top flange (Sec. 33, p. 85)",
                          longitudinal_reinforcement=dict(top=b["top_long"], bottom="1/2 in bars at 6 in (12 bars)",
                                                          area_top_in2=round(b["top_long_area"], 3), area_bottom_in2=round(long_bot_area, 3),
                                                          cover_in=1.0, centroid_depth_assumed_in=dict(top=1.25, bottom=round(b["t_slab"] - 1.25, 2)),
                                                          rho_l=round(rho, 4),
                                                          source="Fig. 32 cross-sections (p. 80); bar areas nominal (1/2 in = 0.196, 3/8 in = 0.110 in2); Table 18 slab area including transformed bars "
                                                                 f"({b['table18']['A_slab_incl_bars']} in2) is consistent with this steel area (72 x t + (n-1) A_s)"),
                          transverse_reinforcement=f"1/2 in bars at 6 in top; {b['bot_trans']} bottom (Fig. 32)",
                          bars="intermediate-grade deformed bars (B24W: mixed stock), Sec. 32 (p. 83); longitudinal bars butt-welded near midspan"),
                shoring=("shored: " + ("rigid intermediate support removed 46 days after casting (Sec. 33, p. 87)" if beam == "B24S" else
                                       "intermediate shore reaction adjusted daily, removed 20 days after casting (Sec. 33, p. 87)")) if b["shored"]
                else "unshored (Sec. 31, pp. 81-82): dead load on the steel section alone",
                table18_section_properties=b["table18"]),
            materials=dict(
                concrete=dict(fc_psi=b["fc"], Ec_measured_psi=b["Ec"], source="Table 14 (p. 84): average of 6 x 12 in cylinders over the test period; E_c initial modulus by compressometer",
                              normal_weight=True),
                steel_beam=dict(flange_fy_ksi=b["fy_f"], flange_fu_ksi=b["fu_f"], web_fy_ksi=b["fy_w"], web_fu_ksi=b["fu_w"],
                                E_table18_ksi=b["Eb"], source="Table 15 (p. 85) static coupon values from unyielded portions after testing; E from Table 18 (p. 94)",
                                residual_stress="about 30 ksi compression at mid-depth of the 21 in beams, flanges -3 to +9 ksi (Sec. 37, p. 97)"),
                channel_steel=dict(fy_ksi_parallel_perpendicular=b["ch_fy"], fu_ksi_parallel_perpendicular=b["ch_fu"], source="Table 16 (p. 85)")),
            connectors=dict(
                type="C4x5.4 (4 in, 5.4 lb) channel, welded to the top flange with 3/16 in continuous fillet welds front and back, all facing the same direction (Sec. 33, p. 85)",
                length_in=b["ch_width"], channel_dims="t_f (average) 0.296 in, t_w 0.184 in (handbook); Table 2 of the bulletin lists web 0.180 in",
                layout=b["positions"], positions_from_left_support_in=b["pos_list"] if b["pos_list"] else
                sorted(set([6, 18, 30, 42, 54, 66, 78, 93, 108, 123, 138, 153, 171, 189, 207, 225] + [450 - x for x in [6, 18, 30, 42, 54, 66, 78, 93, 108, 123, 138, 153, 171, 189, 207]])),
                per_half_span_excluding_midspan_connector=n_half,
                layout_source="Fig. 32 elevations (p. 80) and Sec. 31 (p. 81); midspan connector confirmed on the crop of Fig. 32",
                push_out_same_programme=dict(
                    specimen=b["pushout"]["ref"],
                    load_per_connector_kip_at_average_slip={"0.003 in": b["pushout"]["q003"], "0.006 in": b["pushout"]["q006"], "0.020 in": b["pushout"]["q020"]},
                    ultimate_kip=b["pushout"]["qu"],
                    secant_stiffness_kip_per_in={k: (round(v / float(k.split()[0]), 0) if v else None) for k, v in
                                                 {"0.003 in": b["pushout"]["q003"], "0.006 in": b["pushout"]["q006"], "0.020 in": b["pushout"]["q020"]}.items()},
                    source="Table 5 (printed pp. 26-27, PDF pp. 30-31), read from the rendered page; concrete strengths Table 3 (p. 18)",
                    note=b["pushout"]["note"]),
                beam_derived_modulus=dict(**b["table25"], source="Table 25 (p. 118), computed by the authors from MEASURED BEAM SLIPS (Bul. 396 Sec. 54 method), not from push-out tests; "
                                                                "footnote: based on k = 6.73e6 lb/in per 6 in width (B24W, B24S) and 4.92e6 lb/in per 6 in width (B21S, B21W), k proportional to width; "
                                                                "Table 18 k/s = k divided by the (average) spacing"),
                strength_for_eta=dict(value_kip=qu, basis="bulletin push-out ultimate (Table 5)",
                                      alternatives=dict(slutter_driscoll_eq12_kip=round(550 * (0.296 + 0.5 * 0.184) * b["ch_width"] * (b["fc"]) ** 0.5 / 1000, 1),
                                                        aisc_channel_formula_kip=round(qn_aisc, 1),
                                                        beam_connector_yield_kip=(38.7 if beam == "B21W" else None)),
                                      alt_note="Table 27 (p. 135): 9,680 lb per inch width is the yield load of the channel connectors at first yielding of B21W (38.7 kips per 4 in channel)"),
                degree_of_connection=dict(C_f=C, n_per_half_span=n_half, eta_pushout=round(eta_po, 3), eta_aisc_channel_formula=round(eta_aisc, 3),
                                          eta_slutter_driscoll_table4=sd_eta,
                                          counting_convention="connectors between a support and the midspan section, excluding the connector on the midspan line (it carries no shear under the symmetric midspan load); including it adds one connector",
                                          authors_statement=("B21W 'built with an extremely weak shear connection' (p. 103), designed so that beam and connectors yield together (Sec. 31, pp. 81-82); "
                                                             "1/C = 11 versus > 20 for practically complete interaction (Sec. 42, pp. 118-119)") if is_partial else
                                          f"1/C = {b['table25']['one_over_C']} > 20: practically complete interaction (Sec. 42, p. 119)")),
            loading=dict(
                pattern="single concentrated load at midspan (module pattern: midspan point load)",
                load_plate="14 x 14 x 2 in",
                sequence=b["tests"],
                instrumentation="deflectometers with 0.001 in dials bearing on the laboratory floor and the bottom flange at midspan (Sec. 34, p. 90); loads by 125 kip ring dynamometer",
                bond=b["bond"],
                repeated_loading=("11 repetitions of 40 kips at midspan to break natural bond before test 1B; all digitised post-bond tests stayed below yield except the 47 and 50 kip readings" if is_partial
                                  else "several elastic tests at midspan and quarter points before the plotted test; all below yield (Sec. 35, pp. 91-93; Table 17, p. 92)")),
            deflection_zero_includes=("applied midspan load only: 'a set of readings of all instruments was taken before application of the first load increment' (Sec. 35, p. 93). "
                                      + ("Unshored: slab and beam self-weight (551-567 lb/ft) was already on the steel section and is not in the record. " if not b["shored"] else
                                         "Shored: self-weight acted on the composite section after shore removal and is not in the record. ")
                                      + "Deflection measured relative to the laboratory floor; bearing compression is not separated (not stated)."),
            elastic_limit=dict(first_yield_measured_kip=b["first_yield_meas"], first_yield_theory_incomplete_interaction_kip=b["first_yield_theory"],
                               source="Table 21 (p. 106): load at first yielding, theory with load spread over 14 in and coupon yield",
                               ultimate_test_kip=b["ult_test"], ultimate_complete_interaction_theory_kip=b["ult_theory"], ultimate_source="Table 24 (p. 116)",
                               rule_used="points below the measured first-yield load are flagged elastic; the modelling step must also apply M/M_p <= 0.6 with its own M_p (and consider the dead-load moment already on the steel for unshored beams)"),
            tabulated_stiffness_ratio=dict(
                midspan_deflection_percent_of_complete_interaction=dict(test=b["t19_defl"], incomplete_interaction_theory=b["t19_incomplete"], no_interaction_theory=b["t19_none"]),
                source="Table 19 (p. 104): concentrated load at midspan, test data are averages; complete-interaction theory = 100",
                note="the authors' complete-interaction deflection appears to include shear deformation: PL^3/48 E_b I with Table 18 values is about 6 % below the plotted complete-interaction line (see notes)"),
            measured=measured,
            context_measured=context,
            digitisation=dict(
                method="600 dpi render of the 150 ppi JPEG scan (4x interpolation); grid lines detected programmatically (>= 5 per axis, labelled lines 0/0.4/0.8 in and 0/20/40... kips included), piecewise-linear pixel-to-value map; open markers by ring-template correlation plus closed-interior test; see digitise_viest1952.py",
                files=["viest1952_digitised/digitise_viest1952.py", "viest1952_digitised/calibration.json", "viest1952_digitised/markers_raw.csv", "viest1952_digitised/overlay_check.png"],
                uncertainty=("Fig. 43: +/-0.006 in deflection and +/-0.3 kip load (grid-fit rms 0.002-0.003 in and 0.12-0.17 kip; marker centroid +/-0.5 scan pixel = +/-0.003 in; hand-drafted grid cells irregular by up to 5 render px, removed by the piecewise map)"
                             + ("; Fig. 53: +/-0.004 in and +/-0.2 kip (1115 px/in, 22.6 px/kip)" if beam == "B21W" else ""))),
            assumptions=[
                "Steel plate dimensions from the AISC WF table (bulletin gives only A and I); web area taken as A - 2 b_f t_f (includes fillets).",
                "C_f = min(0.85 f'c b t, A_f F_y,flange + A_w F_y,web) with coupon yields; slab reinforcement ignored in C_f.",
                "Longitudinal bar centroids assumed 1.25 in from the slab faces (1 in cover + half bar); the relative position of longitudinal and transverse bars is not stated.",
                "Connector strength for eta and the model: bulletin push-out ultimate of the matching channel (Table 5); stiffness: Table 5 load-slip points (or the beam-derived Table 25 modulus as a sensitivity case).",
            ] + ([
                "B21W push-out 4C3W2 had natural bond intact and weld failure; its initial stiffness is therefore an upper bound for the bond-broken beam. Sensitivity: k = 3.28e3 kip/in (Table 25, beam slips), 3.74e3 kip/in (4C3C9 secant at 0.006 in x 4/6), 4.93e3 kip/in (4C3W2 secant at 0.006 in).",
                "eta counting: 6 channels per half span (midspan channel excluded); including it gives 7 x 81.9 / C_f = %.3f." % (7 * qu / C["C_f_kip"]),
            ] if is_partial else []),
        )
        specimens.append(spec)

    data = dict(
        programme="Viest, Siess, Appleton and Newmark (1952): full-scale composite T-beams with channel shear connectors, University of Illinois (Studies of Slab and Beam Highway Bridges, Part IV)",
        key="viest1952",
        accessible=True,
        included=True,
        citations=[
            "Viest, I.M., Siess, C.P., Appleton, J.H. and Newmark, N.M. (1952). Full-scale tests of channel shear connectors and composite T-beams. University of Illinois Engineering Experiment Station Bulletin Series No. 405, Vol. 50, No. 21. Urbana, IL. 162 pp.",
            "Slutter, R.G. and Driscoll, G.C. Jr. (1963). The flexural strength of steel and concrete composite beams. Fritz Engineering Laboratory Report 279.15, Lehigh University (Table 4: sum(q_u)/C for B21S, B21W, B24S, B24W).",
        ],
        sources=[dict(url="https://www.ideals.illinois.edu/items/4841 (PDF bitstream https://www.ideals.illinois.edu/items/4841/bitstreams/18942/data.pdf; handle https://hdl.handle.net/2142/4365)",
                      document="Bulletin 405, open access scan (2007 UIUC digitisation), 162 PDF pages, embedded page images 150 ppi JPEG; md5 bc4b997648d5e27dab5b3947f88e1ce0",
                      local_copy="session scratchpad lit_sources/viest1952/viest1952_bulletin405.pdf (not in the repository)",
                      page_mapping="PDF page = printed page + 4",
                      pages_used={"push-out Table 3 (concrete), Table 5 (load-slip, ultimate)": "printed 18, 26-27",
                                  "channel formulas, connector modulus (Sec. 27, App. B)": "printed 74-75, 150",
                                  "specimens, Fig. 32, Sec. 31, materials Tables 14-16": "printed 80-85",
                                  "construction, shoring, bond (Sec. 33)": "printed 85-87",
                                  "instrumentation, test procedure, Table 17 (Secs. 34-35)": "printed 90-93",
                                  "Table 18 section properties": "printed 94",
                                  "Fig. 43, Table 19, Tables 20-21": "printed 102-106",
                                  "Tables 24-25, Sec. 42 shear connection": "printed 116-119",
                                  "Fig. 53 effect of bond (Sec. 44)": "printed 122-123",
                                  "Tables 26-28 connector forces and slips": "printed 130-137"}),
                 dict(url="https://web.archive.org/web/20200318193301id_/https://preserve.lehigh.edu/cgi/viewcontent.cgi?article=2805&context=engr-civil-environmental-fritz-lab-reports",
                      document="Slutter and Driscoll (1963) Fritz Lab. Report 279.15, Table 1 (p. 27) and Table 4 (p. 31)")],
        screening=("All four beams are bridge-scale, solid cast-in-place slab, simply supported 37.5 ft, rolled 21WF68/24WF76, f'c 5.5-6.5 ksi, "
                   "slab 6.11-6.25 in, 6 ft slab width: inside the design space except flange F_y 34.3-35.8 ksi (web 37.9-41.8 ksi) and channel rather than stud connectors. "
                   "B21W is the partial-connection specimen (eta 0.66 with the bulletin's push-out strength, 0.50 per Slutter-Driscoll Table 4, 0.59 with the AISC channel formula); "
                   "B24W, B24S and B21S are over-connected (eta 1.4-2.3) and are included as full-composite references."),
        specimens=specimens,
        data_files=dict(digitisation_script="viest1952_digitised/digitise_viest1952.py", builder="viest1952_digitised/build_viest1952_json.py",
                        raw_markers="viest1952_digitised/markers_raw.csv", calibration="viest1952_digitised/calibration.json",
                        overlay_check="viest1952_digitised/overlay_check.png"),
        notes=[
            "Cross-check of the digitisation against Table 19: at 40 kips the flexural PL^3/48EI with Table 18 E_b I gives 0.564 (B21W), 0.552 (B21S), 0.416 (B24W) in; digitised Fig. 43 values 0.675, 0.591, 0.451 in; adding shear deformation PL/(4 G A_w) with G = 11,300 ksi and A_w = d t_w (0.044 in for 21WF68, 0.038 in for 24WF76) gives ratios 1.11, 0.99, 0.99 versus Table 19 113, 101, 102: the tabulated complete-interaction deflection includes shear deformation and the digitised points reproduce Table 19 within 1-3 %.",
            "Connector modulus premise corrected: the bulletin's k/s (Table 18) and k (Table 25) were back-calculated from measured beam slips; push-out stiffness is available only as loads at 0.003, 0.006 and 0.020 in average slip (Table 5).",
            "B21W: only four elastic load levels exist in each record (10-40 kips); first yield was measured at 47 kips and all connectors yielded in the yield test. Two independent B21W records are provided: test 1B (Fig. 53, after breaking bond) and test 5A/5B (Fig. 43).",
            "B24W and B24S curves continue to 85 and 90 kips beyond the 1.0 in sub-panel edge; those post-yield points were not digitised.",
        ],
    )
    OUT.write_text(json.dumps(data, indent=1))
    for sp in specimens:
        print(sp["id"], json.dumps(sp["spec_summary"])[:600])


if __name__ == "__main__":
    build()
