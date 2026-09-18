"""Measured vs predicted stiffness: Kwon et al. (2007) / Kwon (2008), post-installed connectors.

Specimens (data/experimental/partial_connection/kwon2007.json): DBLNB-30BS,
HASAA-30BS, HTFGB-30BS (flagged: friction-grip), HASAA-30BS1 (flagged:
connectors concentrated near the supports), and NON-00BS as the eta = 0
reference, checked against the bare-steel stiffness only.

W30x99 (AISC handbook A = 29.1 in2, I = 3990 in4, reproduced by equivalent
plates), 456 in span, single mid-span load, 84 x 7 in solid slab, two mats
of #4 at 12 in (2.80 in2, bar depths scaled from Fig. 4.1). E_c from the
ACI expression at the test-day f'c (not measured). Connector law: the
programme's mean direct-shear load-slip backbone per connector type (pairs
per row), with the programme's mean Q_u as the strength that eta uses.

Method: src/validation/partial_connection_windowed.py. Window: points
flagged elastic in the JSON with load >= 25 % of the highest elastic load
(the low readings carry the zero-reading shift and a +/-0.01 in reading
uncertainty that is > 20 % of the deflection).

Outputs
-------
reports/model_validation/partial_connection/kwon2007.csv
reports/model_validation/partial_connection/kwon2007_summary.json

Usage
-----
/opt/anaconda3/envs/ops_x86/bin/python scripts/validate_partial_kwon2007.py
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.validation import partial_connection as pc  # noqa: E402
from src.validation import partial_connection_windowed as pw  # noqa: E402

KEY = "kwon2007"
DATA = REPO / "data/experimental/partial_connection" / f"{KEY}.json"
OUT = REPO / "reports/model_validation/partial_connection"
WINDOW_FRACTION = 0.25


def build_spec(s: dict, prog: dict, rows_x=None) -> pc.Specimen:
    g = s["geometry"]
    sec = g["steel_section"]
    tf, tw = pc.equivalent_plate_thicknesses(sec["d_in"], sec["bf_in"], sec["A_in2"], sec["Ix_in4"])
    m = s["materials"]
    steel = pc.i_section(sec["d_in"], sec["bf_in"], tf, tw, m["steel_beam"]["flange_Fy_ksi"],
                         fy_web_ksi=m["steel_beam"]["web_Fy_ksi"])
    a_mat = 0.5 * g["reinforcement"]["A_long_total_in2"]
    rebar = [pc.RebarLayer(a_mat, 2.4, m["rebar_no4"]["Fy_ksi"]),      # top mat, Fig. 4.1 scaled
             pc.RebarLayer(a_mat, 7.0 - 2.0, m["rebar_no4"]["Fy_ksi"])]  # bottom mat
    conns = []
    ctype = s["connectors"]["type"]
    if ctype in prog["connector_direct_shear_tests"]["per_type"]:
        t = prog["connector_direct_shear_tests"]["per_type"][ctype]
        lay = s["connectors"]["layout"]
        xs = lay["row_x_in_from_left_support"] if rows_x is None else rows_x
        for x in xs:
            conns.append(pc.Connector.from_curve(float(x), t["mean_backbone_points_slip_in_force_kips"],
                                                 n=int(lay["per_row"]), qu_kip=t["Qu_mean_kips"]))
    return pc.Specimen(
        label=s["id"], span_in=g["span_in"], slab_width_in=g["slab"]["width_in"],
        slab_thickness_in=g["slab"]["thickness_in"], fc_ksi=m["concrete"]["fc_test_day_ksi"],
        connectors=conns, load_pattern="midspan", section_type="W", rebar=rebar,
        source="Kwon (2008) dissertation; Kwon et al. (2007) TxDOT 0-4124-1", **steel)


def merge_rows(xs) -> list:
    """Mirror-symmetric rows may coincide at mid-span; keep them as
    separate positions (the beam model lumps same-position connectors)."""
    return sorted(float(x) for x in xs)


def uniform_rows(end_distance: float, pitch: float = 28.5, per_half: int = 8, span: float = 456.0) -> list:
    half = [end_distance + i * pitch for i in range(per_half)]
    return merge_rows(half + [span - x for x in half])


def near_support_rows(first: float, pitch: float = 12.0, per_end: int = 8, span: float = 456.0) -> list:
    half = [first + i * pitch for i in range(per_end)]
    return merge_rows(half + [span - x for x in half])


def classification(ident: str) -> dict:
    friction = ident.startswith("HTFGB")
    return dict(programme="Kwon et al. (2007); Kwon (2008)", slab_type="solid cast-in-place",
                connection_system=("friction-grip bolts" if friction else
                                   "post-installed rods in solid cast-in-place slab"),
                scale="bridge", role="partial-connection")


def make_case(s: dict, prog: dict) -> pw.Case:
    meas = s["measured"]
    load = np.array([r["load_kips"] for r in meas])
    defl = np.array([r["deflection_in"] for r in meas])
    el = np.array([bool(r["elastic"]) for r in meas])
    win, p_el, _thr = pw.default_window(load, el)
    spec = build_spec(s, prog)
    sec = s["geometry"]["steel_section"]
    a_clear, a_full = pw.web_shear_areas(sec["d_in"], sec["tf_in"], sec["tw_in"])
    flags = ["post-installed connectors (7/8 in rods/bolts in oversized holes, pretension or torque): "
             "friction and bond at low load", "mid-span point load (database: its own load pattern)"]
    ident = s["id"]
    if ident.startswith("HTFGB"):
        flags.append("friction-grip bolts: interface friction-locked up to the 55 kip limit "
                     "(slip < 0.001 in at 50 kips); tests near-full interaction; eta 0.32-0.51 by definition")
    if ident == "HASAA-30BS1":
        flags.append("connectors concentrated near the supports (12 in pitch, first-row distance not stated); "
                     "R_EI table assumes uniform connection; C_f governed by concrete (f'c 3.22 ksi)")
    if ident == "DBLNB-30BS":
        flags.append("load-relaxation reading at 46.1 kips excluded (elastic=false in JSON)")
    ctype = s["connectors"]["type"]
    return pw.Case(
        key=KEY, spec=spec, loads=load, source_load=load, source_load_unit="kip (mid-span load)",
        delta_meas_in=defl, elastic=el, window=win,
        window_rule=(f"JSON elastic flag and load >= {WINDOW_FRACTION:.0%} of the highest elastic load (5 % step tolerance) "
                     f"({p_el:.1f} kips): {load[win].min():.1f}-{load[win].max():.1f} kips"),
        rho_l=s["geometry"]["reinforcement"]["A_long_total_in2"] / (84.0 * 7.0),
        shear_area_clear_in2=a_clear, shear_area_full_in2=a_full, classification=classification(ident),
        rho_l_note="gross, both mats / (84 x 7 in)",
        defl_uncertainty_in=s["digitisation"]["uncertainty"]["deflection_in"],
        connector_note=(f"{ctype}: programme mean direct-shear backbone (Kwon 2008 Figs. 3.15-3.17), "
                        f"pairs at each row, Q_u = programme mean "
                        f"{prog['connector_direct_shear_tests']['per_type'][ctype]['Qu_mean_kips']} kips"),
        flags=flags,
        notes=["dead-load moment 124.7 kip-ft acts on the steel alone (unshored, connectors post-installed); "
               "M/M_p uses the applied moment on the composite section",
               "10 x 20 in loading plate treated as a point load (< 0.1 % on deflection)",
               "rebar included in the beam model (fibre layers) but not in K_full (AASHTO/database convention)",
               "shear areas from the W30x99 handbook d, t_f, t_w"])


def non_composite_check(s: dict, prog: dict):
    meas = s["measured"]
    load = np.array([r["load_kips"] for r in meas])
    defl = np.array([r["deflection_in"] for r in meas])
    el = np.array([bool(r["elastic"]) for r in meas])
    bond = np.array([r["interface_state"] == "bond_intact" for r in meas])
    spec = build_spec(s, prog)
    sec = s["geometry"]["steel_section"]
    a_clear, a_full = pw.web_shear_areas(sec["d_in"], sec["tf_in"], sec["tw_in"])
    u = pw.unit_compliances(spec, a_clear, a_full)
    win = el & ~bond
    fit = pw.slope_intercept_free(load[win], defl[win])
    lo = el & bond & (load >= 10.0)
    fit_bond = pw.slope_intercept_free(load[lo], defl[lo])
    k_steel = 1.0 / u["steel"]
    theory = prog["programme_theory_lines_report_fig5_20"]["noncomposite_bare_steel_kips_per_in"]
    r_flex = pw.flexural_ratio(fit["K"], u["steel"], u["shear_clear"])
    r_flex_full = pw.flexural_ratio(fit["K"], u["steel"], u["shear_full"])
    rows = pd.DataFrame(dict(specimen=s["id"], load=load, load_unit="kip (total applied, module convention)",
                             source_load=load, source_load_unit="kip (mid-span load)",
                             m_over_mp=np.nan, regime="", delta_meas_in=defl, delta_full_in=load * u["full"],
                             delta_steel_in=load * u["steel"], delta_noncomposite_in=load * u["noncomposite"],
                             delta_shear_clear_web_in=load * u["shear_clear"],
                             delta_shear_full_web_in=load * u["shear_full"],
                             delta_beam_in=np.nan, delta_beam_E0_in=np.nan, R_EI_at_load=np.nan,
                             elastic=el, in_window=win))
    return rows, dict(
        specimen=s["id"],
        classification=dict(programme="Kwon et al. (2007); Kwon (2008)", slab_type="solid cast-in-place",
                            connection_system="none (four safety studs at mid-span)", scale="bridge",
                            role="eta = 0 reference (bare-steel check)"),
        window=dict(rule="elastic readings after bond break (interface_state bond_broken), "
                         f"{load[win].min():.1f}-{load[win].max():.1f} kips", n_points=int(win.sum())),
        K_meas=dict(slope=fit["K"], slope_standard_error=fit["se"],
                    implied_deflection_zero_offset_in=fit["defl_intercept_in"],
                    secant_from_zero=pw.secant_through_origin(load[win], defl[win])),
        K_meas_bond_intact_10_to_40_kips=fit_bond["K"],
        K_steel=k_steel, K_noncomposite=1.0 / u["noncomposite"],
        authors_bare_steel_line_kips_per_in=theory,
        ratio_K_meas_over_K_steel=fit["K"] / k_steel,
        flexural_ratio_steel=r_flex, flexural_ratio_steel_full_depth_web=r_flex_full,
        flexural_ratio_note="C_steel,flex / (1/K_meas - C_sh): 1.0 means the bare-steel flexural stiffness is recovered",
        C_sh_clear_over_C_steel=u["shear_clear"] / u["steel"],
        I_steel_in4=u["ts"].I_steel_in4,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prog = json.loads(DATA.read_text())
    frames, summaries = [], {}
    for s in prog["specimens"]:
        if s["id"] == prog["baseline_reference_specimen"]:
            df, summ = non_composite_check(s, prog)
            frames.append(df)
            summaries[s["id"]] = summ
            print(f"{s['id']}: K_meas {summ['K_meas']['slope']:.1f}  K_steel {summ['K_steel']:.1f}  "
                  f"ratio {summ['ratio_K_meas_over_K_steel']:.3f}  flexural ratio {summ['flexural_ratio_steel']:.3f} "
                  f"(full-depth web {summ['flexural_ratio_steel_full_depth_web']:.3f})")
            continue
        if not s["included"]:
            continue
        case = make_case(s, prog)
        df, summ = pw.analyse(case)
        k_full = summ["K_full"]
        beam_cases, rei_cases = [], []
        ident = s["id"]
        if ident in ("DBLNB-30BS", "HASAA-30BS", "HTFGB-30BS"):
            for e in (0.0, 28.5):
                sp = build_spec(s, prog, rows_x=uniform_rows(e))
                beam_cases.append(pw.sensitivity_beam(case, f"end distance {e} in (8 pairs per half at 28.5 in)",
                                                      sp, k_full))
        if ident == "HASAA-30BS1":
            for first in (6.0, 0.0):
                sp = build_spec(s, prog, rows_x=near_support_rows(first))
                beam_cases.append(pw.sensitivity_beam(
                    case, f"first row {first} in from support (rows {first:g}-{first + 84:g} in)", sp, k_full))
        if ident == "HTFGB-30BS":
            for o in s["connectors"]["degree_of_connection"]["options"]:
                rei_cases.append(pw.sensitivity_rei(case, f"eta {o['eta']}: {o['label']}", o["eta"]))
        # connector toe: replace the unresolved first segments by the
        # programme's secant at 10 kips (JSON mean_backbone_note)
        t = prog["connector_direct_shear_tests"]["per_type"][s["connectors"]["type"]]
        k10 = t["secant_stiffness_kips_per_in"]["at_10_kips_mean"]
        conns = [pc.Connector.from_curve(c.x_in, t["mean_backbone_points_slip_in_force_kips"], n=c.n,
                                         qu_kip=t["Qu_mean_kips"], k_initial_kip_per_in=k10)
                 for c in case.spec.connectors]
        beam_cases.append(pw.sensitivity_beam(
            case, f"toe replaced by the programme secant at 10 kips ({k10:g} kip/in per connector)",
            dataclasses.replace(case.spec, connectors=conns), k_full))
        summ["sensitivity"] = pw.ranges(summ, beam_cases, rei_cases)
        frames.append(df)
        summaries[ident] = summ
        pw.print_summary(summ)
    pd.concat(frames, ignore_index=True).to_csv(OUT / f"{KEY}.csv", index=False)
    meta = dict(generated_by=f"scripts/validate_partial_{KEY}.py",
                method="src/validation/partial_connection_windowed.py (windowed intercept-free slope, flexural ratios)",
                data=str(DATA.relative_to(REPO)), specimens=summaries)
    (OUT / f"{KEY}_summary.json").write_text(json.dumps(pw.to_jsonable(meta), indent=2))


if __name__ == "__main__":
    main()
