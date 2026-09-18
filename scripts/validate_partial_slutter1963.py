"""Measured vs predicted stiffness: Slutter and Driscoll (1963), Lehigh AISC project 279 beams.

Specimens (data/experimental/partial_connection/slutter1963.json), first
(virgin) loading only: B3-S1, B5-S1, B6-S1, B7-S1, B9-S1 (two loads P/2 at
81 and 99 in from the left support, i.e. symmetric about mid-span, 18 in
apart) and B10, B11, B12 (five equal loads P/5 at 30, 60, 90, 120, 150 in).
12WF27 (handbook A = 7.97 in2, I = 204.1 in4, equivalent plates; measured
flange and web F_y), 180 in span, 48 x 4 in solid slab with a 6 x 6 in mesh
of 1/4 in rods at mid-depth (rho_l 0.2 %). Laboratory scale, flagged just
outside the design space (span, deck, spacing) like the Chapman beams.

Moduli: E_s 29 000 ksi, E_c from the ACI expression at f'c (not measured).

Five-point loading is not a pattern of the shared module. It is handled by
src/validation/partial_connection_windowed.MultiPointSpecimen: transformed
section deflection by superposition of the point-load closed form and the
beam model with the same point loads (the shared module's load-position and
mid-span-moment helpers are wrapped for that subclass only). The wrapper is
verified here against the closed form with near-rigid connectors.

Connector law: the programme's own digitised push-out envelopes: 1/2 in
L-studs = mean slip of P1 and P4 at equal force; 3U4.1 channel (B5) = P2;
3/4 in headed stud (B9) = P3; strength = the programme push-out ultimate.

Bond: natural bond was not prevented. For B3, B5 and B6 every elastic
reading is at or below the first-slip load ("bond intact" flag); for B7, B9,
B10, B11 and B12 the ratio over the readings above first slip is reported
separately.

Outputs
-------
reports/model_validation/partial_connection/slutter1963.csv
reports/model_validation/partial_connection/slutter1963_summary.json
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

KEY = "slutter1963"
DATA = REPO / "data/experimental/partial_connection" / f"{KEY}.json"
OUT = REPO / "reports/model_validation/partial_connection"
FIVE_POINT = (30.0, 60.0, 90.0, 120.0, 150.0)
MESH_FY_KSI = 60.0      # 1/4 in rods; yield not reported (irrelevant in the elastic range)


def envelope(env: dict, name: str) -> np.ndarray:
    return np.array([(p["slip_in"], p["load_per_connector_kip"]) for p in env[name]])


def mean_envelope(env: dict, names=("P1", "P4")) -> list:
    """Mean slip of several push-out envelopes at equal force (common force range)."""
    curves = [envelope(env, n) for n in names]
    f_lo = max(c[0, 1] for c in curves)
    f_hi = min(c[-1, 1] for c in curves)
    forces = np.unique(np.concatenate([[f_lo], np.arange(np.ceil(f_lo), np.floor(f_hi) + 1e-9, 1.0), [f_hi]]))
    slips = np.mean([np.interp(forces, c[:, 1], c[:, 0]) for c in curves], axis=0)
    return [(float(s), float(f)) for s, f in zip(slips, forces)]


def backbone(s: dict, which=None) -> tuple[list, str]:
    c = s["connectors"]
    env = c["push_out_envelopes_digitised"]
    t = c["type"]
    if "channel" in t:
        return envelope(env, "P2").tolist(), "push-out P2 (3U4.1 channel)"
    if "3/4 in headed" in t:
        return envelope(env, "P3").tolist(), "push-out P3 (3/4 in headed stud)"
    if which is not None:
        return envelope(env, which).tolist(), f"push-out {which} alone (1/2 in L-stud)"
    return mean_envelope(env), "mean of push-outs P1 and P4 at equal force (1/2 in L-stud)"


def build_spec(s: dict, *, which=None) -> pc.Specimen:
    g = s["geometry"]
    sec = g["steel_section"]
    tf, tw = pc.equivalent_plate_thicknesses(sec["d"], sec["bf"], sec["A"], sec["Ix"])
    m = s["materials"]
    steel = pc.i_section(sec["d"], sec["bf"], tf, tw, m["steel_beam"]["flange_fy_ksi"],
                         fy_web_ksi=m["steel_beam"]["web_fy_ksi"])
    c = s["connectors"]
    pts, _ = backbone(s, which)
    qu = c["strength"]["pushout_kip"]
    conns = [pc.Connector.from_curve(float(x), pts, n=int(c["per_position"]), qu_kip=qu)
             for x in c["positions_from_left_support_in"]]
    kw = dict(label=s["id"], span_in=g["span_in"], slab_width_in=g["slab"]["width_in"],
              slab_thickness_in=g["slab"]["thickness_in"], fc_ksi=m["concrete"]["fc_psi"] / 1000.0,
              connectors=conns, rebar=[pc.RebarLayer(g["slab"]["longitudinal_area_in2"],
                                                     g["slab"]["longitudinal_depth_from_top_in"], MESH_FY_KSI)],
              section_type="W", overhang_in=3.0,
              source="Slutter and Driscoll (1963); Lehigh Fritz Lab reports 279.2, 279.6, 279.15", **steel)
    if "five equal loads" in s["loading"]["pattern"]:
        return pw.MultiPointSpecimen(load_pattern="midspan", point_positions_in=FIVE_POINT, **kw)
    return pc.Specimen(load_pattern="two_point", load_offset_in=81.0, **kw)


def connection_system(s: dict) -> str:
    return ("channel connectors in solid cast-in-place slab" if "channel" in s["connectors"]["type"]
            else "welded studs in solid cast-in-place slab")


def make_case(s: dict) -> pw.Case:
    rows = s["measured"]
    load = np.array([r["load"] for r in rows])
    defl = np.array([r["deflection"] for r in rows])
    el = np.array([bool(r["elastic"]) for r in rows])
    win, p_el, _ = pw.default_window(load, el)
    spec = build_spec(s)
    sec = s["geometry"]["steel_section"]
    a_clear, a_full = pw.web_shear_areas(sec["d"], sec["tf"], sec["tw"])
    slip = s["loading"]["first_slip"]["load_kip"]
    above = el & np.array([not any("first-slip" in f for f in r["flags"]) for r in rows])
    five = isinstance(spec, pw.MultiPointSpecimen)
    flags = ["laboratory scale: span 15 ft, slab 4 in, 4 ft spacing below the design space (as Chapman)",
             "natural bond not prevented"]
    if not above.any():
        flags.append(f"bond intact: every elastic reading at or below the first-slip load ({slip} kips)")
    else:
        flags.append(f"first slip at {slip} kips: {int(above.sum())} elastic reading(s) above it")
    if five:
        flags.append("five equal point loads (L/6 spacing), modelled explicitly (MultiPointSpecimen)")
    if "channel" in s["connectors"]["type"]:
        flags.append("channel connectors")
    _, bb_note = backbone(s)
    return pw.Case(
        key=KEY, spec=spec, loads=load, source_load=load, source_load_unit="kip (total jack load P)",
        delta_meas_in=defl, elastic=el, window=win,
        window_rule=(f"first loading, JSON elastic flag (up to the cycle load) and load >= 25 % of the highest "
                     f"elastic load ({p_el:.1f} kips, 5 % step tolerance): {load[win].min():.1f}-"
                     f"{load[win].max():.1f} kips, {int(win.sum())} points"),
        rho_l=s["geometry"]["slab"]["rho_l"], shear_area_clear_in2=a_clear, shear_area_full_in2=a_full,
        classification=dict(programme="Slutter and Driscoll (1963)", slab_type="solid cast-in-place",
                            connection_system=connection_system(s), scale="laboratory", role="partial-connection"),
        rho_l_note="0.2 % stated (279.6 p. 18); mesh at slab mid-depth",
        eta_note=("shear span support to load point (81 in)" if not five else "support to mid-span (five-point)"),
        defl_uncertainty_in=rows[-1]["uncertainty"]["deflection_in"],
        connector_note=f"{s['connectors']['type']}: {bb_note}; Q_u = {s['connectors']['strength']['pushout_kip']} kips",
        flags=flags,
        notes=["applied jack load only; self-weight (about 77 kip-in) not in the record; M/M_p uses the applied moment",
               s["connectors"]["layout_source"]]), above


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prog = json.loads(DATA.read_text())
    frames, summaries, verification = [], {}, None
    for s in prog["specimens"]:
        if not s.get("included"):
            continue
        case, above = make_case(s)
        if isinstance(case.spec, pw.MultiPointSpecimen) and verification is None:
            verification = pw.verify_multipoint_rigid(case.spec, 30.0)
            verification["specimen"] = s["id"]
            print("five-point wrapper check (rigid connectors):", verification)
        df, summ = pw.analyse(case)
        above = above & (df.m_over_mp.to_numpy() <= pc.M_OVER_MP_EXTENDED + 1e-12)   # same M/M_p cap as the window
        c_full = 1.0 / summ["K_full"]
        csh_c = summ["shear"]["C_sh_clear_over_C_full"] * c_full
        csh_f = summ["shear"]["C_sh_full_over_C_full"] * c_full
        summ["above_first_slip"] = (pw.sub_window_ratios(case, above, "elastic readings above first slip",
                                                         c_full, csh_c, csh_f)
                                    if above.sum() >= 2 else
                                    dict(n_points=int(above.sum()), note="fewer than 2 readings above first slip"))
        summ["first_slip_load_kip"] = s["loading"]["first_slip"]["load_kip"]
        beam_cases = []
        if "L-stud" in s["connectors"]["type"]:
            for which in ("P1", "P4"):
                beam_cases.append(pw.sensitivity_beam(case, f"connector backbone {which} only",
                                                      build_spec(s, which=which), summ["K_full"]))
        dc = s["connectors"]["degree_of_connection"]
        rei_cases = [pw.sensitivity_rei(case, "AISC Q_n", dc["eta_aisc"]),
                     pw.sensitivity_rei(case, "Slutter-Driscoll convention", dc["eta_slutter_driscoll"])]
        summ["sensitivity"] = pw.ranges(summ, beam_cases, rei_cases)
        frames.append(df)
        summaries[s["id"]] = summ
        pw.print_summary(summ)
        a = summ["above_first_slip"]
        print(f"      above first slip: n {a.get('n_points')} R_meas {a.get('R_meas', float('nan')):.3f} "
              f"R_meas,flex {a.get('R_meas_flex', float('nan')):.3f}")
    pd.concat(frames, ignore_index=True).to_csv(OUT / f"{KEY}.csv", index=False)
    meta = dict(generated_by=f"scripts/validate_partial_{KEY}.py",
                method="src/validation/partial_connection_windowed.py (windowed intercept-free slope, flexural ratios)",
                data=str(DATA.relative_to(REPO)), five_point_wrapper_verification=verification,
                specimens=summaries)
    (OUT / f"{KEY}_summary.json").write_text(json.dumps(pw.to_jsonable(meta), indent=2))


if __name__ == "__main__":
    main()
