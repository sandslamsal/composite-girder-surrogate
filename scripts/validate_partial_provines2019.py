"""Measured vs predicted stiffness: Provines et al. (2019), FHWA-HRT-20-005, precast deck beams.

Specimens (data/experimental/partial_connection/provines2019.json): 1S1,
2S1, 3S1, 4S1: W27x84 (AISC handbook A = 24.7 in2, I = 2850 in4, equivalent
plates; measured F_y flange 54.0 / web 57.4 ksi), 348 in span, two loads
138 in from the bearings, 48 x 8 in solid precast deck on a 1 in grout
haunch over a greased flange, 7/8 in studs in grouted pockets: 12 per
shear span (single, or in clusters of 2, 3, 4) plus 3 per half in the
constant-moment region (drawn, not counted by the programme).

Moduli: E_s 29 000 ksi, E_c from the ACI expression at each deck's f'c
(not measured). Haunch: 1 in offset, haunch concrete not counted (it sits
at the neutral axis; < 0.1 % on I_full). Rebar: 12 No. 4 (2.40 in2) in two
mats, centroid depths approximate (3.0 and 6.5 in), beam model only.

Connector law: the programme's PC-longitudinal push-out mean backbone per
stud (Fig. 93, five curves) to 45 kips, held flat beyond, with the
programme's mean PC push-out strength Q_u = 50 kips for eta (the curve
stops before the peak).

Record: Fig. 68 applied mid-span moment (kip-ft) vs mid-span LVDT; total
load W = 2 P = 2 M / 138 in. The record zero is uncertain by 0.05-0.17 in
(stiff initial toe), so the window is 300-700 kip-ft (the JSON elastic
limit), where the intercept-free slope is offset-independent.

Outputs
-------
reports/model_validation/partial_connection/provines2019.csv
reports/model_validation/partial_connection/provines2019_summary.json
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

KEY = "provines2019"
DATA = REPO / "data/experimental/partial_connection" / f"{KEY}.json"
OUT = REPO / "reports/model_validation/partial_connection"
WINDOW_KIPFT = (300.0, 700.0)


def stud_positions(s: dict, span: float, include_cm: bool = True) -> list:
    lay = s["connectors"]["layout"]
    half = list(lay["positions_from_bearing_in_per_half"])
    if include_cm:
        half += list(lay["constant_moment_region_studs_per_half"]["positions_from_bearing_in"])
    return sorted([float(x) for x in half] + [span - float(x) for x in half])


def build_spec(s: dict, *, include_cm: bool = True, law: str = "curve") -> pc.Specimen:
    g = s["geometry"]
    hb = g["steel"]["handbook_dimensions"]
    tf, tw = pc.equivalent_plate_thicknesses(hb["d_in"], hb["bf_in"], hb["A_in2"], hb["Ix_in4"])
    m = s["materials"]
    steel = pc.i_section(hb["d_in"], hb["bf_in"], tf, tw, m["steel"]["Fy_flange_ksi"],
                         fy_web_ksi=m["steel"]["Fy_web_ksi"])
    L = g["span_in"]
    po = s["connectors"]["pushout_programme_own"]
    qu = po["recommended_model_input"]["Q_u_per_stud_kip"]
    xs = stud_positions(s, L, include_cm)
    if law == "curve":
        pts = [(p["slip_mean_in"], p["Q_per_stud_kip"])
               for p in po["PC_longitudinal_mean_backbone_per_stud"]["points"]]
        conns = [pc.Connector.from_curve(x, pts, n=1, qu_kip=qu) for x in xs]
    else:   # push-out stiffness and strength, Ollgaard-shaped law of the database
        k = po["recommended_model_input"]["k_s_per_stud_kip_per_in"]
        conns = [pc.Connector(x, k, qu, 1, "ollgaard") for x in xs]
    a_mat = 0.5 * 2.40
    rebar = [pc.RebarLayer(a_mat, 3.0, 60.0), pc.RebarLayer(a_mat, 6.5, 60.0)]
    return pc.Specimen(
        label=s["id"], span_in=L, slab_width_in=float(g["slab"]["width_in"]),
        slab_thickness_in=float(g["slab"]["thickness_in"]), fc_ksi=m["concrete"]["fc_ksi"],
        connectors=conns, load_pattern="two_point", load_offset_in=g["load_points_from_bearings_in"],
        haunch_height_in=1.0, haunch_width_in=0.0, rebar=rebar, section_type="W",
        source="Provines et al. (2019) FHWA-HRT-20-005", **steel)


def make_case(s: dict) -> pw.Case:
    meas = s["measured"]
    a = s["geometry"]["load_points_from_bearings_in"]
    mk = np.array([r["load"] for r in meas])                  # kip-ft
    W = 2.0 * mk * 12.0 / a                                   # total load, kip
    defl = np.array([r["deflection"] for r in meas])
    el = np.array([bool(r["elastic"]) for r in meas])
    win = el & (mk >= WINDOW_KIPFT[0] - 1e-9) & (mk <= WINDOW_KIPFT[1] + 1e-9)
    spec = build_spec(s)
    hb = s["geometry"]["steel"]["handbook_dimensions"]
    a_clear, a_full = pw.web_shear_areas(hb["d_in"], hb["tf_in"], hb["tw_in"])
    return pw.Case(
        key=KEY, spec=spec, loads=W, source_load=mk, source_load_unit="kip-ft (applied mid-span moment)",
        delta_meas_in=defl, elastic=el, window=win,
        window_rule=(f"JSON elastic flag and applied moment {WINDOW_KIPFT[0]:.0f}-{WINDOW_KIPFT[1]:.0f} kip-ft "
                     f"(fixed start: record zero uncertain by 0.05-0.17 in below 300 kip-ft); "
                     f"{mk[win].min():.0f}-{mk[win].max():.0f} kip-ft"),
        rho_l=s["geometry"]["slab"]["reinforcement"]["rho_l_pct"] / 100.0,
        rho_l_note="gross, 12 No. 4 / (48 x 8 in)",
        shear_area_clear_in2=a_clear, shear_area_full_in2=a_full,
        classification=dict(programme="Provines et al. (2019) FHWA-HRT-20-005", slab_type="solid precast panels",
                            connection_system="studs in grouted pockets of precast panels", scale="bridge",
                            role="partial-connection"),
        defl_uncertainty_in=0.025,
        connector_note=("PC longitudinal push-out mean backbone per stud (Fig. 93, 5 curves) to 45 kips, held beyond; "
                        "Q_u = 50 kips (mean PC longitudinal push-out, Table 16); 15 studs per half as drawn "
                        "(12 per shear span + 3 in the constant-moment region)"),
        flags=["precast full-depth panels, studs in grouted pockets through a 1 in grout haunch, greased flange "
               "(stud-in-grout connector, softer than CIP: push-out k about 1150 vs 1960 kip/in)",
               "slab width 48 in = 4 ft equivalent spacing (below the 5 ft minimum)",
               "f'c 8.5-10.0 ksi (upper end of the design space)",
               "record zero uncertain by 0.05-0.17 in (secant from zero unreliable; slope used)"],
        notes=["dead-load moment carried by the steel alone (panels set on levelling bolts, no shoring stated); "
               "M/M_p uses the applied moment",
               "3S1 window levels are nearest visible source rows (edge rule, +/-0.025 in)"
               if s["id"] == "3S1" else "window levels interpolated at fixed moments"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prog = json.loads(DATA.read_text())
    frames, summaries = [], {}
    for s in prog["specimens"]:
        if not s.get("included") or not isinstance(s.get("measured"), list):
            continue
        case = make_case(s)
        df, summ = pw.analyse(case)
        k_full = summ["K_full"]
        dc = s["connectors"]["degree_of_connection"]
        # stud count 12 vs 15 per half
        beam_cases = [
            pw.sensitivity_beam(case, "12 studs per half (constant-moment studs removed)",
                                build_spec(s, include_cm=False), k_full),
            pw.sensitivity_beam(case, "push-out k_s = 1146 kip/in, Q_u = 50 kips, Ollgaard-shaped law",
                                build_spec(s, law="ollgaard"), k_full),
        ]
        eta_15 = pc.eta_plastic(case.spec, two_point_region="half_span")
        qn_ratio = dc["Qn_AISC_kip"] / 50.0
        rei_cases = [
            pw.sensitivity_rei(case, "15 studs per half counted (half-span region), Q_u 50", eta_15),
            pw.sensitivity_rei(case, "12 per shear span, AISC Q_n = A_sc F_u = 44.2 kips (programme)",
                               summ["eta_plastic"] * qn_ratio),
            pw.sensitivity_rei(case, "15 per half, AISC Q_n 44.2 kips", eta_15 * qn_ratio),
        ]
        summ["sensitivity"] = pw.ranges(summ, beam_cases, rei_cases)
        summ["programme_tangent_stiffness"] = s.get("tangent_stiffness")
        frames.append(df)
        summaries[s["id"]] = summ
        pw.print_summary(summ)
    pd.concat(frames, ignore_index=True).to_csv(OUT / f"{KEY}.csv", index=False)
    meta = dict(generated_by=f"scripts/validate_partial_{KEY}.py",
                method="src/validation/partial_connection_windowed.py (windowed intercept-free slope)",
                data=str(DATA.relative_to(REPO)), specimens=summaries)
    (OUT / f"{KEY}_summary.json").write_text(json.dumps(pw.to_jsonable(meta), indent=2))


if __name__ == "__main__":
    main()
