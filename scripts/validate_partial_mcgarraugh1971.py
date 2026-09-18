"""Measured vs predicted stiffness: McGarraugh and Baldwin (1971), lightweight-concrete composite beams.

Specimens (data/experimental/partial_connection/mcgarraugh1971.json): B2
(36 in slab, 112 pcf sanded lightweight) and B4 (72 in slab, 85 pcf
all-lightweight). W14x30 (AISC handbook A = 8.85 in2, I = 291 in4,
equivalent plates), 264 in span, two loads 78 in from the supports, 4.5 in
solid slab. MEASURED E_c (2200 and 1700 ksi) is used for both the
transformed section and the beam model, as the programme recommends for
lightweight concrete. Slab reinforcement is not stated: no rebar in the
model, rho_l = 0 for R_EI (0.7 % reported as a sensitivity).

Connector law: the programme's companion push-out curve (Fig. 7, digitised,
simplified to 8 points; for B2 the traced part starts at 0.0101 in because
the curve below about 9 kips is hidden under markers) with 2 studs per
group, and the Table 2(a) push-out ultimate (20 / 19 kips per stud) as the
strength that eta uses.

Record: Fig. 5 end reaction (= load at each point, kip) vs mid-span
deflection; total load W = 2 x reaction. Few elastic points (3 for B2, 5
for B4).

Outputs
-------
reports/model_validation/partial_connection/mcgarraugh1971.csv
reports/model_validation/partial_connection/mcgarraugh1971_summary.json
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

KEY = "mcgarraugh1971"
DIR = REPO / "data/experimental/partial_connection"
DATA = DIR / f"{KEY}.json"
OUT = REPO / "reports/model_validation/partial_connection"
WINDOW_FRACTION = 0.25
CURVE_POINTS = 8


def pushout_points(s: dict) -> np.ndarray:
    st = s["connectors"]["stiffness"]["pushout_curve"]
    df = pd.read_csv(DIR / st["curve_file"])
    df = df[df.slip_in >= st["slip_range_in"][0] - 1e-9]      # traced range stated in the JSON
    return df[["slip_in", "load_per_stud_kip"]].to_numpy()


def build_spec(s: dict, *, studs_per_group: int = None, law: str = "curve") -> pc.Specimen:
    g = s["geometry"]
    sec = g["steel_section"]
    tf, tw = pc.equivalent_plate_thicknesses(sec["d_in"], sec["bf_in"], sec["A_in2"], sec["Ix_in4"])
    m = s["materials"]
    steel = pc.i_section(sec["d_in"], sec["bf_in"], tf, tw, m["steel"]["fy_static_ksi"])
    c = s["connectors"]
    n = c["studs_per_group"] if studs_per_group is None else studs_per_group
    L = g["span_in"]
    xs = sorted([float(x) for x in c["group_positions_from_support_in"]]
                + [L - float(x) for x in c["group_positions_from_support_in"]])
    qu = c["strength"]["pushout_ultimate_per_stud_kip"]
    if law == "curve":
        conns = [pc.Connector.from_curve(x, pushout_points(s), n=n, qu_kip=qu, max_points=CURVE_POINTS)
                 for x in xs]
    else:
        k = c["stiffness"]["pushout_curve"]["secant_stiffness_per_stud"]["0.5 x Table 2a ultimate"]["secant_k_kip_per_in"]
        conns = [pc.Connector(x, k, qu, n, "ollgaard") for x in xs]
    return pc.Specimen(
        label=s["id"], span_in=L, slab_width_in=g["slab"]["width_in"],
        slab_thickness_in=g["slab"]["thickness_in"], fc_ksi=m["concrete"]["fc_ksi"],
        ec_ksi=m["concrete"]["Ec_measured_ksi"], connectors=conns, load_pattern="two_point",
        load_offset_in=78.0, section_type="W", source="McGarraugh and Baldwin (1971)", **steel)


def make_case(s: dict) -> pw.Case:
    meas = s["measured"]
    react = np.array([r["load"] for r in meas])
    defl = np.array([r["deflection"] for r in meas])
    el = np.array([bool(r["elastic"]) for r in meas])
    win, p_el, _thr = pw.default_window(react, el)
    spec = build_spec(s)
    conc = s["materials"]["concrete"]
    sec = s["geometry"]["steel_section"]
    a_clear, a_full = pw.web_shear_areas(sec["d_in"], sec["tf_in"], sec["tw_in"])
    return pw.Case(
        key=KEY, spec=spec, loads=2.0 * react, source_load=react, source_load_unit="kip (end reaction = each point load)",
        delta_meas_in=defl, elastic=el, window=win,
        window_rule=(f"JSON elastic flag and reaction >= {WINDOW_FRACTION:.0%} of the highest elastic reaction (5 % step tolerance) "
                     f"({p_el:.2f} kips): {react[win].min():.2f}-{react[win].max():.2f} kips, {int(win.sum())} points"),
        rho_l=0.0, rho_l_note="not stated in the source; 0 used, 0.7 % as sensitivity",
        shear_area_clear_in2=a_clear, shear_area_full_in2=a_full,
        classification=dict(programme="McGarraugh and Baldwin (1971)", slab_type="solid cast-in-place lightweight",
                            connection_system="headed studs in cast-in-place lightweight slab", scale="laboratory",
                            role="partial-connection"),
        defl_uncertainty_in=s["digitisation"]["uncertainty"]["deflection_in"],
        connector_note=(f"companion push-out curve (Fig. 7, {CURVE_POINTS}-point simplification), 2 studs per group "
                        f"(inferred), Q_u = Table 2(a) {s['connectors']['strength']['pushout_ultimate_per_stud_kip']} kips"),
        flags=[f"LIGHTWEIGHT concrete ({conc['unit_weight_pcf']} pcf), measured E_c {conc['Ec_measured_ksi']:.0f} ksi "
               f"(n = {29000.0 / conc['Ec_measured_ksi']:.1f}); R_EI table derived at normal-weight n",
               "few elastic points; +/-0.02 in reading uncertainty",
               "studs per group (2) inferred from the stated ~50 % connection; rho_l unknown",
               "steel depth 13.8 in and slab 4.5 in near the lower bounds"]
              + (["slab width 36 in = 3 ft equivalent spacing (below the 5 ft minimum)"] if s["id"] == "B2" else []),
        notes=["authors' Fig. 5 reference lines do not match an elastic calculation (JSON check_warning); not used",
               "shoring not stated; applied-load record starts at the origin"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prog = json.loads(DATA.read_text())
    frames, summaries = [], {}
    for s in prog["specimens"]:
        if not s.get("included"):
            continue
        case = make_case(s)
        df, summ = pw.analyse(case)
        k_full = summ["K_full"]
        beam_cases = [pw.sensitivity_beam(case, "push-out secant at 0.5 Q_u, Ollgaard-shaped law",
                                          build_spec(s, law="ollgaard"), k_full)]
        alt = s["degree_of_connection"]["eta_plastic_alternatives"]
        rei_cases = [pw.sensitivity_rei(case, "rho_l = 0.7 %", summ["eta_plastic"], rho=0.007)]
        for label, eta in alt.items():
            rei_cases.append(pw.sensitivity_rei(case, label, eta))
        # window without the lowest point (stiff toe / bond at the first reading)
        w_idx = np.nonzero(case.window)[0]
        if w_idx.size > 2:
            win2 = case.window.copy()
            win2[w_idx[0]] = False
            fit2 = pw.slope_intercept_free(case.loads[win2], case.delta_meas_in[win2])
            summ["K_meas"]["slope_without_lowest_window_point"] = fit2["K"]
            summ["R_meas_without_lowest_window_point"] = fit2["K"] / k_full
            u = pw.unit_compliances(case.spec, case.shear_area_clear_in2, case.shear_area_full_in2)
            summ["R_meas_flex_without_lowest_window_point"] = pw.flexural_ratio(fit2["K"], u["full"], u["shear_clear"])
        summ["sensitivity"] = pw.ranges(summ, beam_cases, rei_cases)
        frames.append(df)
        summaries[s["id"]] = summ
        pw.print_summary(summ)
        print(f"      R_meas,flex without the lowest window point: {summ.get('R_meas_flex_without_lowest_window_point', float('nan')):.3f}")
    pd.concat(frames, ignore_index=True).to_csv(OUT / f"{KEY}.csv", index=False)
    meta = dict(generated_by=f"scripts/validate_partial_{KEY}.py",
                method="src/validation/partial_connection_windowed.py (windowed intercept-free slope)",
                data=str(DATA.relative_to(REPO)), specimens=summaries)
    (OUT / f"{KEY}_summary.json").write_text(json.dumps(pw.to_jsonable(meta), indent=2))


if __name__ == "__main__":
    main()
