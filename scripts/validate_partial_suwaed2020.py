"""Measured vs predicted stiffness: Suwaed (2017) thesis / Suwaed and Karavasilis (2020), FBSC beam.

Specimen (data/experimental/partial_connection/suwaed2020.json): one
full-scale beam, UB457x191x89 (S355; A = 11 377 mm2 and I = 41 015 cm4
with root fillets, equivalent plates), 8500 mm span, two loads 2750 mm
from the supports, 1250 x 150 mm solid precast slab (two panels plus a
0.9 m in-situ stitch), 30 friction-based demountable connectors (M16 8.8
bolts, pairs in 15 rows at 600 mm, first row 50 mm inside each support).

Moduli: database convention (E_s 29 000 ksi, E_c from the ACI expression at
f'c = 52 MPa, cylinder peak of the thesis's own slab cylinders); measured
E_s 210.1 GPa and E_c 31 GPa are reported as a sensitivity of K_full.
Rebar: 679 mm2 at 25 mm above the soffit and 471 mm2 at about 17 mm below
the top (beam model only), rho_l = 0.61 %.

Connector law: the programme's own push-out law, P = 97 S kN (S <= 1 mm,
secant used by the author for the beam) and Eq. 6.1 for S >= 1 mm (fitted
to push-out Tests 5, 6, 11), per bolt, to the 15.7 mm mean slip capacity;
Q_u = 190 kN (mean push-out resistance) for eta.

Record: Fig. 7.3 cycle-1 loading branch, jack load P (kN, total of the
two point loads) vs mid-span deflection with the 2.07 mm self-weight
deflection subtracted. The beam was propped during construction, so the
57 kN.m self-weight and spreader moment act on the composite section and
are added to M for M/M_p only.

Outputs
-------
reports/model_validation/partial_connection/suwaed2020.csv
reports/model_validation/partial_connection/suwaed2020_summary.json
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

KEY = "suwaed2020"
DATA = REPO / "data/experimental/partial_connection" / f"{KEY}.json"
OUT = REPO / "reports/model_validation/partial_connection"
WINDOW_FRACTION = 0.25
MM = pc.IN_PER_MM
KN = pc.KIP_PER_KN


def eq_6_1(s_mm: float) -> float:
    """Thesis Eq. 6.1, push-out load per bolt (kN) for slip S >= 1 mm (p. 132)."""
    return -1.04e-2 * s_mm ** 4 + 3.88e-1 * s_mm ** 3 - 4.93 * s_mm ** 2 + 29.5 * s_mm + 72.0


def programme_law_points(toe: str = "secant") -> list:
    """(slip mm, force kN) per bolt. ``secant``: 97 S to 1 mm then Eq. 6.1.
    ``test6``: idealised Test 6 curve of Fig. 7.19 (P/Pu = 2.1 S to 0.13 mm,
    0.22 S + 0.24 to 1 mm, P_u = 206 kN), then Eq. 6.1."""
    tail = [(s, eq_6_1(s)) for s in (2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 15.7)]
    if toe == "secant":
        head = [(1.0, 97.0)]
    else:
        head = [(0.13, 206.0 * 2.1 * 0.13), (1.0, 206.0 * (0.22 * 1.0 + 0.24))]
    return head + tail          # Eq. 6.1 rises monotonically to 189.6 kN at 15.7 mm


def build_spec(s: dict, *, toe: str = "secant", es_ksi: float = pc.E_STEEL_KSI,
               ec_ksi=None) -> pc.Specimen:
    g = s["geometry"]
    sec = g["steel_section"]
    d, b = sec["h_in"], sec["b_in"]
    A = sec["area_mm2_with_root_fillets_computed"] * MM ** 2
    I = sec["Iy_cm4_computed_with_fillets"] * 1e4 * MM ** 4
    tf, tw = pc.equivalent_plate_thicknesses(d, b, A, I)
    fy = s["materials"]["steel_beam"]["fy_ksi"]
    steel = pc.i_section(d, b, tf, tw, fy)
    L = g["span_mm"] * MM
    rows_mm = [50.0 + 600.0 * i for i in range(15)]
    pts = programme_law_points(toe)
    conns = [pc.Connector.from_curve(x * MM, pts, n=2, qu_kip=190.0 * KN, slip_scale=MM, force_scale=KN)
             for x in rows_mm]
    fyr = 500.0 * pc.KSI_PER_MPA
    t = g["slab"]["thickness_mm"]
    rebar = [pc.RebarLayer(679.0 * MM ** 2, (t - 25.0) * MM, fyr), pc.RebarLayer(471.0 * MM ** 2, 17.0 * MM, fyr)]
    return pc.Specimen(
        label="Suwaed-FBSC", span_in=L, slab_width_in=g["slab"]["width_mm"] * MM,
        slab_thickness_in=t * MM, fc_ksi=s["materials"]["concrete_slab_panels"]["fc_prime_ksi_adopted"],
        connectors=conns, load_pattern="two_point", load_offset_in=2750.0 * MM, section_type="W",
        rebar=rebar, es_ksi=es_ksi, ec_ksi=ec_ksi,
        source="Suwaed (2017) PhD thesis, Warwick; Suwaed and Karavasilis (2020) JCSR 171:106152", **steel)


def make_case(s: dict) -> pw.Case:
    meas = [r for r in s["measured"] if "moment_jack_kNm" in r]     # digitised Fig. 7.3 levels only
    P = np.array([r["load"] for r in meas])                          # kN
    defl = np.array([r["deflection"] for r in meas]) * MM
    el = np.array([bool(r["elastic"]) for r in meas])
    win, p_el, _thr = pw.default_window(P, el)
    spec = build_spec(s)
    sec = s["geometry"]["steel_section"]
    a_clear, a_full = pw.web_shear_areas(sec["h_in"], sec["tf_in"], sec["tw_in"])
    return pw.Case(
        key=KEY, spec=spec, loads=P * KN, source_load=P, source_load_unit="kN (jack load, sum of both points)",
        delta_meas_in=defl, elastic=el, window=win,
        window_rule=(f"JSON elastic flag and jack load >= {WINDOW_FRACTION:.0%} of the highest elastic load (5 % step tolerance) "
                     f"({p_el:.0f} kN): {P[win].min():.0f}-{P[win].max():.0f} kN"),
        rho_l=s["geometry"]["slab"]["reinforcement"]["rho_l_total_percent"] / 100.0,
        rho_l_note="gross, 6 x 12 mm + 6 x 10 mm (inferred) / (1250 x 150 mm)",
        shear_area_clear_in2=a_clear, shear_area_full_in2=a_full,
        classification=dict(programme="Suwaed (2017); Suwaed and Karavasilis (2020)", slab_type="solid precast panels",
                            connection_system="friction-grip bolts", scale="bridge", role="partial-connection"),
        moment_offset_kip_in=57.0 * KN * 1000.0 * MM,
        moment_offset_note="57 kN.m self-weight and spreader moment on the composite section (propped construction)",
        defl_uncertainty_in=0.35 * MM,
        connector_note=("programme push-out law per bolt: 97 kN/mm secant to 1 mm, Eq. 6.1 beyond (Tests 5, 6, 11); "
                        "pairs in 15 rows; Q_u = 190 kN"),
        flags=["friction-based demountable bolt connector (FBSC), pretensioned: friction phase 230-900 kN/mm to "
               "55-70 kN per bolt; no measurable end slip to about 500 kN.m",
               "precast panels with in-situ stitch; slab not bonded to the flange",
               "slab width 1250 mm = 4.1 ft equivalent spacing (below the 5 ft minimum)",
               "eta 0.32-0.70 depending on definition (0.47 AISC shear-span count with mean push-out Q_u)",
               "deflection transducer referenced to the loading rig column (support bedding included)",
               "single specimen"],
        notes=["digitised levels only; the rounded text points are cross-checks (not used)",
               "window 300-445 kN carries the softening that starts near 470 kN.m total moment"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prog = json.loads(DATA.read_text())
    s = prog["specimens"][0]
    case = make_case(s)
    df, summ = pw.analyse(case)
    k_full = summ["K_full"]
    beam_cases = [pw.sensitivity_beam(case, "Test 6 idealised trilinear toe (433 kN/mm to 56 kN, Fig. 7.19) then Eq. 6.1",
                                      build_spec(s, toe="test6"), k_full)]
    opts = s["connectors"]["degree_of_connection"]["plastic_sumQn_over_Cf_options"]
    rei_cases = [pw.sensitivity_rei(case, f"eta option {k}", v) for k, v in opts.items()]
    rei_cases.append(pw.sensitivity_rei(case, "thesis eta (15 bolts per half x 190 kN)",
                                        s["connectors"]["degree_of_connection"]["thesis_value"]["eta"]))
    summ["sensitivity"] = pw.ranges(summ, beam_cases, rei_cases)
    # moduli sensitivity of K_full (not part of the database convention)
    es_m = s["materials"]["steel_beam"]["Es_ksi"]
    ec_m = s["materials"]["concrete_slab_panels"]["Ec_ksi"]
    alt = {}
    for label, sp in (("measured E_c 31 GPa, E_s 29 000 ksi", build_spec(s, ec_ksi=ec_m)),
                      ("measured E_s 210.1 GPa and E_c 31 GPa", build_spec(s, es_ksi=es_m, ec_ksi=ec_m))):
        kf = 1.0 / pw.unit_compliances(sp, case.shear_area_clear_in2, case.shear_area_full_in2)["full"]
        uu = pw.unit_compliances(sp, case.shear_area_clear_in2, case.shear_area_full_in2)
        alt[label] = dict(K_full=kf, R_meas=summ["K_meas"]["slope"] / kf,
                          R_meas_flex=pw.flexural_ratio(summ["K_meas"]["slope"], uu["full"], uu["shear_clear"]))
    summ["K_full_moduli_sensitivity"] = alt
    summaries = {summ["specimen"]: summ}
    pw.print_summary(summ)
    print("   moduli:", alt)
    df.to_csv(OUT / f"{KEY}.csv", index=False)
    meta = dict(generated_by=f"scripts/validate_partial_{KEY}.py",
                method="src/validation/partial_connection_windowed.py (windowed intercept-free slope)",
                data=str(DATA.relative_to(REPO)), specimens=summaries)
    (OUT / f"{KEY}_summary.json").write_text(json.dumps(pw.to_jsonable(meta), indent=2))


if __name__ == "__main__":
    main()
