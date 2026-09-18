"""Measured vs predicted stiffness: Viest, Siess, Appleton and Newmark (1952), channel-connector T-beams.

Specimens (data/experimental/partial_connection/viest1952.json): B21W, the
partial-connection beam (21WF68, 6 C4x5.4 x 4 in channels per half span at
36 in, eta_plastic 0.657 on the bulletin's push-out strength), and the
near-full references B24W, B24S (24WF76) and B21S (21WF68), eta 1.7-2.3,
reported like the Chapman anchor and kept out of the partial-connection
statistics. 450 in span, single mid-span load on a 14 in plate, 72 in
solid cast-in-place slab (6.11-6.25 in).

Section: equivalent plates reproducing the handbook A and I; measured
flange and web F_y (flange 1-5 % below 36 ksi, flagged). E_s 29 000 ksi and
E_c from the ACI expression at the cylinder f'c (database convention; the
bulletin's measured E_c and E_b are not used). Rebar: the JSON bar areas and
assumed depths (beam model only).

Connector law: the bulletin's push-out loads at 0.003, 0.006 and 0.020 in
average slip (Table 5) for the matching channel, held beyond 0.020 in, with
the push-out ultimate as the strength for eta. For B21W the only 4 in
channel push-out (4C3W2) had natural bond intact: its stiffness is an upper
bound, and the stated range k = 3.28e3 to 4.93e3 kip/in (per channel) is run
as a sensitivity by replacing the toe with that initial stiffness.

eta counting: connectors between a support and mid-span, excluding the
channel on the mid-span line (it carries no shear under the symmetric
mid-span load); counting it is an alternative.

B21W records: Fig. 53 test 1B after bond was broken by eleven repetitions
(PRIMARY, the bond-free condition the model represents), Fig. 43 test 5A/5B
(secondary) and Fig. 53 test 1A with bond intact (context).

Outputs
-------
reports/model_validation/partial_connection/viest1952.csv
reports/model_validation/partial_connection/viest1952_summary.json
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

KEY = "viest1952"
DATA = REPO / "data/experimental/partial_connection" / f"{KEY}.json"
OUT = REPO / "reports/model_validation/partial_connection"
PARTIAL = "B21W"
REBAR_FY_KSI = 40.0     # intermediate-grade bars; yield not reported (irrelevant in the elastic range)


def pushout_points(s: dict) -> list:
    po = s["connectors"]["push_out_same_programme"]["load_per_connector_kip_at_average_slip"]
    pts = [(float(k.split()[0]), v) for k, v in po.items() if v is not None]
    return sorted(pts)


def build_spec(s: dict, *, k_initial=None, drop_midspan: bool = False) -> pc.Specimen:
    g = s["geometry"]
    sec = g["steel_section"]
    tf, tw = pc.equivalent_plate_thicknesses(sec["d"], sec["bf"], sec["A"], sec["Ix"])
    m = s["materials"]
    steel = pc.i_section(sec["d"], sec["bf"], tf, tw, m["steel_beam"]["flange_fy_ksi"],
                         fy_web_ksi=m["steel_beam"]["web_fy_ksi"])
    L = g["span_in"]
    c = s["connectors"]
    qu = c["strength_for_eta"]["value_kip"]
    xs = [float(x) for x in c["positions_from_left_support_in"]]
    if drop_midspan:
        xs = [x for x in xs if abs(x - 0.5 * L) > 1e-6]
    conns = [pc.Connector.from_curve(x, pushout_points(s), n=1, qu_kip=qu, k_initial_kip_per_in=k_initial)
             for x in xs]
    lr = g["slab"]["longitudinal_reinforcement"]
    dep = lr["centroid_depth_assumed_in"]
    rebar = [pc.RebarLayer(lr["area_top_in2"], dep["top"], REBAR_FY_KSI),
             pc.RebarLayer(lr["area_bottom_in2"], dep["bottom"], REBAR_FY_KSI)]
    return pc.Specimen(
        label=s["id"], span_in=L, slab_width_in=g["slab"]["width_in"], slab_thickness_in=g["slab"]["thickness_in"],
        fc_ksi=m["concrete"]["fc_psi"] / 1000.0, connectors=conns, load_pattern="midspan",
        bearing_length_in=14.0, rebar=rebar, section_type="W",
        source="Viest, Siess, Appleton and Newmark (1952), Univ. Illinois Eng. Exp. Station Bulletin 405", **steel)


def records(s: dict) -> dict:
    """Measured series by name (B21W has two, plus the bond-intact context)."""
    out = {}
    for r in s["measured"]:
        name = "fig53_after_bond_break" if "Fig. 53" in r["series"] else "fig43"
        out.setdefault(name, []).append(r)
    for r in s.get("context_measured") or []:
        out.setdefault("fig53_bond_intact_context", []).append(r)
    return out


def make_case(s: dict, rows: list, *, role: str, series_label: str) -> pw.Case:
    load = np.array([r["load"] for r in rows])
    defl = np.array([r["deflection"] for r in rows])
    el = np.array([bool(r["elastic"]) for r in rows])
    win, p_el, _ = pw.default_window(load, el)
    spec = build_spec(s)
    sec = s["geometry"]["steel_section"]
    a_clear, a_full = pw.web_shear_areas(sec["d"], sec["tf"], sec["tw"])
    m = s["materials"]["steel_beam"]
    shored = s["geometry"]["shoring"].startswith("shored")
    w_dl = s["geometry"]["table18_section_properties"]["weight_lb_ft"] / 1000.0 / 12.0   # kip/in
    offset = w_dl * spec.span_in ** 2 / 8.0 if shored else 0.0
    dc = s["connectors"]["degree_of_connection"]
    flags = [f"flange F_y {m['flange_fy_ksi']} ksi below the 36 ksi floor by "
             f"{(36.0 - m['flange_fy_ksi']) / 36.0:.0%}",
             "channel connectors (C4x5.4), not studs; push-out load-slip at three slips only",
             "repeated elastic loading before the plotted test; 14 in load plate"]
    if s["id"] == PARTIAL:
        flags += ["connector push-out 4C3W2 had natural bond intact: model connector stiffness is an upper bound",
                  "primary record Fig. 53 test 1B after bond was broken by 11 repetitions of 40 kips; "
                  "only 4 elastic load levels (10-40 kips); first yield measured at 47 kips"]
    else:
        flags += [f"near-full reference: eta_plastic {dc['eta_pushout']} > 1 (over-connected)"]
    return pw.Case(
        key=KEY, spec=spec, loads=load, source_load=load, source_load_unit="kip (mid-span load)",
        delta_meas_in=defl, elastic=el, window=win,
        window_rule=(f"{series_label}: JSON elastic flag (below measured first yield) and load >= 25 % of the "
                     f"highest elastic load ({p_el:.1f} kips, 5 % step tolerance): "
                     f"{load[win].min():.1f}-{load[win].max():.1f} kips, {int(win.sum())} points"),
        rho_l=s["geometry"]["slab"]["longitudinal_reinforcement"]["rho_l"],
        shear_area_clear_in2=a_clear, shear_area_full_in2=a_full,
        classification=dict(programme="Viest et al. (1952)", slab_type="solid cast-in-place",
                            connection_system="channel connectors in solid cast-in-place slab", scale="bridge",
                            role=role),
        rho_l_note="gross, top + bottom longitudinal bars / (72 x t_s)",
        eta_spec=build_spec(s, drop_midspan=True),
        eta_note="connectors between a support and mid-span, the channel on the mid-span line excluded",
        moment_offset_kip_in=offset,
        moment_offset_note=("shored: slab and beam self-weight on the composite section" if shored
                            else "unshored: dead load on the steel alone, applied moment only"),
        defl_uncertainty_in=rows[-1]["uncertainty"]["deflection_in"],
        connector_note=(f"bulletin push-out {s['connectors']['push_out_same_programme']['specimen']}: "
                        f"{pushout_points(s)} (slip in, kip per connector), held beyond; "
                        f"Q_u = {s['connectors']['strength_for_eta']['value_kip']} kips"),
        flags=flags,
        notes=[f"shoring: {s['geometry']['shoring']}", s["deflection_zero_includes"]])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prog = json.loads(DATA.read_text())
    frames, summaries = [], {}
    for s in prog["specimens"]:
        if not s.get("included"):
            continue
        recs = records(s)
        partial = s["id"] == PARTIAL
        primary = "fig53_after_bond_break" if partial else "fig43"
        role = "partial-connection" if partial else "near-full reference"
        case = make_case(s, recs[primary], role=role,
                         series_label=("Fig. 53 test 1B after bond break" if partial else "Fig. 43"))
        df, summ = pw.analyse(case)
        df.insert(1, "series", primary)
        k_full = summ["K_full"]
        c_full = 1.0 / k_full
        csh_c = summ["shear"]["C_sh_clear_over_C_full"] * c_full
        csh_f = summ["shear"]["C_sh_full_over_C_full"] * c_full
        beam_cases, rei_cases, secondary = [], [], []
        if partial:
            for k in (3.28e3, 3.74e3, 4.93e3):
                beam_cases.append(pw.sensitivity_beam(
                    case, f"connector initial stiffness {k:.3g} kip/in (toe replaced)", build_spec(s, k_initial=k),
                    k_full))
            dc = s["connectors"]["degree_of_connection"]
            for label, eta in (("Slutter-Driscoll convention", dc["eta_slutter_driscoll_table4"]),
                               ("AISC channel formula", dc["eta_aisc_channel_formula"]),
                               ("mid-span channel counted", 0.766),
                               ("module half-span convention (mid-span channel weighted 0.5)",
                                pc.eta_plastic(case.spec))):
                rei_cases.append(pw.sensitivity_rei(case, label, eta))
            for name, label in (("fig43", "Fig. 43 test 5A/5B (secondary record)"),
                                ("fig53_bond_intact_context", "Fig. 53 test 1A, natural bond intact (context)")):
                rows = recs[name]
                sub = make_case(s, rows, role=role, series_label=label)
                secondary.append(pw.sub_window_ratios(sub, sub.window, label, c_full, csh_c, csh_f))
                d2 = pd.DataFrame(dict(specimen=s["id"], series=name, load=sub.loads, source_load=sub.loads,
                                       delta_meas_in=sub.delta_meas_in, elastic=sub.elastic, in_window=False,
                                       delta_full_in=sub.loads * c_full))
                frames.append(d2)
        summ["secondary_records"] = secondary
        summ["sensitivity"] = pw.ranges(summ, beam_cases, rei_cases)
        summ["programme_table19_percent_of_complete_interaction"] = s["tabulated_stiffness_ratio"]
        frames.append(df)
        summaries[s["id"]] = summ
        pw.print_summary(summ)
        for sec in secondary:
            print(f"      secondary {sec['label']}: n {sec['n_points']} R_meas {sec.get('R_meas', float('nan')):.3f} "
                  f"R_meas,flex {sec.get('R_meas_flex', float('nan')):.3f}")
    pd.concat(frames, ignore_index=True).to_csv(OUT / f"{KEY}.csv", index=False)
    meta = dict(generated_by=f"scripts/validate_partial_{KEY}.py",
                method="src/validation/partial_connection_windowed.py (windowed intercept-free slope, flexural ratios)",
                data=str(DATA.relative_to(REPO)), specimens=summaries)
    (OUT / f"{KEY}_summary.json").write_text(json.dumps(pw.to_jsonable(meta), indent=2))


if __name__ == "__main__":
    main()
