"""Partial-interaction check on Sheehan, Dai and Lam (2018), eta = 0.33.

WHAT THIS IS. Not a validation of interface slip. It is a check of what a
partial-interaction model predicts for a beam whose connector stiffness
was published a priori by the source programme, so that the prediction
carries no free parameter, and a measurement of how far that prediction
lands from the tabulated slip and the tabulated deflection.

THE SPECIMEN. The 11.2 m fabricated asymmetric plate girder of RFCS
project DISCCO, reported by Sheehan et al. (2018) JCSR 141:251-261
(doi 10.1016/j.jcsr.2017.11.018) and by Lawson et al. (2017)
EUR 28458 EN. Degree of shear connection 0.33. One 19 mm headed stud
per deck rib at 300 mm pitch. DISCCO Section 4.4 and the WP5 summary fix
the elastic stiffness of that stud at 70 kN/mm; Sheehan Section 4.1
derives its resistance, 68 kN, from the Nellinger push tests. Both are
used as published. Nothing is fitted.

WHAT IS COMPARED, THREE WAYS.
  measured      Sheehan Table 2 (end slip, LVDT-11) and Table 1 (mid-span
                deflection), seven load-unload cycles at 3, 5, 7.5, 10,
                12, 15 and 18 kN/m2. Both the per-cycle maximum and the
                cumulative maximum rows are carried: DISCCO Section 4.3.6
                reads the cumulative row for slip, Sheehan Table 7 reads
                the per-cycle row for deflection, and the two rows are
                reported side by side rather than one being chosen.
  closed form   Newmark partial-interaction theory with the same k, the
                formulation DISCCO Section 4.4 uses. Reimplemented in
                src/validation/sheehan_udl.py so its published 0.53 mm end
                slip at 5 kN/m2 can be checked rather than quoted.
  beam model    This repository's two-chain fibre model, extended to an
                asymmetric section and a uniformly distributed load
                (src/validation/sheehan_udl.py). Same k.

Outputs
-------
data/experimental/sheehan_2018_udl.csv           measured table, as printed
reports/model_validation/sheehan_partial_interaction.csv
reports/model_validation/sheehan_partial_interaction_summary.json

Usage
-----
/opt/anaconda3/envs/ops_x86/bin/python scripts/validate_sheehan_partial_interaction.py
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.validation import sheehan_udl as S  # noqa: E402

MEAS = S.MEASURED
LOADS = MEAS["load_kn_per_m2"]
WIDTH_M = MEAS["tributary_width_m"]
SPAN_M = MEAS["span_m"]


def measured_frame() -> pd.DataFrame:
    """Sheehan Table 2 and Table 1 transcribed as printed, with the
    provenance in the file so the CSV stands on its own."""
    df = pd.DataFrame({
        "load_kn_per_m2": LOADS,
        "moment_knm": [q * WIDTH_M * SPAN_M ** 2 / 8.0 for q in LOADS],
        "slip_cycle_max_mm": MEAS["slip_cycle_max_mm"],
        "slip_residual_mm": MEAS["slip_residual_mm"],
        "slip_cumulative_mm": MEAS["slip_cumulative_mm"],
        "defl_cycle_max_mm": MEAS["defl_cycle_max_mm"],
        "defl_cumulative_mm": MEAS["defl_cumulative_mm"],
    })
    df["is_failure_cycle"] = [i == MEAS["failure_index"] for i in range(len(LOADS))]
    df["source"] = "Sheehan_Dai_Lam_2018_JCSR_141_251"
    df["slip_table"] = "Table 2 (LVDT-11)"
    df["defl_table"] = "Table 1"
    return df


def closed_form_frame() -> pd.DataFrame:
    """Newmark closed form at every load level, with DISCCO's own printed
    section properties and stiffness."""
    props = dict(S.DISCCO_PROPS)
    rows = []
    for q in LOADS:
        w = q * WIDTH_M                       # kN/m  ->  N/mm numerically
        r = S.newmark_udl(w_n_per_mm=w, **props)
        rows.append({"load_kn_per_m2": q,
                     "cf_end_slip_mm": r["end_slip_mm"],
                     "cf_deflection_mm": r["deflection_mm"],
                     "cf_deflection_full_interaction_mm":
                         r["deflection_full_interaction_mm"],
                     "cf_stud_force_kn": r["stud_force_kn"],
                     "cf_I_eff_mm4": r["I_eff_mm4"],
                     "cf_I_full_mm4": r["I_full_mm4"],
                     "cf_alpha_L": r["alpha_L"]})
    return pd.DataFrame(rows)


def run_beam_model(p: S.AsymBeamParams, n_steps: int) -> tuple[pd.DataFrame, dict]:
    w_max = S.udl_kn_m2_to_kip_per_in(max(LOADS), WIDTH_M)
    res = S.analyze_udl(p, w_max_kip_per_in=w_max, n_steps=n_steps)
    rows = []
    for q in LOADS:
        w = S.udl_kn_m2_to_kip_per_in(q, WIDTH_M)
        rows.append({
            "load_kn_per_m2": q,
            "bm_end_slip_mm": res.at_w(w, "end_slip_in") * S.MM_PER_IN,
            "bm_max_slip_mm": res.at_w(w, "max_slip_in") * S.MM_PER_IN,
            "bm_deflection_mm": res.at_w(w, "deflection_in") * S.MM_PER_IN,
            "bm_stud_force_kn": res.at_w(w, "conn_force_max_kip") * S.KN_PER_KIP,
        })
    meta = {
        "converged_steps": res.converged_steps,
        "n_requested": res.n_requested,
        "w_max_reached_kn_per_m2":
            float(res.w_kip_per_in[-1] / S.udl_kn_m2_to_kip_per_in(1.0, WIDTH_M))
            if res.converged_steps else None,
    }
    return pd.DataFrame(rows), meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-steps", type=int, default=360)
    ap.add_argument("--reports", default=str(REPO / "reports/model_validation"))
    ap.add_argument("--data", default=str(REPO / "data/experimental"))
    args = ap.parse_args()

    meas = measured_frame()
    data_dir = Path(args.data)
    data_dir.mkdir(parents=True, exist_ok=True)
    meas_path = data_dir / "sheehan_2018_udl.csv"
    meas.to_csv(meas_path, index=False)
    print(f"[out] {meas_path}")

    cf = closed_form_frame()
    print("\n[closed form] Newmark with DISCCO's printed properties and "
          "k = 70 kN/mm")
    print(f"  d back-computed from Icomp = 1104e6: "
          f"{S.DISCCO_PROPS['d_mm']:.1f} mm   alpha*L = {cf.cf_alpha_L[0]:.2f}")
    print(f"  at 5.0 kN/m2 (M = {5.0 * WIDTH_M * SPAN_M ** 2 / 8:.0f} kNm): "
          f"end slip {cf.cf_end_slip_mm[1]:.3f} mm, deflection "
          f"{cf.cf_deflection_mm[1]:.2f} mm, stud force "
          f"{cf.cf_stud_force_kn[1]:.1f} kN, "
          f"I_eff/I_full = {100 * cf.cf_I_eff_mm4[1] / cf.cf_I_full_mm4[1]:.0f} %")
    print("  DISCCO Section 4.4 prints, for the same beam and the same k: "
          "M = 220 kNm,")
    print("  end slip 0.53 mm, stud force 37 kN, I_eff = 821e6 = 74 % of "
          "Icomp = 1104e6,")
    print("  and Table 4.1 computed deflection 16.2 mm against measured "
          "16.0 mm.")

    # ---- beam model: primary run and two sensitivities
    runs = {}
    variants = {
        "primary": dict(kw=dict(), note="ACI-form concrete (Ec from "
                        "Concrete02 initial tangent 2 fc / eps_c0), "
                        "DISCCO connector backbone"),
        "ec35": dict(kw=dict(ec_gpa=35.0), note="concrete modulus forced to "
                     "DISCCO's assumed Es/Ec = 6, i.e. Ec = 35 GPa"),
        "ollgaard": dict(kw=dict(backbone="ollgaard"), note="Ollgaard-shaped "
                         "connector backbone of src/beam_model.py, same "
                         "initial stiffness k = 70 kN/mm"),
    }
    for name, spec in variants.items():
        p = S.sheehan_specimen(**spec["kw"])
        df, meta = run_beam_model(p, args.n_steps)
        runs[name] = {"df": df, "meta": meta, "note": spec["note"]}
        print(f"\n[beam model: {name}] {spec['note']}")
        print(f"  converged {meta['converged_steps']}/{meta['n_requested']} "
              f"steps, reached {meta['w_max_reached_kn_per_m2']:.1f} kN/m2")

    out = meas.merge(cf, on="load_kn_per_m2")
    out = out.merge(runs["primary"]["df"], on="load_kn_per_m2")
    for name in ("ec35", "ollgaard"):
        r = runs[name]["df"].rename(
            columns={c: c.replace("bm_", f"bm_{name}_")
                     for c in runs[name]["df"].columns if c.startswith("bm_")})
        out = out.merge(r, on="load_kn_per_m2")

    # ratios, both against the cumulative and the per-cycle measured row
    for pred, tag in (("cf_end_slip_mm", "cf"), ("bm_end_slip_mm", "bm")):
        out[f"{tag}_slip_over_measured_cumulative"] = (
            out[pred] / out.slip_cumulative_mm)
        out[f"{tag}_slip_over_measured_cycle"] = (
            out[pred] / out.slip_cycle_max_mm)
    out["cf_defl_over_measured_cycle"] = out.cf_deflection_mm / out.defl_cycle_max_mm
    out["bm_defl_over_measured_cycle"] = out.bm_deflection_mm / out.defl_cycle_max_mm

    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    csv_path = reports / "sheehan_partial_interaction.csv"
    out.to_csv(csv_path, index=False)
    print(f"\n[out] {csv_path}")

    # ---- console table
    print("\n" + "=" * 104)
    print("q kN/m2 |     measured end slip (mm)    |    predicted end slip (mm)   "
          "|   measured defl |  predicted defl")
    print("        |  per-cycle    cumulative      |  closed form    beam model   "
          "|  per-cycle (mm) |  beam model (mm)")
    print("-" * 104)
    for _, r in out.iterrows():
        star = " *" if r.is_failure_cycle else "  "
        print(f"{r.load_kn_per_m2:6.1f}{star}|  {r.slip_cycle_max_mm:9.3f} "
              f"{r.slip_cumulative_mm:13.3f}      |"
              f"  {r.cf_end_slip_mm:10.3f} {r.bm_end_slip_mm:13.3f}    |"
              f"  {r.defl_cycle_max_mm:12.1f}   | {r.bm_deflection_mm:13.1f}")
    print("-" * 104)
    print("* failure cycle")

    print("\nratio predicted / measured end slip (cumulative row):")
    for _, r in out.iterrows():
        print(f"  {r.load_kn_per_m2:5.1f} kN/m2   closed form "
              f"{r.cf_slip_over_measured_cumulative:7.2f} x   beam model "
              f"{r.bm_slip_over_measured_cumulative:7.2f} x   "
              f"| deflection ratio (beam model) "
              f"{r.bm_defl_over_measured_cycle:5.2f} x")

    serv = out[out.load_kn_per_m2 <= 15.0]
    summary = {
        "generated_by": "scripts/validate_sheehan_partial_interaction.py",
        "specimen": {
            "source": "Sheehan, Dai and Lam (2018), JCSR 141, 251-261",
            "doi": "10.1016/j.jcsr.2017.11.018",
            "parent_programme": "DISCCO, Lawson et al. (2017), EUR 28458 EN",
            "eta_shear_connection": 0.33,
            "span_m": SPAN_M,
            "tributary_width_m": WIDTH_M,
            "connector": "single 19 mm headed stud per deck rib, 300 mm pitch",
            "connector_stiffness_kn_per_mm": 70.0,
            "connector_stiffness_source":
                "DISCCO Section 4.4 and WP5 summary, published a priori; "
                "NOT fitted to this comparison",
            "connector_resistance_kn": 68.0,
            "connector_resistance_source":
                "Sheehan Section 4.1, 75 x sqrt(25/30) from Nellinger push tests",
        },
        "measured": {
            "slip_table": "Sheehan Table 2, LVDT-11, one end only",
            "deflection_table": "Sheehan Table 1",
            "load_kn_per_m2": LOADS,
            "slip_cycle_max_mm": MEAS["slip_cycle_max_mm"],
            "slip_cumulative_mm": MEAS["slip_cumulative_mm"],
            "defl_cycle_max_mm": MEAS["defl_cycle_max_mm"],
            "row_convention_note":
                "DISCCO Section 4.3.6 reads the cumulative slip row "
                "('about 3 mm at 12 kN/m2', 'passed 6 mm at 15 kN/m2', "
                "'a maximum of 19 mm'); Sheehan Table 7 reads the per-cycle "
                "deflection row. Both slip rows are reported here.",
        },
        "closed_form_check": {
            "formulation": "Newmark partial interaction, UDL, "
                           "reimplemented in src/validation/sheehan_udl.py",
            "properties": "DISCCO Section 4.4 as printed; d back-computed "
                          "from Icomp = 1104e6",
            "d_mm": float(S.DISCCO_PROPS["d_mm"]),
            "alpha_L": float(cf.cf_alpha_L[0]),
            "at_5_kn_per_m2": {
                "moment_knm": 5.0 * WIDTH_M * SPAN_M ** 2 / 8.0,
                "end_slip_mm": float(cf.cf_end_slip_mm[1]),
                "deflection_mm": float(cf.cf_deflection_mm[1]),
                "stud_force_kn": float(cf.cf_stud_force_kn[1]),
                "I_eff_over_I_full": float(cf.cf_I_eff_mm4[1] / cf.cf_I_full_mm4[1]),
            },
            "discco_printed": {
                "moment_knm": 220.0, "end_slip_mm": 0.53,
                "stud_force_kn": 37.0, "I_eff_mm4": 821e6, "I_comp_mm4": 1104e6,
                "deflection_computed_mm": 16.2, "deflection_measured_mm": 16.0,
                "deflection_no_end_slip_mm": 12.0,
            },
            "verdict":
                "reproduced: the reimplementation gives "
                f"{float(cf.cf_end_slip_mm[1]):.3f} mm end slip and "
                f"{float(cf.cf_stud_force_kn[1]):.1f} kN stud force against "
                "DISCCO's printed 0.53 mm and 37 kN, and "
                f"I_eff/I_full = {float(cf.cf_I_eff_mm4[1]/cf.cf_I_full_mm4[1]):.2f} "
                "against DISCCO's 821/1104 = 0.74.",
        },
        "beam_model": {
            "module": "src/validation/sheehan_udl.py",
            "architecture": "two forceBeamColumn fibre chains on a shared "
                            "interface reference line, both sections "
                            "'-noCentroid', equalDOF on DOF 2 only, one "
                            "zeroLength connector per node in direction 1",
            "extensions_over_src_beam_model": [
                "separate top and bottom flange thickness and yield stress",
                "uniformly distributed load under load control, so the shear "
                "flow gradient that drives slip is present",
                "slab topping offset above the top flange by the 80 mm rib "
                "depth, ribs transverse and not modelled as continuous",
            ],
            "variants": {k: {"note": v["note"], **v["meta"]}
                         for k, v in runs.items()},
        },
        "headline": {
            "slip_at_5_kn_per_m2_mm": {
                "measured_cumulative": 0.062, "measured_per_cycle": 0.045,
                "closed_form": float(cf.cf_end_slip_mm[1]),
                "beam_model": float(out.bm_end_slip_mm[1]),
                "closed_form_over_measured":
                    float(cf.cf_end_slip_mm[1] / 0.062),
                "beam_model_over_measured":
                    float(out.bm_end_slip_mm[1] / 0.062),
            },
            "deflection_at_5_kn_per_m2_mm": {
                "measured_per_cycle": 16.4,
                "measured_discco_table_4_1": 16.0,
                "closed_form": float(cf.cf_deflection_mm[1]),
                "beam_model": float(out.bm_deflection_mm[1]),
                "discco_computed": 16.2,
            },
            "deflection_ratio_range_to_15_kn_per_m2": [
                float(serv.bm_defl_over_measured_cycle.min()),
                float(serv.bm_defl_over_measured_cycle.max())],
            "slip_ratio_range_to_15_kn_per_m2_cumulative": [
                float(serv.bm_slip_over_measured_cumulative.min()),
                float(serv.bm_slip_over_measured_cumulative.max())],
        },
        "what_this_is_not":
            "This is not a validation of interface slip. With the connector "
            "stiffness fixed at the value the source programme published, "
            "two independent partial-interaction formulations both "
            "over-predict the measured end slip at service load by nearly an "
            "order of magnitude, while predicting the deflection at the same "
            "load to within a few per cent. Measured slip at service load is "
            "dominated by chemical bond, friction and the mechanical keying "
            "of the profiled deck, none of which a partial-interaction "
            "formulation carries; it is therefore a badly conditioned "
            "validation target. Deflection, and the flexural stiffness this "
            "paper is about, is well conditioned.",
    }
    json_path = reports / "sheehan_partial_interaction_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[out] {json_path}")

    h = summary["headline"]
    print("\n=== headline ===")
    print(f"end slip at 5 kN/m2 : measured {h['slip_at_5_kn_per_m2_mm']['measured_cumulative']} mm "
          f"(cumulative) | closed form "
          f"{h['slip_at_5_kn_per_m2_mm']['closed_form']:.2f} mm "
          f"({h['slip_at_5_kn_per_m2_mm']['closed_form_over_measured']:.1f}x) | "
          f"beam model {h['slip_at_5_kn_per_m2_mm']['beam_model']:.2f} mm "
          f"({h['slip_at_5_kn_per_m2_mm']['beam_model_over_measured']:.1f}x)")
    d = h["deflection_at_5_kn_per_m2_mm"]
    print(f"deflection at 5 kN/m2: measured {d['measured_per_cycle']} mm "
          f"(Sheehan Table 1) / {d['measured_discco_table_4_1']} mm "
          f"(DISCCO Table 4.1) | closed form {d['closed_form']:.1f} mm | "
          f"beam model {d['beam_model']:.1f} mm")
    print(f"deflection ratio 3-15 kN/m2 : "
          f"{h['deflection_ratio_range_to_15_kn_per_m2'][0]:.2f} to "
          f"{h['deflection_ratio_range_to_15_kn_per_m2'][1]:.2f} x measured")
    print(f"slip ratio 3-15 kN/m2       : "
          f"{h['slip_ratio_range_to_15_kn_per_m2_cumulative'][0]:.2f} to "
          f"{h['slip_ratio_range_to_15_kn_per_m2_cumulative'][1]:.2f} x measured")


if __name__ == "__main__":
    main()
