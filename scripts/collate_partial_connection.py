"""Collate the partial-connection stiffness comparison into one table and statistics.

Reads every reports/model_validation/partial_connection/<key>_summary.json
written by scripts/validate_partial_<key>.py (run those first; new
programmes are picked up automatically), adds Sheehan, Dai and Lam (2018)
as a separate PROFILED-slab row, and writes

    summary.csv              one row per specimen, primary columns first
    summary_statistics.json  statistics over the solid-slab partial-connection
                             specimens (role "partial-connection"; NON-00BS,
                             near-full references, the Chapman anchor and
                             Sheehan are excluded)

Primary quantities (definitions in src/validation/partial_connection_windowed.py):
R_meas_flex (shear compliance removed, clear-web shear area), R_beam_aci (beam
model at the transformed-section E_c), R_EI_window_mean (mean table value
over the in-window points).

Scale rule: "bridge" if span >= 25 ft and steel depth >= 18 in, else
"laboratory" (set in each validate script's classification).

Sheehan row: windowed slope over 3-7.5 kN/m2 (below the 8 kN/m2 linearity
limit, per-cycle deflection row of reports/model_validation/
sheehan_partial_interaction.csv); K_full,flex from the shared module's
transformed section with the 70 mm solid topping as the slab, the 80 mm
ribs as a void offset, f'c 20 MPa (ACI E_c, the modulus of the Sheehan beam
model) and E_s 29 000 ksi; beam-model deflections from the same CSV; the
manuscript's Section 4.5 ratios (Sheehan Table 7) are carried alongside.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.validation import partial_connection as pc  # noqa: E402
from src.validation import partial_connection_windowed as pw  # noqa: E402

OUT = REPO / "reports/model_validation/partial_connection"
MM = pc.IN_PER_MM


def group_of(cls: dict) -> str:
    slab = cls.get("slab_type", "")
    if "precast" in slab:
        return "precast solid slab"
    if cls.get("scale") == "bridge":
        return "bridge-scale cast-in-place solid slab"
    return "laboratory-scale solid slab"


def rng(v):
    return "" if not v else f"{v[0]:.3f}-{v[1]:.3f}"


def g(d, *path, default=np.nan):
    for p in path:
        if not isinstance(d, dict) or p not in d or d[p] is None:
            return default
        d = d[p]
    return d


def specimen_row(key: str, ident: str, s: dict) -> dict:
    cls = s.get("classification", {})
    role = cls.get("role", "")
    row = dict(programme=cls.get("programme", key), key=key, specimen=ident, role=role,
               slab_type=cls.get("slab_type", ""), connection_system=cls.get("connection_system", ""),
               scale=cls.get("scale", ""), group=group_of(cls) if role == "partial-connection" else "")
    if "ratio_K_meas_over_K_steel" in s:          # NON-00BS style reference
        row.update(window=s["window"]["rule"], K_meas_slope=s["K_meas"]["slope"], K_steel=s["K_steel"],
                   K_meas_over_K_steel=s["ratio_K_meas_over_K_steel"],
                   flexural_ratio_over_steel=s["flexural_ratio_steel"],
                   flexural_ratio_over_steel_full_depth_web=s["flexural_ratio_steel_full_depth_web"])
        return row
    sens = s.get("sensitivity", {})
    r_flex = s.get("R_meas_flex", np.nan)
    r_beam = s.get("R_beam_aci", np.nan)
    rei = g(s, "R_EI", "window_mean")
    row.update(
        # ---------------- primary
        eta_plastic=s.get("eta_plastic", np.nan), rho_l=s.get("rho_l", np.nan),
        window=g(s, "window", "rule", default=""), n_window=g(s, "window", "n_points"),
        m_over_mp_mid=g(s, "window", "m_over_mp_mid"), m_over_mp_min=g(s, "window", "m_over_mp_min"),
        m_over_mp_max=g(s, "window", "m_over_mp_max"),
        R_meas_flex=r_flex, R_meas_flex_range_shear_area=rng(s.get("R_meas_flex_range_shear_area")),
        R_beam_aci=r_beam, R_EI_window_mean=rei,
        R_meas_flex_minus_R_EI_window_mean=r_flex - rei,
        R_meas_flex_minus_R_beam_aci=r_flex - r_beam,
        R_EI_mid=g(s, "R_EI", "mid_window"), R_EI_window_min=g(s, "R_EI", "window_min"),
        R_EI_window_max=g(s, "R_EI", "window_max"),
        R_beam_aci_sensitivity_range=rng(sens.get("R_beam_aci_range")),
        R_EI_window_mean_sensitivity_range=rng(sens.get("R_EI_window_mean_range")),
        R_meas_flex_range_K_uncertainty=rng(s.get("R_meas_flex_range_K_uncertainty")),
        # ---------------- secondary
        R_meas_flex_secant=s.get("R_meas_flex_secant", np.nan),
        R_meas=s.get("R_meas", np.nan), R_meas_secant=s.get("R_meas_secant", np.nan),
        Delta_meas=s.get("Delta_meas", np.nan), R_beam_E0=s.get("R_beam_E0", np.nan),
        R_meas_flex_above_first_slip=g(s, "above_first_slip", "R_meas_flex"),
        n_above_first_slip=g(s, "above_first_slip", "n_points"),
        eta_emergent_mid_window=s.get("eta_emergent_mid_window", np.nan),
        K_meas_slope=g(s, "K_meas", "slope"), K_meas_secant=g(s, "K_meas", "secant_from_zero"),
        K_meas_uncertainty=g(s, "K_meas", "uncertainty_used_in_checks"),
        implied_zero_offset_in=g(s, "K_meas", "implied_deflection_zero_offset_in"),
        K_full_flex=s.get("K_full", np.nan), K_steel=s.get("K_steel", np.nan),
        K_noncomposite=s.get("K_noncomposite", np.nan), K_beam_aci=s.get("K_beam_aci", np.nan),
        shear_compliance_over_full_flexural=g(s, "shear", "C_sh_clear_over_C_full"),
        ec_ksi=g(s, "section", "ec_ksi"), ec_source=g(s, "section", "ec_source", default=""),
        checks_pass=(all(s["sanity_checks"].values()) if "sanity_checks" in s else np.nan),
        flags=" | ".join(s.get("flags", [])))
    for sec in s.get("secondary_records", []):
        tag = "fig43" if "Fig. 43" in sec["label"] else "bond_intact_context"
        row[f"R_meas_flex_{tag}"] = sec.get("R_meas_flex", np.nan)
    if role == "full-connection anchor":
        row.update(R_meas_flex_20t=s.get("R_meas_flex_20t"),
                   R_meas_flex_repo_convention_5t=s.get("R_meas_flex_repo_convention_5t"),
                   R_meas_flex_repo_convention_20t=s.get("R_meas_flex_repo_convention_20t"),
                   R_meas_flex_minus_R_EI_window_mean=np.nan, R_meas_flex_minus_R_beam_aci=np.nan)
    return row


def sheehan_row() -> dict:
    df = pd.read_csv(REPO / "reports/model_validation/sheehan_partial_interaction.csv")
    el = df[(df.load_kn_per_m2 <= 7.5) & (~df.is_failure_cycle)]
    q = el.load_kn_per_m2.to_numpy()
    w = q * 2.8 * pc.KIP_PER_KN / (1000.0 * MM)                 # kN/m2 x 2.8 m -> kip/in
    dm = el.defl_cycle_max_mm.to_numpy() * MM
    dbeam = el.bm_deflection_mm.to_numpy() * MM
    steel = pc.i_section(450 * MM, 180 * MM, 10 * MM, 10 * MM, 440 * pc.KSI_PER_MPA,
                         bot_flange_thickness_in=15 * MM, fy_bot_flange_ksi=400 * pc.KSI_PER_MPA)
    sp = pc.Specimen(label="Sheehan-2018", span_in=11200 * MM, slab_width_in=2800 * MM, slab_thickness_in=70 * MM,
                     fc_ksi=20 * pc.KSI_PER_MPA, load_pattern="udl", haunch_height_in=80 * MM,
                     section_type="welded", **steel)
    u = pw.unit_compliances(sp, *pw.web_shear_areas(450 * MM, 12.5 * MM, 10 * MM))
    # clear web 425 mm (flanges 10 and 15 mm): t_w (d - t_ft - t_fb) = t_w (d - 2 x 12.5 mm)
    fit = pw.slope_intercept_free(w, dm)
    ks = pw.secant_through_origin(w, dm)
    fitb = pw.slope_intercept_free(w, dbeam)
    c_full = u["full"]
    eta = 0.33
    mp = pc.plastic_moment_database_definition(sp, eta)
    mr = np.array([pc.midspan_moment(sp, x) / mp for x in w])
    rho = 14 * 0.25 * np.pi * 7.0 ** 2 / (2800.0 * 70.0)
    rei = np.array([pc.r_ei_table(eta, m, rho) for m in mr])
    mid = 0.5 * (w.min() + w.max())
    r_flex = pw.flexural_ratio(fit["K"], c_full, u["shear_clear"])
    r_beam = fitb["K"] * c_full
    k_repo_full = float(np.mean(q / el.cf_deflection_full_interaction_mm.to_numpy()))
    return dict(
        programme="Sheehan, Dai and Lam (2018)", key="sheehan2018", specimen="DISCCO 11.2 m girder",
        role="profiled-slab case (reported separately)", slab_type="profiled (80 mm trapezoidal deck, 70 mm topping)",
        connection_system="headed studs, one per rib", scale="laboratory", group="",
        eta_plastic=eta, rho_l=rho, window="3-7.5 kN/m2 (below the 8 kN/m2 linearity limit), 3 load cycles",
        n_window=int(len(w)), m_over_mp_mid=pc.midspan_moment(sp, mid) / mp, m_over_mp_min=float(mr.min()),
        m_over_mp_max=float(mr.max()),
        R_meas_flex=r_flex,
        R_meas_flex_range_shear_area=rng(sorted([r_flex, pw.flexural_ratio(fit["K"], c_full, u["shear_full"])])),
        R_beam_aci=r_beam, R_EI_window_mean=float(rei.mean()),
        R_meas_flex_minus_R_EI_window_mean=r_flex - float(rei.mean()), R_meas_flex_minus_R_beam_aci=r_flex - r_beam,
        R_EI_mid=pc.r_ei_table(eta, pc.midspan_moment(sp, mid) / mp, rho), R_EI_window_min=float(rei.min()),
        R_EI_window_max=float(rei.max()),
        R_meas_flex_secant=pw.flexural_ratio(ks, c_full, u["shear_clear"]),
        R_meas=fit["K"] * c_full, R_meas_secant=ks * c_full, Delta_meas=1.0 - fit["K"] * c_full,
        K_meas_slope=fit["K"], K_meas_secant=ks, K_full_flex=1.0 / c_full, K_steel=1.0 / u["steel"],
        K_noncomposite=1.0 / u["noncomposite"], K_beam_aci=fitb["K"],
        shear_compliance_over_full_flexural=u["shear_clear"] / c_full, ec_ksi=u["ts"].ec_ksi,
        ec_source=u["ts"].ec_source,
        implied_zero_offset_in=fit["defl_intercept_in"],
        R_meas_repo_closed_form_full=float(pw.slope_intercept_free(q, el.defl_cycle_max_mm.to_numpy())["K"] / k_repo_full),
        manuscript_sec45_R_secant_per_level="0.82 / 0.75 / 0.70",
        flags=("PROFILED slab: reported separately; K_full,flex from the 70 mm topping with ribs as a void offset "
               "(ACI E_c at f'c 20 MPa, E_s 29 000 ksi); beam model from scripts/validate_sheehan_partial_interaction.py "
               "(ACI E_c); manuscript Section 4.5 uses Sheehan Table 7 predictions (I = 1148e6 mm4) and DISCCO I = 1104e6 mm4"))


def stats(v) -> dict:
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return dict(n=0)
    return dict(n=int(v.size), mean=float(v.mean()), median=float(np.median(v)), min=float(v.min()),
                max=float(v.max()))


def statistics(df: pd.DataFrame) -> dict:
    pop = df[(df.role == "partial-connection") & df.slab_type.str.startswith("solid")].copy()
    def block(sub):
        return dict(
            specimens=sub.specimen.tolist(),
            R_meas_flex_minus_R_EI_window_mean=stats(sub.R_meas_flex_minus_R_EI_window_mean),
            R_meas_flex_minus_R_beam_aci=stats(sub.R_meas_flex_minus_R_beam_aci),
            R_meas_flex=stats(sub.R_meas_flex), R_EI_window_mean=stats(sub.R_EI_window_mean),
            R_beam_aci=stats(sub.R_beam_aci),
            count_R_meas_flex_below_1=int((sub.R_meas_flex < 1.0).sum()),
            secondary_R_meas_minus_R_EI_window_mean=stats(sub.R_meas - sub.R_EI_window_mean))
    lab_slip = pop[pop.group == "laboratory-scale solid slab"].copy()
    has = lab_slip.R_meas_flex_above_first_slip.notna()
    lab_slip.loc[has, "R_meas_flex"] = lab_slip.loc[has, "R_meas_flex_above_first_slip"]
    lab_slip["R_meas_flex_minus_R_EI_window_mean"] = lab_slip.R_meas_flex - lab_slip.R_EI_window_mean
    lab_slip["R_meas_flex_minus_R_beam_aci"] = lab_slip.R_meas_flex - lab_slip.R_beam_aci
    bond = pop["flags"].fillna("").str.contains("bond intact: every elastic reading")
    out = dict(
        population=("role partial-connection with a solid slab; excluded: NON-00BS (eta = 0 reference), "
                    "Viest B24W/B24S/B21S (near-full references), Chapman A1-A6 (full-connection anchor), "
                    "Sheehan (profiled slab)"),
        pooled=block(pop),
        by_group={k: block(v) for k, v in pop.groupby("group")},
        by_connection_system={k: block(v) for k, v in pop.groupby("connection_system")},
        by_scale={k: block(v) for k, v in pop.groupby("scale")},
        by_programme={k: block(v) for k, v in pop.groupby("key")},
        sensitivity_pooled_excluding_bond_intact=block(pop[~bond]),
        sensitivity_laboratory_above_first_slip=dict(
            note=("laboratory-scale group with R_meas,flex replaced by the value over readings above first slip "
                  "where >= 2 exist (Slutter B7, B9, B10, B11, B12); R_EI and R_beam kept at the full-window values"),
            **block(lab_slip)),
        references=dict(
            near_full_references=df[df.role == "near-full reference"][["specimen", "R_meas_flex", "R_beam_aci",
                                                                        "R_EI_window_mean"]].to_dict("records"),
            chapman_anchor=df[df.role == "full-connection anchor"][["specimen", "eta_plastic", "R_meas_flex",
                                                                     "R_meas_flex_20t",
                                                                     "R_meas_flex_repo_convention_5t",
                                                                     "R_meas_flex_repo_convention_20t"]].to_dict("records")),
    )
    return out


def main() -> None:
    rows = []
    for f in sorted(OUT.glob("*_summary.json")):
        key = f.name[: -len("_summary.json")]
        d = json.loads(f.read_text())
        for ident, s in d.get("specimens", {}).items():
            rows.append(specimen_row(key, ident, s))
    rows.append(sheehan_row())
    df = pd.DataFrame(rows)
    order = {"partial-connection": 0, "near-full reference": 1, "full-connection anchor": 2}
    df["_o"] = df.role.map(lambda r: order.get(r, 3))
    df = df.sort_values(["_o", "group", "key"], kind="stable").drop(columns="_o")
    df.to_csv(OUT / "summary.csv", index=False)
    st = statistics(df)
    (OUT / "summary_statistics.json").write_text(json.dumps(pw.to_jsonable(st), indent=2))
    cols = ["key", "specimen", "role", "group", "eta_plastic", "rho_l", "m_over_mp_mid", "R_meas_flex",
            "R_meas_flex_range_shear_area", "R_beam_aci", "R_EI_window_mean", "R_meas_flex_minus_R_EI_window_mean",
            "R_meas_flex_minus_R_beam_aci"]
    with pd.option_context("display.width", 300, "display.max_columns", 40, "display.precision", 3):
        print(df[cols].to_string(index=False))
    for name, blk in [("pooled", st["pooled"]), ("pooled excl. bond-intact", st["sensitivity_pooled_excluding_bond_intact"]),
                      ("lab above first slip", st["sensitivity_laboratory_above_first_slip"])] \
            + [(f"group: {k}", v) for k, v in st["by_group"].items()] \
            + [(f"system: {k}", v) for k, v in st["by_connection_system"].items()] \
            + [(f"scale: {k}", v) for k, v in st["by_scale"].items()] \
            + [(f"programme: {k}", v) for k, v in st["by_programme"].items()]:
        a, b = blk["R_meas_flex_minus_R_EI_window_mean"], blk["R_meas_flex_minus_R_beam_aci"]
        print(f"{name}: n {a['n']}  dREI mean {a['mean']:+.3f} med {a['median']:+.3f} [{a['min']:+.3f}, {a['max']:+.3f}]"
              f"  dBeam mean {b['mean']:+.3f} med {b['median']:+.3f} [{b['min']:+.3f}, {b['max']:+.3f}]"
              f"  R_flex<1: {blk['count_R_meas_flex_below_1']}")


if __name__ == "__main__":
    main()
