"""Full-connection anchor: Chapman and Balakrishnan (1964) beams A1-A6.

No new digitising: Balakrishnan's thesis tabulates the measured mid-span
deflection against the full-interaction calculation at W = 5 tons (p. 138,
with the separate bending and shear terms) and 20 tons (p. 143), central
point load, 18 ft span (data/experimental/partial_connection/chapman1964.json).

PRIMARY (thesis basis, bending part only):
    R_flex = delta_bending,full / (delta_expt - delta_shear,thesis)
SECONDARY (this study's convention, as for the partial-connection set):
    transformed section with ACI E_c at the cylinder f'c and E_s 29 000 ksi,
    9 in load spread, shear compliance with the clear web carrying all the
    shear (G = E_s / 2.6): R_flex = delta_full,flex / (delta_expt - delta_sh).

eta_plastic from the literature_tests.csv geometry and the paper's stud
layout (pairs at the Table 1 spacing, symmetric about mid-span, pairs
outside the span not counted) with the programme push-out ultimate
(PA1-PA3 mean 12.5 tons per stud).

These beams are a full-connection anchor and are excluded from the
partial-connection statistics.

Outputs
-------
reports/model_validation/partial_connection/chapman1964.csv
reports/model_validation/partial_connection/chapman1964_summary.json
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

KEY = "chapman1964"
DATA = REPO / "data/experimental/partial_connection" / f"{KEY}.json"
CSV = REPO / "data/experimental/literature_tests.csv"
OUT = REPO / "reports/model_validation/partial_connection"
TON = pc.KIP_PER_LONG_TON
BEARING_IN = 9.0


def build_spec(beam: str, row: pd.Series, prog: dict) -> pc.Specimen:
    st = prog["studs"]
    n_pairs = st["number"][beam] // 2
    p = st["pair_spacing_in"][beam]
    L = float(row.span_in)
    x0 = 0.5 * L - 0.5 * (n_pairs - 1) * p
    qu = float(np.mean(list(st["pushout_ultimate_per_stud_tons"].values()))) * TON
    conns = [pc.Connector(x0 + i * p, 1000.0, qu, 2) for i in range(n_pairs)]
    steel = pc.i_section(float(row.steel_depth_in), float(row.flange_width_in), float(row.flange_thickness_in),
                         float(row.web_thickness_in), float(row.fy_ksi))
    return pc.Specimen(label=beam, span_in=L, slab_width_in=float(row.deck_width_in),
                       slab_thickness_in=float(row.deck_thickness_in), fc_ksi=float(row.fc_deck_ksi),
                       connectors=conns, load_pattern="midspan", bearing_length_in=BEARING_IN,
                       overhang_in=0.5 * (st["overall_beam_length_in"] - L), section_type="W",
                       source="Chapman and Balakrishnan (1964); Balakrishnan (1963) thesis", **steel)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prog = json.loads(DATA.read_text())
    lit = pd.read_csv(CSV)
    rows, summaries = [], {}
    t5, t20 = prog["thesis_p138_W5tons"], prog["thesis_p143_W20tons"]
    for beam in ("A1", "A2", "A3", "A4", "A5", "A6"):
        row = lit[lit.test_id == f"CB-{beam}-40T"].iloc[0]
        spec = build_spec(beam, row, prog)
        eta_d = pc.eta_plastic_details(spec)
        a_clear, a_full = pw.web_shear_areas(spec.steel_depth_in, spec.top_flange_thickness_in, spec.web_thickness_in)
        u = pw.unit_compliances(spec, a_clear, a_full)
        mp = pc.plastic_moment_database_definition(spec, eta_d["eta"])
        rho = (20 * 0.25 * np.pi * 0.625 ** 2 if beam == "A1" else 8 * 0.25 * np.pi * 0.3125 ** 2) / (48.0 * 6.0)
        per_load = {}
        v5 = dict(zip(t5["columns"], t5[beam]))
        v20 = dict(zip(t20["columns"], t20[beam]))
        for label, W_t, bend, shear, total, expt, scale in (
                ("W5t", 5.0, v5["delta_bending"], v5["delta_shear"], v5["delta_total"], v5["delta_expt"], 1e-4),
                ("W20t", 20.0, t20["full_interaction_bending"],
                 t20["full_interaction_total"] - t20["full_interaction_bending"],
                 t20["full_interaction_total"], v20["delta_expt"], 1e-3)):
            W = W_t * TON
            d_meas = expt * scale
            r_thesis = bend / (expt - shear)
            d_full = W * u["full"]
            r_repo = pw.flexural_ratio(W / d_meas, u["full"], u["shear_clear"])
            r_repo_full = pw.flexural_ratio(W / d_meas, u["full"], u["shear_full"])
            mr = pc.midspan_moment(spec, W) / mp
            rei = pc.r_ei_table(eta_d["eta"], mr, rho)
            per_load[label] = dict(
                W_tons=W_t, W_kip=W, delta_expt_in=d_meas,
                thesis_delta_bending_in=bend * scale, thesis_delta_shear_in=shear * scale,
                thesis_delta_total_in=total * scale,
                R_meas_flex_thesis=r_thesis, R_meas_total_thesis=total / expt,
                R_meas_flex_repo_convention=r_repo, R_meas_flex_repo_full_depth_web=r_repo_full,
                repo_delta_full_flex_in=d_full, repo_delta_shear_clear_web_in=W * u["shear_clear"],
                m_over_mp=mr, R_EI_reference=rei,
                partial_interaction_theory_total_in=(v20["partial_total"] * scale if label == "W20t" else None))
            rows.append(dict(specimen=beam, load=W, load_unit="kip (central point load)", source_load=W_t,
                             source_load_unit="ton (British)", m_over_mp=mr, delta_meas_in=d_meas,
                             delta_full_in=d_full, delta_full_thesis_bending_in=bend * scale,
                             delta_shear_thesis_in=shear * scale, delta_shear_clear_web_in=W * u["shear_clear"],
                             delta_steel_in=W * u["steel"], delta_beam_in=np.nan,
                             R_meas_flex_thesis=r_thesis, R_meas_flex_repo_convention=r_repo,
                             R_EI_at_load=rei, elastic=True, in_window=True))
        summ = dict(
            specimen=beam, source=spec.source,
            classification=dict(programme="Chapman and Balakrishnan (1964)", slab_type="solid cast-in-place",
                                connection_system="welded studs in solid cast-in-place slab", scale="laboratory",
                                role="full-connection anchor"),
            window=dict(rule="thesis tables at W = 5 and 20 tons", n_points=2, m_over_mp_mid=per_load["W5t"]["m_over_mp"],
                        m_over_mp_min=per_load["W5t"]["m_over_mp"], m_over_mp_max=per_load["W20t"]["m_over_mp"]),
            R_meas_flex=per_load["W5t"]["R_meas_flex_thesis"],
            R_meas_flex_note="thesis basis at W = 5 tons (primary); the 20 ton value is R_meas_flex_20t",
            R_meas_flex_20t=per_load["W20t"]["R_meas_flex_thesis"],
            R_meas_flex_repo_convention_5t=per_load["W5t"]["R_meas_flex_repo_convention"],
            R_meas_flex_repo_convention_20t=per_load["W20t"]["R_meas_flex_repo_convention"],
            R_meas_flex_repo_convention_5t_range_shear_area=sorted([per_load["W5t"]["R_meas_flex_repo_convention"],
                                                                    per_load["W5t"]["R_meas_flex_repo_full_depth_web"]]),
            R_EI=dict(window_mean=float(np.mean([per_load[k]["R_EI_reference"] for k in per_load])),
                      mid_window=per_load["W5t"]["R_EI_reference"],
                      window_min=min(per_load[k]["R_EI_reference"] for k in per_load),
                      window_max=max(per_load[k]["R_EI_reference"] for k in per_load)),
            eta_plastic=eta_d["eta"], eta_plastic_details=eta_d, rho_l=rho,
            rho_l_note="approximate, from the thesis bar description (reference only)",
            per_load=per_load, first_end_slip_W_tons=prog["first_end_slip_W_tons"][beam],
            section=dict(I_full_in4=u["ts"].I_full_in4, n_modular=u["ts"].n, ec_ksi=u["ts"].ec_ksi,
                         thesis_I_transformed_in4=v5["I_transformed_in4"], thesis_m=v5["m"]),
            flags=["full-connection anchor (eta_plastic >= 0.9); excluded from partial-connection statistics",
                   "laboratory scale; span 18 ft and 4 ft spacing below the design space",
                   "chemical bond not prevented"],
        )
        summaries[beam] = summ
        print(f"{beam}: eta {eta_d['eta']:.3f}  R_flex thesis 5t {per_load['W5t']['R_meas_flex_thesis']:.3f} "
              f"20t {per_load['W20t']['R_meas_flex_thesis']:.3f}  | repo convention 5t "
              f"{per_load['W5t']['R_meas_flex_repo_convention']:.3f} 20t {per_load['W20t']['R_meas_flex_repo_convention']:.3f}"
              f"  | total thesis 5t {per_load['W5t']['R_meas_total_thesis']:.3f}  M/Mp {per_load['W5t']['m_over_mp']:.2f}/"
              f"{per_load['W20t']['m_over_mp']:.2f}  I_full {u['ts'].I_full_in4:.0f} (thesis {v5['I_transformed_in4']})"
              f"  shear/flex repo {u['shear_clear'] / u['full']:.3f} thesis {v5['delta_shear'] / v5['delta_bending']:.3f}")
    pd.DataFrame(rows).to_csv(OUT / f"{KEY}.csv", index=False)
    meta = dict(generated_by=f"scripts/validate_partial_{KEY}.py", data=str(DATA.relative_to(REPO)),
                role="full-connection anchor", specimens=summaries)
    (OUT / f"{KEY}_summary.json").write_text(json.dumps(pw.to_jsonable(meta), indent=2))


if __name__ == "__main__":
    main()
