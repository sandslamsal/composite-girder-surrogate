"""Measured-to-predicted stiffness on a common deflection basis.

Reads reports/model_validation/partial_connection/summary.csv (written by
scripts/collate_partial_connection.py) and compares the measured stiffness
of every specimen with three predictions, each including the same web-shear
compliance, so the shear assumption affects every prediction identically
and nothing is subtracted from a measurement:

    full composite   1 / (C_flex,full + C_sh)
    R_EI corrected   1 / (C_flex,full / R_EI + C_sh)
    beam model       1 / (1 / K_beam + C_sh)

K_meas is the least-squares slope over the elastic window (free intercept),
C_flex,full = 1 / K_full (uncracked transformed section, ACI E_c unless a
measured modulus is documented), R_EI is the mean of Table
tab:design-correction over the in-window M/M_p values, and K_beam is the
discrete-connector beam model run with the programme's own connector data
at the same E_c.

Outputs
-------
reports/model_validation/partial_connection/deflection_basis.csv
reports/model_validation/partial_connection/deflection_basis_stats.json

Usage
-----
/opt/anaconda3/envs/ops_x86/bin/python scripts/partial_connection_deflection_basis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DIR = REPO / "reports/model_validation/partial_connection"

PRECAST = "precast solid slab"


def add_ratios(d: pd.DataFrame) -> pd.DataFrame:
    c_full = 1.0 / d.K_full_flex
    c_sh = d.shear_compliance_over_full_flexural * c_full
    d = d.copy()
    d["meas_over_full"] = d.K_meas_slope * (c_full + c_sh)
    d["meas_over_REI"] = d.K_meas_slope * (c_full / d.R_EI_window_mean + c_sh)
    d["meas_over_beam"] = d.K_meas_slope * (1.0 / d.K_beam_aci + c_sh)
    return d


def stats(x: pd.Series) -> dict:
    x = x.dropna()
    q1, q3 = np.percentile(x, [25, 75])
    return {"n": int(len(x)), "median": round(float(x.median()), 3),
            "mean": round(float(x.mean()), 3), "q1": round(float(q1), 3),
            "q3": round(float(q3), 3), "min": round(float(x.min()), 3),
            "max": round(float(x.max()), 3),
            "below_1": int((x < 1).sum()),
            "within_10pct": int(((x - 1).abs() <= 0.10).sum())}


SERIES_CITE = {"Kwon": r"\citet{kwon2007txdot}",
               "Viest": r"\citet{viest1952fullscale}",
               "Slutter": r"\citet{culver1960tests,culver1961tests}",
               "McGarraugh": r"\citet{mcgarraugh1971lightweight}",
               "Provines": r"\citet{provines2019clustered}",
               "Suwaed": r"\citet{suwaed2020demountable}"}
CONNECTOR = {"post-installed rods in solid cast-in-place slab": "post-installed rods",
             "friction-grip bolts": "friction-grip bolts",
             "channel connectors in solid cast-in-place slab": "channels",
             "headed studs in cast-in-place lightweight slab": "studs, lightweight slab",
             "welded studs in solid cast-in-place slab": "welded studs",
             "studs in grouted pockets of precast panels": "grouted-pocket studs"}
TABLE = REPO / "paper/revision_2/submission/sources/tables/tab_partial_specimens.tex"


def write_specimen_table(part: pd.DataFrame) -> None:
    """Appendix table: one row per specimen, written from the analysis file
    so that no value is transcribed by hand."""
    rows = []
    last = None
    for r in part.itertuples():
        key = next(k for k in SERIES_CITE if r.programme.startswith(k))
        series = SERIES_CITE[key] if key != last else ""
        last = key
        unc = 100.0 * r.K_meas_uncertainty / r.K_meas_slope
        scale = "bridge" if str(r.scale).startswith("bridge") else "laboratory"
        rows.append(
            f"    {series} & {r.specimen} & {scale} & {CONNECTOR[r.connection_system]}"
            f" & {r.eta_plastic:.2f} & {r.m_over_mp_min:.2f}--{r.m_over_mp_max:.2f}"
            f" & {int(r.n_window)} & $\\pm${unc:.0f} & {r.meas_over_full:.2f}"
            f" & {r.meas_over_REI:.2f} & {r.meas_over_beam:.2f} \\\\")
    body = "\n".join(rows)
    TABLE.write_text(r"""\begin{table}[pos=h]
  \centering
  \caption{Specimens of the partial-connection comparison of
    Table~\ref{tab:partial-validation}: elastic window, number of
    readings, uncertainty of the measured stiffness and ratio of measured
    to predicted stiffness.}
  \label{tab:partial-specimens}
  \footnotesize
  \setlength{\tabcolsep}{2.8pt}
  \begin{tabular}{@{}llllccccccc@{}}
    \toprule
    Test series & Specimen & Scale & Connector & $\eta$ & $M/M_p$ & $n$
      & $K$ & \multicolumn{3}{c}{Measured / predicted} \\
    \cmidrule(l){9-11}
    & & & & & & & (\%) & AASHTO & AASHTO & Beam \\
    & & & & & & & & & $\times\,R_{EI}$ & model \\
    \midrule
""" + body + r"""
    \bottomrule
  \end{tabular}
  \\[3pt]
  {\footnotesize $n$, readings in the elastic window; $K$, uncertainty of
   the measured stiffness (twice the slope standard error combined with
   the reading uncertainty of the source record); AASHTO, full-composite
   transformed section.}
\end{table}
""")


def main() -> None:
    d = add_ratios(pd.read_csv(DIR / "summary.csv"))
    d.to_csv(DIR / "deflection_basis.csv", index=False)

    part = d[d.group.notna()]
    sets = {
        "all_solid_partial": part,
        "cast_in_place": part[part.group != PRECAST],
        "precast": part[part.group == PRECAST],
        "bridge_scale_cast_in_place": part[part.group.str.startswith("bridge")],
        "laboratory_scale": part[part.group.str.startswith("laboratory")],
        "without_bond_intact": part[~part["flags"].fillna("").str.contains(
            "bond intact: every elastic reading")],
    }
    out = {name: {col: stats(sub[col]) for col in
                  ("meas_over_full", "meas_over_REI", "meas_over_beam")}
           for name, sub in sets.items()}
    refs = d[d.role.fillna("").str.contains("reference|anchor", case=False)
             & d.meas_over_full.notna()]
    out["full_connection_references"] = {
        r.specimen: round(float(r.meas_over_full), 3) for r in refs.itertuples()}
    prof = d[d.slab_type.fillna("").str.contains("profiled")]
    out["profiled_slab_sheehan"] = {
        c: round(float(prof[c].iloc[0]), 3) for c in
        ("meas_over_full", "meas_over_REI", "meas_over_beam")} if len(prof) else {}
    (DIR / "deflection_basis_stats.json").write_text(json.dumps(out, indent=2))

    write_specimen_table(part)

    by_prog = part.groupby("programme").agg(
        n=("specimen", "size"),
        eta_min=("eta_plastic", "min"), eta_max=("eta_plastic", "max"),
        full=("meas_over_full", "median"), rei=("meas_over_REI", "median"),
        beam=("meas_over_beam", "median"))
    print(by_prog.round(3).to_string())
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
