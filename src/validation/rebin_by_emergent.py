"""Re-bin the Table tab:sensitivity sections by their emergent eta_c.

Reviewer 3 (revision 2, Section 4.4) notes that the emergent degree of
composite action tracks the target ratio at only r ~ 0.8 and asks how
that bin-assignment blur affects the Table tab:sensitivity bin means.
This script re-aggregates the SAVED sensitivity run (no beam
re-analysis) with each section assigned to the bin of its emergent
eta_c instead of its target eta_c. The per-(bin, regime, metric)
statistics are computed by importing and calling the identical
``summarise`` / ``cluster_boot_se`` code of ``sensitivity_sweep.py``,
so the only thing that changes between the two assignments is the
``eta_bin`` label carried by each section.

Assignment rule
---------------
* Bin edges are ``sensitivity_sweep.ETA_BINS`` (25-50, 50-70, 70-90,
  90-100 %), applied to ``eta_c_emergent`` through the same
  ``eta_bin_label`` helper.
* Emergent values above 1.0 are clipped to 1.0 and therefore fall in
  the 90-100 % bin.
* Sections with emergent eta_c below 0.25 lie outside all four bins
  and are dropped (counted and reported).

Inputs (read only)
------------------
reports/section_vs_beam_revision/sensitivity_rows.parquet
reports/section_vs_beam_revision/sensitivity_sections.csv

Outputs
-------
reports/section_vs_beam_revision/rebin_by_emergent.csv
reports/section_vs_beam_revision/rebin_by_emergent.md
paper/revision_2/submission/sources/tables/tab_sensitivity_emergent.tex

Usage
-----
/opt/anaconda3/envs/ops_x86/bin/python \
    src/validation/rebin_by_emergent.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sensitivity_sweep as ss  # noqa: E402  (summarise, ETA_BINS, ...)

REPO_ROOT = ss.REPO_ROOT
REPORT_DIR = REPO_ROOT / "reports/section_vs_beam_revision"
TEX_PATH = (REPO_ROOT / "paper/revision_2/submission/sources/tables"
            / "tab_sensitivity_emergent.tex")

METRICS = ["section_matched", "beam_defl", "beam_curv"]
REGIMES = ["service", "extended"]
DROP_LABEL = "<25% (dropped)"
BIN_TEX = {"25-50%": "25--\\SI{50}{\\percent}",
           "50-70%": "50--\\SI{70}{\\percent}",
           "70-90%": "70--\\SI{90}{\\percent}",
           "90-100%": "90--\\SI{100}{\\percent}"}


# ---------------------------------------------------------------------------
# reassignment
# ---------------------------------------------------------------------------

def reassign_by_emergent(secs: pd.DataFrame) -> pd.DataFrame:
    """Return ``secs`` with ``eta_bin_target`` (the original label) and
    ``eta_bin_emergent`` (bin of the clipped emergent eta_c; '' when the
    emergent value lies below the lowest bin edge)."""
    out = secs.copy()
    out["eta_bin_target"] = out["eta_bin"]
    clipped = out["eta_c_emergent"].clip(upper=1.0)
    out["eta_c_emergent_clipped"] = clipped
    out["eta_bin_emergent"] = clipped.map(ss.eta_bin_label)
    return out


def crosstab(secs: pd.DataFrame) -> pd.DataFrame:
    col = secs["eta_bin_emergent"].replace({"": DROP_LABEL})
    ct = pd.crosstab(secs["eta_bin_target"], col)
    ct = ct.reindex(index=ss.BIN_LABELS,
                    columns=ss.BIN_LABELS + [DROP_LABEL], fill_value=0)
    ct.index.name = "target bin \\ emergent bin"
    ct["total"] = ct.sum(axis=1)
    ct.loc["total"] = ct.sum(axis=0)
    return ct


# ---------------------------------------------------------------------------
# tabulation
# ---------------------------------------------------------------------------

def _stat(summary: pd.DataFrame, lab: str, regime: str, metric: str,
          field: str) -> float:
    s = summary[(summary.eta_bin == lab) & (summary.regime == regime)
                & (summary.metric == metric)]
    return float(s[field].iloc[0]) if len(s) else float("nan")


def wide_table(summ_target: pd.DataFrame, summ_em: pd.DataFrame,
               n_target: dict, n_em: dict) -> pd.DataFrame:
    recs = []
    for lab in ss.BIN_LABELS:
        for regime in REGIMES:
            rec = {"eta_bin": lab, "regime": regime,
                   "n_sections_target": n_target[lab],
                   "n_sections_emergent": n_em[lab],
                   "n_sections_matched_target":
                       int(_stat(summ_target, lab, regime,
                                 "section_matched", "n_sections")),
                   "n_sections_matched_emergent":
                       int(_stat(summ_em, lab, regime,
                                 "section_matched", "n_sections"))}
            for m in METRICS:
                mt = _stat(summ_target, lab, regime, m, "mean_pct")
                me = _stat(summ_em, lab, regime, m, "mean_pct")
                rec[f"{m}_mean_target"] = mt
                rec[f"{m}_se_target"] = _stat(summ_target, lab, regime,
                                              m, "se_pct")
                rec[f"{m}_mean_emergent"] = me
                rec[f"{m}_se_emergent"] = _stat(summ_em, lab, regime,
                                                m, "se_pct")
                rec[f"{m}_shift"] = me - mt
            rec["gap_sec_minus_defl_target"] = (
                rec["section_matched_mean_target"]
                - rec["beam_defl_mean_target"])
            rec["gap_sec_minus_defl_emergent"] = (
                rec["section_matched_mean_emergent"]
                - rec["beam_defl_mean_emergent"])
            rec["gap_shift"] = (rec["gap_sec_minus_defl_emergent"]
                                - rec["gap_sec_minus_defl_target"])
            recs.append(rec)
    return pd.DataFrame(recs)


def _ct_markdown(ct: pd.DataFrame) -> str:
    """Plain markdown rendering of the crosstab (no tabulate dependency)."""
    head = "| " + str(ct.index.name) + " | " + " | ".join(map(str, ct.columns)) + " |"
    sep = "|" + "---|" * (len(ct.columns) + 1)
    body = ["| " + str(idx) + " | " + " | ".join(str(int(v)) for v in row) + " |"
            for idx, row in ct.iterrows()]
    return "\n".join([head, sep, *body])


def to_markdown(wide: pd.DataFrame, ct: pd.DataFrame, r_target: float,
                r_layout: float, rho_target: float, rho_layout: float,
                n_dropped: int, n_clipped: int, n_em: dict) -> str:
    L = ["# Table tab:sensitivity re-binned by emergent eta_c", "",
         "Same 400 sections, same saved beam runs, same `summarise()` "
         "code; only the bin label per section changes.", "",
         f"* Pearson r(eta_c_target, eta_c_emergent) = {r_target:.2f} "
         f"(Spearman {rho_target:.2f})",
         f"* Pearson r(eta_c_layout, eta_c_emergent) = {r_layout:.2f} "
         f"(Spearman {rho_layout:.2f})",
         f"* Emergent values above 1.0 clipped into the 90-100% bin: "
         f"{n_clipped}",
         f"* Sections with emergent eta_c < 0.25 (outside all bins, "
         f"dropped): {n_dropped}",
         "* Sections per emergent bin: "
         + ", ".join(f"{lab} n={n_em[lab]}" for lab in ss.BIN_LABELS),
         ""]
    small = [lab for lab in ss.BIN_LABELS if n_em[lab] < 20]
    if small:
        L += ["**Bins with fewer than 20 sections (emergent assignment): "
              + ", ".join(small) + "**", ""]
    L += ["## Crosstab: target bin (rows) vs emergent bin (columns)", "",
          _ct_markdown(ct), ""]
    L += ["## Bin means (%) under both assignments", "",
          "Mean ± section-clustered bootstrap SE (500 resamples). "
          "Shift = emergent minus target. Gap = section_matched minus "
          "beam_defl.", "",
          "| bin | regime | n (target / emergent) | metric | target "
          "assignment | emergent assignment | shift |",
          "|---|---|---|---|---|---|---|"]
    for _, r in wide.iterrows():
        first = True
        for m in METRICS:
            L.append(
                f"| {r.eta_bin if first else ''} | "
                f"{r.regime if first else ''} | "
                f"{(str(r.n_sections_target) + ' / ' + str(r.n_sections_emergent)) if first else ''} | "
                f"{m} | {r[f'{m}_mean_target']:+.2f} ± "
                f"{r[f'{m}_se_target']:.2f} | "
                f"{r[f'{m}_mean_emergent']:+.2f} ± "
                f"{r[f'{m}_se_emergent']:.2f} | {r[f'{m}_shift']:+.2f} |")
            first = False
        L.append(f"| | | | gap (sec - beam_defl) | "
                 f"{r.gap_sec_minus_defl_target:+.2f} | "
                 f"{r.gap_sec_minus_defl_emergent:+.2f} | "
                 f"{r.gap_shift:+.2f} |")
    return "\n".join(L) + "\n"


def to_latex(summ_em: pd.DataFrame, n_em: dict, n_dropped: int) -> str:
    def cell(lab, regime, metric):
        return (f"${_stat(summ_em, lab, regime, metric, 'mean_pct'):.1f} "
                f"\\pm {_stat(summ_em, lab, regime, metric, 'se_pct'):.1f}$")

    rows = []
    for lab in ss.BIN_LABELS:
        rows.append(
            f"    {BIN_TEX[lab]} & {n_em[lab]} & "
            f"{cell(lab, 'service', 'section_matched')} & "
            f"{cell(lab, 'service', 'beam_defl')} & "
            f"{cell(lab, 'service', 'beam_curv')} & & "
            f"{cell(lab, 'extended', 'section_matched')} & "
            f"{cell(lab, 'extended', 'beam_defl')} & "
            f"{cell(lab, 'extended', 'beam_curv')} \\\\")
    return "\n".join([
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Table~\\ref{tab:sensitivity} recomputed with each",
        "    section assigned to the bin of its emergent $\\eta_c$ rather",
        "    than its target $\\eta_c$. Mean $\\pm$ section-clustered",
        "    bootstrap standard error (500 resamples); $n$ is the number",
        f"    of sections in each emergent bin. The {n_dropped} sections",
        "    whose emergent $\\eta_c$ lies below \\SI{25}{\\percent} fall",
        "    outside the four bins and are excluded.}",
        "  \\label{tab:sensitivity-emergent}",
        "  \\begin{tabular}{lcccccccc}",
        "    \\toprule",
        "    \\multirow{2}{*}{Emergent $\\eta_c$ bin} & "
        "\\multirow{2}{*}{$n$} & "
        "\\multicolumn{3}{c}{Service load ($M/M_p\\le 0.4$)} & & "
        "\\multicolumn{3}{c}{Extended elastic ($M/M_p\\le 0.6$)} \\\\",
        "    \\cmidrule(lr){3-5}\\cmidrule(lr){7-9}",
        "     & & Section \\% & Beam$_{\\delta}$ \\% & Beam$_{\\varphi}$ \\% & &"
        " Section \\% & Beam$_{\\delta}$ \\% & Beam$_{\\varphi}$ \\% \\\\",
        "    \\midrule",
        *rows,
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        ""])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    rows = pd.read_parquet(REPORT_DIR / "sensitivity_rows.parquet")
    secs = pd.read_csv(REPORT_DIR / "sensitivity_sections.csv")
    print(f"[load] {len(secs)} sections, {len(rows)} rows")

    # (1) correlations
    r_target = float(secs.eta_c_target.corr(secs.eta_c_emergent))
    r_layout = float(secs.eta_c_layout.corr(secs.eta_c_emergent))
    rho_target = float(secs.eta_c_target.corr(secs.eta_c_emergent,
                                              method="spearman"))
    rho_layout = float(secs.eta_c_layout.corr(secs.eta_c_emergent,
                                              method="spearman"))
    print(f"[corr] Pearson r(target, emergent) = {r_target:.4f} "
          f"(Spearman {rho_target:.4f})")
    print(f"[corr] Pearson r(layout, emergent) = {r_layout:.4f} "
          f"(Spearman {rho_layout:.4f})")

    # (2) reassignment
    secs = reassign_by_emergent(secs)
    n_clipped = int((secs.eta_c_emergent > 1.0).sum())
    n_dropped = int((secs.eta_bin_emergent == "").sum())
    assert n_dropped == int((secs.eta_c_emergent < ss.ETA_BINS[0][0]).sum())
    n_target = {lab: int((secs.eta_bin_target == lab).sum())
                for lab in ss.BIN_LABELS}
    n_em = {lab: int((secs.eta_bin_emergent == lab).sum())
            for lab in ss.BIN_LABELS}
    print(f"[rebin] clipped above 1.0 into 90-100%: {n_clipped}; "
          f"dropped below 0.25: {n_dropped}")
    print("[rebin] sections per emergent bin: "
          + ", ".join(f"{lab}={n_em[lab]}" for lab in ss.BIN_LABELS))
    for lab in ss.BIN_LABELS:
        if n_em[lab] < 20:
            print(f"[warn] emergent bin {lab} has only {n_em[lab]} "
                  "sections (< 20)")

    # (3) crosstab
    ct = crosstab(secs)
    print("\n=== crosstab: target bin (rows) vs emergent bin (cols) ===")
    print(ct.to_string())

    # (4) identical summarise() on both assignments
    secs_target = secs.copy()
    secs_target["eta_bin"] = secs_target["eta_bin_target"]
    secs_em = secs[secs.eta_bin_emergent != ""].copy()
    secs_em["eta_bin"] = secs_em["eta_bin_emergent"]
    summ_target = ss.summarise(rows, secs_target)
    summ_em = ss.summarise(rows, secs_em)

    wide = wide_table(summ_target, summ_em, n_target, n_em)
    show = ["eta_bin", "regime", "n_sections_target", "n_sections_emergent"]
    for m in METRICS:
        show += [f"{m}_mean_target", f"{m}_mean_emergent", f"{m}_shift"]
    show += ["gap_sec_minus_defl_target", "gap_sec_minus_defl_emergent"]
    print("\n=== bin means (%), target vs emergent assignment ===")
    with pd.option_context("display.width", 250, "display.precision", 2):
        print(wide[show].to_string(index=False))

    # (5) outputs
    csv_path = REPORT_DIR / "rebin_by_emergent.csv"
    md_path = REPORT_DIR / "rebin_by_emergent.md"
    wide.to_csv(csv_path, index=False)
    md_path.write_text(to_markdown(wide, ct, r_target, r_layout,
                                   rho_target, rho_layout, n_dropped,
                                   n_clipped, n_em))
    TEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEX_PATH.write_text(to_latex(summ_em, n_em, n_dropped))
    print(f"\n[out] {csv_path}\n[out] {md_path}\n[out] {TEX_PATH}")


if __name__ == "__main__":
    main()
