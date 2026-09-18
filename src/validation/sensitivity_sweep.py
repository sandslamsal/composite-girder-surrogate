"""Scaled section-vs-beam sensitivity study for the journal revision.

For ~400 Paper-1 sections stratified across the four eta_c bins
(25-50, 50-70, 70-90, 90-100 %), runs the FIXED beam-level model
(discrete shear connectors, '-noCentroid' fiber sections) with a
connector layout matched to the section's eta_c, and compares BOTH
partial-composite representations against the AASHTO transformed-
section stiffness at the SAME (section, moment) points:

* Delta_section : from reports/aashto_full/aashto_comparison.parquet
  (the width-scaled section model already in the manuscript), where
  Delta = 1 - EI_model/EI_AASHTO = -phi_error_pct/100.
* Delta_beam    : from the beam model, with EI recovered two ways --
  curvature-based (M/phi at midspan; contaminated by the point-load
  slip boundary layer for short deep spans) and deflection-based
  (P L^3 / 48 delta; a span-integrated stiffness, less contaminated).

Both regimes reported: service (M/Mp <= 0.4) and extended-elastic
(M/Mp <= 0.6), using the dataset's moment_ratio so the regime
membership is identical to Table tab:aashto in the manuscript.

The beam model imposes NOTHING about stiffness reduction: eta_c
emerges from connector equilibrium. Agreement in trend/magnitude with
the width-scaling model is therefore evidence that the section-level
assumption does not predefine the reported AASHTO deviations.

Outputs
-------
reports/section_vs_beam_revision/sensitivity_rows.parquet    per-row
reports/section_vs_beam_revision/sensitivity_sections.csv    per-section
reports/section_vs_beam_revision/sensitivity_summary.csv     per-bin
reports/section_vs_beam_revision/sensitivity_summary.md      markdown
paper/revision_1/submission/sources/tables/tab_sensitivity.tex  booktabs
paper/revision_1/figures/fig_section_vs_beam.png             figure

Usage
-----
/opt/anaconda3/envs/ops_x86/bin/python \
    src/validation/sensitivity_sweep.py \
    [--n-per-bin 100] [--workers 7] [--seed 20260802]
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
import traceback
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

ETA_BINS = [(0.25, 0.50), (0.50, 0.70), (0.70, 0.90), (0.90, 1.001)]
BIN_LABELS = ["25-50%", "50-70%", "70-90%", "90-100%"]
_E_STEEL_KSI = 29_000.0
_EC_COEF_KSI = 57.0 * math.sqrt(1000.0)

PARAM_COLS = [
    "span_in", "deck_thickness_in", "deck_width_in", "fc_deck_ksi",
    "fy_ksi", "steel_depth_in", "flange_width_in", "flange_thickness_in",
    "web_thickness_in", "section_type", "shear_stud_stiffness_ratio",
    "composite_action", "total_depth_in", "mp_estimate_kip_in",
]


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------

def _rising(m: np.ndarray, *others: np.ndarray):
    """Strictly increasing prefix of the moment history (to its peak)."""
    if m.size == 0:
        return (m,) + others
    i_peak = int(np.argmax(m))
    mm = m[: i_peak + 1]
    keep = np.zeros(mm.size, dtype=bool)
    run = -np.inf
    for k, v in enumerate(mm):
        if v > run:
            keep[k] = True
            run = v
    return (mm[keep],) + tuple(o[: i_peak + 1][keep] for o in others)


def run_one(payload: dict) -> dict:
    """Run the fixed beam model for one section and evaluate it at the
    section's aashto-comparison moments. Runs in a worker process."""
    from src.beam_model import analyze_beam, BeamParams
    from src.validation.section_vs_beam import (build_beam_params,
                                                connector_layout_for_eta)
    sid = payload["sample_id"]
    row = pd.Series(payload["params"])
    out = {"sample_id": sid, "eta_c_target": float(row.composite_action),
           "ok": False, "error": ""}
    try:
        p = build_beam_params(row)
        p = connector_layout_for_eta(p, float(row.composite_action))
        t0 = time.time()
        res = analyze_beam(p, n_steps=80)          # slip cap 0.30 in default
        out["wall_s"] = time.time() - t0
        if res.converged_steps < 5:
            out["error"] = "too few converged steps"
            return out
        m_r, phi_r, dfl_r = _rising(res.moment, res.curvature,
                                    res.deflection_in)
        if m_r.size < 3:
            out["error"] = "degenerate rising branch"
            return out

        i_tr = payload["i_tr"]
        ei_aashto = _E_STEEL_KSI * i_tr
        L = float(row.span_in)
        rows = []
        for m_row, mr, phi_a, dsec in zip(payload["m"], payload["mr"],
                                          payload["phi_aashto"],
                                          payload["delta_sec_pct"]):
            rec = {"sample_id": sid, "moment_kip_in": m_row,
                   "moment_ratio": mr, "delta_sec_pct": dsec,
                   "in_beam_range": bool(m_row <= m_r[-1])}
            if m_row <= m_r[-1]:
                phi_b = float(np.interp(m_row, m_r, phi_r))
                dfl_b = float(np.interp(m_row, m_r, dfl_r))
                rec["phi_beam"] = phi_b
                # curvature-based EI -> Delta = 1 - EI_beam/EI_AASHTO
                if phi_b > 0:
                    rec["delta_beam_curv_pct"] = 100.0 * (
                        1.0 - (m_row / phi_b) / ei_aashto)
                # deflection-based EI = P L^3/48d = m L^2 / (12 d)
                if dfl_b > 0:
                    ei_defl = m_row * L * L / (12.0 * dfl_b)
                    rec["delta_beam_defl_pct"] = 100.0 * (
                        1.0 - ei_defl / ei_aashto)
                # direct phi deviation beam vs section model
                phi_os = phi_a / (1.0 - dsec / 100.0) if dsec < 100.0 else np.nan
                if np.isfinite(phi_os) and phi_os > 0:
                    rec["phi_beam_vs_sec_pct"] = 100.0 * (phi_b - phi_os) / phi_os
            rows.append(rec)

        # Newmark interaction parameter alpha*L and slenderness L/d
        ec = _EC_COEF_KSI * math.sqrt(max(row.fc_deck_ksi, 1e-6))
        a_deck = row.deck_width_in * row.deck_thickness_in
        a_steel = (2.0 * row.flange_width_in * row.flange_thickness_in
                   + (row.steel_depth_in - 2.0 * row.flange_thickness_in)
                   * row.web_thickness_in)
        i_deck = row.deck_width_in * row.deck_thickness_in ** 3 / 12.0
        bf, tf, ds, tw = (row.flange_width_in, row.flange_thickness_in,
                          row.steel_depth_in, row.web_thickness_in)
        web_h = ds - 2.0 * tf
        i_steel = (tw * web_h ** 3 / 12.0
                   + 2.0 * (bf * tf ** 3 / 12.0
                            + bf * tf * ((ds - tf) / 2.0) ** 2))
        z = 0.5 * row.deck_thickness_in + 0.5 * row.steel_depth_in
        from src.beam_model import stud_stiffness_kip_per_in
        k_len = (stud_stiffness_kip_per_in(p.stud_diameter_in, p.ks_ratio)
                 * p.n_studs_per_row / p.stud_pitch_in)
        alpha2 = k_len * (1.0 / (ec * a_deck) + 1.0 / (_E_STEEL_KSI * a_steel)
                          + z * z / (ec * i_deck + _E_STEEL_KSI * i_steel))
        out.update({
            "ok": True, "rows": rows,
            "eta_c_layout": float(res.eta_c),
            "eta_c_emergent": float(res.eta_c_emergent),
            "slip_at_mp040_in": float(res.slip_at_mp040_in),
            "slip_at_mp060_in": float(res.slip_at_mp060_in),
            "max_slip_in": float(res.max_slip_in.max()),
            "max_m_over_mp_beam": float(res.moment.max()
                                        / max(res.mp_estimate_kip_in, 1e-9)),
            "max_m_over_mp_dataset": float(m_r[-1]
                                           / max(row.mp_estimate_kip_in, 1e-9)),
            "converged_steps": int(res.converged_steps),
            "alpha_L": float(math.sqrt(max(alpha2, 0.0)) * L),
            "L_over_d": float(L / max(row.total_depth_in, 1e-9)),
            "stud_pitch_in": float(p.stud_pitch_in),
            "n_studs_per_row": int(p.n_studs_per_row),
            "stud_diameter_in": float(p.stud_diameter_in),
        })
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()
    return out


# ---------------------------------------------------------------------------
# sampling and payload assembly
# ---------------------------------------------------------------------------

def build_payloads(n_per_bin: int, seed: int) -> list[dict]:
    full = pd.read_parquet(
        REPO_ROOT / "data/raw/full_50k.parquet",
        columns=["sample_id", "step_index"] + PARAM_COLS)
    first = full[full.step_index == 0].set_index("sample_id")
    aa = pd.read_parquet(
        REPO_ROOT / "reports/aashto_full/aashto_comparison.parquet")
    valid = first.index.intersection(aa.sample_id.unique())
    first = first.loc[valid]
    first = first[first.section_type.isin(("W", "plate"))]

    rng = np.random.default_rng(seed)
    chosen = []
    for lo, hi in ETA_BINS:
        elig = first[(first.composite_action >= lo)
                     & (first.composite_action < hi)].index.to_numpy()
        take = min(n_per_bin, elig.size)
        chosen.extend(rng.choice(elig, size=take, replace=False).tolist())

    aa_g = {sid: g for sid, g in
            aa[aa.sample_id.isin(chosen)].groupby("sample_id")}
    payloads = []
    for sid in chosen:
        g = aa_g[sid].sort_values("moment_ratio")
        payloads.append({
            "sample_id": int(sid),
            "params": first.loc[sid, PARAM_COLS].to_dict(),
            "i_tr": float(g.transformed_inertia_in4.iloc[0]),
            "m": g.moment_kip_in.to_numpy(),
            "mr": g.moment_ratio.to_numpy(),
            "phi_aashto": g.phi_aashto_1_per_in.to_numpy(),
            # Delta = 1 - EI_OS/EI_AASHTO = -phi_error_pct (manuscript conv.)
            "delta_sec_pct": (-g.phi_error_pct).to_numpy(),
        })
    return payloads


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def eta_bin_label(eta: float) -> str:
    for (lo, hi), lab in zip(ETA_BINS, BIN_LABELS):
        if lo <= eta < hi:
            return lab
    return ""


def cluster_boot_se(df: pd.DataFrame, col: str, n_boot: int = 500,
                    seed: int = 0) -> float:
    """Section-clustered bootstrap SE of the row-pooled mean of ``col``."""
    sub = df.dropna(subset=[col])
    if sub.empty:
        return float("nan")
    groups = {sid: g[col].to_numpy() for sid, g in sub.groupby("sample_id")}
    sids = np.array(list(groups))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(sids, size=sids.size, replace=True)
        vals = np.concatenate([groups[s] for s in pick])
        means[b] = vals.mean()
    return float(means.std(ddof=1))


def summarise(rows: pd.DataFrame, secs: pd.DataFrame) -> pd.DataFrame:
    """Per (bin, regime, metric) stats.

    ``section`` pools ALL dataset rows in the regime (the manuscript's
    Table tab:aashto population). ``section_matched`` restricts the
    section-model rows to those the beam sweep actually covered
    (in_beam_range with a finite deflection-based Delta), which is the
    apples-to-apples population for the section-vs-beam comparison --
    beams that hit the 0.30-in slip cap early never reach the highest
    moment ratios, and those high-M/Mp rows carry the largest section
    Delta."""
    metrics = [("section", "delta_sec_pct", False),
               ("section_matched", "delta_sec_pct", True),
               ("beam_curv", "delta_beam_curv_pct", True),
               ("beam_defl", "delta_beam_defl_pct", True),
               ("phi_beam_vs_sec", "phi_beam_vs_sec_pct", True)]
    regimes = [("service", 0.4), ("extended", 0.6)]
    recs = []
    for lab in BIN_LABELS:
        bin_sids = secs[secs.eta_bin == lab].sample_id
        sub_bin = rows[rows.sample_id.isin(bin_sids)]
        for rname, mr_max in regimes:
            sub_all = sub_bin[(sub_bin.moment_ratio > 0)
                              & (sub_bin.moment_ratio <= mr_max)]
            matched = sub_all[sub_all.delta_beam_defl_pct.notna()]
            for mname, col, use_matched in metrics:
                sub = matched if use_matched else sub_all
                v = sub[col].dropna()
                recs.append({
                    "eta_bin": lab, "regime": rname, "metric": mname,
                    "n_sections": int(sub.dropna(subset=[col])
                                      .sample_id.nunique()),
                    "n_rows": int(len(v)),
                    "mean_pct": float(v.mean()) if len(v) else np.nan,
                    "median_pct": float(v.median()) if len(v) else np.nan,
                    "se_pct": cluster_boot_se(sub, col),
                    "coverage": float(sub_all.in_beam_range.mean())
                                if len(sub_all) else np.nan,
                })
    return pd.DataFrame(recs)


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

def to_markdown_table(summary: pd.DataFrame) -> str:
    lines = ["| eta_c bin | n sect | regime | Delta section, all rows (%) | "
             "Delta section, matched rows (%) | "
             "Delta beam, curvature EI (%) | Delta beam, deflection EI (%) |",
             "|---|---|---|---|---|---|---|"]
    for lab in BIN_LABELS:
        for regime in ("service", "extended"):
            s = summary[(summary.eta_bin == lab)
                        & (summary.regime == regime)].set_index("metric")
            def cell(m):
                r = s.loc[m]
                return f"{r.mean_pct:+.1f} ± {r.se_pct:.1f} (med {r.median_pct:+.1f})"
            n_sec = int(s.loc["section_matched"].n_sections)
            lines.append(f"| {lab} | {n_sec} | {regime} | {cell('section')} | "
                         f"{cell('section_matched')} | "
                         f"{cell('beam_curv')} | {cell('beam_defl')} |")
    return "\n".join(lines)


def to_latex_table(summary: pd.DataFrame, n_sections: dict,
                   failure_note: str) -> str:
    def cell(lab, regime, metric):
        s = summary[(summary.eta_bin == lab) & (summary.regime == regime)
                    & (summary.metric == metric)].iloc[0]
        return f"${s.mean_pct:.1f} \\pm {s.se_pct:.1f}$"

    rows = []
    for lab, tex in zip(BIN_LABELS,
                        ["25--\\SI{50}{\\percent}", "50--\\SI{70}{\\percent}",
                         "70--\\SI{90}{\\percent}", "90--\\SI{100}{\\percent}"]):
        rows.append(
            f"    {tex} & {cell(lab,'service','section_matched')} & "
            f"{cell(lab,'service','beam_defl')} & "
            f"{cell(lab,'service','beam_curv')} & & "
            f"{cell(lab,'extended','section_matched')} & "
            f"{cell(lab,'extended','beam_defl')} & "
            f"{cell(lab,'extended','beam_curv')} \\\\")
    n_line = ("    $n$ (sections) & \\multicolumn{3}{c}{"
              + ", ".join(str(n_sections[l]) for l in BIN_LABELS)
              + " per bin} & & \\multicolumn{3}{c}{same sections} \\\\")
    return "\n".join([
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Sensitivity of the AASHTO stiffness deviation $\\Delta$ to",
        "    the partial-composite representation, by $\\eta_c$ bin and load",
        "    regime. Mean $\\pm$ section-clustered bootstrap standard error",
        "    (500 resamples)." + failure_note,
        "    }",
        "  \\label{tab:sensitivity}",
        "  \\begin{tabular}{lccccccc}",
        "    \\toprule",
        "    \\multirow{2}{*}{$\\eta_c$ bin} & "
        "\\multicolumn{3}{c}{Service load ($M/M_p\\le 0.4$)} & & "
        "\\multicolumn{3}{c}{Extended elastic ($M/M_p\\le 0.6$)} \\\\",
        "    \\cmidrule(lr){2-4}\\cmidrule(lr){6-8}",
        "     & Section \\% & Beam$_{\\delta}$ \\% & Beam$_{\\varphi}$ \\% & &"
        " Section \\% & Beam$_{\\delta}$ \\% & Beam$_{\\varphi}$ \\% \\\\",
        "    \\midrule",
        *rows,
        "    \\midrule",
        n_line,
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        ""])


def make_figure(summary: pd.DataFrame, out_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "paper_plotting", REPO_ROOT / "src/utils/plotting.py")
    plotting = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plotting)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plotting.apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(plotting.COL_SINGLE_IN, 3.0),
                             sharey=True)
    metric_order = [("section_matched",
                     "Section model (width-scaled $\\eta_c$)"),
                    ("beam_defl", "Beam model, deflection $EI$"),
                    ("beam_curv", "Beam model, curvature $EI$")]
    colors = plotting.COLORS[:3]
    width = 0.26
    x = np.arange(len(BIN_LABELS))
    for ax, (regime, title) in zip(
            axes, [("service", "Service ($M/M_p \\leq 0.4$)"),
                   ("extended", "Extended elastic ($M/M_p \\leq 0.6$)")]):
        for j, ((metric, label), c) in enumerate(zip(metric_order, colors)):
            s = summary[(summary.regime == regime)
                        & (summary.metric == metric)].set_index("eta_bin")
            means = [s.loc[l].mean_pct for l in BIN_LABELS]
            ses = [s.loc[l].se_pct for l in BIN_LABELS]
            ax.bar(x + (j - 1) * width, means, width * 0.92, yerr=ses,
                   capsize=2, color=c, edgecolor="white", linewidth=0.5,
                   error_kw={"lw": 0.8},
                   label=label if regime == "service" else None)
        ax.axhline(0.0, color="0.3", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([l.replace("%", "\\%") if False else l
                            for l in BIN_LABELS], fontsize=8)
        ax.set_xlabel("$\\eta_c$ bin")
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("$\\Delta$ vs AASHTO (%)")
    fig.legend(loc="upper center", ncol=3, fontsize=6.2, frameon=False,
               columnspacing=1.0, handlelength=1.4,
               bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    plotting.savefig(fig, out_path)
    print(f"[fig] {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-per-bin", type=int, default=100)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--from-cache", action="store_true",
                    help="Skip the beam runs; re-aggregate from the saved "
                         "sensitivity_rows.parquet / sensitivity_sections.csv")
    args = ap.parse_args()

    out_dir = REPO_ROOT / "reports/section_vs_beam_revision"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "run_meta.json"

    if args.from_cache:
        import json
        rows = pd.read_parquet(out_dir / "sensitivity_rows.parquet")
        secs = pd.read_csv(out_dir / "sensitivity_sections.csv")
        meta = (json.loads(meta_path.read_text())
                if meta_path.exists() else {"n_failed": 0, "n_total": len(secs)})
        n_failed, n_total = meta["n_failed"], meta["n_total"]
        failed = [None] * n_failed
        print(f"[cache] {len(secs)} sections re-aggregated "
              f"({n_failed}/{n_total} had failed)")
    else:
        payloads = build_payloads(args.n_per_bin, args.seed)
        print(f"[sample] {len(payloads)} sections "
              f"({args.n_per_bin}/bin, seed={args.seed})", flush=True)

        t0 = time.time()
        ctx = get_context("spawn")
        with ctx.Pool(args.workers) as pool:
            results = []
            for i, r in enumerate(pool.imap_unordered(run_one, payloads,
                                                      chunksize=4)):
                results.append(r)
                if (i + 1) % 25 == 0:
                    print(f"  [{i+1}/{len(payloads)}] "
                          f"{time.time()-t0:.0f}s elapsed", flush=True)
        dt = time.time() - t0
        ok = [r for r in results if r["ok"]]
        failed = [r for r in results if not r["ok"]]
        print(f"[run] {len(ok)} ok / {len(failed)} failed in {dt:.0f}s")
        for r in failed[:10]:
            print(f"  FAIL sid={r['sample_id']}: {r['error']}")

        rows = pd.DataFrame([rr for r in ok for rr in r["rows"]])
        sec_cols = [k for k in ok[0] if k not in ("rows", "ok", "error",
                                                  "traceback")]
        secs = pd.DataFrame([{k: r[k] for k in sec_cols} for r in ok])
        secs["eta_bin"] = secs.eta_c_target.map(eta_bin_label)
        rows.to_parquet(out_dir / "sensitivity_rows.parquet", index=False)
        secs.to_csv(out_dir / "sensitivity_sections.csv", index=False)
        import json
        n_failed, n_total = len(failed), len(payloads)
        meta_path.write_text(json.dumps(
            {"n_failed": n_failed, "n_total": n_total, "seed": args.seed,
             "n_per_bin": args.n_per_bin, "wall_s": dt}))

    summary = summarise(rows, secs)
    summary.to_csv(out_dir / "sensitivity_summary.csv", index=False)
    print(summary.to_string(index=False))

    md = to_markdown_table(summary)
    (out_dir / "sensitivity_summary.md").write_text(md + "\n")
    print(md)

    n_sections = {lab: int((secs.eta_bin == lab).sum()) for lab in BIN_LABELS}
    failure_note = (f" {n_failed} of {n_total} beam analyses "
                    "failed to converge and are excluded."
                    if n_failed else "")
    tex = to_latex_table(summary, n_sections, failure_note)
    tex_path = (REPO_ROOT
                / "paper/revision_1/submission/sources/tables/tab_sensitivity.tex")
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(tex)
    print(f"[tex] {tex_path}")

    make_figure(summary, REPO_ROOT
                / "paper/revision_1/figures/fig_section_vs_beam.png")

    # quick diagnostics for the write-up
    print("\n=== emergent eta_c vs target (per bin) ===")
    print(secs.groupby("eta_bin")[["eta_c_target", "eta_c_layout",
                                   "eta_c_emergent"]].mean().to_string())
    print("\n=== slip at fixed load levels (median, in) ===")
    print(secs.groupby("eta_bin")[["slip_at_mp040_in", "slip_at_mp060_in",
                                   "max_slip_in"]].median().to_string())
    print("\n=== beam-curve coverage: max M/Mp (dataset Mp) reached ===")
    print(secs.groupby("eta_bin")["max_m_over_mp_dataset"]
          .describe().to_string())

    # boundary-layer footprint: correlate per-section curv-vs-defl gap and
    # section-vs-beam gap with alpha*L and L/d
    per_sec = rows.merge(secs[["sample_id", "eta_bin", "alpha_L",
                               "L_over_d"]], on="sample_id")
    per_sec = per_sec[(per_sec.moment_ratio > 0)
                      & (per_sec.moment_ratio <= 0.6)]
    g = per_sec.groupby("sample_id").agg(
        gap_curv=("delta_beam_curv_pct", "mean"),
        gap_defl=("delta_beam_defl_pct", "mean"),
        dsec=("delta_sec_pct", "mean"),
        alpha_L=("alpha_L", "first"), L_over_d=("L_over_d", "first"))
    g["bl_gap"] = g.gap_curv - g.gap_defl          # boundary-layer footprint
    g["sb_gap_defl"] = g.gap_defl - g.dsec
    g["sb_gap_curv"] = g.gap_curv - g.dsec
    print("\n=== boundary-layer / representation-gap correlations ===")
    for a in ("bl_gap", "sb_gap_curv", "sb_gap_defl"):
        for b in ("alpha_L", "L_over_d"):
            r = g[[a, b]].dropna().corr().iloc[0, 1]
            print(f"  corr({a}, {b}) = {r:+.3f}")
    g.to_csv(out_dir / "sensitivity_gaps_per_section.csv")
    print(f"[done] outputs -> {out_dir}")


if __name__ == "__main__":
    main()
