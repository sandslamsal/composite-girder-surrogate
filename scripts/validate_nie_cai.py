#!/usr/bin/env python
"""Apply Nie & Cai (2003) effective-rigidity formula to the LHS-sampled
dataset and compare against AASHTO and OpenSeesPy.

Usage:
    python scripts/validate_nie_cai.py \\
        --data data/raw/full_50k.parquet \\
        --aashto reports/aashto_full/aashto_comparison.parquet \\
        --out reports/niecai/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.validation.nie_cai import apply_nie_cai


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True,
                   help="Full LHS dataset (e.g. data/raw/full_50k.parquet)")
    p.add_argument("--aashto", required=True,
                   help="AASHTO comparison parquet from validate_aashto.py")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] full dataset = {args.data}")
    full = pd.read_parquet(args.data)
    print(f"  rows: {len(full):,}, samples: {full['sample_id'].nunique():,}")

    print(f"[load] AASHTO comparison = {args.aashto}")
    aashto = pd.read_parquet(args.aashto)
    print(f"  rows: {len(aashto):,}")

    # Pull per-sample geometry from the full dataset (use step 0 since
    # geometry is constant per sample). Materials, span, deck dims, etc.
    geom_cols = [
        "sample_id", "span_in", "deck_thickness_in", "deck_width_in",
        "fc_deck_ksi", "fy_ksi", "steel_depth_in", "flange_width_in",
        "flange_thickness_in", "web_thickness_in",
        "shear_stud_stiffness_ratio",
    ]
    geom = full.loc[full["step_index"] == 0, geom_cols].drop_duplicates("sample_id")
    print(f"[geom] {len(geom):,} unique samples")

    # Join with AASHTO comparison rows
    df = aashto.merge(geom, on="sample_id", how="left", validate="m:1")
    df = df.dropna(subset=geom_cols[1:])
    print(f"[merge] {len(df):,} rows after join")

    # Compute Nie & Cai
    print("[compute] Nie & Cai effective rigidity...")
    nc = apply_nie_cai(df)

    # Comparison columns
    # phi_aashto already there. phi_opensees already there.
    # phi_niecai computed.
    nc["niecai_error_pct_vs_opensees"] = 100.0 * (
        nc["phi_niecai_1_per_in"] - nc["phi_opensees_1_per_in"]
    ) / nc["phi_opensees_1_per_in"]
    nc["aashto_error_pct_vs_opensees"] = nc["phi_error_pct"]   # already this convention

    # Save full-row output
    out_parquet = out_dir / "niecai_comparison.parquet"
    nc.to_parquet(out_parquet, index=False)
    print(f"[write] {out_parquet}")

    # Bin-wise summary
    bins = ["25-50%", "50-70%", "70-90%", "90-100%"]
    rows = []
    for b in bins:
        sub = nc[nc["eta_bin"] == b]
        rows.append({
            "eta_bin": b,
            "n_rows": len(sub),
            "aashto_err_mean_pct": sub["aashto_error_pct_vs_opensees"].mean(),
            "aashto_err_median_pct": sub["aashto_error_pct_vs_opensees"].median(),
            "niecai_err_mean_pct": sub["niecai_error_pct_vs_opensees"].mean(),
            "niecai_err_median_pct": sub["niecai_error_pct_vs_opensees"].median(),
            "xi_mean": sub["xi_slip_correction"].mean(),
            "xi_median": sub["xi_slip_correction"].median(),
        })
    summary = pd.DataFrame(rows)
    out_summary = out_dir / "niecai_summary.csv"
    summary.to_csv(out_summary, index=False)
    print(f"[write] {out_summary}")
    print()
    print("[Nie & Cai vs AASHTO vs OpenSeesPy summary]")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
