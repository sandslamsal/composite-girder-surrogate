#!/usr/bin/env python
"""Run the AASHTO transformed-section comparison against an OpenSees parquet.

Outputs:
  - <out_dir>/aashto_comparison.parquet  per-row comparison table
  - <out_dir>/aashto_summary.csv         eta-bin summary statistics
  - printed eta-bin summary

Usage:
    python scripts/validate_aashto.py --data data/raw/full_50k.parquet \\
        --out reports/aashto_v1/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.validation.aashto import compare_aashto_vs_opensees, summarise_by_eta_bin


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="OpenSees parquet file")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument(
        "--moment-ratio-max", type=float, default=0.6,
        help="restrict comparison to rows with M/Mp <= this (elastic regime)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    data_path = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_path)
    print(f"[data] {data_path.name}: {len(df)} rows, "
          f"{df['sample_id'].nunique()} sections")

    comparison = compare_aashto_vs_opensees(
        df, moment_ratio_max=args.moment_ratio_max
    )
    print(f"[compare] {len(comparison)} elastic-regime rows "
          f"(moment_ratio <= {args.moment_ratio_max})")

    comparison.to_parquet(out_dir / "aashto_comparison.parquet", index=False)

    summary = summarise_by_eta_bin(comparison)
    summary.to_csv(out_dir / "aashto_summary.csv", index=False)

    print()
    print(summary.to_string(index=False))
    print()
    print(f"[done] wrote {out_dir / 'aashto_comparison.parquet'}")
    print(f"[done] wrote {out_dir / 'aashto_summary.csv'}")


if __name__ == "__main__":
    main()
