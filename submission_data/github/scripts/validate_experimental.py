#!/usr/bin/env python
"""Run Tier-3 experimental validation: surrogate vs published beam test data.

Usage:
    python scripts/validate_experimental.py \\
        --checkpoint checkpoints/full_300/best.pt \\
        --csv data/experimental/literature_tests.csv \\
        --out reports/tier3/

The CSV must follow the schema documented in
``src/validation/experimental.py``. The shipped CSV contains placeholder
rows; replace them with digitised values from Oehlers & Bradford (1995)
and Nie & Cai (2003) before drawing conclusions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.inference import SurrogatePredictor
from src.validation.experimental import (
    compare_surrogate_to_experiment,
    load_experimental_csv,
    summarise,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] checkpoint = {args.checkpoint}")
    predictor = SurrogatePredictor.load(args.checkpoint)
    print(f"[load] experimental data = {args.csv}")
    df = load_experimental_csv(args.csv)
    print(f"[compare] {len(df)} test points across {df['source'].nunique()} sources")

    comparison = compare_surrogate_to_experiment(predictor, df)
    summary = summarise(comparison)

    comparison.to_csv(out_dir / "tier3_comparison.csv", index=False)
    summary.to_csv(out_dir / "tier3_summary.csv", index=False)

    print("\n[Tier-3 summary]")
    print(summary.to_string(index=False))
    print()
    print(f"[done] wrote {out_dir / 'tier3_comparison.csv'}")
    print(f"[done] wrote {out_dir / 'tier3_summary.csv'}")


if __name__ == "__main__":
    main()
