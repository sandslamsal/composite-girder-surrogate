#!/usr/bin/env python
"""Stratified deck-reinforcement subset for the AASHTO comparison.

The released dataset omits deck longitudinal reinforcement from the
section model. This script quantifies the effect of that omission: it
draws a stratified subset of sections (even coverage of the four
composite-action bins) and runs each one through the moment-curvature
analysis at several total deck-reinforcement ratios, including the
no-reinforcement baseline.

For every reinforcement ratio it reports the extended-elastic
($M/M_p \\le 0.6$) AASHTO stiffness deviation by composite-action bin,
so the reinforcement-inclusive deviation can be compared directly with
the no-reinforcement values reported in the paper.

The no-reinforcement column is run here as well (rather than reusing the
published numbers) so that the reinforcement effect is measured as a
within-run delta, independent of the OpenSeesPy version used.

Usage:
    python scripts/run_rebar_subset.py --per-bin 150 --out reports/rebar/
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data_generation.lhs_sampler import SectionParams, load_config, sample
from src.data_generation.moment_curvature import analyze
from src.data_generation.generate_dataset import to_rows
from src.validation.aashto import compare_aashto_vs_opensees

ETA_BINS = [(0.25, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0001)]
RHO_VALUES = [0.0, 0.004, 0.007, 0.010]


def _bin_index(eta: float) -> int:
    for i, (lo, hi) in enumerate(ETA_BINS):
        if lo <= eta < hi:
            return i
    return -1


def _rows_for(params: SectionParams, n_steps: int, rho: float) -> list[dict]:
    """Run one section at reinforcement ratio ``rho`` and expand to rows."""
    result = analyze(params, n_steps=n_steps, deck_rho_long=rho)
    if not result.physically_valid:
        return []
    features = asdict(params)
    features["total_depth_in"] = result.section_info.total_depth_in
    features["mp_estimate_kip_in"] = result.section_info.plastic_moment_kip_in
    return to_rows(result, features)


def _deviation_by_bin(rows: list[dict], moment_ratio_max: float,
                      n_boot: int = 500, seed: int = 0) -> dict:
    """Bin-mean AASHTO stiffness over-prediction Delta (percent, positive).

    The point estimate is the row-level (per-curvature-point) bin mean,
    matching the methodology of the main AASHTO comparison table. Because
    rows are correlated within a section, the standard error is computed
    by a section-clustered bootstrap: sections are resampled with
    replacement and the row-level mean recomputed, ``n_boot`` times.
    """
    df = pd.DataFrame(rows)
    comp = compare_aashto_vs_opensees(df, moment_ratio_max=moment_ratio_max)
    comp = comp.copy()
    # phi_error_pct < 0 means AASHTO is stiffer; Delta = stiffness
    # over-prediction = -phi_error_pct.
    comp["delta_pct"] = -comp["phi_error_pct"]
    rng = np.random.default_rng(seed)
    out = {}
    for b, g in comp.groupby("eta_bin"):
        mean = float(g["delta_pct"].mean())
        median = float(g["delta_pct"].median())
        sec_arrays = [sub["delta_pct"].to_numpy()
                      for _, sub in g.groupby("sample_id")]
        n_sec = len(sec_arrays)
        boot = np.empty(n_boot)
        for k in range(n_boot):
            idx = rng.integers(0, n_sec, n_sec)
            boot[k] = np.concatenate([sec_arrays[j] for j in idx]).mean()
        out[b] = {
            "delta_mean_pct": mean,
            "delta_se_pct": float(boot.std(ddof=1)),
            "delta_median_pct": median,
            "n_sections": n_sec,
            "n_rows": int(len(g)),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data_gen.yaml")
    p.add_argument("--per-bin", type=int, default=150,
                   help="target sections per composite-action bin")
    p.add_argument("--pool", type=int, default=3000,
                   help="LHS pool size to stratify from")
    p.add_argument("--seed", type=int, default=20260514)
    p.add_argument("--out", default="reports/rebar")
    args = p.parse_args()

    cfg = load_config(args.config)
    n_steps = cfg["analysis"]["n_curvature_steps"]
    pool = sample(cfg, n=args.pool, seed=args.seed)

    # Stratify: keep up to per-bin sections in each composite-action bin.
    kept: list[SectionParams] = []
    counts = [0, 0, 0, 0]
    for prm in pool:
        b = _bin_index(prm.composite_action)
        if b < 0 or counts[b] >= args.per_bin:
            continue
        kept.append(prm)
        counts[b] += 1
        if all(c >= args.per_bin for c in counts):
            break
    print(f"[stratify] kept {len(kept)} sections, per-bin counts = {counts}")

    # Run every kept section at each reinforcement ratio.
    rows_by_rho: dict[float, list[dict]] = {rho: [] for rho in RHO_VALUES}
    n_valid = 0
    for i, prm in enumerate(kept):
        per_section = {}
        ok = True
        for rho in RHO_VALUES:
            r = _rows_for(prm, n_steps, rho)
            if not r:
                ok = False
                break
            per_section[rho] = r
        if not ok:
            continue
        n_valid += 1
        for rho in RHO_VALUES:
            rows_by_rho[rho].extend(per_section[rho])
        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{len(kept)}] {n_valid} valid")
    print(f"[run] {n_valid} sections valid at all reinforcement ratios")

    # Summarise extended-elastic (M/Mp <= 0.6) and service-load (<= 0.4).
    report = {"n_sections": n_valid, "per_bin_target": args.per_bin,
              "seed": args.seed, "rho_values": RHO_VALUES, "regimes": {}}
    for label, mrm in (("service_load", 0.4), ("extended_elastic", 0.6)):
        report["regimes"][label] = {
            f"rho_{rho}": _deviation_by_bin(rows_by_rho[rho], mrm)
            for rho in RHO_VALUES
        }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "rebar_summary.json", "w") as f:
        json.dump(report, f, indent=2)

    # Console table for the extended-elastic regime.
    for regime in ("service_load", "extended_elastic"):
        print(f"\n{regime} AASHTO stiffness over-prediction Delta (%), "
              f"bin-mean +/- SE:")
        hdr = f"  {'eta_c bin':<12}" + "".join(
            f"{('rho=' + format(r, '.3f')):>16}" for r in RHO_VALUES)
        print(hdr)
        rr = report["regimes"][regime]
        bins = sorted(rr[f"rho_{RHO_VALUES[0]}"].keys())
        for b in bins:
            line = f"  {b:<12}"
            for rho in RHO_VALUES:
                cell = rr[f"rho_{rho}"][b]
                line += f"{cell['delta_mean_pct']:>10.1f}+-{cell['delta_se_pct']:<4.1f}"
            print(line)
    print(f"\n[written] {out_dir / 'rebar_summary.json'}")


if __name__ == "__main__":
    main()
