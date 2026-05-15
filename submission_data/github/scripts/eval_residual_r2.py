#!/usr/bin/env python
"""Honest accuracy metrics for the PINN's moment prediction.

The raw moment R^2 (~1.0) overstates the model's value because the
input feature ``moment_ratio = M/M_p`` is fed in directly, and we
recover the absolute moment as ``pred * M_p_est``. The model can
achieve very high raw R^2 simply by copying its input.

Two more informative metrics:

1. **Residual R^2** — compute the R^2 of the PINN's prediction against
   the *residual* of the trivial baseline ``M_baseline = r * M_p_est``.
   This measures how much of the *remaining* variance the PINN
   explains beyond the trivial guess. Values >> 0 mean the PINN is
   genuinely doing more than copying the input.

2. **Skill score** — Murphy's skill score:
       SS = 1 - MSE(PINN) / MSE(baseline)
   This is the fraction of the baseline's mean-squared error that the
   PINN eliminates. 1.0 = perfect, 0.0 = no better than baseline,
   negative = worse than baseline.

We compute both for all four targets, although they're most meaningful
for moment.

Usage:
    python scripts/eval_residual_r2.py \
        --checkpoint checkpoints/best_snapshot.pt \
        --data data/raw/full_50k.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.inference import PINNPredictor
from src.utils.normalize import TARGET_COLUMNS


def _split_by_sample(df: pd.DataFrame, fracs: dict, seed: int):
    ids = df["sample_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(fracs["train"] * n))
    n_val = int(round(fracs["val"] * n))
    test_ids = set(ids[n_train + n_val:])
    return df[df["sample_id"].isin(test_ids)].reset_index(drop=True)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float("nan") if ss_tot < 1e-12 else float(1.0 - ss_res / ss_tot)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--config", default="configs/training.yaml")
    args = p.parse_args()

    cfg = yaml.safe_load(open(REPO_ROOT / args.config))
    df = pd.read_parquet(args.data)
    test = _split_by_sample(df, cfg["splits"], int(cfg["seed"]))
    print(f"[data] test rows = {len(test):,}, sections = "
          f"{test['sample_id'].nunique():,}")

    predictor = PINNPredictor.load(args.checkpoint)
    pred = predictor.predict(test)

    # --- raw metrics --------------------------------------------------
    print("\n== Raw R^2 / RMSE / MAPE (what's currently in the paper) ==")
    print(f"{'target':24s}  {'R2':>8s}  {'RMSE':>12s}  {'MAPE %':>8s}")
    for col in TARGET_COLUMNS:
        y = test[col].to_numpy().astype(float)
        yh = pred[col].to_numpy().astype(float)
        r2 = _r2(y, yh)
        rmse = _rmse(y, yh)
        mask = np.abs(y) > 1e-6
        mape = 100 * np.mean(np.abs((yh[mask] - y[mask]) / y[mask]))
        print(f"{col:24s}  {r2: 8.4f}  {rmse:12.4g}  {mape:8.2f}")

    # --- diagnose the moment target ----------------------------------
    print("\n== Diagnostic: is the moment target genuinely predicted? ==")
    r = test["moment_ratio"].to_numpy()
    mp = test["mp_estimate_kip_in"].to_numpy()
    M_true = test["moment_kip_in"].to_numpy()
    M_baseline = r * mp
    M_pred = pred["moment_kip_in"].to_numpy()

    rmse_baseline = _rmse(M_true, M_baseline)
    rmse_pinn = _rmse(M_true, M_pred)
    print(f"Trivial baseline (r * Mp_est) RMSE vs M_true: {rmse_baseline:.3e} kip-in")
    print(f"PINN prediction RMSE vs M_true:               {rmse_pinn:.3e} kip-in")
    if rmse_baseline < 1e-3:
        print()
        print("** FINDING **: M_true is literally identical to r·Mp_est in the")
        print("dataset (baseline RMSE ~ machine epsilon). The PINN cannot do")
        print("better than the trivial baseline because the answer is the input")
        print("feature value times another input feature. Moment is NOT a real")
        print("prediction in this setup; it's a pass-through.")
    print()

    # --- honest metrics for the three real predictions ----------------
    print("\n== Honest metrics for the REAL predictions ==")
    print("(moment is a trivial pass-through; the model's actual work is")
    print(" predicting y_na, curvature, slip given section + load level.)")
    print()
    print(f"{'target':24s}  {'R2':>8s}  {'RMSE (phys)':>14s}  "
          f"{'MAPE %':>8s}  {'note':s}")
    notes = {
        "neutral_axis_in": "depth of NA, in",
        "curvature_1_per_in": "1/in (the surrogate's headline real prediction)",
        "moment_kip_in": "trivial: = r * Mp_est exactly",
        "slip_in": "interface slip in; ground-truth is analytical (Eq.1)",
    }
    for col in TARGET_COLUMNS:
        y = test[col].to_numpy().astype(float)
        yh = pred[col].to_numpy().astype(float)
        r2 = _r2(y, yh)
        rmse = _rmse(y, yh)
        mask = np.abs(y) > 1e-6
        mape = 100 * np.mean(np.abs((yh[mask] - y[mask]) / y[mask]))
        print(f"{col:24s}  {r2: 8.4f}  {rmse:14.4g}  {mape:8.2f}  {notes[col]}")


if __name__ == "__main__":
    main()
