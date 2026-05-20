#!/usr/bin/env python
"""XGBoost baseline only (no torch). Runs fast on subsampled data."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.normalize import FEATURE_COLUMNS, SECTION_TYPES

COMPARE_TARGETS = ["neutral_axis_in", "curvature_1_per_in"]


def split_by_sample(df, fracs, seed):
    ids = df["sample_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(fracs["train"] * n))
    n_val = int(round(fracs["val"] * n))
    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:n_train + n_val])
    test_ids = set(ids[n_train + n_val:])
    return (
        df[df["sample_id"].isin(train_ids)].reset_index(drop=True),
        df[df["sample_id"].isin(val_ids)].reset_index(drop=True),
        df[df["sample_id"].isin(test_ids)].reset_index(drop=True),
    )


def build_X(df):
    cont = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    onehot = np.zeros((len(df), len(SECTION_TYPES)), dtype=np.float32)
    for k, s in enumerate(SECTION_TYPES):
        onehot[:, k] = (df["section_type"].to_numpy() == s).astype(np.float32)
    return np.concatenate([cont, onehot], axis=1)


def metrics(y, yp):
    err = yp - y
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(y), 1e-12))) * 100.0
    return {"r2": r2, "rmse": rmse, "mape_pct": mape}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default="configs/training.yaml")
    args = p.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(Path(args.config).read_text())
    print(f"[xgb] loading {args.data} ...", flush=True)
    df = pd.read_parquet(args.data)
    print(f"[xgb] loaded {len(df)} rows; splitting by sample id...", flush=True)
    df_tr, df_val, df_te = split_by_sample(df, cfg["splits"], int(cfg["seed"]))
    print(f"[xgb] train={len(df_tr)} val={len(df_val)} test={len(df_te)}", flush=True)

    # Compute-budget-matched: subsample the train split to 200k rows, the
    # same budget and the same RNG draw used by the residual/plain MLP
    # baselines, so the three baselines are directly comparable.
    n_sub = 200_000
    if len(df_tr) > n_sub:
        sub_idx = np.random.default_rng(int(cfg["seed"])).choice(
            len(df_tr), size=n_sub, replace=False)
        df_tr = df_tr.iloc[sub_idx].reset_index(drop=True)
    print(f"[xgb] subsampled train -> {len(df_tr)} rows", flush=True)

    X_tr = build_X(df_tr); X_val = build_X(df_val); X_te = build_X(df_te)
    results = {}
    total_time = 0.0
    for j, target in enumerate(COMPARE_TARGETS):
        print(f"[xgb] {target}: fitting...", flush=True)
        t0 = time.time()
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            tree_method="hist",
            n_jobs=1,
            random_state=int(cfg["seed"]) + j,
            verbosity=0,
        )
        model.fit(X_tr, df_tr[target].to_numpy(dtype=np.float32))
        elapsed = time.time() - t0
        total_time += elapsed
        yp = model.predict(X_te)
        y = df_te[target].to_numpy(dtype=np.float32)
        m = metrics(y, yp)
        m["train_time_s"] = elapsed
        results[target] = m
        print(f"[xgb] {target}: r2={m['r2']:.4f}  rmse={m['rmse']:.4g}  "
              f"mape={m['mape_pct']:.2f}%  ({elapsed:.1f}s)", flush=True)

    (out_dir / "xgboost_metrics.json").write_text(json.dumps(results, indent=2))
    print(f"[xgb] total time {total_time:.1f}s; results -> "
          f"{out_dir / 'xgboost_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
