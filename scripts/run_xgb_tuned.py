#!/usr/bin/env python
"""Optuna-tuned XGBoost baseline on the 200k compute-matched subsample
(no torch in this process — avoids OpenMP thread pool conflict).

Per-target TPE search (60 trials) maximising validation-split R2, then a
retrain of the best config and evaluation on the held-out test split.

Outputs:
  reports/baselines_revision/xgb_tuned_metrics.json
  reports/baselines_revision/xgb_tuned_trials_<target>.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.normalize import FEATURE_COLUMNS, SECTION_TYPES, TARGET_COLUMNS
from scripts.revision_common import (
    DATA_PATH, OUT_DIR, SEED, load_and_split, metrics,
)

N_TRIALS = 60
EARLY_STOP = 50
N_JOBS = max(1, (os.cpu_count() or 4) - 1)


def build_X(df):
    """Raw physical features + section-type one-hot (same as run_xgb_only)."""
    cont = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    onehot = np.zeros((len(df), len(SECTION_TYPES)), dtype=np.float32)
    for k, s in enumerate(SECTION_TYPES):
        onehot[:, k] = (df["section_type"].to_numpy() == s).astype(np.float32)
    return np.concatenate([cont, onehot], axis=1)


def make_model(params: dict, seed: int) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        tree_method="hist",
        n_jobs=N_JOBS,
        random_state=seed,
        verbosity=0,
        early_stopping_rounds=EARLY_STOP,
        eval_metric="rmse",
        **params,
    )


def r2_only(y, yh):
    y = np.asarray(y, dtype=np.float64)
    yh = np.asarray(yh, dtype=np.float64)
    return float(1.0 - np.sum((y - yh) ** 2)
                 / (np.sum((y - y.mean()) ** 2) + 1e-12))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DATA_PATH))
    p.add_argument("--out", default=str(OUT_DIR))
    p.add_argument("--trials", type=int, default=N_TRIALS)
    p.add_argument("--smoke", action="store_true",
                   help="2 trials on a 5k-row train subsample.")
    args = p.parse_args()

    n_trials = 2 if args.smoke else args.trials
    n_sub = 5_000 if args.smoke else 200_000
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "_smoke" if args.smoke else ""

    _, df_tr, df_va, df_te = load_and_split(args.data, n_sub=n_sub)
    X_tr, X_va, X_te = build_X(df_tr), build_X(df_va), build_X(df_te)
    print(f"[xgb-tuned] n_jobs={N_JOBS}  trials/target={n_trials}", flush=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    out = {"meta": {
        "n_train": int(len(df_tr)), "n_val": int(len(df_va)),
        "n_test": int(len(df_te)), "seed": SEED,
        "n_trials_per_target": n_trials, "sampler": "TPE",
        "objective": "validation R2", "early_stopping_rounds": EARLY_STOP,
        "smoke": bool(args.smoke),
        "search_space": {
            "max_depth": [4, 12], "learning_rate": [0.01, 0.3, "log"],
            "n_estimators": [200, 1500], "subsample": [0.5, 1.0],
            "colsample_bytree": [0.5, 1.0], "min_child_weight": [1, 20],
            "reg_lambda": [1e-3, 10.0, "log"],
        },
    }}

    t_all = time.time()
    for j, target in enumerate(TARGET_COLUMNS):
        y_tr = df_tr[target].to_numpy(dtype=np.float32)
        y_va = df_va[target].to_numpy(dtype=np.float32)
        y_te = df_te[target].to_numpy(dtype=np.float64)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "max_depth": trial.suggest_int("max_depth", 4, 12),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int(
                    "min_child_weight", 1, 20),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", 1e-3, 10.0, log=True),
            }
            t0 = time.time()
            model = make_model(params, seed=SEED + j)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            r2 = r2_only(y_va, model.predict(X_va))
            print(f"  [{target}] trial {trial.number:02d}  val-R2={r2:.5f}  "
                  f"({time.time()-t0:.1f}s)  {params}", flush=True)
            return r2

        print(f"\n[xgb-tuned] {target}: TPE search ({n_trials} trials)...",
              flush=True)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED + j),
        )
        t0 = time.time()
        study.optimize(objective, n_trials=n_trials)
        search_time = time.time() - t0
        study.trials_dataframe().to_csv(
            out_dir / f"xgb_tuned_trials_{target}{tag}.csv", index=False)

        # Retrain best config, evaluate on the held-out test split.
        best = study.best_params
        print(f"[xgb-tuned] {target}: best val-R2={study.best_value:.5f}  "
              f"params={best}", flush=True)
        t0 = time.time()
        model = make_model(best, seed=SEED + j)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        retrain_time = time.time() - t0
        m = metrics(y_te, model.predict(X_te).astype(np.float64))
        m.update({
            "best_params": best,
            "best_val_r2": float(study.best_value),
            "best_iteration": int(getattr(model, "best_iteration", -1)),
            "search_time_s": search_time,
            "retrain_time_s": retrain_time,
        })
        out[target] = m
        print(f"[xgb-tuned] {target}: TEST r2={m['r2']:.4f}  "
              f"rmse={m['rmse']:.4g}  mape={m['mape_pct']:.2f}%", flush=True)

    out["meta"]["total_time_s"] = time.time() - t_all
    out_path = out_dir / f"xgb_tuned_metrics{tag}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[xgb-tuned] saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
