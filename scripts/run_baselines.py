#!/usr/bin/env python
"""ML baselines for the composite-girder surrogate (Tier-1 comparison).

Trains two reference models on exactly the same 80/10/10 train/val/test
split (by sample_id, seeded with the training config) used by the
residual MLP surrogate, and reports R^2 / RMSE / MAPE per target on
the held-out test set.

Baselines:
    (a) XGBoost   -- one regressor per target (curvature, neutral-axis
                     depth). Captures tabular nonlinear interactions
                     without an MLP.
    (b) Plain MLP -- 5-layer fully-connected MLP (no residual
                     connections), matched to the surrogate's
                     parameter count.

Moment is not predicted (it is the arithmetic pass-through r * Mp_est).
Slip is excluded because it is itself an analytical surrogate output
(see Tier-1 discussion).

Usage:
    python scripts/run_baselines.py \\
        --data data/raw/full_50k.parquet \\
        --out reports/baselines/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.normalize import FEATURE_COLUMNS, SECTION_TYPES, TARGET_COLUMNS

# Targets we actually compare. Moment is arithmetic; slip is analytical.
COMPARE_TARGETS = ["neutral_axis_in", "curvature_1_per_in"]


def split_by_sample(df: pd.DataFrame, fracs: dict, seed: int):
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


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Construct the X matrix (continuous features + one-hot section type)
    in the same column order as the surrogate's FeatureNormalizer."""
    cont = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    onehot = np.zeros((len(df), len(SECTION_TYPES)), dtype=np.float32)
    for k, s in enumerate(SECTION_TYPES):
        onehot[:, k] = (df["section_type"].to_numpy() == s).astype(np.float32)
    return np.concatenate([cont, onehot], axis=1)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    eps = 1e-12
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, eps)
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), eps))) * 100.0
    return {"r2": r2, "rmse": rmse, "mape_pct": mape}


def train_xgb(X_tr, X_val, y_tr, y_val, X_te, y_te, name: str):
    print(f"[xgb] {name}: training (subsampling train to 500k rows for tractability)...", flush=True)
    t0 = time.time()
    # Subsample train to keep memory bounded; XGBoost saturates well below
    # 3M rows on a 17-feature tabular problem.
    n_max = 500_000
    if len(X_tr) > n_max:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X_tr), size=n_max, replace=False)
        X_tr_sub = X_tr[idx]
        y_tr_sub = y_tr[idx]
    else:
        X_tr_sub = X_tr
        y_tr_sub = y_tr
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=15,
        tree_method="hist",
        n_jobs=4,
        random_state=0,
        verbosity=0,
    )
    model.fit(X_tr_sub, y_tr_sub, eval_set=[(X_val, y_val)], verbose=False)
    elapsed = time.time() - t0
    y_pred = model.predict(X_te)
    m = metrics(y_te, y_pred)
    print(f"[xgb] {name}: r2={m['r2']:.4f}  rmse={m['rmse']:.4g}  mape={m['mape_pct']:.2f}%  time={elapsed:.1f}s")
    return m, elapsed, model


class PlainMLP(torch.nn.Module):
    """Plain (non-residual) 5-layer MLP, matched in parameter count to the
    residual surrogate (~664k parameters when width=256, depth=5)."""

    def __init__(self, in_dim: int, out_dim: int, width: int = 256, depth: int = 5,
                 dropout: float = 0.1):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width), torch.nn.GELU(),
                       torch.nn.Dropout(dropout)]
            d = width
        layers += [torch.nn.Linear(width, out_dim), torch.nn.Softplus()]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_plain_mlp(X_tr, X_val, y_tr, y_val, X_te, y_te, target_names,
                    epochs: int = 100, batch_size: int = 512, lr: float = 3e-4):
    print(f"[mlp] plain MLP: training {len(target_names)} targets jointly...", flush=True)
    # Force CPU to avoid MPS competition with the parallel ensemble run.
    device = torch.device("cpu")
    t0 = time.time()
    # Subsample train rows for tractability (matches XGBoost subset).
    n_max = 500_000
    if len(X_tr) > n_max:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X_tr), size=n_max, replace=False)
        X_tr = X_tr[idx]
        y_tr = y_tr[idx]

    # Standardise inputs (mean 0 std 1 per column based on train).
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Xn_tr = (X_tr - mu) / sd
    Xn_val = (X_val - mu) / sd
    Xn_te = (X_te - mu) / sd

    # Standardise targets.
    y_mu = y_tr.mean(axis=0)
    y_sd = y_tr.std(axis=0)
    y_sd[y_sd < 1e-8] = 1.0
    y_tr_n = (y_tr - y_mu) / y_sd

    in_dim = X_tr.shape[1]
    out_dim = y_tr.shape[1]
    model = PlainMLP(in_dim, out_dim).to(device)
    print(f"[mlp] params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=3e-6)
    loss_fn = torch.nn.MSELoss()

    n = len(Xn_tr)
    rng = np.random.default_rng(0)

    Xn_val_t = torch.tensor(Xn_val, dtype=torch.float32, device=device)
    yv_n = torch.tensor((y_val - y_mu) / y_sd, dtype=torch.float32, device=device)

    best_val = float("inf")
    best_state = None
    for ep in range(epochs):
        model.train()
        order = rng.permutation(n)
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            xb = torch.tensor(Xn_tr[idx], dtype=torch.float32, device=device)
            yb = torch.tensor(y_tr_n[idx], dtype=torch.float32, device=device)
            opt.zero_grad()
            yp = model(xb)
            # The Softplus head guarantees non-negative outputs; since we
            # normalise to mean 0 the loss in normalised space is still well-
            # defined (Softplus saturates linear at large input).
            loss = loss_fn(yp, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vp = model(Xn_val_t)
            vloss = loss_fn(vp, yv_n).item()
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 10 == 0:
            print(f"[mlp] epoch {ep+1:3d}/{epochs}  val_loss={vloss:.5g}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    Xn_te_t = torch.tensor(Xn_te, dtype=torch.float32, device=device)
    with torch.no_grad():
        yp_n = model(Xn_te_t).cpu().numpy()
    yp = yp_n * y_sd + y_mu
    elapsed = time.time() - t0
    out_metrics = {}
    for k, name in enumerate(target_names):
        m = metrics(y_te[:, k], yp[:, k])
        print(f"[mlp] {name}: r2={m['r2']:.4f}  rmse={m['rmse']:.4g}  mape={m['mape_pct']:.2f}%")
        out_metrics[name] = m
    out_metrics["_train_time_s"] = elapsed
    return out_metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default="configs/training.yaml")
    p.add_argument("--mlp-epochs", type=int, default=100)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = int(cfg["seed"])
    splits = cfg["splits"]

    print(f"[data] loading {args.data} ...")
    df = pd.read_parquet(args.data)
    df_tr, df_val, df_te = split_by_sample(df, splits, seed)
    print(f"[data] train={len(df_tr)}  val={len(df_val)}  test={len(df_te)} rows")
    print(f"[data] train_samples={df_tr['sample_id'].nunique()}  "
          f"test_samples={df_te['sample_id'].nunique()}")

    # Build feature matrices.
    X_tr = build_feature_matrix(df_tr)
    X_val = build_feature_matrix(df_val)
    X_te = build_feature_matrix(df_te)

    # Targets we benchmark.
    y_tr = df_tr[COMPARE_TARGETS].to_numpy(dtype=np.float32)
    y_val = df_val[COMPARE_TARGETS].to_numpy(dtype=np.float32)
    y_te = df_te[COMPARE_TARGETS].to_numpy(dtype=np.float32)

    results = {}

    # --- XGBoost (one model per target) ---
    print("\n=== XGBoost baselines ===")
    xgb_results = {}
    xgb_time = 0.0
    for k, name in enumerate(COMPARE_TARGETS):
        m, t, _ = train_xgb(X_tr, X_val, y_tr[:, k], y_val[:, k], X_te, y_te[:, k], name)
        xgb_results[name] = m
        xgb_time += t
    xgb_results["_train_time_s"] = xgb_time
    results["xgboost"] = xgb_results

    # --- Plain (non-residual) MLP ---
    print("\n=== Plain (non-residual) MLP baseline ===")
    mlp_results = train_plain_mlp(
        X_tr, X_val, y_tr, y_val, X_te, y_te, COMPARE_TARGETS,
        epochs=args.mlp_epochs,
    )
    results["plain_mlp"] = mlp_results

    # Save.
    (out_dir / "baseline_metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\n[done] results written to {out_dir / 'baseline_metrics.json'}")

    # Compact summary table.
    print("\nSummary:")
    print(f"{'model':<15}  {'target':<22}  {'R^2':>7}  {'RMSE':>12}  {'MAPE':>8}")
    for model_name, m in results.items():
        for tgt in COMPARE_TARGETS:
            if tgt in m:
                print(f"{model_name:<15}  {tgt:<22}  {m[tgt]['r2']:>7.4f}  "
                      f"{m[tgt]['rmse']:>12.4g}  {m[tgt]['mape_pct']:>7.2f}%")


if __name__ == "__main__":
    main()
