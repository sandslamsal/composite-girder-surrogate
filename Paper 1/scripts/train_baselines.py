#!/usr/bin/env python
"""Train ML baselines for comparison against the residual-MLP surrogate.

Trains and evaluates two baselines on the same train/val/test split and
the same feature encoding as the headline residual MLP, so that the
held-out test R^2 / RMSE / MAPE values are directly comparable:

  - XGBoost regressor (one model per target, mature defaults)
  - Plain (non-residual) MLP at matched parameter count

Outputs:
  - reports/baselines/baselines_test_metrics.csv (target x model)

Usage:
    python scripts/train_baselines.py \\
        --data data/raw/full_50k.parquet \\
        --out reports/baselines/
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.normalize import FeatureNormalizer, TARGET_COLUMNS


# ---------------------------------------------------------------- split
def _split_by_sample(df: pd.DataFrame, fracs, seed: int):
    ids = df["sample_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_tr = int(round(fracs["train"] * n))
    n_va = int(round(fracs["val"] * n))
    tr = set(ids[:n_tr]); va = set(ids[n_tr:n_tr + n_va]); te = set(ids[n_tr + n_va:])
    return (
        df[df["sample_id"].isin(tr)].reset_index(drop=True),
        df[df["sample_id"].isin(va)].reset_index(drop=True),
        df[df["sample_id"].isin(te)].reset_index(drop=True),
    )


# ---------------------------------------------------------------- metrics
def _metrics(y_true: np.ndarray, y_pred: np.ndarray):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    eps = 1e-6
    denom = np.where(np.abs(y_true) > eps, np.abs(y_true), eps)
    mape = float(100.0 * np.mean(np.abs((y_pred - y_true) / denom)))
    return float(r2), rmse, mape


# ---------------------------------------------------------------- plain MLP
class PlainMLP(nn.Module):
    """Non-residual MLP. Layers 15 -> 512 -> 384 -> 256 -> 128 -> 2 give
    ~6.5e5 parameters, within ~3% of the headline residual MLP's 664k."""
    def __init__(self, n_in: int = 15, n_out: int = 2, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 384), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(384, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, n_out),
            nn.Softplus(),
        )

    def forward(self, x):
        return self.net(x)


def _train_plain_mlp(X_tr, Y_tr_n, X_va, Y_va_n, X_te,
                     device, epochs=100, batch=512, lr=3e-4, wd=1e-5):
    model = PlainMLP(n_in=X_tr.shape[1], n_out=Y_tr_n.shape[1]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[plain-mlp] params = {n_params:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=3e-6)
    loss_fn = nn.MSELoss()

    Xtr = torch.tensor(X_tr, dtype=torch.float32, device=device)
    Ytr = torch.tensor(Y_tr_n, dtype=torch.float32, device=device)
    Xva = torch.tensor(X_va, dtype=torch.float32, device=device)
    Yva = torch.tensor(Y_va_n, dtype=torch.float32, device=device)
    Xte = torch.tensor(X_te, dtype=torch.float32, device=device)
    loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=batch, shuffle=True)
    best_val = float("inf"); best_state = None
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva), Yva).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 20 == 0:
            print(f"  [plain-mlp] epoch {ep+1:3d}/{epochs} val={val_loss:.4e}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(Xte).cpu().numpy()


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260513)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--data-gen-yaml", default="configs/data_gen.yaml")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "mps":
        device = torch.device("mps")
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[setup] device = {device}")
    print(f"[load] data = {args.data}")
    df = pd.read_parquet(args.data)
    print(f"  {len(df):,} rows / {df['sample_id'].nunique():,} samples")

    splits = {"train": 0.8, "val": 0.1, "test": 0.1}
    tr_df, va_df, te_df = _split_by_sample(df, splits, args.seed)
    print(f"[split] train={len(tr_df):,} val={len(va_df):,} test={len(te_df):,}")

    norm = FeatureNormalizer(data_gen_yaml=args.data_gen_yaml)
    norm.fit(tr_df)

    X_tr = norm.transform_features(tr_df)
    X_va = norm.transform_features(va_df)
    X_te = norm.transform_features(te_df)
    print(f"[encode] feature dim = {X_tr.shape[1]}")

    Y_tr_n = norm.transform_targets(tr_df)
    Y_va_n = norm.transform_targets(va_df)
    Y_te = te_df[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    te_mp = te_df["mp_estimate_kip_in"].to_numpy()

    # ---- XGBoost ----------------------------------------------------------
    print("\n[xgboost] training one model per target...")
    t0 = time.time()
    xgb_preds_n = np.zeros_like(Y_te, dtype=np.float64)
    for j, tgt in enumerate(TARGET_COLUMNS):
        m = xgb.XGBRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85,
            tree_method="hist", n_jobs=-1, random_state=args.seed + j,
        )
        m.fit(X_tr, Y_tr_n[:, j], eval_set=[(X_va, Y_va_n[:, j])], verbose=False)
        xgb_preds_n[:, j] = m.predict(X_te)
        print(f"  [xgb] {tgt}: trained")
    t_xgb = time.time() - t0
    print(f"[xgboost] total time = {t_xgb:.1f}s")

    # ---- Plain MLP --------------------------------------------------------
    print("\n[plain-mlp] training...")
    t0 = time.time()
    plain_preds_n = _train_plain_mlp(
        X_tr, Y_tr_n, X_va, Y_va_n, X_te, device, epochs=args.epochs,
    )
    t_pmlp = time.time() - t0
    print(f"[plain-mlp] total time = {t_pmlp:.1f}s")

    # ---- Denormalise (per-row mp for moment) -----------------------------
    xgb_pred = norm.inverse_transform_targets(xgb_preds_n, mp_estimate_kip_in=te_mp)
    plain_pred = norm.inverse_transform_targets(plain_preds_n, mp_estimate_kip_in=te_mp)

    # ---- Metrics ----------------------------------------------------------
    rows = []
    for j, tgt in enumerate(TARGET_COLUMNS):
        r2_x, rmse_x, mape_x = _metrics(Y_te[:, j], xgb_pred[:, j])
        r2_p, rmse_p, mape_p = _metrics(Y_te[:, j], plain_pred[:, j])
        rows.append({
            "target": tgt,
            "xgb_r2": r2_x, "xgb_rmse": rmse_x, "xgb_mape_pct": mape_x,
            "plain_mlp_r2": r2_p, "plain_mlp_rmse": rmse_p, "plain_mlp_mape_pct": mape_p,
        })
    out_df = pd.DataFrame(rows)
    out_csv = out_dir / "baselines_test_metrics.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\n[done] wrote {out_csv}")
    print(out_df.to_string(index=False))
    print(f"\nTimes: XGBoost={t_xgb:.0f}s  PlainMLP={t_pmlp:.0f}s")


if __name__ == "__main__":
    main()
