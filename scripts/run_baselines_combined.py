#!/usr/bin/env python
"""Combined XGBoost + Plain MLP baselines, subsampled to fit in memory.

Same train/val/test split as the residual MLP (by sample_id, seed
20260513). XGBoost trains on a fixed 500k-row training subsample to
stay within available RAM; the plain MLP trains on the full split.
Test set is held constant for both, so test metrics are directly
comparable to the headline residual MLP.

Outputs:
  - reports/baselines/baselines_metrics.json
"""
from __future__ import annotations

import json
import os
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


SEED = 20260513
SUBSAMPLE = 500_000


def split(df):
    ids = df.sample_id.unique()
    rng = np.random.default_rng(SEED); rng.shuffle(ids); n = len(ids)
    n_tr = int(round(0.8 * n)); n_va = int(round(0.1 * n))
    return (
        df[df.sample_id.isin(set(ids[:n_tr]))].reset_index(drop=True),
        df[df.sample_id.isin(set(ids[n_tr:n_tr + n_va]))].reset_index(drop=True),
        df[df.sample_id.isin(set(ids[n_tr + n_va:]))].reset_index(drop=True),
    )


def metrics(y, yh):
    r2 = 1 - np.sum((y - yh) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12)
    rmse = float(np.sqrt(np.mean((y - yh) ** 2)))
    eps = 1e-6
    denom = np.where(np.abs(y) > eps, np.abs(y), eps)
    mape = float(100.0 * np.mean(np.abs((yh - y) / denom)))
    return float(r2), rmse, mape


class PlainMLP(nn.Module):
    def __init__(self, n_in=17, n_out=4, dropout=0.1):
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


def train_plain_mlp(X_tr, Y_tr_n, X_va, Y_va_n, X_te, device, epochs=100):
    model = PlainMLP(n_in=X_tr.shape[1], n_out=Y_tr_n.shape[1]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[plain-mlp] params = {n_params:,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=3e-6)
    loss_fn = nn.MSELoss()

    Xtr = torch.tensor(X_tr, dtype=torch.float32, device=device)
    Ytr = torch.tensor(Y_tr_n, dtype=torch.float32, device=device)
    Xva = torch.tensor(X_va, dtype=torch.float32, device=device)
    Yva = torch.tensor(Y_va_n, dtype=torch.float32, device=device)
    Xte = torch.tensor(X_te, dtype=torch.float32, device=device)

    loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=512, shuffle=True)
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
        if (ep + 1) % 10 == 0:
            print(f"  [plain-mlp] epoch {ep + 1}/{epochs}  val={val_loss:.4e}", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(Xte).cpu().numpy()


def main():
    out_dir = Path("reports/baselines"); out_dir.mkdir(parents=True, exist_ok=True)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[setup] device = {device}", flush=True)

    print("[load]", flush=True)
    df = pd.read_parquet("data/raw/full_50k.parquet")
    print(f"  {len(df):,} rows", flush=True)
    tr_df, va_df, te_df = split(df)
    print(f"[split] tr={len(tr_df):,} va={len(va_df):,} te={len(te_df):,}", flush=True)

    norm = FeatureNormalizer(data_gen_yaml="configs/data_gen.yaml").fit(tr_df)
    X_tr_full = norm.transform_features(tr_df)
    X_va = norm.transform_features(va_df)
    X_te = norm.transform_features(te_df)
    Y_tr_n_full = norm.transform_targets(tr_df)
    Y_va_n = norm.transform_targets(va_df)
    Y_te = te_df[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    te_mp = te_df["mp_estimate_kip_in"].to_numpy()

    # Subsample training set for XGBoost (memory)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X_tr_full), size=min(SUBSAMPLE, len(X_tr_full)), replace=False)
    X_tr_sub = X_tr_full[idx]
    Y_tr_n_sub = Y_tr_n_full[idx]
    print(f"[xgb subsample] {len(X_tr_sub):,} rows", flush=True)

    # ---- XGBoost (subsampled training)
    print("\n[xgb] training on subsample, 4 targets, n_jobs=2", flush=True)
    xgb_preds_n = np.zeros_like(Y_te)
    for j, tgt in enumerate(TARGET_COLUMNS):
        t0 = time.time()
        m = xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85,
            tree_method="hist", n_jobs=2, random_state=SEED + j,
        )
        m.fit(X_tr_sub, Y_tr_n_sub[:, j], eval_set=[(X_va, Y_va_n[:, j])], verbose=False)
        xgb_preds_n[:, j] = m.predict(X_te)
        print(f"  {tgt}: {time.time() - t0:.1f}s", flush=True)

    # ---- Plain MLP (full training)
    print("\n[plain-mlp] training on full split", flush=True)
    t0 = time.time()
    plain_preds_n = train_plain_mlp(
        X_tr_full, Y_tr_n_full, X_va, Y_va_n, X_te, device, epochs=100,
    )
    print(f"[plain-mlp] total time = {time.time() - t0:.1f}s", flush=True)

    # ---- Denormalise + metrics
    xgb_pred = norm.inverse_transform_targets(xgb_preds_n, mp_estimate_kip_in=te_mp)
    plain_pred = norm.inverse_transform_targets(plain_preds_n, mp_estimate_kip_in=te_mp)

    print("\nTARGET                  XGB (R2/RMSE/MAPE)        plain-MLP (R2/RMSE/MAPE)")
    out = {}
    for j, tgt in enumerate(TARGET_COLUMNS):
        r2_x, rm_x, mp_x = metrics(Y_te[:, j], xgb_pred[:, j])
        r2_p, rm_p, mp_p = metrics(Y_te[:, j], plain_pred[:, j])
        out[tgt] = {
            "xgb": {"r2": r2_x, "rmse": rm_x, "mape_pct": mp_x},
            "plain_mlp": {"r2": r2_p, "rmse": rm_p, "mape_pct": mp_p},
        }
        print(f"  {tgt:22s}  {r2_x:.4f} / {rm_x:.3e} / {mp_x:5.1f}%   "
              f"{r2_p:.4f} / {rm_p:.3e} / {mp_p:5.1f}%")

    out["meta"] = {
        "xgb_train_subsample": int(len(X_tr_sub)),
        "plain_mlp_train_full": int(len(X_tr_full)),
        "test_rows": int(len(X_te)),
        "split_seed": SEED,
        "device": str(device),
    }
    with open(out_dir / "baselines_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {out_dir/'baselines_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
