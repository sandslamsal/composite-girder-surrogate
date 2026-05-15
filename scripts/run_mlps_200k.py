#!/usr/bin/env python
"""Train Plain MLP and Residual MLP on the 200k subsample (no XGBoost
in the same process — avoids OpenMP thread pool conflict).

Outputs: reports/baselines/mlps_200k_metrics.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.normalize import FeatureNormalizer, TARGET_COLUMNS
from src.models.surrogate import CompositeGirderSurrogate

SEED = 20260513
N_SUB = 200_000
EPOCHS = 100


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


def train_torch(model, X_tr, Y_tr_n, X_va, Y_va_n, X_te, device, epochs, name):
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{name}] params = {n_params:,}", flush=True)
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
    t_total = time.time()
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
            print(f"  [{name}] ep {ep+1:3d}/{epochs}  val={val_loss:.4e}  best={best_val:.4e}  t={time.time()-t_total:.0f}s", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(Xte).cpu().numpy(), n_params


def main():
    out_dir = Path("reports/baselines"); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    print(f"[setup] device = {device}", flush=True)

    df = pd.read_parquet("data/raw/full_50k.parquet")
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

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X_tr_full), size=N_SUB, replace=False)
    X_tr = X_tr_full[idx]
    Y_tr_n = Y_tr_n_full[idx]
    print(f"[subsample] tr -> {len(X_tr):,} rows", flush=True)

    out = {"meta": {
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_va)),
        "n_test": int(len(X_te)),
        "seed": SEED, "epochs": EPOCHS, "device": str(device),
    }}

    # Plain MLP
    print("\n[plain-mlp]", flush=True)
    plain_preds_n, plain_params = train_torch(
        PlainMLP(n_in=X_tr.shape[1], n_out=Y_tr_n.shape[1]),
        X_tr, Y_tr_n, X_va, Y_va_n, X_te, device, EPOCHS, "plain-mlp",
    )
    plain_pred = norm.inverse_transform_targets(plain_preds_n, mp_estimate_kip_in=te_mp)

    # Residual MLP (same arch as headline)
    print("\n[residual-mlp] (matched data)", flush=True)
    res_preds_n, res_params = train_torch(
        CompositeGirderSurrogate(input_dim=X_tr.shape[1], output_dim=Y_tr_n.shape[1],
                            width=256, n_blocks=5, dropout=0.1),
        X_tr, Y_tr_n, X_va, Y_va_n, X_te, device, EPOCHS, "residual-mlp",
    )
    res_pred = norm.inverse_transform_targets(res_preds_n, mp_estimate_kip_in=te_mp)

    print("\nTARGET                  Plain MLP            Residual MLP (200k)")
    for j, tgt in enumerate(TARGET_COLUMNS):
        r2_p, rm_p, mp_p = metrics(Y_te[:, j], plain_pred[:, j])
        r2_r, rm_r, mp_r = metrics(Y_te[:, j], res_pred[:, j])
        out[tgt] = {
            "plain_mlp": {"r2": r2_p, "rmse": rm_p, "mape_pct": mp_p},
            "residual_mlp_200k": {"r2": r2_r, "rmse": rm_r, "mape_pct": mp_r},
        }
        print(f"  {tgt:22s}  R2={r2_p:.4f}  /  R2={r2_r:.4f}")

    out["meta"]["plain_mlp_params"] = plain_params
    out["meta"]["residual_mlp_params"] = res_params
    with open(out_dir / "mlps_200k_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {out_dir/'mlps_200k_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
