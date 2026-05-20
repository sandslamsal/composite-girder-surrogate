#!/usr/bin/env python
"""Plain (non-residual) MLP baseline, CPU-only to avoid MPS contention."""
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


class PlainMLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, width=256, depth=5, dropout=0.1):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default="configs/training.yaml")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=512)
    args = p.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(Path(args.config).read_text())
    print(f"[mlp] loading {args.data}...", flush=True)
    df = pd.read_parquet(args.data)
    df_tr, df_val, df_te = split_by_sample(df, cfg["splits"], int(cfg["seed"]))
    print(f"[mlp] train={len(df_tr)} val={len(df_val)} test={len(df_te)}", flush=True)

    X_tr = build_X(df_tr); X_val = build_X(df_val); X_te = build_X(df_te)
    y_tr = df_tr[COMPARE_TARGETS].to_numpy(dtype=np.float32)
    y_val = df_val[COMPARE_TARGETS].to_numpy(dtype=np.float32)
    y_te = df_te[COMPARE_TARGETS].to_numpy(dtype=np.float32)

    # Standardise.
    mu_x = X_tr.mean(0); sd_x = X_tr.std(0); sd_x[sd_x < 1e-8] = 1.0
    mu_y = y_tr.mean(0); sd_y = y_tr.std(0); sd_y[sd_y < 1e-8] = 1.0

    Xn_tr = (X_tr - mu_x) / sd_x
    Xn_val = (X_val - mu_x) / sd_x
    Xn_te = (X_te - mu_x) / sd_x
    yn_tr = (y_tr - mu_y) / sd_y

    device = torch.device("cpu")
    torch.set_num_threads(4)
    model = PlainMLP(Xn_tr.shape[1], y_tr.shape[1]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[mlp] params: {n_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=3e-6)
    loss_fn = torch.nn.MSELoss()

    Xt = torch.tensor(Xn_tr, dtype=torch.float32, device=device)
    Yt = torch.tensor(yn_tr, dtype=torch.float32, device=device)
    Xv = torch.tensor(Xn_val, dtype=torch.float32, device=device)
    Yv = torch.tensor((y_val - mu_y) / sd_y, dtype=torch.float32, device=device)
    Xe = torch.tensor(Xn_te, dtype=torch.float32, device=device)

    n = len(Xt)
    rng = np.random.default_rng(0)
    best_val = float("inf")
    best_state = None
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(n)
        for i in range(0, n, args.batch):
            idx = order[i:i + args.batch]
            xb = Xt[idx]; yb = Yt[idx]
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xv), Yv).item()
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 5 == 0:
            print(f"[mlp] epoch {ep+1:3d}/{args.epochs}  "
                  f"val_loss={vloss:.5g}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        yp = model(Xe).cpu().numpy() * sd_y + mu_y
    elapsed = time.time() - t0

    results = {"train_time_s": elapsed, "params": n_params}
    for k, name in enumerate(COMPARE_TARGETS):
        m = metrics(y_te[:, k], yp[:, k])
        print(f"[mlp] {name}: r2={m['r2']:.4f}  rmse={m['rmse']:.4g}  "
              f"mape={m['mape_pct']:.2f}%", flush=True)
        results[name] = m

    (out_dir / "plain_mlp_metrics.json").write_text(json.dumps(results, indent=2))
    print(f"[mlp] total {elapsed:.1f}s -> {out_dir / 'plain_mlp_metrics.json'}",
          flush=True)


if __name__ == "__main__":
    main()
