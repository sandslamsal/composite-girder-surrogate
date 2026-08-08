#!/usr/bin/env python
"""Deep ensemble (practical BNN baseline): 5 independently-initialised
copies of the paper's residual MLP (CompositeGirderSurrogate, width 256,
5 blocks — identical to run_mlps_200k.py), seeds 0-4, each with the same
training budget as the 200k runs (100 epochs, AdamW 3e-4, cosine).

Reports ensemble-mean test R2/MAPE, per-member spread, and Gaussian NLL
from the ensemble mean/variance (physical units).

Outputs:
  reports/baselines_revision/ensemble_metrics.json
  checkpoints/revision_ensemble/member_seed<k>.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.normalize import FeatureNormalizer, TARGET_COLUMNS
from src.models.surrogate import CompositeGirderSurrogate
from scripts.revision_common import (
    DATA_PATH, OUT_DIR, SEED, load_and_split, metrics,
)

EPOCHS = 100          # same budget as run_mlps_200k.py
N_MEMBERS = 5
BATCH = 512
CKPT_DIR = REPO_ROOT / "checkpoints" / "revision_ensemble"


def train_member(seed, X_tr, Y_tr_n, X_va, Y_va_n, X_te, epochs, norm):
    """Same loop as run_mlps_200k.train_torch, seeded per member."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    model = CompositeGirderSurrogate(
        input_dim=X_tr.shape[1], output_dim=Y_tr_n.shape[1],
        width=256, n_blocks=5, dropout=0.1,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=3e-6)
    loss_fn = nn.MSELoss()
    Xtr = torch.tensor(X_tr, dtype=torch.float32)
    Ytr = torch.tensor(Y_tr_n, dtype=torch.float32)
    Xva = torch.tensor(X_va, dtype=torch.float32)
    Yva = torch.tensor(Y_va_n, dtype=torch.float32)
    Xte = torch.tensor(X_te, dtype=torch.float32)
    loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=BATCH,
                        shuffle=True)
    best_val = float("inf"); best_state = None
    t0 = time.time()
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
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
        if (ep + 1) % 10 == 0:
            print(f"  [member-{seed}] ep {ep+1:3d}/{epochs}  "
                  f"val={val_loss:.4e}  best={best_val:.4e}  "
                  f"t={time.time()-t0:.0f}s", flush=True)
    model.load_state_dict(best_state)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "normalizer_state": norm.state_dict(),
        "seed": seed, "epochs": epochs, "best_val_mse": best_val,
    }, CKPT_DIR / f"member_seed{seed}.pt")
    model.eval()
    with torch.no_grad():
        preds_n = model(Xte).numpy()
    return preds_n, best_val, time.time() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DATA_PATH))
    p.add_argument("--out", default=str(OUT_DIR))
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--members", type=int, default=N_MEMBERS)
    p.add_argument("--smoke", action="store_true",
                   help="1 member, 2 epochs, 5k rows.")
    args = p.parse_args()

    epochs = 2 if args.smoke else args.epochs
    members = 1 if args.smoke else args.members
    n_sub = 5_000 if args.smoke else 200_000
    tag = "_smoke" if args.smoke else ""
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    df_tr_full, df_tr, df_va, df_te = load_and_split(args.data, n_sub=n_sub)
    norm = FeatureNormalizer(
        data_gen_yaml=str(REPO_ROOT / "configs/data_gen.yaml"))
    norm.fit(df_tr_full)
    X_tr = norm.transform_features(df_tr)
    X_va = norm.transform_features(df_va)
    X_te = norm.transform_features(df_te)
    Y_tr_n = norm.transform_targets(df_tr)
    Y_va_n = norm.transform_targets(df_va)
    Y_te = df_te[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    te_mp = df_te["mp_estimate_kip_in"].to_numpy()

    member_preds = []   # physical units, (M, N, 2)
    member_stats = []
    for seed in range(members):
        print(f"\n[ensemble] member seed={seed}", flush=True)
        preds_n, best_val, ttime = train_member(
            seed, X_tr, Y_tr_n, X_va, Y_va_n, X_te, epochs, norm)
        pred_phys = norm.inverse_transform_targets(
            preds_n, mp_estimate_kip_in=te_mp)
        member_preds.append(pred_phys)
        stats = {"seed": seed, "best_val_mse": best_val,
                 "train_time_s": ttime}
        for j, tgt in enumerate(TARGET_COLUMNS):
            stats[tgt] = metrics(Y_te[:, j], pred_phys[:, j])
            print(f"  [member-{seed}] {tgt}: r2={stats[tgt]['r2']:.4f}",
                  flush=True)
        member_stats.append(stats)

    P = np.stack(member_preds, axis=0)          # (M, N, 2)
    mu = P.mean(axis=0)                          # ensemble mean
    var = P.var(axis=0)                          # across-member variance

    out = {"meta": {
        "n_train": int(len(df_tr)), "n_val": int(len(df_va)),
        "n_test": int(len(df_te)), "seed_base": 0, "members": members,
        "epochs": epochs, "batch_size": BATCH,
        "architecture": "CompositeGirderSurrogate w=256 blocks=5 do=0.1 "
                        "(identical to run_mlps_200k residual MLP)",
        "smoke": bool(args.smoke),
        "nll_note": "Gaussian NLL from ensemble mean/variance in physical "
                    "units; variance floored at (1e-4 * train std)^2 since "
                    "members are deterministic (no aleatoric head).",
    }, "members": member_stats}

    for j, tgt in enumerate(TARGET_COLUMNS):
        y = Y_te[:, j]
        m = metrics(y, mu[:, j])
        floor = (1e-4 * float(df_tr[tgt].to_numpy().std())) ** 2
        v = var[:, j] + floor
        nll = float(np.mean(0.5 * np.log(2.0 * np.pi * v)
                            + (y - mu[:, j]) ** 2 / (2.0 * v)))
        member_r2 = [s[tgt]["r2"] for s in member_stats]
        member_mape = [s[tgt]["mape_pct"] for s in member_stats]
        out[tgt] = {
            "ensemble_mean": m,
            "mean_nll": nll,
            "variance_floor": floor,
            "member_r2_mean": float(np.mean(member_r2)),
            "member_r2_std": float(np.std(member_r2)),
            "member_mape_mean": float(np.mean(member_mape)),
            "member_mape_std": float(np.std(member_mape)),
            "mean_predictive_std": float(np.mean(np.sqrt(v))),
        }
        print(f"\n[ensemble] {tgt}: mean r2={m['r2']:.4f}  "
              f"mape={m['mape_pct']:.2f}%  nll={nll:.4f}  "
              f"member-r2 spread={np.std(member_r2):.5f}", flush=True)

    out_path = out_dir / f"ensemble_metrics{tag}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[ensemble] saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
