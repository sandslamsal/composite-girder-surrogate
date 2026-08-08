#!/usr/bin/env python
"""Sparse variational GP (SVGP, gpytorch) baseline on the 200k
compute-matched subsample. Per-target: 1024 inducing points, RBF-ARD
kernel, minibatch 4096, Adam 1e-2, 60 epochs, standardized inputs and
outputs. Reports test R2, MAPE and mean Gaussian NLL (physical units).

If a timing probe projects >90 min/target at 200k rows, the train set is
subsampled to 100k and this is recorded in the JSON.

Outputs: reports/baselines_revision/svgp_metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import gpytorch
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.normalize import FeatureNormalizer, TARGET_COLUMNS
from scripts.revision_common import (
    DATA_PATH, OUT_DIR, SEED, load_and_split, metrics,
)

EPOCHS = 60
N_INDUCING = 1024
BATCH = 4096
LR = 1e-2
FALLBACK_N = 100_000
PROBE_BATCHES = 10
BUDGET_MIN_PER_TARGET = 90.0


class SVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor):
        var_dist = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0))
        var_strat = gpytorch.variational.VariationalStrategy(
            self, inducing_points, var_dist, learn_inducing_locations=True)
        super().__init__(var_strat)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=inducing_points.size(1)))

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))


def train_one_target(Xtr, ytr, Xte, epochs, n_inducing, batch, seed, name):
    """Train an SVGP on standardized (Xtr, ytr); return (mu, var) on Xte in
    standardized space, plus timing info."""
    torch.manual_seed(seed)
    n = Xtr.size(0)
    idx = torch.randperm(n)[:n_inducing]
    model = SVGPModel(Xtr[idx].clone())
    lik = gpytorch.likelihoods.GaussianLikelihood()
    model.train(); lik.train()
    opt = torch.optim.Adam(
        list(model.parameters()) + list(lik.parameters()), lr=LR)
    mll = gpytorch.mlls.VariationalELBO(lik, model, num_data=n)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch,
                        shuffle=True)

    t0 = time.time()
    for ep in range(epochs):
        ep_loss, nb = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = -mll(model(xb), yb)
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach()); nb += 1
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  [{name}] ep {ep+1:3d}/{epochs}  elbo-loss="
                  f"{ep_loss/max(nb,1):.4f}  t={time.time()-t0:.0f}s",
                  flush=True)
    train_time = time.time() - t0

    model.eval(); lik.eval()
    mus, vars_ = [], []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i in range(0, Xte.size(0), 8192):
            post = lik(model(Xte[i:i + 8192]))
            mus.append(post.mean)
            vars_.append(post.variance)
    return (torch.cat(mus).numpy(), torch.cat(vars_).numpy(), train_time)


def probe_min_per_epoch(Xtr, ytr, n_inducing, batch, seed):
    """Time PROBE_BATCHES optimisation steps and project minutes/epoch."""
    torch.manual_seed(seed)
    idx = torch.randperm(Xtr.size(0))[:n_inducing]
    model = SVGPModel(Xtr[idx].clone())
    lik = gpytorch.likelihoods.GaussianLikelihood()
    model.train(); lik.train()
    opt = torch.optim.Adam(
        list(model.parameters()) + list(lik.parameters()), lr=LR)
    mll = gpytorch.mlls.VariationalELBO(lik, model, num_data=Xtr.size(0))
    nb = 0
    t0 = time.time()
    for i in range(0, Xtr.size(0), batch):
        if nb >= PROBE_BATCHES:
            break
        xb, yb = Xtr[i:i + batch], ytr[i:i + batch]
        opt.zero_grad()
        loss = -mll(model(xb), yb)
        loss.backward()
        opt.step()
        nb += 1
    per_batch = (time.time() - t0) / max(nb, 1)
    batches_per_epoch = int(np.ceil(Xtr.size(0) / batch))
    return per_batch * batches_per_epoch / 60.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DATA_PATH))
    p.add_argument("--out", default=str(OUT_DIR))
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--smoke", action="store_true",
                   help="2 epochs, 5k rows, 256 inducing points.")
    args = p.parse_args()

    epochs = 2 if args.smoke else args.epochs
    n_sub = 5_000 if args.smoke else 200_000
    n_inducing = 256 if args.smoke else N_INDUCING
    batch = 1024 if args.smoke else BATCH
    tag = "_smoke" if args.smoke else ""
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    df_tr_full, df_tr, df_va, df_te = load_and_split(args.data, n_sub=n_sub)

    # Same feature encoding as the paper's MLP baselines, then z-scored.
    norm = FeatureNormalizer(data_gen_yaml=str(REPO_ROOT / "configs/data_gen.yaml"))
    norm.fit(df_tr_full)
    X_tr = norm.transform_features(df_tr).astype(np.float64)
    X_te = norm.transform_features(df_te).astype(np.float64)
    x_mu, x_sd = X_tr.mean(axis=0), X_tr.std(axis=0)
    x_sd = np.where(x_sd < 1e-8, 1.0, x_sd)
    Xtr = torch.tensor((X_tr - x_mu) / x_sd, dtype=torch.float32)
    Xte = torch.tensor((X_te - x_mu) / x_sd, dtype=torch.float32)

    out = {"meta": {
        "n_train": int(len(df_tr)), "n_val": int(len(df_va)),
        "n_test": int(len(df_te)), "seed": SEED, "epochs": epochs,
        "n_inducing": n_inducing, "batch_size": batch, "lr": LR,
        "kernel": "RBF-ARD", "smoke": bool(args.smoke),
        "inputs": "FeatureNormalizer features, z-scored on train",
        "outputs": "per-target z-scored on train",
        "nll_units": "physical (de-standardized predictive Gaussian)",
    }}

    for j, target in enumerate(TARGET_COLUMNS):
        y_tr_phys = df_tr[target].to_numpy(dtype=np.float64)
        y_te_phys = df_te[target].to_numpy(dtype=np.float64)
        y_mu, y_sd = y_tr_phys.mean(), y_tr_phys.std()
        ytr = torch.tensor((y_tr_phys - y_mu) / y_sd, dtype=torch.float32)

        # Timing probe: fall back to 100k train rows if projection > budget.
        Xtr_t, ytr_t = Xtr, ytr
        subsampled_to = None
        if not args.smoke:
            mpe = probe_min_per_epoch(Xtr, ytr, n_inducing, batch, SEED + j)
            proj = mpe * epochs
            print(f"[svgp] {target}: probe {mpe:.2f} min/epoch -> "
                  f"projected {proj:.0f} min/target", flush=True)
            if proj > BUDGET_MIN_PER_TARGET and len(df_tr) > FALLBACK_N:
                keep = torch.randperm(Xtr.size(0),
                                      generator=torch.Generator().manual_seed(SEED)
                                      )[:FALLBACK_N]
                Xtr_t, ytr_t = Xtr[keep], ytr[keep]
                subsampled_to = FALLBACK_N
                print(f"[svgp] {target}: projected >{BUDGET_MIN_PER_TARGET:.0f}"
                      f" min -> train subsampled to {FALLBACK_N:,} rows",
                      flush=True)

        print(f"[svgp] {target}: training on {Xtr_t.size(0):,} rows...",
              flush=True)
        mu_n, var_n, train_time = train_one_target(
            Xtr_t, ytr_t, Xte, epochs, n_inducing, batch, SEED + j, target)

        mu = mu_n * y_sd + y_mu
        var = var_n * (y_sd ** 2)
        var = np.maximum(var, 1e-300)
        nll = float(np.mean(0.5 * np.log(2.0 * np.pi * var)
                            + (y_te_phys - mu) ** 2 / (2.0 * var)))
        m = metrics(y_te_phys, mu)
        m.update({
            "mean_nll": nll,
            "train_time_s": train_time,
            "train_rows_used": int(Xtr_t.size(0)),
            "train_subsampled_to": subsampled_to,
        })
        out[target] = m
        print(f"[svgp] {target}: TEST r2={m['r2']:.4f}  "
              f"mape={m['mape_pct']:.2f}%  nll={nll:.4f}  "
              f"({train_time:.0f}s)", flush=True)

    out_path = out_dir / f"svgp_metrics{tag}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[svgp] saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
