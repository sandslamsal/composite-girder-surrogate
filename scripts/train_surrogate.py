#!/usr/bin/env python
"""Train the composite-girder surrogate.

Usage:
    python scripts/train_surrogate.py --data data/raw/smoke_100.parquet --smoke \\
        --out checkpoints/smoke
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Repo root onto sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.surrogate import CompositeGirderSurrogate, count_parameters
from src.physics.losses import PhysicsLossContext, total_loss
from src.utils.normalize import FeatureNormalizer, TARGET_COLUMNS


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Path to a parquet file.")
    p.add_argument("--config", type=str, default="configs/training.yaml")
    p.add_argument("--out", type=str, required=True, help="Checkpoint output dir.")
    p.add_argument("--epochs", type=int, default=None, help="Override epochs.")
    p.add_argument("--seed", type=int, default=None,
                   help="Override config seed for different initialisations.")
    p.add_argument("--equil-mode", choices=("fibre", "proxy", "none"),
                   default="fibre",
                   help="Equilibrium loss formulation. 'fibre' is the "
                        "physics-informed fibre-stress integration (default); "
                        "'proxy' is the trivial M_pred = r*Mp baseline (used "
                        "for ablations); 'none' disables the term.")
    p.add_argument("--no-compat", action="store_true",
                   help="Disable the compatibility physics loss for ablation.")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: 3 epochs, no checkpoint pruning.")
    return p.parse_args()


def _resolve_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (REPO_ROOT / path)


def _split_by_sample(df: pd.DataFrame, fracs: Dict[str, float], seed: int):
    ids = df["sample_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(fracs["train"] * n))
    n_val = int(round(fracs["val"] * n))
    # remainder -> test
    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:n_train + n_val])
    test_ids = set(ids[n_train + n_val:])
    mask_train = df["sample_id"].isin(train_ids)
    mask_val = df["sample_id"].isin(val_ids)
    mask_test = df["sample_id"].isin(test_ids)
    return df[mask_train].reset_index(drop=True), df[mask_val].reset_index(drop=True), df[mask_test].reset_index(drop=True)


def _build_context(df: pd.DataFrame, normalizer: FeatureNormalizer, device: torch.device) -> PhysicsLossContext:
    return PhysicsLossContext(
        total_depth_in=torch.tensor(df["total_depth_in"].to_numpy(), dtype=torch.float32, device=device),
        mp_estimate_kip_in=torch.tensor(df["mp_estimate_kip_in"].to_numpy(), dtype=torch.float32, device=device),
        moment_ratio=torch.tensor(df["moment_ratio"].to_numpy(), dtype=torch.float32, device=device),
        target_scale=torch.tensor(normalizer.target_scale(), dtype=torch.float32, device=device),
        target_offset=torch.tensor(normalizer.target_offset(), dtype=torch.float32, device=device),
    )


# Physics-context columns carried alongside (X, Y) so each batch can
# rebuild a PhysicsLossContext. Order is fixed; see _ctx_from_batch.
_CTX_COLS = [
    "total_depth_in",       # 0
    "mp_estimate_kip_in",   # 1
    "moment_ratio",         # 2
    "deck_thickness_in",    # 3
    "deck_width_in",        # 4 -- raw effective width (NOT eta_c scaled)
    "composite_action",     # 5
    "steel_depth_in",       # 6
    "flange_width_in",      # 7
    "flange_thickness_in",  # 8
    "web_thickness_in",     # 9
    "fc_deck_ksi",          # 10
    "fy_ksi",               # 11
]


def _make_loaders(splits, normalizer, batch_size, device):
    loaders = {}
    for name, df in splits.items():
        X = torch.tensor(normalizer.transform_features(df), dtype=torch.float32)
        Y = torch.tensor(normalizer.transform_targets(df), dtype=torch.float32)
        ctx = torch.tensor(df[_CTX_COLS].to_numpy(), dtype=torch.float32)
        ds = TensorDataset(X, Y, ctx)
        shuffle = (name == "train")
        loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    return loaders


def _ctx_from_batch(ctx_batch: torch.Tensor, target_scale: torch.Tensor, target_offset: torch.Tensor) -> PhysicsLossContext:
    """Rebuild PhysicsLossContext from the carried-column tensor.

    Columns must match ``_CTX_COLS`` order. ``deck_width_in`` is the *full*
    effective width; we scale by ``composite_action`` here so the
    fibre-integration equilibrium loss sees the partial-composite width
    already applied (matching the dataset generator).
    """
    eta_c = ctx_batch[:, 5]
    deck_width_full = ctx_batch[:, 4]
    deck_width_effective = deck_width_full * eta_c
    return PhysicsLossContext(
        total_depth_in=ctx_batch[:, 0],
        mp_estimate_kip_in=ctx_batch[:, 1],
        moment_ratio=ctx_batch[:, 2],
        target_scale=target_scale,
        target_offset=target_offset,
        deck_thickness_in=ctx_batch[:, 3],
        deck_width_in=deck_width_effective,
        composite_action=eta_c,
        steel_depth_in=ctx_batch[:, 6],
        flange_width_in=ctx_batch[:, 7],
        flange_thickness_in=ctx_batch[:, 8],
        web_thickness_in=ctx_batch[:, 9],
        fc_deck_ksi=ctx_batch[:, 10],
        fy_ksi=ctx_batch[:, 11],
    )


def _lambda_schedule(epoch: int, total_epochs: int, lo: float, hi: float, ramp_frac: float) -> float:
    ramp_epochs = max(1, int(round(ramp_frac * total_epochs)))
    if epoch >= ramp_epochs:
        return hi
    return lo + (hi - lo) * (epoch / ramp_epochs)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """Mean absolute percentage error. Rows where |y_true| < eps are dropped
    so a near-zero target doesn't blow up the mean."""
    mask = np.abs(y_true) > eps
    if not mask.any():
        return float("nan")
    return float(100.0 * np.mean(np.abs(
        (y_pred[mask] - y_true[mask]) / y_true[mask]
    )))


def main():
    args = _parse_args()
    cfg = yaml.safe_load(open(_resolve_path(args.config)))
    seed = int(args.seed) if args.seed is not None else int(cfg.get("seed", 0))
    cfg["seed"] = seed     # propagate the chosen seed into the saved config
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[setup] device={device}")

    # ---- data ------------------------------------------------------------
    data_path = _resolve_path(args.data)
    df = pd.read_parquet(data_path)
    print(f"[data] {data_path.name}: {len(df)} rows, {df['sample_id'].nunique()} samples")
    train_df, val_df, test_df = _split_by_sample(df, cfg["splits"], seed)
    print(f"[data] split rows: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    normalizer = FeatureNormalizer(_resolve_path(cfg["data_gen_config"]))
    normalizer.fit(train_df)

    loaders = _make_loaders(
        {"train": train_df, "val": val_df, "test": test_df},
        normalizer, batch_size=cfg["batch_size"], device=device,
    )

    # ---- model -----------------------------------------------------------
    model = CompositeGirderSurrogate(
        input_dim=normalizer.input_dim,
        output_dim=normalizer.output_dim,
        width=cfg["model"]["width"],
        n_blocks=cfg["model"]["n_blocks"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    print(f"[model] params={count_parameters(model):,}")

    epochs = int(args.epochs) if args.epochs is not None else int(cfg["epochs"])
    if args.smoke:
        epochs = min(epochs, 3) if args.epochs is None else int(args.epochs)
        if args.epochs is None:
            epochs = 3
    print(f"[train] epochs={epochs}")

    optim = torch.optim.AdamW(model.parameters(), lr=cfg["optimizer"]["lr"],
                              weight_decay=cfg["optimizer"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=epochs, eta_min=cfg["lr_scheduler"]["eta_min"]
    )

    data_weights = torch.tensor(cfg["data_weights"], dtype=torch.float32, device=device)
    target_scale = torch.tensor(normalizer.target_scale(), dtype=torch.float32, device=device)
    target_offset = torch.tensor(normalizer.target_offset(), dtype=torch.float32, device=device)

    lambda_equil = float(cfg["lambda_equil"])

    # Ablation flags from CLI
    equil_mode = "none" if args.equil_mode == "none" else args.equil_mode
    compat_active = not args.no_compat
    print(f"[loss] equilibrium-mode={equil_mode}  "
          f"compat-active={compat_active}")

    out_dir = _resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = math.inf
    history = []

    for epoch in range(epochs):
        lambda_compat = _lambda_schedule(
            epoch, epochs,
            cfg["lambda_compat"]["start"],
            cfg["lambda_compat"]["end"],
            cfg["lambda_compat"]["ramp_fraction"],
        )
        if not compat_active:
            lambda_compat = 0.0

        # ---- train ----
        model.train()
        agg = {"total": 0.0, "data": 0.0, "compat": 0.0, "equil": 0.0, "n": 0}
        for xb, yb, cb in loaders["train"]:
            xb = xb.to(device)
            yb = yb.to(device)
            cb = cb.to(device)
            ctx = _ctx_from_batch(cb, target_scale, target_offset)
            pred = model(xb)
            losses = total_loss(
                pred, yb, ctx, lambda_compat, lambda_equil, data_weights,
                equil_mode=equil_mode,
            )
            optim.zero_grad()
            losses["total"].backward()
            # Clip to keep MPS happy when early-training physics residuals are large.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()
            bs = xb.size(0)
            for k in ("total", "data", "compat", "equil"):
                agg[k] += float(losses[k].detach().cpu()) * bs
            agg["n"] += bs
        train_log = {k: agg[k] / agg["n"] for k in ("total", "data", "compat", "equil")}

        # ---- val ----
        model.eval()
        vagg = {"total": 0.0, "data": 0.0, "compat": 0.0, "equil": 0.0, "n": 0}
        with torch.no_grad():
            for xb, yb, cb in loaders["val"]:
                xb = xb.to(device); yb = yb.to(device); cb = cb.to(device)
                ctx = _ctx_from_batch(cb, target_scale, target_offset)
                pred = model(xb)
                losses = total_loss(
                    pred, yb, ctx, lambda_compat, lambda_equil, data_weights,
                    equil_mode=equil_mode,
                )
                bs = xb.size(0)
                for k in ("total", "data", "compat", "equil"):
                    vagg[k] += float(losses[k].detach().cpu()) * bs
                vagg["n"] += bs
        val_log = {k: vagg[k] / max(vagg["n"], 1) for k in ("total", "data", "compat", "equil")}

        lr_now = optim.param_groups[0]["lr"]
        sched.step()

        log_line = {
            "epoch": epoch,
            "lr": lr_now,
            "lambda_compat": lambda_compat,
            "train": train_log,
            "val": val_log,
        }
        history.append(log_line)
        print(
            f"[epoch {epoch:03d}] lr={lr_now:.2e} l1={lambda_compat:.3f} "
            f"| train total={train_log['total']:.4e} data={train_log['data']:.4e} "
            f"compat={train_log['compat']:.4e} equil={train_log['equil']:.4e} "
            f"| val total={val_log['total']:.4e} data={val_log['data']:.4e}"
        )

        if val_log["total"] < best_val:
            best_val = val_log["total"]
            torch.save({
                "model_state": model.state_dict(),
                "normalizer_state": normalizer.state_dict(),
                "epoch": epoch,
                "val_loss": best_val,
                "config": cfg,
            }, out_dir / "best.pt")

    # final checkpoint
    torch.save({
        "model_state": model.state_dict(),
        "normalizer_state": normalizer.state_dict(),
        "epoch": epochs - 1,
        "config": cfg,
    }, out_dir / "final.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ---- test set evaluation in physical units ----
    model.eval()
    preds_phys = []
    targets_phys = []
    with torch.no_grad():
        for xb, yb, cb in loaders["test"]:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            # ctx layout: [total_depth, mp_estimate, moment_ratio]
            mp_est = cb[:, 1].numpy()
            preds_phys.append(normalizer.inverse_transform_targets(pred, mp_est))
            targets_phys.append(normalizer.inverse_transform_targets(yb.numpy(), mp_est))
    if preds_phys:
        preds_phys = np.concatenate(preds_phys, axis=0)
        targets_phys = np.concatenate(targets_phys, axis=0)
        print("\n[test] per-target metrics (physical units):")
        for j, name in enumerate(TARGET_COLUMNS):
            r2 = _r2(targets_phys[:, j], preds_phys[:, j])
            rmse = _rmse(targets_phys[:, j], preds_phys[:, j])
            mape = _mape(targets_phys[:, j], preds_phys[:, j])
            print(f"  {name:24s}  R2={r2: .4f}  RMSE={rmse: .4g}  MAPE={mape: .2f}%")
    else:
        print("[test] empty test split, skipping metrics.")


if __name__ == "__main__":
    main()
