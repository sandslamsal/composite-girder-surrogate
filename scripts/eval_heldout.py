#!/usr/bin/env python
"""Recompute held-out test metrics for a trained surrogate checkpoint.

Reproduces the by-sample split used in training, evaluates the
checkpoint on the test split, and reports per-target R2/RMSE/MAPE plus
load-level-stratified curvature MAPE (service-load M/Mp<=0.4,
extended-elastic M/Mp<=0.6, full curve).

Usage:
    python scripts/eval_heldout.py \
        --checkpoint checkpoints/run_2output/best.pt \
        --data data/raw/full_50k.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.surrogate import CompositeGirderSurrogate, count_parameters
from src.utils.normalize import FeatureNormalizer, TARGET_COLUMNS
from scripts.train_surrogate import (
    _split_by_sample, _make_loaders, _resolve_path, _r2, _rmse, _mape,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--config", default="configs/training.yaml")
    args = p.parse_args()

    cfg = yaml.safe_load(open(_resolve_path(args.config)))
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    df = pd.read_parquet(_resolve_path(args.data))
    train_df, val_df, test_df = _split_by_sample(df, cfg["splits"], seed)
    print(f"[data] test rows={len(test_df)} samples={test_df['sample_id'].nunique()}")

    normalizer = FeatureNormalizer(_resolve_path(cfg["data_gen_config"]))
    normalizer.fit(train_df)
    loaders = _make_loaders(
        {"test": test_df}, normalizer, batch_size=cfg["batch_size"], device=device,
    )

    model = CompositeGirderSurrogate(
        input_dim=normalizer.input_dim,
        output_dim=normalizer.output_dim,
        width=cfg["model"]["width"],
        n_blocks=cfg["model"]["n_blocks"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    ckpt = torch.load(_resolve_path(args.checkpoint), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[model] params={count_parameters(model):,} "
          f"epoch={ckpt.get('epoch')} val_loss={ckpt.get('val_loss')}")

    preds_phys, targets_phys = [], []
    with torch.no_grad():
        for xb, yb, cb in loaders["test"]:
            pred = model(xb.to(device)).cpu().numpy()
            mp_est = cb[:, 1].numpy()
            preds_phys.append(normalizer.inverse_transform_targets(pred, mp_est))
            targets_phys.append(normalizer.inverse_transform_targets(yb.numpy(), mp_est))
    preds_phys = np.concatenate(preds_phys, axis=0)
    targets_phys = np.concatenate(targets_phys, axis=0)

    print("\n[test] per-target metrics (physical units):")
    for j, name in enumerate(TARGET_COLUMNS):
        r2 = _r2(targets_phys[:, j], preds_phys[:, j])
        rmse = _rmse(targets_phys[:, j], preds_phys[:, j])
        mape = _mape(targets_phys[:, j], preds_phys[:, j])
        print(f"  {name:24s}  R2={r2: .4f}  RMSE={rmse: .4g}  MAPE={mape: .2f}%")

    # ---- load-level-stratified curvature metrics ----
    mr = test_df["moment_ratio"].to_numpy()
    cur_idx = TARGET_COLUMNS.index("curvature_1_per_in")
    na_idx = TARGET_COLUMNS.index("neutral_axis_in")
    print("\n[test] load-level-stratified metrics:")
    for label, lim in [("service-load M/Mp<=0.4", 0.4),
                        ("extended-elastic M/Mp<=0.6", 0.6),
                        ("full curve", np.inf)]:
        m = mr <= lim
        cur_mape = _mape(targets_phys[m, cur_idx], preds_phys[m, cur_idx])
        cur_r2 = _r2(targets_phys[m, cur_idx], preds_phys[m, cur_idx])
        na_mape = _mape(targets_phys[m, na_idx], preds_phys[m, na_idx])
        na_r2 = _r2(targets_phys[m, na_idx], preds_phys[m, na_idx])
        print(f"  {label:28s} n={m.sum():7d}  "
              f"curvature R2={cur_r2:.4f} MAPE={cur_mape:.2f}%  "
              f"y_na R2={na_r2:.4f} MAPE={na_mape:.2f}%")


if __name__ == "__main__":
    main()
