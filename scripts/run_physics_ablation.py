#!/usr/bin/env python
"""Physics-loss ablation on the 200k compute-matched subsample.

Three matched-budget variants of the paper's residual MLP, replicating
scripts/train_surrogate.py's loss wiring (src/physics/losses.total_loss,
configs/training.yaml lambdas — train_surrogate.py has no subsample /
short-budget mode, so the wiring is reproduced here):

  a) data_only     — data MSE only
  b) compat        — + compatibility penalty (lambda ramp 0.01 -> 0.10)
  c) compat_equil  — + compatibility + fibre-integration equilibrium (0.05)

For each variant, on the held-out test split:
  (i)  constraint-violation metrics — fraction of predictions with the
       neutral axis outside the section depth, and extreme-fibre
       strain-sign inconsistency (same checks as compatibility_loss in
       src/physics/losses.py);
  (ii) an extrapolation probe — variants (a) and (b) are retrained with
       all rows in the top decile of span_in excluded, then scored on
       that excluded decile.

Outputs: reports/baselines_revision/physics_ablation_metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.normalize import FeatureNormalizer, TARGET_COLUMNS
from src.models.surrogate import CompositeGirderSurrogate
from src.physics.losses import PhysicsLossContext, total_loss
from scripts.revision_common import (
    DATA_PATH, OUT_DIR, SEED, load_and_split, metrics,
)

EPOCHS_DEFAULT = 100      # matched to the 200k baseline budget
BATCH = 512

# Physics-context columns (same order as scripts/train_surrogate.py).
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

VARIANTS = {
    "data_only":    {"compat": False, "equil_mode": "none"},
    "compat":       {"compat": True,  "equil_mode": "none"},
    "compat_equil": {"compat": True,  "equil_mode": "fibre"},
}


def _ctx_from_batch(cb, target_scale, target_offset):
    """Rebuild PhysicsLossContext (same as train_surrogate.py: deck width
    is scaled by composite_action here)."""
    eta_c = cb[:, 5]
    return PhysicsLossContext(
        total_depth_in=cb[:, 0],
        mp_estimate_kip_in=cb[:, 1],
        moment_ratio=cb[:, 2],
        target_scale=target_scale,
        target_offset=target_offset,
        deck_thickness_in=cb[:, 3],
        deck_width_in=cb[:, 4] * eta_c,
        composite_action=eta_c,
        steel_depth_in=cb[:, 6],
        flange_width_in=cb[:, 7],
        flange_thickness_in=cb[:, 8],
        web_thickness_in=cb[:, 9],
        fc_deck_ksi=cb[:, 10],
        fy_ksi=cb[:, 11],
    )


def _lambda_schedule(epoch, total_epochs, lo, hi, ramp_frac):
    ramp_epochs = max(1, int(round(ramp_frac * total_epochs)))
    if epoch >= ramp_epochs:
        return hi
    return lo + (hi - lo) * (epoch / ramp_epochs)


def make_tensors(df, norm):
    X = torch.tensor(norm.transform_features(df), dtype=torch.float32)
    Y = torch.tensor(norm.transform_targets(df), dtype=torch.float32)
    C = torch.tensor(df[_CTX_COLS].to_numpy(), dtype=torch.float32)
    return X, Y, C


def train_variant(name, variant, cfg, epochs, norm, tensors_tr, tensors_va,
                  seed=SEED):
    """Train one ablation variant; returns the best-val model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, Ytr, Ctr = tensors_tr
    Xva, Yva, Cva = tensors_va
    model = CompositeGirderSurrogate(
        input_dim=Xtr.shape[1], output_dim=Ytr.shape[1],
        width=cfg["model"]["width"], n_blocks=cfg["model"]["n_blocks"],
        dropout=cfg["model"]["dropout"],
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["optimizer"]["lr"],
                            weight_decay=cfg["optimizer"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=cfg["lr_scheduler"]["eta_min"])
    data_weights = torch.tensor(cfg["data_weights"], dtype=torch.float32)
    target_scale = torch.tensor(norm.target_scale(), dtype=torch.float32)
    target_offset = torch.tensor(norm.target_offset(), dtype=torch.float32)
    lambda_equil = float(cfg["lambda_equil"]) \
        if variant["equil_mode"] != "none" else 0.0

    loader = DataLoader(TensorDataset(Xtr, Ytr, Ctr), batch_size=BATCH,
                        shuffle=True)
    ctx_va = _ctx_from_batch(Cva, target_scale, target_offset)
    best_val = float("inf"); best_state = None
    t0 = time.time()
    for ep in range(epochs):
        lam_c = _lambda_schedule(
            ep, epochs, cfg["lambda_compat"]["start"],
            cfg["lambda_compat"]["end"],
            cfg["lambda_compat"]["ramp_fraction"],
        ) if variant["compat"] else 0.0

        model.train()
        for xb, yb, cb in loader:
            ctx = _ctx_from_batch(cb, target_scale, target_offset)
            losses = total_loss(model(xb), yb, ctx, lam_c, lambda_equil,
                                data_weights, equil_mode=variant["equil_mode"])
            opt.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()

        # Model selection on val DATA loss so the criterion is identical
        # across variants (physics terms would change "total" per variant).
        model.eval()
        with torch.no_grad():
            val_losses = total_loss(model(Xva), Yva, ctx_va, lam_c,
                                    lambda_equil, data_weights,
                                    equil_mode=variant["equil_mode"])
            val_data = float(val_losses["data"])
        if val_data < best_val:
            best_val = val_data
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  [{name}] ep {ep+1:3d}/{epochs}  lam_c={lam_c:.3f}  "
                  f"val-data={val_data:.4e}  best={best_val:.4e}  "
                  f"t={time.time()-t0:.0f}s", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    return model, best_val, time.time() - t0


def predict_phys(model, X, norm, mp):
    with torch.no_grad():
        preds_n = model(X).numpy()
    return norm.inverse_transform_targets(preds_n, mp_estimate_kip_in=mp)


def constraint_metrics(pred_phys, d_total):
    """Reuses the checks inside compatibility_loss (src/physics/losses.py):
    NA inside the section, and opposite extreme-fibre strain signs
    (top compression / bottom tension)."""
    y_na = pred_phys[:, TARGET_COLUMNS.index("neutral_axis_in")]
    phi = pred_phys[:, TARGET_COLUMNS.index("curvature_1_per_in")]
    na_outside = (y_na < 0.0) | (y_na > d_total)
    eps_top = phi * (0.0 - y_na)      # must be <= 0 (compression)
    eps_bot = phi * (d_total - y_na)  # must be >= 0 (tension)
    sign_bad = (eps_top > 0.0) | (eps_bot < 0.0)
    return {
        "na_outside_section_frac": float(np.mean(na_outside)),
        "strain_sign_inconsistent_frac": float(np.mean(sign_bad)),
    }


def eval_model(model, df, norm):
    X = torch.tensor(norm.transform_features(df), dtype=torch.float32)
    pred = predict_phys(model, X, norm,
                        df["mp_estimate_kip_in"].to_numpy())
    Y = df[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    res = {tgt: metrics(Y[:, j], pred[:, j])
           for j, tgt in enumerate(TARGET_COLUMNS)}
    res["constraints"] = constraint_metrics(
        pred, df["total_depth_in"].to_numpy(dtype=np.float64))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DATA_PATH))
    p.add_argument("--out", default=str(OUT_DIR))
    p.add_argument("--config", default=str(REPO_ROOT / "configs/training.yaml"))
    p.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    p.add_argument("--smoke", action="store_true",
                   help="2 epochs, 5k rows (all five trainings).")
    args = p.parse_args()

    epochs = 2 if args.smoke else args.epochs
    n_sub = 5_000 if args.smoke else 200_000
    tag = "_smoke" if args.smoke else ""
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text())

    df_tr_full, df_tr, df_va, df_te = load_and_split(args.data, n_sub=n_sub)
    norm = FeatureNormalizer(
        data_gen_yaml=str(REPO_ROOT / "configs/data_gen.yaml"))
    norm.fit(df_tr_full)

    tensors_tr = make_tensors(df_tr, norm)
    tensors_va = make_tensors(df_va, norm)

    out = {"meta": {
        "n_train": int(len(df_tr)), "n_val": int(len(df_va)),
        "n_test": int(len(df_te)), "seed": SEED, "epochs": epochs,
        "batch_size": BATCH, "smoke": bool(args.smoke),
        "lambda_compat": cfg["lambda_compat"],
        "lambda_equil": cfg["lambda_equil"],
        "data_weights": cfg["data_weights"],
        "model_selection": "best epoch by validation data-loss "
                           "(identical criterion across variants)",
        "epochs_note": ("trimmed from 100 to fit the <=6h chain budget"
                        if epochs not in (EPOCHS_DEFAULT, 2) else "full budget"),
    }, "variants": {}, "extrapolation_probe": {}}

    # ---- three matched-budget variants on the full 200k subsample --------
    for name, variant in VARIANTS.items():
        print(f"\n[ablation] variant {name}: compat={variant['compat']} "
              f"equil={variant['equil_mode']}", flush=True)
        model, best_val, ttime = train_variant(
            name, variant, cfg, epochs, norm, tensors_tr, tensors_va)
        res = eval_model(model, df_te, norm)
        res["best_val_data_mse"] = best_val
        res["train_time_s"] = ttime
        out["variants"][name] = res
        for tgt in TARGET_COLUMNS:
            print(f"  [{name}] {tgt}: r2={res[tgt]['r2']:.4f}  "
                  f"mape={res[tgt]['mape_pct']:.2f}%", flush=True)
        print(f"  [{name}] constraints: {res['constraints']}", flush=True)

    # ---- extrapolation probe: exclude top decile of span_in --------------
    thr = float(np.quantile(df_tr["span_in"].to_numpy(), 0.9))
    tr_in = df_tr[df_tr["span_in"] < thr].reset_index(drop=True)
    tr_out = df_tr[df_tr["span_in"] >= thr].reset_index(drop=True)
    te_out = df_te[df_te["span_in"] >= thr].reset_index(drop=True)
    va_in = df_va[df_va["span_in"] < thr].reset_index(drop=True)
    print(f"\n[extrapolation] span_in 90th pct = {thr:.1f} in; "
          f"train kept {len(tr_in):,}, excluded {len(tr_out):,}; "
          f"test rows above thr = {len(te_out):,}", flush=True)
    out["extrapolation_probe"]["span_threshold_in"] = thr
    out["extrapolation_probe"]["n_train_kept"] = int(len(tr_in))
    out["extrapolation_probe"]["n_train_excluded"] = int(len(tr_out))
    out["extrapolation_probe"]["n_test_above_thr"] = int(len(te_out))

    tensors_tr_in = make_tensors(tr_in, norm)
    tensors_va_in = make_tensors(va_in, norm)
    for name in ("data_only", "compat"):
        variant = VARIANTS[name]
        print(f"\n[extrapolation] variant {name} (top-decile span "
              f"excluded)...", flush=True)
        model, best_val, ttime = train_variant(
            f"{name}-noext", variant, cfg, epochs, norm,
            tensors_tr_in, tensors_va_in)
        probe = {
            "excluded_train_decile": eval_model(model, tr_out, norm),
            "test_above_threshold": eval_model(model, te_out, norm),
            "test_full": eval_model(model, df_te, norm),
            "best_val_data_mse": best_val,
            "train_time_s": ttime,
        }
        out["extrapolation_probe"][name] = probe
        for tgt in TARGET_COLUMNS:
            print(f"  [{name}] excluded-decile {tgt}: "
                  f"r2={probe['excluded_train_decile'][tgt]['r2']:.4f}",
                  flush=True)

    out_path = out_dir / f"physics_ablation_metrics{tag}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[ablation] saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
