"""Shared helpers for the journal-revision baseline scripts.

Reproduces EXACTLY the data pipeline of the paper's compute-matched
baselines (scripts/run_mlps_200k.py and scripts/run_xgb_only.py):

* 80/10/10 split by ``sample_id`` with ``np.random.default_rng(SEED)``
* 200,000-row train subsample drawn with the SAME rng call
  (``default_rng(SEED).choice(len(train_split), size=200_000)``)
* R2 / RMSE / MAPE metric definitions from run_mlps_200k.py

Deliberately torch-free so the XGBoost runner can import it without
pulling a second OpenMP runtime into the process.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "raw" / "full_50k.parquet"
OUT_DIR = REPO_ROOT / "reports" / "baselines_revision"

SEED = 20260513
N_SUB = 200_000
SPLIT_FRACS = {"train": 0.8, "val": 0.1, "test": 0.1}


def split_by_sample(df: pd.DataFrame, seed: int = SEED):
    """80/10/10 split by sample_id — identical to run_mlps_200k.split()."""
    ids = df["sample_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_tr = int(round(SPLIT_FRACS["train"] * n))
    n_va = int(round(SPLIT_FRACS["val"] * n))
    return (
        df[df["sample_id"].isin(set(ids[:n_tr]))].reset_index(drop=True),
        df[df["sample_id"].isin(set(ids[n_tr:n_tr + n_va]))].reset_index(drop=True),
        df[df["sample_id"].isin(set(ids[n_tr + n_va:]))].reset_index(drop=True),
    )


def subsample_train(df_tr: pd.DataFrame, n_sub: int = N_SUB, seed: int = SEED):
    """Row-level train subsample with the SAME rng draw as the existing
    200k baselines (run_mlps_200k.py / run_xgb_only.py)."""
    if len(df_tr) <= n_sub:
        return df_tr.reset_index(drop=True)
    idx = np.random.default_rng(seed).choice(len(df_tr), size=n_sub, replace=False)
    return df_tr.iloc[idx].reset_index(drop=True)


def metrics(y: np.ndarray, yh: np.ndarray) -> dict:
    """R2 / RMSE / MAPE, matching run_mlps_200k.metrics()."""
    y = np.asarray(y, dtype=np.float64)
    yh = np.asarray(yh, dtype=np.float64)
    r2 = 1.0 - np.sum((y - yh) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12)
    rmse = float(np.sqrt(np.mean((y - yh) ** 2)))
    eps = 1e-6
    denom = np.where(np.abs(y) > eps, np.abs(y), eps)
    mape = float(100.0 * np.mean(np.abs((yh - y) / denom)))
    return {"r2": float(r2), "rmse": rmse, "mape_pct": mape}


def load_and_split(data_path: str | Path = DATA_PATH, n_sub: int = N_SUB,
                   seed: int = SEED, verbose: bool = True):
    """Load parquet, split by sample_id, subsample train to n_sub rows.

    Returns (df_tr_full, df_tr_sub, df_va, df_te). ``df_tr_full`` is the
    un-subsampled train split — run_mlps_200k.py fits its FeatureNormalizer
    on that (target min-max from the full train split), so scripts that
    normalise should do the same for bit-compatible target scaling.
    """
    df = pd.read_parquet(data_path)
    if verbose:
        print(f"[data] {Path(data_path).name}: {len(df):,} rows, "
              f"{df['sample_id'].nunique():,} samples", flush=True)
    df_tr_full, df_va, df_te = split_by_sample(df, seed)
    if verbose:
        print(f"[split] tr={len(df_tr_full):,} va={len(df_va):,} "
              f"te={len(df_te):,}", flush=True)
    df_tr_sub = subsample_train(df_tr_full, n_sub, seed)
    if verbose:
        print(f"[subsample] tr -> {len(df_tr_sub):,} rows", flush=True)
    return df_tr_full, df_tr_sub, df_va, df_te
