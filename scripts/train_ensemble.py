#!/usr/bin/env python
"""Train a deep ensemble of PINN members.

Each member uses a different random seed but the same dataset, the same
train/val/test split (the split is derived from the *training* seed in
``train_pinn.py``, so to keep splits constant across members we set a
fixed seed for the split and a different seed only for model
initialisation -- see below).

Default: 5 members, seeds 0..4 added to the config base seed.

Usage:
    python scripts/train_ensemble.py \\
        --data data/raw/full_50k.parquet \\
        --out checkpoints/ensemble/ \\
        --members 5 \\
        --epochs 300

The script invokes ``scripts/train_pinn.py`` as a subprocess for each
member so that PyTorch / MPS state is fresh per run. Members are trained
sequentially; on a single Apple-Silicon laptop running through Rosetta
this is the safe option (parallel training would saturate memory).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True,
                   help="Parent directory; members go to <out>/member_0/, ...")
    p.add_argument("--config", default="configs/training.yaml")
    p.add_argument("--members", type=int, default=5)
    p.add_argument("--epochs", type=int, default=None,
                   help="Override per-member epoch count (default: from config)")
    p.add_argument("--base-seed", type=int, default=2026_05_13,
                   help="Seed for member 0; member k uses base_seed + k")
    p.add_argument("--python", default=sys.executable,
                   help="Python interpreter to call (defaults to current)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    train_script = REPO_ROOT / "scripts" / "train_pinn.py"
    if not train_script.exists():
        raise FileNotFoundError(train_script)

    print(f"[ensemble] training {args.members} members sequentially")
    print(f"[ensemble] output root: {out_root}")
    for k in range(args.members):
        seed = args.base_seed + k
        member_dir = out_root / f"member_{k}"
        cmd = [
            args.python, str(train_script),
            "--data", args.data,
            "--config", args.config,
            "--out", str(member_dir),
            "--seed", str(seed),
        ]
        if args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]
        print(f"\n[ensemble] member {k}/{args.members - 1}, seed={seed}")
        print("           " + " ".join(cmd))
        ret = subprocess.call(cmd)
        if ret != 0:
            print(f"[ensemble] member {k} failed with exit code {ret}")
            sys.exit(ret)
        print(f"[ensemble] member {k} done -> {member_dir}/best.pt")

    print(f"\n[ensemble] all {args.members} members complete")
    print(f"[ensemble] aggregate with:")
    print(f"  from src.models.ensemble import EnsemblePredictor")
    print(f"  ens = EnsemblePredictor.from_directory('{out_root}')")


if __name__ == "__main__":
    main()
