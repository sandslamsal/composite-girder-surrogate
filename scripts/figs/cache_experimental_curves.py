#!/usr/bin/env python
"""Re-run the section- and beam-level models on the 24 published test
points and cache the M-phi curves so the figure script can be iterated
without paying for the OpenSees sweeps again.

Reuses the loading / model-entry / stud-layout logic of
scripts/validate_models_experimental.py verbatim (imported, not copied),
so the cached curves are exactly the ones behind
reports/model_validation/per_specimen.csv.

Output
------
<scratch>/exp_curves.npz  (one M, phi array pair per specimen key)

Usage
-----
/opt/anaconda3/envs/ops_x86/bin/python \
    scripts/figs/cache_experimental_curves.py --out <path.npz>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from validate_models_experimental import (  # noqa: E402
    load_tests, run_section_model, run_beam_model, specimen_key,
    stud_count_by_specimen,
)

CSV = REPO / "data/experimental/literature_tests.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-steps-section", type=int, default=160)
    ap.add_argument("--n-steps-beam", type=int, default=100)
    args = ap.parse_args()

    df = load_tests(CSV)
    counts = stud_count_by_specimen(df)
    arrays: dict = {}
    index: dict = {}

    seen: dict = {}
    for _, row in df.iterrows():
        key = specimen_key(row)
        if key not in seen:
            seen[key] = f"sp{len(seen):02d}"
            t0 = time.time()
            sec = run_section_model(row, args.n_steps_section)
            arrays[f"{seen[key]}_sec_M"] = np.asarray(sec.moment, float)
            arrays[f"{seen[key]}_sec_phi"] = np.asarray(sec.curvature, float)
            dt_s = time.time() - t0
            t0 = time.time()
            try:
                bres, _bp, note = run_beam_model(row, args.n_steps_beam,
                                                 counts.get(key))
                arrays[f"{seen[key]}_beam_M"] = np.asarray(bres.moment, float)
                arrays[f"{seen[key]}_beam_phi"] = np.asarray(bres.curvature,
                                                             float)
                arrays[f"{seen[key]}_beam_slip"] = np.asarray(
                    bres.max_slip_in, float)
            except Exception as exc:  # pragma: no cover
                note = f"FAILED {exc}"
            dt_b = time.time() - t0
            print(f"  {seen[key]} {row.test_id:10s} sec {dt_s:5.1f}s "
                  f"beam {dt_b:5.1f}s  [{note}]", flush=True)
        index[str(row.test_id)] = seen[key]

    out = Path(args.out)
    np.savez_compressed(out, **arrays)
    out.with_suffix(".index.json").write_text(json.dumps(index, indent=1))
    print(f"[out] {out}  ({len(seen)} specimens)")


if __name__ == "__main__":
    main()
