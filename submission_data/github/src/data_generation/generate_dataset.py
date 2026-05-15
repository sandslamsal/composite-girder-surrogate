"""Drive the full LHS -> OpenSees -> parquet dataset generation pipeline.

Each LHS sample (one composite section) becomes one moment-curvature
sweep of ``n_curvature_steps`` (default 80) points. The resulting wide
table has *one row per (sample, curvature step)* with feature columns for
the section design parameters and target columns for the OpenSees output.

The PINN can use this directly as a per-point regression target; the
section grouping is preserved via the ``sample_id`` column so train/test
splits can be made by section (avoiding leakage across the M-phi curve).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from .lhs_sampler import SectionParams, load_config, sample
from .moment_curvature import MomentCurvatureResult, analyze


# Column order is fixed so downstream PINN code can rely on it.
_FEATURE_COLUMNS = (
    "sample_id",
    "section_type",
    "span_in",
    "deck_thickness_in",
    "deck_width_in",
    "girder_spacing_in",
    "fc_deck_ksi",
    "composite_action",
    "shear_stud_stiffness_ratio",
    "fy_ksi",
    "steel_depth_in",
    "flange_width_in",
    "flange_thickness_in",
    "web_thickness_in",
    "pcb_depth_in",
    "pcb_fc_ksi",
    "pcb_prestress_ratio",
    "total_depth_in",
    "mp_estimate_kip_in",
)
_TARGET_COLUMNS = (
    "step_index",
    "curvature_1_per_in",
    "moment_kip_in",
    "axial_strain",
    "neutral_axis_in",
    "slip_in",
    "moment_ratio",
)


def run_one(params: SectionParams, n_steps: int) -> tuple[MomentCurvatureResult, dict]:
    """Run one section and return the analysis result + a feature dict
    that captures the section-level design parameters."""
    result = analyze(params, n_steps=n_steps)
    features = asdict(params)
    features["total_depth_in"] = result.section_info.total_depth_in
    features["mp_estimate_kip_in"] = result.section_info.plastic_moment_kip_in
    return result, features


def to_rows(
    result: MomentCurvatureResult, features: dict
) -> list[dict]:
    """Expand one analysis result into per-step rows for the wide table."""
    mp_est = max(features["mp_estimate_kip_in"], 1.0)
    rows = []
    for k in range(result.converged_steps):
        row = {col: features.get(col, np.nan) for col in _FEATURE_COLUMNS}
        row["step_index"] = k
        row["curvature_1_per_in"] = result.curvature[k]
        row["moment_kip_in"] = result.moment[k]
        row["axial_strain"] = result.axial_strain[k]
        row["neutral_axis_in"] = result.neutral_axis_in[k]
        row["slip_in"] = result.slip_in[k]
        row["moment_ratio"] = result.moment[k] / mp_est
        rows.append(row)
    return rows


def generate(
    config_path: str | Path,
    n_samples: int,
    out_path: str | Path,
    *,
    seed: int | None = None,
    progress: bool = True,
) -> dict:
    """Generate ``n_samples`` sections, run M-phi on each, write parquet.

    Returns a small dict of summary statistics for the run."""
    cfg = load_config(config_path)
    n_steps = cfg["analysis"]["n_curvature_steps"]
    samples = sample(cfg, n=n_samples, seed=seed)

    all_rows: list[dict] = []
    n_invalid = 0
    n_partial = 0

    iterator: Iterable[SectionParams] = samples
    if progress:
        iterator = tqdm(samples, desc="OpenSeesPy M-phi", unit="section")

    for params in iterator:
        result, features = run_one(params, n_steps=n_steps)
        if not result.physically_valid:
            n_invalid += 1
            continue
        if not result.fully_converged:
            n_partial += 1
        all_rows.extend(to_rows(result, features))

    if not all_rows:
        raise RuntimeError(
            "no physically valid samples — check parameter ranges and section model"
        )

    df = pd.DataFrame(all_rows)
    df = df[list(_FEATURE_COLUMNS) + list(_TARGET_COLUMNS)]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    stats = {
        "n_requested": n_samples,
        "n_invalid": n_invalid,
        "n_partial_converged": n_partial,
        "n_rows": len(df),
        "n_sections_kept": df["sample_id"].nunique(),
        "out_path": str(out_path),
    }
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate composite-girder M-phi dataset.")
    p.add_argument("--config", default="configs/data_gen.yaml")
    p.add_argument("--n", type=int, default=100, help="number of sections to sample")
    p.add_argument("--out", default="data/raw/sections.parquet")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    stats = generate(
        config_path=args.config,
        n_samples=args.n,
        out_path=args.out,
        seed=args.seed,
        progress=not args.no_progress,
    )
    print()
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
