"""Tier-3 experimental comparison: surrogate vs published beam test data.

The shipped CSV contains digitised values from Chapman & Balakrishnan
(1964), Nie & Cai (2003), and Ansourian (1982). Note that all three of
those test programmes used laboratory-scale specimens (~10-15 ft spans,
~8 in steel depth, f_y ~32-37 ksi), which are below the bridge-girder
training distribution; comparisons here are therefore extrapolation,
not validation. The pipeline is kept as supplementary code.

CSV schema (one row per test point — a section + a specific load level):

    test_id, source, citation, section_type,
    span_in, deck_thickness_in, deck_width_in, girder_spacing_in,
    fc_deck_ksi, composite_action, shear_stud_stiffness_ratio,
    fy_ksi, steel_depth_in, flange_width_in, flange_thickness_in,
    web_thickness_in,
    measured_moment_kip_in, measured_curvature_1_per_in,
    measured_slip_in, notes

The first sixteen columns are exactly the section design + load-state
features the surrogate consumes (after deriving ``total_depth_in`` and
``moment_ratio`` here). The three ``measured_*`` columns are the
ground-truth experimental observations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.inference import PINNPredictor


_E_C_KSI = 1802.0  # placeholder coefficient; only used if mp_estimate is
                   # not supplied externally.


def _estimate_plastic_moment_steel_i(row: pd.Series) -> float:
    """Quick M_p estimate matching ``composite_section._estimate_plastic_moment_steel_i``.

    Necessary because the experimental CSV does not include
    ``mp_estimate_kip_in`` (we'd never have that for a literature test);
    we recompute it from the section geometry so the same per-row
    moment-ratio framing the PINN was trained on still applies."""
    fy = row["fy_ksi"]
    fc = row["fc_deck_ksi"]
    b_eff = row["deck_width_in"] * row["composite_action"]
    d_s = row["steel_depth_in"]
    b_f = row["flange_width_in"]
    t_f = row["flange_thickness_in"]
    t_w = row["web_thickness_in"]
    a_s = 2.0 * b_f * t_f + (d_s - 2.0 * t_f) * t_w
    c_force_steel = a_s * fy
    a_block = c_force_steel / (0.85 * fc * b_eff) if b_eff > 0 else row["deck_thickness_in"]
    a_block = min(a_block, row["deck_thickness_in"])
    y_steel_centroid = row["deck_thickness_in"] + d_s / 2.0
    y_block_centroid = a_block / 2.0
    return float(c_force_steel * (y_steel_centroid - y_block_centroid))


def load_experimental_csv(path: str | Path) -> pd.DataFrame:
    """Load the experimental CSV and add derived columns needed by the
    surrogate predictor (``total_depth_in``, ``mp_estimate_kip_in``,
    ``moment_ratio``, ``sample_id``, ``step_index``)."""
    df = pd.read_csv(path)
    df["total_depth_in"] = df["deck_thickness_in"] + df["steel_depth_in"]
    df["mp_estimate_kip_in"] = df.apply(_estimate_plastic_moment_steel_i, axis=1)
    df["moment_ratio"] = df["measured_moment_kip_in"] / df["mp_estimate_kip_in"]
    df["moment_ratio"] = df["moment_ratio"].clip(0.0, 1.2)
    # Placeholders so the row passes the normaliser
    df["sample_id"] = np.arange(len(df))
    df["step_index"] = 0
    return df


def compare_surrogate_to_experiment(
    predictor: PINNPredictor, experimental_df: pd.DataFrame,
) -> pd.DataFrame:
    """Run the surrogate on the experimental rows and return a side-by-side
    comparison dataframe with relative errors."""
    pred = predictor.predict(experimental_df)
    out = experimental_df[[
        "test_id", "source", "section_type", "composite_action",
        "measured_moment_kip_in", "measured_curvature_1_per_in",
        "measured_slip_in",
    ]].copy()
    out["surrogate_neutral_axis_in"] = pred["neutral_axis_in"].to_numpy()
    out["surrogate_curvature_1_per_in"] = pred["curvature_1_per_in"].to_numpy()
    out["surrogate_slip_in"] = pred["slip_in"].to_numpy()

    def _rel(meas, model, eps=1e-9):
        return 100.0 * (model - meas) / np.where(np.abs(meas) > eps, meas, np.nan)

    out["curvature_rel_error_pct"] = _rel(
        experimental_df["measured_curvature_1_per_in"].to_numpy(),
        pred["curvature_1_per_in"].to_numpy(),
    )
    out["slip_rel_error_pct"] = _rel(
        experimental_df["measured_slip_in"].to_numpy(),
        pred["slip_in"].to_numpy(),
    )
    return out


def summarise(comparison: pd.DataFrame) -> pd.DataFrame:
    """Mean / median / MAPE per measurement column for the Tier-3 table."""
    rows = []
    for col_meas, col_model, col_rel, label in [
        ("measured_curvature_1_per_in", "surrogate_curvature_1_per_in",
         "curvature_rel_error_pct", "curvature (1/in)"),
        ("measured_slip_in", "surrogate_slip_in",
         "slip_rel_error_pct", "slip (in)"),
    ]:
        y = comparison[col_meas].to_numpy()
        yh = comparison[col_model].to_numpy()
        finite = np.isfinite(y) & np.isfinite(yh)
        y, yh = y[finite], yh[finite]
        rel = comparison[col_rel].to_numpy()[finite]
        rows.append({
            "quantity": label,
            "n_tests": int(finite.sum()),
            "rmse": float(np.sqrt(np.mean((yh - y) ** 2))) if len(y) else float("nan"),
            "mape_pct": float(np.mean(np.abs(rel))) if len(rel) else float("nan"),
            "max_abs_rel_pct": float(np.max(np.abs(rel))) if len(rel) else float("nan"),
        })
    return pd.DataFrame(rows)
