"""AASHTO LRFD transformed-section comparison (Validation Tier 2).

The proposal's headline contribution is a *quantification* of the error
introduced by the AASHTO simplified transformed-section approach when
partial composite action is present. This module implements the AASHTO
approach for a steel-I + concrete-deck section and exposes a single API
for sweeping a parquet of OpenSees results and computing the elastic-
range disagreement, stratified by degree-of-composite-action.

AASHTO assumes *full* composite action regardless of η_c, plus linear-
elastic material behaviour and a fixed neutral axis. Our OpenSees data
captures both partial composite action and material nonlinearity, so the
AASHTO error grows with both (1) decreasing η_c and (2) the moment
ratio approaching plastic capacity.

References
----------
AASHTO LRFD Bridge Design Specifications (9th ed., 2020) §6.10 (composite
flexural members). E_c per ACI 318: E_c = 1802 √f'c (ksi).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import math
import numpy as np
import pandas as pd


# Concrete modulus per the ACI simple formula in ksi units:
#   E_c (psi) = 57000 √f'c (psi)
#   -> E_c (ksi) = 57 √(f'c [psi]) = 57 √(1000 · f'c [ksi]) ≈ 1802 √f'c [ksi]
_EC_COEF_KSI = 57.0 * math.sqrt(1000.0)   # ≈ 1802.0

_E_STEEL_KSI = 29_000.0


@dataclass
class AASHTOPrediction:
    """Elastic transformed-section properties + predictions at a load level.

    All quantities use the same kip/inch unit system as the rest of the
    pipeline. ``y_na_in`` is measured from the top of the deck.
    """

    section_id: int
    eta_c: float
    n_modular: float                 # E_s / E_c
    transformed_area_in2: float
    transformed_inertia_in4: float   # I_tr
    y_na_in: float                   # depth from top of deck
    # Curvature at the applied moment ratio (elastic prediction):
    moment_kip_in: float             # input applied moment
    curvature_aashto_1_per_in: float


def concrete_modulus_ksi(fc_ksi: float) -> float:
    """ACI-style elastic modulus for normal-weight concrete (ksi)."""
    return _EC_COEF_KSI * math.sqrt(max(fc_ksi, 1e-6))


def steel_i_area_in2(d_s: float, b_f: float, t_f: float, t_w: float) -> float:
    """Cross-sectional area of a steel I section with two equal flanges and
    a single web."""
    web_height = max(d_s - 2.0 * t_f, 0.0)
    return 2.0 * b_f * t_f + web_height * t_w


def steel_i_centroid_from_bottom_in(
    d_s: float, b_f: float, t_f: float, t_w: float
) -> float:
    """For a doubly-symmetric I section, the centroid is at half-depth."""
    return d_s / 2.0


def steel_i_inertia_about_own_centroid_in4(
    d_s: float, b_f: float, t_f: float, t_w: float
) -> float:
    """Second moment of area of a doubly-symmetric I about its own
    horizontal centroidal axis. Uses the standard textbook decomposition:
    two flanges + web."""
    web_height = max(d_s - 2.0 * t_f, 0.0)
    i_web = (t_w * web_height ** 3) / 12.0
    flange_offset = (d_s - t_f) / 2.0
    i_flange = (b_f * t_f ** 3) / 12.0
    i_flanges = 2.0 * (i_flange + b_f * t_f * flange_offset ** 2)
    return i_web + i_flanges


def transformed_section_properties(
    deck_thickness_in: float,
    deck_width_in: float,
    fc_deck_ksi: float,
    steel_depth_in: float,
    flange_width_in: float,
    flange_thickness_in: float,
    web_thickness_in: float,
) -> tuple[float, float, float, float]:
    """Compute AASHTO transformed-section (deck-as-steel-equivalent) elastic
    properties for a positive-moment composite section.

    Returns ``(area_tr, inertia_tr, y_na_from_top_in, n)`` where ``y_na`` is
    measured from the top of the deck downward, ``n = E_s/E_c``.
    """
    e_c = concrete_modulus_ksi(fc_deck_ksi)
    n = _E_STEEL_KSI / e_c

    # Deck: equivalent steel width = b_eff / n, full thickness t_s.
    a_deck_eq = (deck_width_in / n) * deck_thickness_in
    y_deck_centroid = deck_thickness_in / 2.0   # from top of deck

    # Steel I: centroid at deck_thickness + d_s/2 below deck top.
    a_steel = steel_i_area_in2(
        steel_depth_in, flange_width_in, flange_thickness_in, web_thickness_in
    )
    y_steel_centroid = deck_thickness_in + steel_depth_in / 2.0

    a_total = a_deck_eq + a_steel
    if a_total < 1e-9:
        return a_total, 0.0, 0.0, n
    y_na = (a_deck_eq * y_deck_centroid + a_steel * y_steel_centroid) / a_total

    # I_tr about NA: use parallel axis theorem.
    i_deck = (deck_width_in / n) * deck_thickness_in ** 3 / 12.0
    i_deck += a_deck_eq * (y_deck_centroid - y_na) ** 2

    i_steel_own = steel_i_inertia_about_own_centroid_in4(
        steel_depth_in, flange_width_in, flange_thickness_in, web_thickness_in
    )
    i_steel = i_steel_own + a_steel * (y_steel_centroid - y_na) ** 2

    return a_total, i_deck + i_steel, y_na, n


def predict_one_section(
    row: pd.Series, applied_moment_kip_in: float
) -> AASHTOPrediction:
    """AASHTO transformed-section prediction for a single (section, load
    level) point. The row carries the section design parameters; the
    applied moment is supplied separately."""
    a_tr, i_tr, y_na, n = transformed_section_properties(
        row["deck_thickness_in"],
        row["deck_width_in"],
        row["fc_deck_ksi"],
        row["steel_depth_in"],
        row["flange_width_in"],
        row["flange_thickness_in"],
        row["web_thickness_in"],
    )
    phi = applied_moment_kip_in / (_E_STEEL_KSI * i_tr) if i_tr > 0 else float("nan")
    return AASHTOPrediction(
        section_id=int(row["sample_id"]),
        eta_c=float(row["composite_action"]),
        n_modular=n,
        transformed_area_in2=a_tr,
        transformed_inertia_in4=i_tr,
        y_na_in=y_na,
        moment_kip_in=applied_moment_kip_in,
        curvature_aashto_1_per_in=phi,
    )


def compare_aashto_vs_opensees(
    df: pd.DataFrame, eta_bins: Iterable[tuple[float, float]] | None = None,
    moment_ratio_max: float = 0.6,
) -> pd.DataFrame:
    """For each row in the OpenSees dataset (elastic regime only by default),
    compute the AASHTO transformed-section prediction at the same applied
    moment and return a comparison table.

    Parameters
    ----------
    df
        Parquet row table produced by :mod:`generate_dataset`.
    eta_bins
        List of ``(lo, hi)`` bins of composite action for stratification.
        Defaults to four bins: 25-50, 50-70, 70-90, 90-100 %.
    moment_ratio_max
        Only include rows with ``moment_ratio <= moment_ratio_max`` in the
        elastic comparison. AASHTO is a linear-elastic prediction; comparing
        it past concrete cracking or steel yielding conflates two distinct
        sources of error.

    Returns
    -------
    pd.DataFrame
        Per-row comparison with columns: sample_id, composite_action,
        moment_ratio, moment_kip_in, phi_opensees, phi_aashto, phi_error,
        phi_error_pct, y_na_opensees, y_na_aashto, y_na_error_in, eta_bin.
    """
    if eta_bins is None:
        eta_bins = [(0.25, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0001)]

    mask = (df["moment_ratio"] > 0.0) & (df["moment_ratio"] <= moment_ratio_max)
    sub = df[mask].copy()

    a_tr = np.empty(len(sub))
    i_tr = np.empty(len(sub))
    y_na_aashto = np.empty(len(sub))
    n_arr = np.empty(len(sub))
    for i, (_, row) in enumerate(sub.iterrows()):
        a, ii, y_na, n = transformed_section_properties(
            row["deck_thickness_in"],
            row["deck_width_in"],
            row["fc_deck_ksi"],
            row["steel_depth_in"],
            row["flange_width_in"],
            row["flange_thickness_in"],
            row["web_thickness_in"],
        )
        a_tr[i] = a
        i_tr[i] = ii
        y_na_aashto[i] = y_na
        n_arr[i] = n

    phi_aashto = sub["moment_kip_in"].to_numpy() / (_E_STEEL_KSI * i_tr)
    phi_opensees = sub["curvature_1_per_in"].to_numpy()
    phi_error = phi_aashto - phi_opensees
    phi_error_pct = 100.0 * phi_error / np.where(np.abs(phi_opensees) > 1e-12,
                                                  phi_opensees, np.nan)

    y_na_opensees = sub["neutral_axis_in"].to_numpy()
    y_na_error_in = y_na_aashto - y_na_opensees

    eta = sub["composite_action"].to_numpy()
    bin_labels = np.empty(len(sub), dtype=object)
    for lo, hi in eta_bins:
        bin_mask = (eta >= lo) & (eta < hi)
        bin_labels[bin_mask] = f"{int(lo*100)}-{int(min(hi, 1.0)*100)}%"

    out = pd.DataFrame({
        "sample_id": sub["sample_id"].to_numpy(),
        "composite_action": eta,
        "moment_ratio": sub["moment_ratio"].to_numpy(),
        "moment_kip_in": sub["moment_kip_in"].to_numpy(),
        "phi_opensees_1_per_in": phi_opensees,
        "phi_aashto_1_per_in": phi_aashto,
        "phi_error_1_per_in": phi_error,
        "phi_error_pct": phi_error_pct,
        "y_na_opensees_in": y_na_opensees,
        "y_na_aashto_in": y_na_aashto,
        "y_na_error_in": y_na_error_in,
        "transformed_inertia_in4": i_tr,
        "modular_ratio_n": n_arr,
        "eta_bin": bin_labels,
    })
    return out


def summarise_by_eta_bin(comparison: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-bin error statistics. AASHTO is expected to
    *overestimate* stiffness when partial composite action is present,
    i.e. AASHTO predicts a *smaller* curvature for a given moment than the
    actual section achieves. Negative phi_error_pct means AASHTO is
    stiffer (under-predicting curvature)."""
    g = comparison.groupby("eta_bin", dropna=True)
    summary = g.agg(
        n_rows=("phi_error_pct", "count"),
        phi_error_pct_mean=("phi_error_pct", "mean"),
        phi_error_pct_median=("phi_error_pct", "median"),
        phi_error_pct_p10=("phi_error_pct", lambda x: np.percentile(x, 10)),
        phi_error_pct_p90=("phi_error_pct", lambda x: np.percentile(x, 90)),
        y_na_error_mean_in=("y_na_error_in", "mean"),
    ).reset_index()
    return summary
