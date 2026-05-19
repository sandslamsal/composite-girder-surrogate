"""Nie & Cai (2003) effective-rigidity formula for partially composite
beams, applied across the LHS-sampled bridge-girder design space.

Reference:
    Nie, J. and Cai, C.S. (2003). "Steel-Concrete Composite Beams
    Considering Shear Slip Effects." Journal of Structural Engineering,
    129(4), 495-506. DOI: 10.1061/(ASCE)0733-9445(2003)129:4(495).

The formula gives an effective flexural rigidity B that accounts for
the stiffness reduction due to interface slip:

    B = EI / (1 + xi)              [Eq. 32]
    xi = eta * (0.4 - 3/(alpha L)^2)    [Eq. 31, general loading]

where EI is the transformed-section rigidity, and (eta, alpha, beta)
are functions of section geometry, materials, stud stiffness K, and
stud pitch p. The formula was calibrated against ~36 simply-supported
and continuous composite-beam tests (Chapman & Balakrishnan 1964;
Davies 1969; Wang & Nie 1992; Li 1984; Newmark et al. 1951;
Hope-Gill & Johnson 1976; Ansourian 1981; plus the authors' own
specimens SCB-1..4 and CCB-1..2).

Parameter mapping from our LHS samples:
    - composite_action eta_c <-> Nie & Cai k_p = Sum(Q_n)/C_f
    - shear_stud_stiffness_ratio scales typical stud stiffness K_max
    - stud pitch p derived from k_p, C_f, and assumed nominal
      stud capacity Q_u (3/4" headed stud, AISC value 30 kip)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# AISC / Ollgaard et al. (1971) values for 3/4" headed-stud connectors
# in normal-weight concrete. These are the typical reference values; a
# sensitivity analysis is included in the supplementary release.
Q_U_PER_STUD_KIP = 30.0      # nominal ultimate shear strength
K_MAX_PER_STUD_KIP_PER_IN = 1500.0   # secant stiffness (Q_u / 0.02")
N_STUDS_PER_ROW = 2
E_S_KSI = 29000.0


def _Ec_ksi(fc_ksi: float) -> float:
    """AASHTO/ACI normal-weight concrete modulus."""
    return 1820.0 * np.sqrt(fc_ksi)


def _steel_I_inertia(d_s: float, b_f: float, t_f: float, t_w: float) -> float:
    """Moment of inertia of a symmetric I-section about its centroidal
    axis, using the box-minus-cutout decomposition."""
    return (b_f * d_s**3) / 12.0 - ((b_f - t_w) * (d_s - 2 * t_f) ** 3) / 12.0


def _steel_area(d_s: float, b_f: float, t_f: float, t_w: float) -> float:
    return 2.0 * b_f * t_f + (d_s - 2.0 * t_f) * t_w


def nie_cai_effective_rigidity_row(row: pd.Series) -> dict:
    """Compute B_eff for a single LHS-sample row.

    Required columns (subset of full_50k.parquet schema):
        span_in, deck_thickness_in, deck_width_in,
        fc_deck_ksi, fy_ksi, composite_action,
        shear_stud_stiffness_ratio, steel_depth_in,
        flange_width_in, flange_thickness_in, web_thickness_in.

    Returns a dict with EI_transformed, xi, B_effective, plus the
    intermediate quantities (alpha L, eta) for diagnostic plots.
    """
    # --- geometry ---
    b_c = float(row["deck_width_in"])
    t_c = float(row["deck_thickness_in"])
    d_s = float(row["steel_depth_in"])
    b_f = float(row["flange_width_in"])
    t_f = float(row["flange_thickness_in"])
    t_w = float(row["web_thickness_in"])
    L = float(row["span_in"])

    # --- materials ---
    f_c = float(row["fc_deck_ksi"])
    f_y = float(row["fy_ksi"])
    E_s = E_S_KSI
    E_c = _Ec_ksi(f_c)
    n_modular = E_s / E_c

    # --- section properties ---
    A_s = _steel_area(d_s, b_f, t_f, t_w)
    A_c = b_c * t_c
    I_s = _steel_I_inertia(d_s, b_f, t_f, t_w)
    I_c = (b_c * t_c**3) / 12.0

    A_0 = (A_s * A_c) / (n_modular * A_s + A_c)
    I_0 = I_c / n_modular + I_s
    y_cb = t_c / 2.0           # concrete centroid distance from interface
    y_st = d_s / 2.0           # steel centroid distance from interface
    d_c = y_cb + y_st
    h = t_c + d_s

    A_1 = A_0 / (I_0 + A_0 * d_c**2)

    # Transformed-section rigidity
    EI = E_s * (I_0 + A_0 * d_c**2)

    # --- shear connection ---
    k_p = float(row["composite_action"])                          # Nie & Cai's k_p
    stiff_ratio = float(row["shear_stud_stiffness_ratio"])
    K = K_MAX_PER_STUD_KIP_PER_IN * stiff_ratio                   # per-stud secant stiffness

    # Compression force at full composite (yield-limited)
    C_f = min(A_s * f_y, 0.85 * f_c * b_c * t_c)
    # Stud pitch (from Nie & Cai Eq. 47): p = (1/k_p) * 0.5 * L * n_s * Q_n / C_f
    p_full = 0.5 * L * N_STUDS_PER_ROW * Q_U_PER_STUD_KIP / max(C_f, 1.0)
    p = max(p_full / max(k_p, 1e-3), 0.5)                        # avoid division blowup

    # Nie & Cai dimensionless groups (Eq. 11)
    alpha_sq = K / (A_1 * E_s * I_0 * p)
    alpha = np.sqrt(max(alpha_sq, 1e-12))
    aL = alpha * L
    beta = A_1 * d_c * p / max(K, 1e-9)

    # eta and xi (Eq. 28-31)
    eta = 24.0 * EI * beta / (L**2 * h)
    # Generalized loading form (Eq. 31): xi = eta * (0.4 - 3/(alpha L)^2)
    xi_raw = eta * (0.4 - 3.0 / max(aL**2, 1e-9))
    xi = max(xi_raw, 0.0)            # physical: rigidity reduction is non-negative

    B_eff = EI / (1.0 + xi)

    return {
        "EI_transformed_kip_in2": EI,
        "xi_slip_correction": xi,
        "B_effective_kip_in2": B_eff,
        "alpha_L": aL,
        "eta_geom": eta,
    }


def apply_nie_cai(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised application of the row-wise function. Returns a new
    DataFrame with the original columns plus EI_transformed,
    xi_slip_correction, B_effective, alpha_L, eta_geom, and the derived
    Nie-Cai curvature phi_niecai = M / B_eff."""
    rows = []
    for _, row in df.iterrows():
        rows.append(nie_cai_effective_rigidity_row(row))
    out = df.copy()
    nc = pd.DataFrame(rows, index=df.index)
    for col in ["EI_transformed_kip_in2", "xi_slip_correction",
                "B_effective_kip_in2", "alpha_L", "eta_geom"]:
        out[col] = nc[col].values
    # Predicted curvature: phi = M / B (elastic regime)
    out["phi_niecai_1_per_in"] = out["moment_kip_in"] / out["B_effective_kip_in2"]
    return out
