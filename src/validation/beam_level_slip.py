"""Beam-level slip validator.

Builds a simply-supported composite beam in OpenSeesPy with two
parallel elastic-beam strings (deck + steel girder) connected by
discrete horizontal shear-connector springs. Under uniform end moment
the model produces a moment-curvature response *and* a true interface
slip field; the midspan slip is the quantity we compare against the
analytical surrogate Eq.~1.

This module is **not** used to train the surrogate; it exists to defend the
analytical slip formula against reviewer scrutiny by showing that the
formula agrees with a physically richer model across the
composite-action range.

Usage
-----

>>> from src.validation.beam_level_slip import (
...     run_beam_level, compare_to_analytical,
... )
>>> result = run_beam_level(params, n_elements=10)
>>> result.slip_midspan_at_moment(M=1500.0)  # kip-in -> in

The full sweep can be driven by ``scripts/validate_slip.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import openseespy.opensees as ops

from src.data_generation.lhs_sampler import SectionParams


# Material tags used by this module (distinct from the section-builder set so
# the two modules don't collide if invoked together)
_MAT_DECK_ELASTIC = 11
_MAT_STEEL_GIRDER_ELASTIC = 12
_MAT_SHEAR_CONNECTOR = 13

_E_STEEL_KSI = 29_000.0
_EC_COEF_KSI = 1802.0   # ACI elastic concrete modulus, ksi


def _elastic_section_props(params: SectionParams) -> tuple[float, float, float, float]:
    """Compute elastic axial and flexural rigidities for the deck and
    girder strings, using their own centroids."""
    fc = params.fc_deck_ksi
    E_c = _EC_COEF_KSI * np.sqrt(max(fc, 1e-6))

    # Deck: rectangle b_eff * t_s about its own centroid
    b_eff = params.deck_width_in * params.composite_action
    t_s = params.deck_thickness_in
    A_deck = b_eff * t_s
    I_deck = b_eff * t_s ** 3 / 12.0
    EA_deck = E_c * A_deck
    EI_deck = E_c * I_deck

    # Steel I-section (W or plate)
    d_s = params.steel_depth_in
    b_f = params.flange_width_in
    t_f = params.flange_thickness_in
    t_w = params.web_thickness_in
    web_h = max(d_s - 2.0 * t_f, 0.0)
    A_steel = 2.0 * b_f * t_f + web_h * t_w
    # I about the I-section's own centroidal axis (doubly symmetric I)
    i_web = t_w * web_h ** 3 / 12.0
    flange_offset = (d_s - t_f) / 2.0
    i_flange = b_f * t_f ** 3 / 12.0
    I_steel = i_web + 2.0 * (i_flange + b_f * t_f * flange_offset ** 2)
    EA_steel = _E_STEEL_KSI * A_steel
    EI_steel = _E_STEEL_KSI * I_steel
    return EA_deck, EI_deck, EA_steel, EI_steel


@dataclass
class BeamLevelResult:
    """Beam-level slip-validation output for one section."""

    section_id: int
    moment_kip_in: np.ndarray         # applied moment at each load step
    curvature_midspan: np.ndarray     # 1/in
    slip_midspan_in: np.ndarray       # in
    eta_c: float

    def slip_at_moment(self, M: float) -> float:
        """Linear interpolation of slip at a given moment level."""
        if M <= self.moment_kip_in[0]:
            return float(self.slip_midspan_in[0])
        if M >= self.moment_kip_in[-1]:
            return float(self.slip_midspan_in[-1])
        return float(np.interp(M, self.moment_kip_in, self.slip_midspan_in))


def run_beam_level(
    params: SectionParams,
    n_elements: int = 10,
    n_load_steps: int = 30,
    moment_ratio_max: float = 0.8,
) -> BeamLevelResult:
    """Build and analyse a simply-supported composite beam with discrete
    shear connectors. Returns sampled (M, phi, slip) at midspan over
    ``n_load_steps`` evenly-spaced applied-moment levels."""
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    L = params.span_in
    n_nodes = n_elements + 1
    dx = L / n_elements

    # Vertical offsets so the deck centroid sits above the girder centroid
    y_deck = -(params.deck_thickness_in / 2.0)
    y_girder = params.deck_thickness_in / 2.0 + params.steel_depth_in / 2.0
    e_offset = y_girder - y_deck            # vertical eccentricity, in

    # ---- nodes ----------------------------------------------------------
    # Steel girder nodes use tags 1..n_nodes, deck nodes use 100+i.
    for i in range(n_nodes):
        ops.node(i + 1, i * dx, 0.0)
        ops.node(100 + i, i * dx, e_offset)

    # ---- supports -------------------------------------------------------
    ops.fix(1, 1, 1, 0)              # girder left: pin
    ops.fix(n_nodes, 0, 1, 0)        # girder right: roller
    ops.fix(101, 1, 0, 0)            # deck left: horizontal restraint to anchor end moment

    # ---- vertical compatibility -----------------------------------------
    # Force deck and girder paired nodes to share vertical displacement
    # via equalDOF. Rotation and axial stay independent.
    for i in range(n_nodes):
        ops.equalDOF(i + 1, 100 + i, 2)

    # ---- materials ------------------------------------------------------
    EA_deck, EI_deck, EA_girder, EI_girder = _elastic_section_props(params)
    ops.uniaxialMaterial("Elastic", _MAT_DECK_ELASTIC, 1.0)
    ops.uniaxialMaterial("Elastic", _MAT_STEEL_GIRDER_ELASTIC, 1.0)
    # Shear-connector stiffness K_s per connector. The
    # shear_stud_stiffness_ratio in [0.1, 1.0] maps to an effective
    # per-connector stiffness of K_s_max * ratio.
    K_s_max = 200.0      # kip/in per connector, typical headed-stud value
    K_s = K_s_max * params.shear_stud_stiffness_ratio
    ops.uniaxialMaterial("Elastic", _MAT_SHEAR_CONNECTOR, K_s)

    # ---- elements: elastic beam-column ----------------------------------
    ops.geomTransf("Linear", 1)
    # Girder elements 1..n_elements
    for i in range(n_elements):
        ops.element(
            "elasticBeamColumn", i + 1, i + 1, i + 2,
            1.0,            # A placeholder; we pass EA, EI via E*I directly
            _E_STEEL_KSI, EI_girder / _E_STEEL_KSI, 1,
        )
        ops.element(
            "elasticBeamColumn", 100 + i, 100 + i, 100 + (i + 1),
            EA_deck / 1.0, 1.0, EI_deck / 1.0, 1,
        )

    # ---- shear connectors at each node (zeroLength) ---------------------
    for i in range(n_nodes):
        ops.element(
            "zeroLength", 200 + i, i + 1, 100 + i,
            "-mat", _MAT_SHEAR_CONNECTOR, "-dir", 1,
        )

    # ---- apply equal-and-opposite end moments ---------------------------
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(1, 0.0, 0.0, -1.0)              # left support: -M (sagging)
    ops.load(n_nodes, 0.0, 0.0, +1.0)        # right support: +M

    # Solver
    ops.system("UmfPack")
    ops.numberer("Plain")
    ops.constraints("Plain")
    ops.test("NormUnbalance", 1e-6, 30)
    ops.algorithm("Newton")

    # Estimate maximum moment for the sweep (use fraction of plastic moment
    # if available; otherwise compute roughly from steel section).
    fy = params.fy_ksi
    Z_plastic = 2.0 * (params.flange_width_in * params.flange_thickness_in *
                       (params.steel_depth_in - params.flange_thickness_in) / 2.0
                       + 0.25 * params.web_thickness_in *
                       (params.steel_depth_in - 2.0 * params.flange_thickness_in) ** 2)
    M_p_est = fy * Z_plastic + 1.0      # avoid division by zero
    M_target = moment_ratio_max * M_p_est

    moments = np.linspace(M_target / n_load_steps, M_target, n_load_steps)
    curvature = np.zeros(n_load_steps)
    slip = np.zeros(n_load_steps)

    ops.integrator("LoadControl", moments[0])
    ops.analysis("Static")

    # First load step
    if ops.analyze(1) != 0:
        return BeamLevelResult(params.sample_id, moments[:1], curvature[:1],
                               slip[:1], params.composite_action)
    midspan_girder = n_nodes // 2 + 1
    midspan_deck = 100 + n_nodes // 2
    # Curvature at midspan from the girder beam: use the rotation of
    # adjacent nodes; phi ~ (theta_{i+1} - theta_{i-1}) / (2 dx).
    def midspan_curvature() -> float:
        theta_left = ops.nodeDisp(midspan_girder - 1, 3)
        theta_right = ops.nodeDisp(midspan_girder + 1, 3)
        return (theta_right - theta_left) / (2.0 * dx)

    def midspan_slip() -> float:
        # Slip = horizontal displacement of deck relative to girder at midspan
        return ops.nodeDisp(midspan_deck, 1) - ops.nodeDisp(midspan_girder, 1)

    curvature[0] = midspan_curvature()
    slip[0] = abs(midspan_slip())

    # Remaining steps
    converged = 1
    for k in range(1, n_load_steps):
        dM = moments[k] - moments[k - 1]
        ops.integrator("LoadControl", dM)
        if ops.analyze(1) != 0:
            break
        curvature[k] = midspan_curvature()
        slip[k] = abs(midspan_slip())
        converged += 1

    return BeamLevelResult(
        section_id=params.sample_id,
        moment_kip_in=moments[:converged],
        curvature_midspan=curvature[:converged],
        slip_midspan_in=slip[:converged],
        eta_c=params.composite_action,
    )


def compare_to_analytical(
    params: SectionParams, beam_result: BeamLevelResult,
) -> dict:
    """Per-section summary of analytical vs. beam-level slip.

    For a fair point-by-point comparison we evaluate both at the *same*
    moment levels (the analytical formula is monotone in M; the
    beam-level result is sampled). We report mean absolute relative
    error, peak relative error, and the slip at 60 % plastic moment.
    """
    from src.data_generation.moment_curvature import _interface_slip

    M = beam_result.moment_kip_in
    if M.size == 0:
        return {"valid": False}

    # Use a representative mp_est for the analytical formula --
    # consistent with the dataset generator.
    mp_est = max(np.median(M) / 0.4, 1.0)
    analytical = _interface_slip(M, params, mp_est)
    bl = beam_result.slip_midspan_in

    mask = np.abs(bl) > 1e-6
    if mask.sum() == 0:
        return {"valid": False}
    rel_err = (analytical[mask] - bl[mask]) / bl[mask]
    return {
        "valid": True,
        "section_id": params.sample_id,
        "eta_c": params.composite_action,
        "span_in": params.span_in,
        "n_points": int(mask.sum()),
        "mean_abs_rel_err_pct": float(100 * np.mean(np.abs(rel_err))),
        "max_abs_rel_err_pct": float(100 * np.max(np.abs(rel_err))),
        "slip_analytical_at_06Mp_in": float(np.interp(0.6 * mp_est, M, analytical)),
        "slip_beam_level_at_06Mp_in": float(np.interp(0.6 * mp_est, M, bl)),
    }
