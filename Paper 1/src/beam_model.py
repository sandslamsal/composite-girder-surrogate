"""Beam-level composite-girder model with discrete shear connectors.

This is the beam-level counterpart to the section-level model in
:mod:`data_generation.composite_section`. Where the section model
approximates partial composite action by scaling the effective deck
width by a prescribed ``eta_c``, this model represents the shear
connection physically: the deck and the steel girder are separate
beam-column chains, coupled by discrete nonlinear springs along the
span. The degree of composite action is therefore an *emergent*
result of the connector stiffness and spacing, not an input.

Model architecture (2D, units: kip, inch, ksi)
----------------------------------------------
* Two coincident node chains run along the span -- a deck chain and a
  steel chain -- both placed on a single reference line at the
  steel/deck interface (y = 0).
* Each chain is a ``forceBeamColumn`` chain with its own fibre section.
  The fibre sections are **not** referenced to their own centroid: the
  deck fibres sit at section-local ``y in [0, t_s]`` (above the
  interface) and the steel fibres at ``y in [-d_s, 0]`` (below it).
  Defining the sections about the shared interface line is what lets a
  bare ``zeroLength`` connector measure true interface slip -- a
  ``zeroLength`` between two centroid nodes at different elevations
  would be geometrically wrong. The off-centroid reference axis
  produces the axial--flexural coupling that carries the
  partial-interaction mechanics.
* At each node the deck and steel share vertical translation
  (``equalDOF`` on DOF 2): equal deflection, no uplift. Rotation is
  left independent, and the horizontal DOF is independent -- its
  relative value *is* the interface slip.
* A ``zeroLength`` shear connector couples the horizontal DOFs at every
  node. Its force--slip law is a ``Hysteretic`` material with a
  three-point backbone shaped to the Ollgaard (1971) push-out curve
  (stiff initial branch, softening knee, near-ultimate plateau).

The beam is simply supported and pushed by a midspan point load under
displacement control.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import openseespy.opensees as ops


# Material constants (consistent with composite_section.py)
_E_STEEL_KSI = 29_000.0
_EPS_C0 = -0.002
_EPS_CU = -0.005
_STUD_FU_KSI = 65.0          # nominal tensile strength of a headed stud
_OLLGAARD_K_BASE = 1500.0    # kip/in secant stiffness, 3/4-in stud (Ollgaard 1971)

# Tag offsets
_MAT_CONCRETE = 1
_MAT_STEEL = 2
_MAT_CONNECTOR_BASE = 5000
_SEC_DECK = 1
_SEC_STEEL = 2
_TRANSF = 1
_INTEG_DECK = 1
_INTEG_STEEL = 2
_DECK_NODE = 1000
_STEEL_NODE = 2000
_DECK_ELE = 3000
_STEEL_ELE = 4000
_CONN_ELE = 6000


@dataclass
class BeamParams:
    """One composite-beam specimen. Connector properties are inputs;
    the degree of composite action is computed after analysis."""

    span_in: float
    deck_thickness_in: float
    deck_width_in: float          # full effective width (no eta_c scaling)
    fc_deck_ksi: float
    fy_ksi: float
    steel_depth_in: float
    flange_width_in: float
    flange_thickness_in: float
    web_thickness_in: float
    stud_diameter_in: float = 0.75
    ks_ratio: float = 1.0         # connector stiffness / Ollgaard base
    n_studs_per_row: int = 1
    stud_pitch_in: float = 12.0
    section_type: str = "W"
    deck_rho_long: float = 0.01   # longitudinal deck reinforcement ratio


@dataclass
class BeamResult:
    """Output of one monotonic midspan push."""

    moment: np.ndarray             # kip-in, midspan
    curvature: np.ndarray          # 1/in, midspan
    neutral_axis_in: np.ndarray    # in, from deck top (operational)
    midspan_slip_in: np.ndarray    # in
    max_slip_in: np.ndarray        # in, peak over the span
    deflection_in: np.ndarray      # in, midspan
    eta_c_emergent: float          # connector force sum / C_f at last step
    mp_estimate_kip_in: float
    converged_steps: int
    n_requested: int


# ---------------------------------------------------------------------------
# connector and section helpers
# ---------------------------------------------------------------------------

def _concrete_modulus_ksi(fc: float) -> float:
    """ACI 318 secant modulus, E_c = 57000 sqrt(f'c[psi]), returned in ksi."""
    return 1.802e3 * math.sqrt(fc)


def stud_strength_kip(d_stud: float, fc: float) -> float:
    """AISC 360 I8.2a nominal shear strength of one headed stud (kip)."""
    a_sc = 0.25 * math.pi * d_stud ** 2
    ec = _concrete_modulus_ksi(fc)
    q_concrete = 0.5 * a_sc * math.sqrt(fc * ec)
    q_cap = 0.75 * a_sc * _STUD_FU_KSI          # R_g R_p A_sc F_u, R_g R_p ~ 0.75
    return min(q_concrete, q_cap)


def stud_stiffness_kip_per_in(d_stud: float, ks_ratio: float) -> float:
    """Per-stud secant stiffness: Ollgaard (1971) base for a 3/4-in stud,
    scaled by stud area and by the sampled stiffness ratio."""
    k_base = _OLLGAARD_K_BASE * (d_stud / 0.75) ** 2
    return k_base * ks_ratio


def _connector_material(tag: int, q_ult: float, k_init: float) -> None:
    """Hysteretic force-slip law for one lumped connector.

    Three-point backbone shaped to the Ollgaard (1971) push-out curve:
    a stiff initial branch of slope ``k_init``, a softening knee, and a
    near-ultimate plateau at ``q_ult``. Monotonic loading here, so the
    pinching/damage parameters are inert."""
    q1 = 0.5 * q_ult
    d1 = q1 / k_init                       # end of the initial linear branch
    q2 = 0.9 * q_ult
    d2 = max(0.05, 2.0 * d1)               # softening knee
    q3 = q_ult
    d3 = max(0.30, 2.0 * d2)               # near-ultimate plateau
    ops.uniaxialMaterial("Hysteretic", tag,
                         q1, d1, q2, d2, q3, d3,
                         -q1, -d1, -q2, -d2, -q3, -d3,
                         1.0, 1.0, 0.0, 0.0, 0.0)


def _steel_area(p: BeamParams) -> float:
    web_clear = p.steel_depth_in - 2.0 * p.flange_thickness_in
    return 2.0 * p.flange_width_in * p.flange_thickness_in + web_clear * p.web_thickness_in


def _compression_force_capacity(p: BeamParams) -> float:
    """C_f: smaller of deck crushing force and steel yield force, the
    horizontal force a fully composite connection would have to carry."""
    deck_crush = 0.85 * p.fc_deck_ksi * p.deck_width_in * p.deck_thickness_in
    steel_yield = p.fy_ksi * _steel_area(p)
    return min(deck_crush, steel_yield)


def estimate_plastic_moment(p: BeamParams) -> float:
    """Full-composite plastic moment estimate (kip-in), for the M/M_p
    normalisation only. The analysis itself does not use it."""
    fy, fc = p.fy_ksi, p.fc_deck_ksi
    a_s = _steel_area(p)
    c_steel = a_s * fy
    a_block = min(c_steel / (0.85 * fc * p.deck_width_in), p.deck_thickness_in)
    y_steel_centroid = p.deck_thickness_in + 0.5 * p.steel_depth_in
    y_block_centroid = 0.5 * a_block
    return c_steel * (y_steel_centroid - y_block_centroid)


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------

def _define_materials(p: BeamParams) -> None:
    fc = p.fc_deck_ksi
    ft = 0.21 * math.sqrt(fc)
    ets = 0.05 * 57.0 * math.sqrt(fc * 1000.0)
    ops.uniaxialMaterial("Concrete02", _MAT_CONCRETE,
                         -fc, _EPS_C0, -0.2 * fc, _EPS_CU, 0.1, ft, ets)
    ops.uniaxialMaterial("Steel02", _MAT_STEEL,
                         p.fy_ksi, _E_STEEL_KSI, 0.01, 20.0, 0.925, 0.15)


def _define_sections(p: BeamParams) -> None:
    """Deck and steel fibre sections, both referenced to the interface
    line y = 0. Deck fibres sit at y in [0, t_s]; steel at y in [-d_s, 0]."""
    t_s = p.deck_thickness_in
    b_eff = p.deck_width_in
    d_s = p.steel_depth_in
    b_f = p.flange_width_in
    t_f = p.flange_thickness_in
    t_w = p.web_thickness_in

    # Deck: concrete rectangle above the interface, with longitudinal
    # reinforcement in a top and a bottom layer. Real bridge decks carry
    # roughly 0.5-1% longitudinal steel; including it also gives the
    # cracked deck a stable, non-zero tangent for the force-based element.
    ops.section("Fiber", _SEC_DECK)
    ops.patch("rect", _MAT_CONCRETE, 40, 1,
              0.0, -b_eff / 2.0, t_s, b_eff / 2.0)
    if p.deck_rho_long > 0.0:
        a_layer = 0.5 * p.deck_rho_long * b_eff * t_s
        for y_bar in (0.15 * t_s, 0.85 * t_s):
            ops.layer("straight", _MAT_STEEL, 1, a_layer,
                      y_bar, -b_eff / 2.0, y_bar, b_eff / 2.0)

    # Steel I-section below the interface (top flange touches y = 0).
    ops.section("Fiber", _SEC_STEEL)
    ops.patch("rect", _MAT_STEEL, 10, 1,
              -t_f, -b_f / 2.0, 0.0, b_f / 2.0)                       # top flange
    ops.patch("rect", _MAT_STEEL, 40, 1,
              -(d_s - t_f), -t_w / 2.0, -t_f, t_w / 2.0)              # web
    ops.patch("rect", _MAT_STEEL, 10, 1,
              -d_s, -b_f / 2.0, -(d_s - t_f), b_f / 2.0)              # bottom flange

    ops.beamIntegration("Lobatto", _INTEG_DECK, _SEC_DECK, 5)
    ops.beamIntegration("Lobatto", _INTEG_STEEL, _SEC_STEEL, 5)


def build_beam_model(p: BeamParams) -> dict:
    """Assemble the beam model. Returns a layout dict the analysis uses."""
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    _define_materials(p)
    _define_sections(p)
    ops.geomTransf("Linear", _TRANSF)

    # Element count: at least 40, even (so midspan is a node), and fine
    # enough that one element spans roughly one stud pitch.
    n_ele = max(40, int(math.ceil(p.span_in / p.stud_pitch_in)))
    if n_ele % 2:
        n_ele += 1
    dx = p.span_in / n_ele
    n_node = n_ele + 1
    mid = n_ele // 2

    for i in range(n_node):
        x = i * dx
        ops.node(_DECK_NODE + i, x, 0.0)
        ops.node(_STEEL_NODE + i, x, 0.0)
        # Deck and steel share vertical translation only (no uplift);
        # the horizontal DOF stays independent (interface slip) and
        # rotation is independent, coupled through the connectors.
        ops.equalDOF(_STEEL_NODE + i, _DECK_NODE + i, 2)

    # Simply supported: steel pinned at the left support, roller at the
    # right. The deck is free to slip horizontally at both supports.
    ops.fix(_STEEL_NODE + 0, 1, 1, 0)
    ops.fix(_STEEL_NODE + n_ele, 0, 1, 0)

    for i in range(n_ele):
        ops.element("forceBeamColumn", _DECK_ELE + i,
                    _DECK_NODE + i, _DECK_NODE + i + 1, _TRANSF, _INTEG_DECK)
        ops.element("forceBeamColumn", _STEEL_ELE + i,
                    _STEEL_NODE + i, _STEEL_NODE + i + 1, _TRANSF, _INTEG_STEEL)

    # Discrete shear connectors: one zeroLength per node, each lumping the
    # studs tributary to that node (n_studs_per_row studs over one pitch).
    q_n = stud_strength_kip(p.stud_diameter_in, p.fc_deck_ksi) * p.n_studs_per_row
    k_s = stud_stiffness_kip_per_in(p.stud_diameter_in, p.ks_ratio) * p.n_studs_per_row
    studs_per_node = dx / p.stud_pitch_in           # tributary stud fraction
    conn_qn = q_n * studs_per_node
    conn_ks = k_s * studs_per_node
    for i in range(n_node):
        mat = _MAT_CONNECTOR_BASE + i
        _connector_material(mat, conn_qn, conn_ks)
        ops.element("zeroLength", _CONN_ELE + i,
                    _STEEL_NODE + i, _DECK_NODE + i, "-mat", mat, "-dir", 1)

    return {
        "n_ele": n_ele, "n_node": n_node, "dx": dx, "mid": mid,
        "conn_qn": conn_qn, "cf": _compression_force_capacity(p),
        "mp": estimate_plastic_moment(p),
    }


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def analyze_beam(p: BeamParams, n_steps: int = 80,
                 max_deflection_in: Optional[float] = None,
                 newton_tol: float = 1e-8,
                 newton_max_iter: int = 30) -> BeamResult:
    """Monotonic displacement-controlled midspan push. Records the
    midspan moment-curvature response and the interface-slip profile."""
    layout = build_beam_model(p)
    n_ele, n_node, dx, mid = (layout[k] for k in ("n_ele", "n_node", "dx", "mid"))

    if max_deflection_in is None:
        max_deflection_in = p.span_in / 50.0
    d_disp = -max_deflection_in / n_steps          # downward (negative y)

    # Reference midspan point load (downward); scaled by DisplacementControl.
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(_STEEL_NODE + mid, 0.0, -1.0, 0.0)

    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("UmfPack")
    ops.test("NormDispIncr", newton_tol, newton_max_iter)
    ops.algorithm("Newton")
    ops.integrator("DisplacementControl", _STEEL_NODE + mid, 2, d_disp)
    ops.analysis("Static")
    mid_node = _STEEL_NODE + mid

    moment = np.zeros(n_steps)
    curvature = np.zeros(n_steps)
    neutral_axis = np.zeros(n_steps)
    mid_slip = np.zeros(n_steps)
    max_slip = np.zeros(n_steps)
    deflection = np.zeros(n_steps)
    conn_force_final = np.zeros(n_node)

    converged = 0
    for step in range(n_steps):
        if not _advance_increment(d_disp, mid_node):
            break

        # Applied midspan load = load factor x reference load (1.0 kip
        # downward). Midspan moment of a simply supported beam under a
        # midspan point load is exactly P*L/4 (statically determinate).
        p_total = ops.getTime()
        moment[step] = p_total * p.span_in / 4.0

        # Midspan curvature / axial strain from the steel element ending
        # at the midspan node (last Lobatto point).
        defo = ops.eleResponse(_STEEL_ELE + mid - 1, "section", 5, "deformation")
        eps0, kappa = defo[0], defo[1]
        curvature[step] = abs(kappa)
        neutral_axis[step] = _neutral_axis_depth(eps0, kappa, p.deck_thickness_in)

        slips = np.array([ops.nodeDisp(_DECK_NODE + i, 1) - ops.nodeDisp(_STEEL_NODE + i, 1)
                          for i in range(n_node)])
        mid_slip[step] = abs(slips[mid])
        max_slip[step] = float(np.abs(slips).max())
        deflection[step] = abs(ops.nodeDisp(_STEEL_NODE + mid, 2))

        for i in range(n_node):
            s = ops.eleResponse(_CONN_ELE + i, "material", 1, "stress")
            conn_force_final[i] = abs(s[0]) if s else 0.0
        converged += 1

    sl = slice(0, converged)
    # Emergent eta_c: horizontal force transferred between a support and
    # midspan = sum of connector forces over the half-span, divided by C_f.
    half_force = float(conn_force_final[:mid + 1].sum())
    eta_c = float(np.clip(half_force / max(layout["cf"], 1e-9), 0.0, 1.05))

    return BeamResult(
        moment=moment[sl], curvature=curvature[sl],
        neutral_axis_in=neutral_axis[sl], midspan_slip_in=mid_slip[sl],
        max_slip_in=max_slip[sl], deflection_in=deflection[sl],
        eta_c_emergent=eta_c, mp_estimate_kip_in=layout["mp"],
        converged_steps=converged, n_requested=n_steps,
    )


def _advance_increment(d_disp: float, mid_node: int, depth: int = 0) -> bool:
    """Advance one displacement increment, robustly.

    Tries Newton, then modified Newton, then a line-search Newton at the
    current step size; failing all three, halves the step and recurses
    (to a floor of d_disp/16). This carries the analysis through the
    yield plateau and the concrete-crushing branch where a plain Newton
    step would diverge.
    """
    for algo in (("Newton",), ("ModifiedNewton",),
                 ("NewtonLineSearch", "-type", "Bisection")):
        ops.algorithm(*algo)
        if ops.analyze(1) == 0:
            ops.algorithm("Newton")
            return True
    ops.algorithm("Newton")
    if depth >= 4:
        return False
    half = d_disp / 2.0
    ops.integrator("DisplacementControl", mid_node, 2, half)
    ok = (_advance_increment(half, mid_node, depth + 1)
          and _advance_increment(half, mid_node, depth + 1))
    ops.integrator("DisplacementControl", mid_node, 2, d_disp)
    return ok


def _neutral_axis_depth(eps0: float, kappa: float, t_s: float) -> float:
    """Depth of zero steel-fibre strain, measured from the deck top.

    Sections use eps(y) = eps0 - y*kappa about the interface (y = 0);
    zero strain is at y = eps0/kappa, and depth = t_s - y. This is an
    operational neutral axis taken from the steel strain field; with
    partial interaction the strain profile is discontinuous at the
    interface, so a single section-wide neutral axis is approximate."""
    if abs(kappa) < 1e-12:
        return float("nan")
    return t_s - eps0 / kappa


# ---------------------------------------------------------------------------
# single-beam test driver
# ---------------------------------------------------------------------------

def _test_beam() -> BeamParams:
    """A representative mid-range composite girder for the Step-1 check."""
    return BeamParams(
        span_in=60.0 * 12.0, deck_thickness_in=8.0, deck_width_in=80.0,
        fc_deck_ksi=4.0, fy_ksi=50.0, steel_depth_in=36.0,
        flange_width_in=12.0, flange_thickness_in=0.75, web_thickness_in=0.5,
        stud_diameter_in=0.75, ks_ratio=0.6, n_studs_per_row=2,
        stud_pitch_in=12.0, section_type="W",
    )


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    p = _test_beam()
    res = analyze_beam(p, n_steps=80)
    mp = res.mp_estimate_kip_in
    m_over_mp = res.moment / mp

    print(f"test beam: L={p.span_in/12:.0f} ft, d_s={p.steel_depth_in:.0f} in, "
          f"f_y={p.fy_ksi:.0f} ksi, K_s ratio={p.ks_ratio}, "
          f"{p.n_studs_per_row} studs/row @ {p.stud_pitch_in:.0f} in")
    print(f"converged steps : {res.converged_steps}/{res.n_requested}")
    print(f"M_p estimate    : {mp:,.0f} kip-in")
    print(f"peak moment     : {res.moment.max():,.0f} kip-in "
          f"(M/M_p = {m_over_mp.max():.2f})")
    print(f"peak curvature  : {res.curvature.max():.3e} 1/in")
    print(f"peak midspan defl: {res.deflection_in.max():.2f} in "
          f"(L/{p.span_in/max(res.deflection_in.max(),1e-9):.0f})")
    print(f"max slip        : {res.max_slip_in.max():.4f} in")
    print(f"emergent eta_c  : {res.eta_c_emergent:.3f}")
    print()
    print(f"{'M/M_p':>8} {'curvature':>12} {'y_na (in)':>10} "
          f"{'defl (in)':>10} {'max slip':>10}")
    idx = np.linspace(0, res.converged_steps - 1, 10, dtype=int)
    for k in idx:
        print(f"{m_over_mp[k]:8.3f} {res.curvature[k]:12.3e} "
              f"{res.neutral_axis_in[k]:10.2f} {res.deflection_in[k]:10.3f} "
              f"{res.max_slip_in[k]:10.4f}")

    out = Path(__file__).resolve().parent.parent / "figures" / "validation"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(res.curvature, res.moment / 1000.0, "-o", ms=3, color="#1f77b4")
    ax[0].set_xlabel("midspan curvature  (1/in)")
    ax[0].set_ylabel("midspan moment  (kip-ft x 12 = kip-in / 1000)")
    ax[0].set_title("Beam-level moment-curvature")
    ax[0].grid(alpha=0.3)
    ax[1].plot(res.deflection_in, res.max_slip_in, "-o", ms=3, color="#d62728")
    ax[1].set_xlabel("midspan deflection  (in)")
    ax[1].set_ylabel("peak interface slip  (in)")
    ax[1].set_title("Slip growth")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "beam_test_mphi.png", dpi=130)
    print(f"\nfigure -> {out / 'beam_test_mphi.png'}")


if __name__ == "__main__":
    main()
