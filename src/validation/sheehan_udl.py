"""Partial-interaction check against the Sheehan, Dai and Lam (2018) beam.

Sheehan, T., Dai, X. and Lam, D. (2018), "Flexural behaviour of asymmetric
composite beam with low degree of shear connection", Journal of
Constructional Steel Research 141, 251-261, doi 10.1016/j.jcsr.2017.11.018.
The specimen is the 11.2 m fabricated beam of RFCS project DISCCO
(Lawson et al., 2017, EUR 28458 EN).

Why this specimen and not another
---------------------------------
It is the only published test found for which all four of the following
hold at once: end slip is tabulated (Table 2, seven load levels), the
degree of shear connection is genuinely partial (eta = 0.33), the
specimen is fully documented, and, decisively, the parent programme
publishes the connector stiffness *a priori*: DISCCO Section 4.4 and the
WP5 summary fix a single 19 mm stud at k = 70 kN/mm. With the stud
layout also known exactly (one stud per deck rib at 300 mm pitch), a
prediction for this beam carries no free parameter. That is the property
the Nie and Cai (2003) comparison structurally lacks.

Two things the existing beam machinery could not do, and what was added
----------------------------------------------------------------------
``src.beam_model`` builds a doubly symmetric I-section and pushes it with
a midspan point load under displacement control. This specimen is an
asymmetric plate girder (10 mm top flange, 15 mm bottom) tested under a
simulated uniformly distributed load. This module therefore rebuilds the
same two-chain architecture with

* separate top- and bottom-flange thickness and separate yield stress for
  the 10 mm and 15 mm plate, and
* a uniformly distributed load applied under load control, so the shear
  flow gradient that actually drives interface slip is present. (Under
  the uniform end moment of ``src.validation.beam_level_slip`` the shear
  is identically zero and end slip is not a meaningful target.)

Everything else follows ``src.beam_model``: two ``forceBeamColumn`` fibre
chains on a shared interface reference line, both sections declared with
``-noCentroid``, ``equalDOF`` on DOF 2 only, and one ``zeroLength``
connector per node in direction 1, so the relative horizontal DOF *is*
the interface slip.

Units are kip, inch and ksi throughout, as in the rest of the repository;
the specimen is defined in SI and converted once, in :func:`sheehan_specimen`.

Nothing here is fitted. The connector stiffness is the published
70 kN/mm; the stud resistance is the 68 kN that Sheehan Section 4.1
derives from the Nellinger push tests; the geometry and the measured
yield stresses are as printed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import openseespy.opensees as ops

# ---------------------------------------------------------------- units
IN_PER_MM = 1.0 / 25.4
KIP_PER_KN = 1.0 / 4.4482216
KSI_PER_MPA = 1.0 / 6.894757
KIPIN_PER_KNM = KIP_PER_KN * 1000.0 * IN_PER_MM  # kN*m -> kip*in (= 8.851)
MM_PER_IN = 25.4
KN_PER_KIP = 4.4482216
KNM_PER_KIPIN = 1.0 / KIPIN_PER_KNM
KN_PER_MM_PER_KIP_PER_IN = KN_PER_KIP * IN_PER_MM   # kip/in -> kN/mm

_E_STEEL_KSI = 29_000.0        # 199.9 GPa
_EPS_CU = -0.005

# tags
_MAT_CONC = 1
_MAT_STEEL_10 = 2              # 10 mm plate: top flange and web, fy = 440 MPa
_MAT_STEEL_15 = 3              # 15 mm plate: bottom flange, fy = 400 MPa
_MAT_MESH = 4
_MAT_CONN_BASE = 5000
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
class AsymBeamParams:
    """One asymmetric composite beam under uniformly distributed load.

    All lengths in inch, stresses in ksi, forces in kip.
    """

    span_in: float
    # steel plate girder
    steel_depth_in: float
    flange_width_top_in: float
    flange_thick_top_in: float
    flange_width_bot_in: float
    flange_thick_bot_in: float
    web_thickness_in: float
    fy_top_ksi: float                  # top flange + web plate
    fy_bot_ksi: float                  # bottom flange plate
    # slab: solid topping only, sitting `rib_depth_in` above the top flange
    deck_width_in: float
    topping_thickness_in: float
    rib_depth_in: float
    fc_deck_ksi: float
    ec_deck_ksi: Optional[float] = None   # override the ACI modulus
    mesh_area_in2: float = 0.0            # total longitudinal mesh area
    mesh_depth_in: float = 0.0            # above the top of the topping datum
    # connectors: published, not fitted
    stud_k_kip_per_in: float = 399.7      # 70 kN/mm
    stud_qu_kip: float = 15.29            # 68 kN
    stud_pitch_in: float = 11.811         # 300 mm
    n_studs_per_row: int = 1
    stud_slip_plateau_in: float = 0.2362  # 6 mm, DISCCO declining branch
    connector_backbone: str = "discco"    # "discco" | "ollgaard"
    label: str = "Sheehan-2018-11.2m"
    notes: str = ""


@dataclass
class UdlResult:
    """Load-controlled UDL sweep output."""

    w_kip_per_in: np.ndarray        # applied UDL
    moment_kip_in: np.ndarray       # midspan moment, w L^2 / 8
    deflection_in: np.ndarray       # midspan
    end_slip_in: np.ndarray         # slip at the support node
    max_slip_in: np.ndarray         # peak over the span
    conn_force_max_kip: np.ndarray  # peak connector force over the span
    converged_steps: int
    n_requested: int
    label: str = ""

    def at_w(self, w: float, key: str) -> float:
        """Linear interpolation of one channel at an applied UDL.

        Returns NaN rather than extrapolating past the last converged step.
        """
        y = getattr(self, key)
        if self.w_kip_per_in.size == 0 or w > self.w_kip_per_in[-1] * 1.0001:
            return float("nan")
        return float(np.interp(w, self.w_kip_per_in, y))


# ------------------------------------------------------------ specimen

def sheehan_specimen(fc_cyl_mpa: float = 20.0,
                     ec_gpa: Optional[float] = None,
                     backbone: str = "discco") -> AsymBeamParams:
    """The 11.2 m DISCCO/Sheehan beam, converted from the printed SI values.

    Geometry and materials: Sheehan et al. (2018) Sections 2.1 to 2.3.
    Fabricated asymmetric I, 450 mm deep; top flange 180 x 10 mm; web
    425 x 10 mm; bottom flange 180 x 15 mm. Slab 2800 mm wide, 150 mm
    total on an 80 mm trapezoidal deck, i.e. 70 mm of solid topping
    (DISCCO Section 4.4 uses exactly hc = 70 mm, Ac = 196e3 mm2). One
    19 mm stud per rib at 300 mm. Measured yield 440 MPa (10 mm plate)
    and 400 MPa (15 mm plate). Concrete 25 MPa measured on cubes;
    ``fc_cyl_mpa`` is the cylinder-equivalent used by the ACI-form
    constitutive stack in this repository (0.8 x 25 = 20 MPa by default).
    A193 mesh, 7 mm wire on a 200 x 200 grid.

    ``ec_gpa`` optionally forces the concrete modulus (DISCCO assumes
    Es/Ec = 6, i.e. Ec = 35 GPa; the ACI form gives about 21 GPa at this
    strength, and the difference is reported rather than hidden).
    """
    mesh_n = math.floor(2800.0 / 200.0)          # 7 mm wires across 2800 mm
    mesh_area_mm2 = mesh_n * 0.25 * math.pi * 7.0 ** 2
    return AsymBeamParams(
        span_in=11200.0 * IN_PER_MM,
        steel_depth_in=450.0 * IN_PER_MM,
        flange_width_top_in=180.0 * IN_PER_MM,
        flange_thick_top_in=10.0 * IN_PER_MM,
        flange_width_bot_in=180.0 * IN_PER_MM,
        flange_thick_bot_in=15.0 * IN_PER_MM,
        web_thickness_in=10.0 * IN_PER_MM,
        fy_top_ksi=440.0 * KSI_PER_MPA,
        fy_bot_ksi=400.0 * KSI_PER_MPA,
        deck_width_in=2800.0 * IN_PER_MM,
        topping_thickness_in=70.0 * IN_PER_MM,
        rib_depth_in=80.0 * IN_PER_MM,
        fc_deck_ksi=fc_cyl_mpa * KSI_PER_MPA,
        ec_deck_ksi=None if ec_gpa is None else ec_gpa * 1000.0 * KSI_PER_MPA,
        mesh_area_in2=mesh_area_mm2 * IN_PER_MM ** 2,
        mesh_depth_in=35.0 * IN_PER_MM,          # mid-depth of the topping
        stud_k_kip_per_in=70.0 / KN_PER_MM_PER_KIP_PER_IN,   # 70 kN/mm
        stud_qu_kip=68.0 * KIP_PER_KN,                       # 68 kN
        stud_pitch_in=300.0 * IN_PER_MM,
        n_studs_per_row=1,
        stud_slip_plateau_in=6.0 * IN_PER_MM,
        connector_backbone=backbone,
        notes="Sheehan, Dai & Lam (2018) JCSR 141:251-261; DISCCO EUR 28458 EN",
    )


# ------------------------------------------------- model construction

def _concrete_modulus_ksi(fc: float) -> float:
    return 1.802e3 * math.sqrt(fc)


def _define_materials(p: AsymBeamParams) -> None:
    fc = p.fc_deck_ksi
    ft = 0.21 * math.sqrt(fc)
    ets = 0.05 * 57.0 * math.sqrt(fc * 1000.0)
    # Concrete02's initial tangent is 2 fc / eps_c0. Choose eps_c0 so the
    # model modulus equals the requested one, otherwise use the ACI value
    # implied by eps_c0 = -0.002 as elsewhere in the repository.
    ec = p.ec_deck_ksi if p.ec_deck_ksi else 2.0 * fc / 0.002
    eps_c0 = -2.0 * fc / ec
    ops.uniaxialMaterial("Concrete02", _MAT_CONC,
                         -fc, eps_c0, -0.2 * fc, _EPS_CU, 0.1, ft, ets)
    ops.uniaxialMaterial("Steel02", _MAT_STEEL_10,
                         p.fy_top_ksi, _E_STEEL_KSI, 0.01, 20.0, 0.925, 0.15)
    ops.uniaxialMaterial("Steel02", _MAT_STEEL_15,
                         p.fy_bot_ksi, _E_STEEL_KSI, 0.01, 20.0, 0.925, 0.15)
    ops.uniaxialMaterial("Steel02", _MAT_MESH,
                         500.0 * KSI_PER_MPA, _E_STEEL_KSI, 0.01, 20.0,
                         0.925, 0.15)


def _define_sections(p: AsymBeamParams) -> None:
    """Both fibre sections are referenced to the interface line y = 0, the
    top surface of the top flange, which is where the studs act.

    Deck fibres therefore sit at y in [rib_depth, rib_depth + topping]:
    the profiled sheeting leaves an 80 mm void between the top flange and
    the underside of the solid topping, and only the topping is modelled
    as continuous concrete (the ribs run transverse to the beam and
    carry no longitudinal stress). Steel fibres sit at y in [-d_s, 0].
    ``-noCentroid`` is what keeps that deliberate offset alive.
    """
    ops.section("Fiber", _SEC_DECK, "-noCentroid")
    y0 = p.rib_depth_in
    y1 = y0 + p.topping_thickness_in
    ops.patch("rect", _MAT_CONC, 20, 1,
              y0, -p.deck_width_in / 2.0, y1, p.deck_width_in / 2.0)
    if p.mesh_area_in2 > 0.0:
        y_mesh = y0 + p.mesh_depth_in
        ops.layer("straight", _MAT_MESH, 1, p.mesh_area_in2,
                  y_mesh, -p.deck_width_in / 2.0,
                  y_mesh, p.deck_width_in / 2.0)

    ops.section("Fiber", _SEC_STEEL, "-noCentroid")
    d_s = p.steel_depth_in
    tft, tfb = p.flange_thick_top_in, p.flange_thick_bot_in
    ops.patch("rect", _MAT_STEEL_10, 6, 1,
              -tft, -p.flange_width_top_in / 2.0,
              0.0, p.flange_width_top_in / 2.0)                    # top flange
    ops.patch("rect", _MAT_STEEL_10, 40, 1,
              -(d_s - tfb), -p.web_thickness_in / 2.0,
              -tft, p.web_thickness_in / 2.0)                      # web
    ops.patch("rect", _MAT_STEEL_15, 6, 1,
              -d_s, -p.flange_width_bot_in / 2.0,
              -(d_s - tfb), p.flange_width_bot_in / 2.0)           # bottom flange

    ops.beamIntegration("Lobatto", _INTEG_DECK, _SEC_DECK, 5)
    ops.beamIntegration("Lobatto", _INTEG_STEEL, _SEC_STEEL, 5)


def _connector_material(tag: int, p: AsymBeamParams,
                        q_ult: float, k_init: float) -> None:
    """Force-slip law for the connectors lumped at one node.

    ``discco``  the law the source programme publishes for this stud:
                linear at k = 70 kN/mm to the 68 kN resistance, plateau to
                the 6 mm ductility limit, then a declining branch. No free
                parameter.
    ``ollgaard`` the three-point backbone used elsewhere in this
                repository (src/beam_model.py), retained as a sensitivity
                on the backbone shape. Its initial slope is still k_init.
    """
    if p.connector_backbone == "discco":
        d1 = q_ult / k_init
        d2 = max(p.stud_slip_plateau_in, 1.5 * d1)
        d3 = 2.0 * d2
        q1, q2, q3 = q_ult, q_ult, 0.8 * q_ult
    else:
        q1 = 0.5 * q_ult
        d1 = q1 / k_init
        q2, d2 = 0.9 * q_ult, max(0.05, 2.0 * d1)
        q3, d3 = q_ult, max(0.30, 2.0 * d2)
    ops.uniaxialMaterial("Hysteretic", tag,
                         q1, d1, q2, d2, q3, d3,
                         -q1, -d1, -q2, -d2, -q3, -d3,
                         1.0, 1.0, 0.0, 0.0, 0.0)


def build_model(p: AsymBeamParams) -> dict:
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    _define_materials(p)
    _define_sections(p)
    ops.geomTransf("Linear", _TRANSF)

    # One element per stud pitch, at least 40, even so midspan is a node.
    n_ele = max(40, int(round(p.span_in / p.stud_pitch_in)))
    if n_ele % 2:
        n_ele += 1
    dx = p.span_in / n_ele
    n_node = n_ele + 1
    mid = n_ele // 2

    support_nodes = (0, n_ele)

    for i in range(n_node):
        x = i * dx
        ops.node(_DECK_NODE + i, x, 0.0)
        ops.node(_STEEL_NODE + i, x, 0.0)
        # At the two supports uy = 0 is imposed directly on both chords
        # rather than tied, so the retained steel node never carries a
        # single-point constraint on a DOF it also retains for a
        # multi-point constraint; the Transformation constraint handler
        # does not resolve that combination. With the tie on DOF 2 only
        # the support value is zero either way, so the change leaves
        # every result here unaltered -- verified to machine precision
        # on all three variants at all seven load levels.
        if i not in support_nodes:
            ops.equalDOF(_STEEL_NODE + i, _DECK_NODE + i, 2)

    ops.fix(_STEEL_NODE + 0, 1, 1, 0)
    ops.fix(_STEEL_NODE + n_ele, 0, 1, 0)
    ops.fix(_DECK_NODE + 0, 0, 1, 0)
    ops.fix(_DECK_NODE + n_ele, 0, 1, 0)

    for i in range(n_ele):
        ops.element("forceBeamColumn", _DECK_ELE + i,
                    _DECK_NODE + i, _DECK_NODE + i + 1, _TRANSF, _INTEG_DECK)
        ops.element("forceBeamColumn", _STEEL_ELE + i,
                    _STEEL_NODE + i, _STEEL_NODE + i + 1, _TRANSF, _INTEG_STEEL)

    # Connectors: each node lumps the studs tributary to it.
    studs_per_node = dx / p.stud_pitch_in * p.n_studs_per_row
    conn_qn = p.stud_qu_kip * studs_per_node
    conn_ks = p.stud_k_kip_per_in * studs_per_node
    for i in range(n_node):
        _connector_material(_MAT_CONN_BASE + i, p, conn_qn, conn_ks)
        ops.element("zeroLength", _CONN_ELE + i,
                    _STEEL_NODE + i, _DECK_NODE + i,
                    "-mat", _MAT_CONN_BASE + i, "-dir", 1)

    return {"n_ele": n_ele, "n_node": n_node, "dx": dx, "mid": mid,
            "conn_qn": conn_qn, "conn_ks": conn_ks,
            "studs_per_node": studs_per_node}


# ------------------------------------------------------------ analysis

def analyze_udl(p: AsymBeamParams, w_max_kip_per_in: float,
                n_steps: int = 200, tol: float = 1e-8,
                max_iter: int = 60) -> UdlResult:
    """Load-controlled sweep of a uniformly distributed load to ``w_max``.

    The test applied the load at eight evenly spaced points through a
    spreader system to simulate a UDL; a true UDL is modelled, which
    changes the midspan moment by well under one per cent and is stated
    as such.
    """
    layout = build_model(p)
    n_ele, n_node, dx, mid = (layout[k] for k in ("n_ele", "n_node", "dx", "mid"))

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    # Consistent nodal loads for a unit UDL of 1 kip/in.
    for i in range(n_node):
        trib = dx if 0 < i < n_ele else dx / 2.0
        ops.load(_STEEL_NODE + i, 0.0, -trib, 0.0)

    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("UmfPack")
    ops.test("NormDispIncr", tol, max_iter)
    ops.algorithm("Newton")
    dw = w_max_kip_per_in / n_steps
    ops.integrator("LoadControl", dw)
    ops.analysis("Static")

    w = np.zeros(n_steps)
    defl = np.zeros(n_steps)
    end_slip = np.zeros(n_steps)
    max_slip = np.zeros(n_steps)
    qmax = np.zeros(n_steps)

    converged = 0
    for step in range(n_steps):
        if not _advance(dw):
            break
        w[step] = ops.getTime()
        defl[step] = abs(ops.nodeDisp(_STEEL_NODE + mid, 2))
        slips = np.array([ops.nodeDisp(_DECK_NODE + i, 1)
                          - ops.nodeDisp(_STEEL_NODE + i, 1)
                          for i in range(n_node)])
        end_slip[step] = max(abs(slips[0]), abs(slips[-1]))
        max_slip[step] = float(np.abs(slips).max())
        forces = []
        for i in range(n_node):
            s = ops.eleResponse(_CONN_ELE + i, "material", 1, "stress")
            forces.append(abs(s[0]) if s else 0.0)
        qmax[step] = max(forces) / layout["studs_per_node"]   # per stud
        converged += 1

    sl = slice(0, converged)
    return UdlResult(
        w_kip_per_in=w[sl],
        moment_kip_in=w[sl] * p.span_in ** 2 / 8.0,
        deflection_in=defl[sl], end_slip_in=end_slip[sl],
        max_slip_in=max_slip[sl], conn_force_max_kip=qmax[sl],
        converged_steps=converged, n_requested=n_steps, label=p.label,
    )


def _advance(dw: float, depth: int = 0) -> bool:
    for algo in (("Newton",), ("ModifiedNewton",),
                 ("NewtonLineSearch", "-type", "Bisection")):
        ops.algorithm(*algo)
        if ops.analyze(1) == 0:
            ops.algorithm("Newton")
            return True
    ops.algorithm("Newton")
    if depth >= 4:
        return False
    half = dw / 2.0
    ops.integrator("LoadControl", half)
    ok = _advance(half, depth + 1) and _advance(half, depth + 1)
    ops.integrator("LoadControl", dw)
    return ok


# ------------------------------------------- Newmark closed form (DISCCO)

def newmark_udl(w_n_per_mm: float, d_mm: float, Is_mm4: float, Ic_mm4: float,
                As_mm2: float, Ac_mm2: float, n_modular: float,
                k_stud_n_per_mm: float, pitch_mm: float,
                span_mm: float = 11200.0,
                Es_mpa: float = 210000.0) -> dict:
    """Newmark (1951) partial-interaction closed form for a simply
    supported beam under UDL, in N and mm. This is the same formulation
    DISCCO Section 4.4 uses; it is reimplemented here so the published
    0.53 mm end slip can be checked rather than quoted.

        s'' - a^2 s = -(d / EI0) V(x),  a^2 = k_sc (1/EA* + d^2/EI0)
        s(x) = A cosh(a x) + B sinh(a x) + C (L/2 - x),  C = d w /(a^2 EI0)
        N(0) = 0 -> s'(0) = 0 -> B = C/a ;  s(L/2) = 0 -> A = -B tanh(aL/2)
    """
    L = span_mm
    Ec = Es_mpa / n_modular
    EA_star = 1.0 / (1.0 / (Es_mpa * As_mm2) + 1.0 / (Ec * Ac_mm2))
    EI0 = Es_mpa * Is_mm4 + Ec * Ic_mm4
    ksc = k_stud_n_per_mm / pitch_mm
    a2 = ksc * (1.0 / EA_star + d_mm * d_mm / EI0)
    a = math.sqrt(a2)
    C = d_mm * w_n_per_mm / (a2 * EI0)
    B = C / a
    A = -B * math.tanh(a * L / 2.0)
    s0 = A + C * L / 2.0

    x = np.linspace(0.0, L, 20001)
    s = A * np.cosh(a * x) + B * np.sinh(a * x) + C * (L / 2.0 - x)
    N = ksc * np.concatenate(
        ([0.0], np.cumsum(0.5 * (s[1:] + s[:-1]) * np.diff(x))))
    M = w_n_per_mm * x * (L - x) / 2.0
    kappa = (M - N * d_mm) / EI0
    m_unit = np.where(x <= L / 2.0, x / 2.0, (L - x) / 2.0)
    delta = float(np.trapz(kappa * m_unit, x))
    A_tr = Ac_mm2 / n_modular
    I_full = Is_mm4 + Ic_mm4 / n_modular + (As_mm2 * A_tr / (As_mm2 + A_tr)) * d_mm ** 2
    return {
        "alpha_L": a * L,
        "end_slip_mm": s0,
        "deflection_mm": delta,
        "deflection_full_interaction_mm":
            5.0 * w_n_per_mm * L ** 4 / (384.0 * Es_mpa * I_full),
        "I_full_mm4": I_full,
        "I_eff_mm4": 5.0 * w_n_per_mm * L ** 4 / (384.0 * Es_mpa * delta),
        "stud_force_kn": k_stud_n_per_mm * s0 / 1000.0,
    }


#: DISCCO Section 4.4 section properties for this beam, exactly as printed
#: (Is = 263e6, Ic = 80e6, As = 8.75e3, Ac = 196e3, Es/Ec = 6,
#: Icomp = 1104e6). ``d`` is not printed and is back-computed from Icomp.
DISCCO_PROPS = dict(Is_mm4=263e6, Ic_mm4=80e6, As_mm2=8.75e3, Ac_mm2=196e3,
                    n_modular=6.0, k_stud_n_per_mm=70000.0, pitch_mm=300.0)
DISCCO_PROPS["d_mm"] = math.sqrt(
    (1104e6 - DISCCO_PROPS["Is_mm4"]
     - DISCCO_PROPS["Ic_mm4"] / DISCCO_PROPS["n_modular"])
    / (DISCCO_PROPS["As_mm2"] * (DISCCO_PROPS["Ac_mm2"] / 6.0)
       / (DISCCO_PROPS["As_mm2"] + DISCCO_PROPS["Ac_mm2"] / 6.0)))


#: Sheehan et al. (2018) Table 2 (end slip, LVDT-11) and Table 1
#: (mid-span deflection), transcribed as printed. Loads in kN/m2.
#: The cumulative rows are the ones the parent programme reads: DISCCO
#: Section 4.3.6 quotes "about 3 mm" at 12 kN/m2 (cumulative 3.04),
#: "passed 6 mm at 15" (5.75) and "a maximum of 19 mm" (19.21).
MEASURED = {
    "load_kn_per_m2": [3.0, 5.0, 7.5, 10.0, 12.0, 15.0, 18.0],
    "slip_cycle_max_mm": [0.017, 0.045, 0.27, 1.41, 2.58, 4.62, 16.87],
    "slip_residual_mm": [0.005, 0.005, 0.08, 0.36, 0.67, 1.20, 9.68],
    "slip_cumulative_mm": [0.017, 0.062, 0.30, 1.52, 3.04, 5.75, 19.21],
    "defl_cycle_max_mm": [11.2, 16.4, 26.7, 37.4, 50.9, 72.3, 182.1],
    "defl_cumulative_mm": [11.2, 19.5, 31.9, 46.2, 64.8, 94.0, 215.0],
    "failure_index": 6,
    "tributary_width_m": 2.8,
    "span_m": 11.2,
}


def udl_kn_m2_to_kip_per_in(q_kn_m2: float, width_m: float = 2.8) -> float:
    """kN/m2 on the slab -> kip/in along the beam."""
    w_kn_per_m = q_kn_m2 * width_m
    return w_kn_per_m * KIP_PER_KN / (1000.0 * IN_PER_MM)
