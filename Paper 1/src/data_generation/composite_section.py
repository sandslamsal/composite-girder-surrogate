"""Build an OpenSeesPy fibre section for a composite girder.

The section is laid out in OpenSees local coordinates ``(y, z)`` where ``y``
is vertical (positive *downward*, so the deck top sits at ``y = 0``) and
``z`` is the horizontal width direction.

Partial composite action is approximated by scaling the *effective deck
width* by the degree-of-composite-action ``eta_c``. Only the fraction
``eta_c`` of the deck width is treated as participating in section
compression — the AISC partial-composite simplification. Interface slip is
*not* captured by the section model itself; it is computed analytically
from the M-phi response in :mod:`moment_curvature`.

Three section archetypes are supported (selected by
``SectionParams.section_type``):

``W`` / ``plate``
    Concrete deck on top of a rolled or welded steel I-section. Geometry
    fields ``steel_depth_in``, ``flange_width_in``, ``flange_thickness_in``,
    ``web_thickness_in`` are used.

``concrete_I``
    Concrete deck on top of a precast concrete I-beam (AASHTO Type II-VI).
    Geometry is parameterised by ``pcb_depth_in``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import openseespy.opensees as ops

from .lhs_sampler import SectionParams


# Material tag offsets (kept distinct so multiple sections can coexist in
# one OpenSees domain if we ever need it).
_MAT_DECK_CONCRETE = 1
_MAT_BEAM_CONCRETE = 2
_MAT_STEEL = 3
_MAT_STRAND_PRESTRESSED = 4  # Steel02 wrapped in InitStrainMaterial

# Material constants
_E_STEEL_KSI = 29_000.0
_F_PU_KSI = 270.0           # nominal ultimate strength of prestressing strand


@dataclass
class SectionInfo:
    """Geometric metadata returned by :func:`build_section`."""

    section_tag: int
    total_depth_in: float           # top of deck to bottom of girder
    deck_depth_in: float
    girder_depth_in: float
    y_top: float                    # always 0.0 (origin at deck top)
    y_bottom: float                 # = total_depth_in
    plastic_moment_kip_in: float    # rough M_p estimate for normalisation


def build_section(params: SectionParams, section_tag: int = 1,
                  deck_rho_long: float = 0.0) -> SectionInfo:
    """Wipe the current OpenSees domain and assemble the fibre section.

    ``deck_rho_long`` is an optional total longitudinal deck-reinforcement
    ratio (steel area / modelled deck area). The default ``0.0`` reproduces
    the released dataset, in which deck reinforcement is omitted; a positive
    value adds top and bottom reinforcement layers (see
    :func:`_build_steel_i_with_deck`).

    Returns geometry metadata used by downstream analysis.
    """
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    _define_materials(params)

    n_fibers_deck = 40
    n_fibers_web = 40
    n_fibers_flange = 10

    if params.section_type in ("W", "plate"):
        info = _build_steel_i_with_deck(
            params, section_tag, n_fibers_deck, n_fibers_web, n_fibers_flange,
            deck_rho_long=deck_rho_long,
        )
    elif params.section_type == "concrete_I":
        # Implementation exists in `_build_concrete_i_with_deck` but the
        # resulting section converges to the hogging branch under
        # sagging-direction DisplacementControl. v1 routes here only if a
        # user re-enables concrete_I in the config; surface that early.
        raise NotImplementedError(
            "concrete_I section type is deferred to v2; see configs/data_gen.yaml"
        )
    else:
        raise ValueError(f"unknown section_type: {params.section_type!r}")

    return info


# ---------------------------------------------------------------------------
# materials
# ---------------------------------------------------------------------------

def _define_materials(params: SectionParams) -> None:
    """Concrete02 for both concretes; Steel02 for steel. Compression-side
    values are passed as negative per OpenSees convention.

    Concrete02 (vs. the previous Concrete01) adds a small tension branch
    with linear softening. This matters for the concrete-I section: when
    nearly the entire cross-section is concrete and only a small strand
    layer provides tension, Concrete01's exactly-zero tension stiffness
    leaves the Newton solver with an ambiguous equilibrium and it can
    settle on the hogging branch under sagging-direction loading. Giving
    the concrete a non-zero, softening tension branch breaks that
    degeneracy without materially changing the post-cracking M-phi shape.

    The prestressing strand is built lazily by the concrete-I builder so
    it can use a section-specific tag (see :func:`_build_concrete_i_with_deck`).
    """
    eps_c0 = -0.002
    eps_u = -0.005

    # Deck concrete
    fc_deck = params.fc_deck_ksi
    ft_deck = 0.21 * math.sqrt(fc_deck)              # tensile strength, ksi
    ets_deck = 0.05 * 57.0 * math.sqrt(fc_deck * 1000.0)  # 0.05 * Ec(ksi)
    ops.uniaxialMaterial(
        "Concrete02",
        _MAT_DECK_CONCRETE,
        -fc_deck, eps_c0, -0.2 * fc_deck, eps_u,
        0.1, ft_deck, ets_deck,
    )

    # Beam concrete (only used for concrete-I girders, but cheap to define)
    fc_beam = params.pcb_fc_ksi
    ft_beam = 0.21 * math.sqrt(fc_beam)
    ets_beam = 0.05 * 57.0 * math.sqrt(fc_beam * 1000.0)
    ops.uniaxialMaterial(
        "Concrete02",
        _MAT_BEAM_CONCRETE,
        -fc_beam, eps_c0, -0.2 * fc_beam, eps_u,
        0.1, ft_beam, ets_beam,
    )

    # Steel (Giuffre-Menegotto-Pinto with bilinear hardening b = 0.01)
    ops.uniaxialMaterial(
        "Steel02",
        _MAT_STEEL,
        params.fy_ksi,
        _E_STEEL_KSI,
        0.01,        # strain hardening ratio
        20.0,        # R0
        0.925,       # cR1
        0.15,        # cR2
    )

    # Prestressing strand: Steel02 wrapped in InitStrainMaterial so the
    # strand carries an effective tensile prestress at zero applied
    # section deformation. OpenSees computes sigma = base(eps_applied -
    # eps_init); to make the inner Steel02 see *positive* (tensile) strain
    # at eps_applied = 0 we must use a *negative* eps_init. A small
    # fraction of the nominal pre-stretch keeps the strand elastic and
    # avoids driving the section into a degenerate axial-tension
    # equilibrium under displacement-controlled sagging.
    init_strain = -0.25 * params.pcb_prestress_ratio * (_F_PU_KSI / _E_STEEL_KSI)
    ops.uniaxialMaterial(
        "InitStrainMaterial", _MAT_STRAND_PRESTRESSED, _MAT_STEEL, init_strain
    )


# ---------------------------------------------------------------------------
# steel I + deck
# ---------------------------------------------------------------------------

def _build_steel_i_with_deck(
    params: SectionParams,
    section_tag: int,
    n_fibers_deck: int,
    n_fibers_web: int,
    n_fibers_flange: int,
    deck_rho_long: float = 0.0,
) -> SectionInfo:
    t_s = params.deck_thickness_in
    b_eff_full = params.deck_width_in
    eta_c = params.composite_action
    b_eff = b_eff_full * eta_c            # partial-composite reduction

    d_s = params.steel_depth_in
    b_f = params.flange_width_in
    t_f = params.flange_thickness_in
    t_w = params.web_thickness_in

    # Vertical extents (y positive downward, origin at deck top)
    y_deck_top, y_deck_bot = 0.0, t_s
    y_tf_top, y_tf_bot = y_deck_bot, y_deck_bot + t_f
    y_web_top, y_web_bot = y_tf_bot, y_tf_bot + (d_s - 2.0 * t_f)
    y_bf_top, y_bf_bot = y_web_bot, y_web_bot + t_f

    total_depth = y_bf_bot

    ops.section("Fiber", section_tag)

    # Deck: rectangular patch in the (y, z) plane
    _rect_patch(
        _MAT_DECK_CONCRETE,
        n_fibers_deck, 1,
        y_deck_top, -b_eff / 2.0,
        y_deck_bot, +b_eff / 2.0,
    )

    # Optional deck longitudinal reinforcement. ``deck_rho_long`` is the
    # total longitudinal steel ratio referred to the modelled
    # (effective-width) deck area; it is split equally between a top and a
    # bottom layer placed 15% of the slab depth in from each face. The
    # default (0.0) reproduces the released dataset, which omits deck
    # reinforcement; a positive value gives the cracked deck non-zero
    # axial stiffness.
    if deck_rho_long > 0.0 and b_eff > 0.0:
        a_layer = 0.5 * deck_rho_long * b_eff * t_s
        ops.layer(
            "straight", _MAT_STEEL, 1, a_layer,
            y_deck_top + 0.15 * t_s, -b_eff / 2.0,
            y_deck_top + 0.15 * t_s, +b_eff / 2.0,
        )
        ops.layer(
            "straight", _MAT_STEEL, 1, a_layer,
            y_deck_bot - 0.15 * t_s, -b_eff / 2.0,
            y_deck_bot - 0.15 * t_s, +b_eff / 2.0,
        )

    # Top flange
    _rect_patch(
        _MAT_STEEL,
        n_fibers_flange, 1,
        y_tf_top, -b_f / 2.0,
        y_tf_bot, +b_f / 2.0,
    )

    # Web
    _rect_patch(
        _MAT_STEEL,
        n_fibers_web, 1,
        y_web_top, -t_w / 2.0,
        y_web_bot, +t_w / 2.0,
    )

    # Bottom flange
    _rect_patch(
        _MAT_STEEL,
        n_fibers_flange, 1,
        y_bf_top, -b_f / 2.0,
        y_bf_bot, +b_f / 2.0,
    )

    mp_est = _estimate_plastic_moment_steel_i(
        params, total_depth, t_s, b_eff, d_s, b_f, t_f, t_w
    )

    return SectionInfo(
        section_tag=section_tag,
        total_depth_in=total_depth,
        deck_depth_in=t_s,
        girder_depth_in=d_s,
        y_top=0.0,
        y_bottom=total_depth,
        plastic_moment_kip_in=mp_est,
    )


# ---------------------------------------------------------------------------
# concrete-I + deck
# ---------------------------------------------------------------------------

def _build_concrete_i_with_deck(
    params: SectionParams,
    section_tag: int,
    n_fibers_deck: int,
    n_fibers_web: int,
) -> SectionInfo:
    """Simplified AASHTO precast I-beam geometry.

    Approximates the cross-section by three stacked rectangles: a top flange,
    a web, and a bottom flange. Dimensions are scaled from the depth using
    proportions roughly representative of AASHTO Type II-VI shapes:

    - top flange:    width = 0.6 * depth, height = 0.10 * depth
    - web:           width = 0.10 * depth, height = 0.75 * depth
    - bottom flange: width = 0.45 * depth, height = 0.15 * depth
    """
    t_s = params.deck_thickness_in
    b_eff = params.deck_width_in * params.composite_action

    d_b = params.pcb_depth_in
    b_tf = 0.6 * d_b
    h_tf = 0.10 * d_b
    b_w = 0.10 * d_b
    h_w = 0.75 * d_b
    b_bf = 0.45 * d_b
    h_bf = d_b - h_tf - h_w

    y_deck_top, y_deck_bot = 0.0, t_s
    y_tf_top, y_tf_bot = y_deck_bot, y_deck_bot + h_tf
    y_web_top, y_web_bot = y_tf_bot, y_tf_bot + h_w
    y_bf_top, y_bf_bot = y_web_bot, y_web_bot + h_bf

    total_depth = y_bf_bot

    ops.section("Fiber", section_tag)

    _rect_patch(
        _MAT_DECK_CONCRETE,
        n_fibers_deck, 1,
        y_deck_top, -b_eff / 2.0,
        y_deck_bot, +b_eff / 2.0,
    )
    # Top flange (beam concrete, not deck concrete — different f'c)
    _rect_patch(
        _MAT_BEAM_CONCRETE,
        max(n_fibers_deck // 2, 6), 1,
        y_tf_top, -b_tf / 2.0,
        y_tf_bot, +b_tf / 2.0,
    )
    _rect_patch(
        _MAT_BEAM_CONCRETE,
        n_fibers_web, 1,
        y_web_top, -b_w / 2.0,
        y_web_bot, +b_w / 2.0,
    )
    _rect_patch(
        _MAT_BEAM_CONCRETE,
        max(n_fibers_deck // 2, 6), 1,
        y_bf_top, -b_bf / 2.0,
        y_bf_bot, +b_bf / 2.0,
    )

    # Deck mild reinforcement (top + bottom layers, each ~1% by area).
    # Real AASHTO bridge decks have substantial longitudinal steel;
    # including it here gives the deck non-zero axial stiffness once the
    # concrete cracks.
    a_deck_steel = 0.01 * b_eff * t_s
    ops.layer(
        "straight", _MAT_STEEL, 1, a_deck_steel,
        y_deck_top + 0.15 * t_s, -b_eff / 2.0,
        y_deck_top + 0.15 * t_s, +b_eff / 2.0,
    )
    ops.layer(
        "straight", _MAT_STEEL, 1, a_deck_steel,
        y_deck_bot - 0.15 * t_s, -b_eff / 2.0,
        y_deck_bot - 0.15 * t_s, +b_eff / 2.0,
    )

    # Tension chord near the bottom of the precast beam, at 0.85 * d_b from
    # the deck top. Sized to the AASHTO 10-40-strand range (0.8-2.4 in^2 of
    # equivalent steel) via prestress_ratio. We use *plain* Steel02 here,
    # not an InitStrainMaterial wrap: applying an initial prestress strain
    # places the section into a small hogging equilibrium at rest that
    # Newton then cannot escape under sagging-direction displacement
    # control. The prestress effect is recovered approximately downstream
    # by treating this layer as a high-strength tension chord; modelling
    # bonded pre-tensioning with proper transfer-length physics is on the
    # v2 backlog.
    strand_area = params.pcb_prestress_ratio * 8.0
    y_strand = y_deck_bot + 0.85 * d_b
    ops.layer(
        "straight", _MAT_STEEL, 1, strand_area,
        y_strand, -b_bf / 4.0, y_strand, +b_bf / 4.0,
    )
    # Non-prestressed mild reinforcement near the bottom of the precast
    # beam. Equal area to the prestressing strand. This concentrates
    # tensile steel at the bottom — without it the section's bottom-fiber
    # stiffness is dominated by concrete (which softens to zero under
    # tension) and the Newton solver locks onto a degenerate axial-tension
    # equilibrium under sagging-direction displacement control.
    a_bottom_bars = strand_area
    y_bottom_bars = y_bf_top + 0.5 * h_bf
    ops.layer(
        "straight", _MAT_STEEL, 1, a_bottom_bars,
        y_bottom_bars, -b_bf / 4.0, y_bottom_bars, +b_bf / 4.0,
    )

    mp_est = _estimate_plastic_moment_concrete_i(
        params, total_depth, t_s, b_eff, d_b, strand_area, y_strand
    )

    return SectionInfo(
        section_tag=section_tag,
        total_depth_in=total_depth,
        deck_depth_in=t_s,
        girder_depth_in=d_b,
        y_top=0.0,
        y_bottom=total_depth,
        plastic_moment_kip_in=mp_est,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _rect_patch(mat_tag: int, n_y: int, n_z: int,
                y1: float, z1: float, y2: float, z2: float) -> None:
    """Thin wrapper around ``ops.patch('rect', ...)`` so the call sites stay
    readable."""
    ops.patch("rect", mat_tag, n_y, n_z, y1, z1, y2, z2)


def _estimate_plastic_moment_steel_i(
    params: SectionParams,
    total_depth: float,
    t_s: float,
    b_eff: float,
    d_s: float,
    b_f: float,
    t_f: float,
    t_w: float,
) -> float:
    """Rough plastic moment for the full-composite section under positive
    bending. Used only for normalisation (M / M_p feature, curvature
    bounds). The actual moment capacity comes from the OpenSees analysis."""
    fy = params.fy_ksi
    fc = params.fc_deck_ksi
    a_s = 2.0 * b_f * t_f + (d_s - 2.0 * t_f) * t_w
    # Whitney compression block depth in deck
    c_force_steel = a_s * fy
    a_block = c_force_steel / (0.85 * fc * b_eff) if b_eff > 0 else t_s
    a_block = min(a_block, t_s)
    # Moment arm from compression block centroid to steel centroid
    y_steel_centroid = t_s + d_s / 2.0
    y_block_centroid = a_block / 2.0
    return c_force_steel * (y_steel_centroid - y_block_centroid)


def _estimate_plastic_moment_concrete_i(
    params: SectionParams,
    total_depth: float,
    t_s: float,
    b_eff: float,
    d_b: float,
    strand_area: float,
    y_strand: float,
) -> float:
    """Very rough M_p estimate for a prestressed concrete-I + deck section.
    Treats the strand layer as the tension chord at f_pe + f_y."""
    fy = params.fy_ksi
    fc = params.fc_deck_ksi
    f_tension = fy + params.pcb_prestress_ratio * 270.0  # 270 ksi nominal strand
    t_force = strand_area * f_tension
    a_block = t_force / (0.85 * fc * b_eff) if b_eff > 0 else t_s * 0.5
    a_block = min(a_block, t_s)
    return t_force * (y_strand - a_block / 2.0)
