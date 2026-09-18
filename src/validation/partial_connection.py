"""Shared comparison machinery for published partial-connection beam tests.

One module, called identically for every specimen, so that every test in
the partial-connection validation set is reduced by the same arithmetic.
Units are kip, inch and ksi throughout, as in the rest of the repository.
Source data stay in their own units in the data files; convert once, with
the constants below, when a :class:`Specimen` is built.

What is computed for a specimen
-------------------------------
``transformed_section``            full-composite transformed section
                                   (AASHTO reading: uncracked, full slab
                                   width as tested, n = E_s / E_c), the bare
                                   steel section and the non-composite pair.
``elastic_deflection``             closed-form flexural mid-span deflection
                                   for the three supported load patterns;
                                   ``shear_deflection`` is a separate term.
``plastic_moment_database_definition``
                                   M_p exactly as the database normalises
                                   M / M_p (see the function docstring).
``eta_plastic``                    AISC degree of composite action
                                   sum(Q_u) / C_f over a half span.
``r_ei_table``                     Table tab:design-correction of the
                                   manuscript with its interpolation rule.
``beam_model``                     two-chain discrete-connector fibre model
                                   (generalised from ``sheehan_udl.py``) under
                                   load control.
``compare_specimen``               assembles all of the above per measured
                                   load level into one table, with the
                                   elastic-range exclusion flags.

Conventions a specimen agent must know
--------------------------------------
* ``x`` is measured from the LEFT SUPPORT; the span is the support-to-
  support distance. Overhangs (``overhang_in``) extend to x < 0 and x > L.
* ``load`` is the TOTAL applied load in kip for ``"midspan"`` and
  ``"two_point"`` (for two-point loading each point carries load / 2), and
  the line load in kip/in for ``"udl"`` (over the span only).
* Two-point loads sit at ``load_offset_in`` = a from EACH support.
* ``bearing_length_in`` spreads each point load uniformly over that length,
  centred on the load point (Balakrishnan spread the 18 ft beams' point
  load over 9 in).
* Connector stiffness ``k`` and strength ``Q_u`` are PER CONNECTOR; ``n``
  connectors share a position (a pair across the flange is ``n = 2``).
  Every position is modelled as its own zeroLength spring (connectors are
  not smeared). Backbones: ``"ollgaard"`` (src/beam_model.py, default),
  ``"discco"`` (sheehan_udl.py), ``"elastic"``, or ``"curve"``, a measured
  load-slip curve per connector (:meth:`Connector.from_curve`,
  :func:`connector_curve`, :func:`curve_connector_rows`) modelled as an
  OpenSees MultiLinear material. ``Q_u`` is what :func:`eta_plastic` sums,
  so for a curve connector state whether it is the curve peak (default) or
  the programme's stated strength.
* Rolled shapes: the section is three rectangular plates. When the
  handbook A and I are the documented properties, get plate thicknesses
  from :func:`equivalent_plate_thicknesses`; nominal flange and web
  thicknesses without fillets can under-state I_s by several per cent
  (6 % for the 12 x 6 in BSB of Chapman and Balakrishnan).
* Moduli. ``es_ksi`` defaults to 29 000 ksi and ``ec_ksi`` to ``None``,
  which is the database convention: the transformed section then uses the
  ACI secant E_c = 1802 sqrt(f'c) (src/validation/aashto.py) and the beam
  model uses the Concrete02 initial tangent E_0 = 2 f'c / 0.002 = 1000 f'c
  of the fibre database (src/data_generation/composite_section.py). The
  ratio of the two, 0.555 sqrt(f'c), is the constitutive offset the
  manuscript discusses (Eq. eq:modulus-ratio); it is carried into
  R_beam = delta_full / delta_beam on purpose. Supplying ``ec_ksi`` (a
  measured modulus, or lightweight concrete) sets BOTH the transformed
  section and the beam-model initial tangent to that value.
* rho_l for ``r_ei_table`` is a FRACTION (0.007 = 0.7 %).

Nothing in this module is fitted to any test.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import openseespy.opensees as ops

from src.validation.aashto import concrete_modulus_ksi as aci_concrete_modulus_ksi

# ------------------------------------------------------------------ units
IN_PER_MM = 1.0 / 25.4
MM_PER_IN = 25.4
IN_PER_FT = 12.0
KIP_PER_KN = 1.0 / 4.4482216152605
KN_PER_KIP = 4.4482216152605
KSI_PER_MPA = 1.0 / 6.894757293168
MPA_PER_KSI = 6.894757293168
KSI_PER_PSI = 1.0e-3
KIP_PER_LB = 1.0e-3
KIP_PER_LONG_TON = 2.240          # 1 long ton = 2240 lb (British "ton")
KIP_PER_SHORT_TON = 2.000
KSI_PER_TSI = 2.240               # long tons per square inch
KIP_PER_IN_PER_KN_PER_M = KIP_PER_KN / (1000.0 * IN_PER_MM)   # line load kN/m -> kip/in


def kn_per_mm_to_kip_per_in(k_kn_per_mm: float) -> float:
    """Connector stiffness kN/mm -> kip/in (1 kN/mm = 5.7101 kip/in)."""
    return k_kn_per_mm * KIP_PER_KN / IN_PER_MM


# ------------------------------------------------------ material constants
E_STEEL_KSI = 29_000.0            # database and AASHTO module value
NU_STEEL = 0.30
EPS_C0_DATABASE = -0.002          # composite_section.py, beam_model.py
EPS_CU = -0.005

LOAD_PATTERNS = ("midspan", "two_point", "udl")
BACKBONES = ("ollgaard", "discco", "elastic", "curve")

# Anchors of Table tab:design-correction (bin mid-points): manuscript
# application procedure, step 4 ("Interpolate"), manuscript.tex line 2727
# ff. of paper/revision_2/submission/sources (line numbers as of
# 2026-09-17; the file is being edited, so find it by the label).
R_EI_ETA_ANCHORS = (0.375, 0.600, 0.800, 0.950)
# Table tab:design-correction (\label at manuscript.tex line 2690, body
# rows at lines 2697-2700 as of 2026-09-17), transcribed as printed.
# Rows here: (regime, rho_l) -> values at the four eta bins [0.25,0.50),
# [0.50,0.70), [0.70,0.90), [0.90,1.00], i.e. the printed COLUMNS 2-5 read
# down.
R_EI_TABLE = {
    ("service", 0.000): (0.83, 0.92, 0.99, 1.00),
    ("service", 0.007): (0.83, 0.93, 1.00, 1.00),
    ("extended", 0.000): (0.50, 0.65, 0.76, 0.84),
    ("extended", 0.007): (0.59, 0.76, 0.87, 0.94),
}
M_OVER_MP_SERVICE = 0.4
M_OVER_MP_EXTENDED = 0.6


# ============================================================ data classes

@dataclass
class Connector:
    """One connector position.

    ``k_kip_per_in`` and ``qu_kip`` are per connector; ``n`` connectors act
    at ``x_in`` (measured from the left support).

    backbone
        ``"ollgaard"``  three-point Hysteretic backbone of src/beam_model.py:
                        0.5 Q_u at slope k, 0.9 Q_u at max(0.05 in, 2 d1),
                        Q_u at max(0.30 in, 2 d2).
        ``"discco"``    law of src/validation/sheehan_udl.py: linear at k to
                        Q_u, plateau to ``slip_plateau_in`` (or 1.5 d1),
                        declining to 0.8 Q_u at twice that slip.
        ``"elastic"``   linear, slope k, no strength limit (idealised checks
                        only).
        ``"curve"``     a measured (digitised) load-slip curve per connector,
                        ``curve`` = ((slip_in, force_kip), ...) with slips
                        strictly increasing and the origin implied; build
                        it with :meth:`Connector.from_curve` or
                        :func:`connector_curve`. The beam model then uses the
                        curve itself (OpenSees ``MultiLinear``); ``k_kip_per_in``
                        is informational (the first-segment slope) and
                        ``qu_kip`` is the strength used by :func:`eta_plastic`
                        (the curve peak unless set otherwise).

    Past the last backbone point every law continues with its last slope
    (OpenSees Hysteretic and MultiLinear both extrapolate); for the Ollgaard
    law that is a mild hardening past 0.30 in, for measured curves the
    default tail appended by :func:`connector_curve` is flat.
    """

    x_in: float
    k_kip_per_in: float
    qu_kip: float
    n: int = 1
    backbone: str = "ollgaard"
    slip_plateau_in: float = 6.0 * IN_PER_MM
    curve: Optional[tuple] = None

    def __post_init__(self) -> None:
        if self.backbone not in BACKBONES:
            raise ValueError(f"backbone must be one of {BACKBONES}, got {self.backbone!r}")
        if self.k_kip_per_in <= 0.0 or self.qu_kip <= 0.0 or self.n < 1:
            raise ValueError("connector k, Q_u must be > 0 and n >= 1")
        if self.backbone == "curve":
            if self.curve is None:
                raise ValueError("backbone 'curve' needs curve=((slip_in, force_kip), ...)")
            pts = np.asarray(self.curve, dtype=float)
            if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 1:
                raise ValueError("curve must be a sequence of (slip_in, force_kip) pairs")
            if pts[0, 0] <= 0.0 or np.any(np.diff(pts[:, 0]) <= 0.0) or np.any(pts[:, 1] < 0.0):
                raise ValueError("curve slips must be > 0 and strictly increasing, forces >= 0")
            if pts[0, 1] <= 0.0:
                raise ValueError("curve must start with a positive force (initial stiffness > 0)")
            self.curve = tuple((float(s), float(f)) for s, f in pts)
        elif self.curve is not None:
            raise ValueError("curve is only used with backbone='curve'")

    @classmethod
    def from_curve(cls, x_in: float, points: Sequence, *, n: int = 1,
                   qu_kip: Optional[float] = None, **curve_kw) -> "Connector":
        """Connector with a measured backbone.

        ``points`` are (slip, force) pairs per connector in any units;
        ``curve_kw`` goes to :func:`connector_curve` (``slip_scale``,
        ``force_scale`` for unit conversion, ``k_initial_kip_per_in``,
        ``max_points``, ``tail``). ``qu_kip`` defaults to the peak force of
        the cleaned curve; pass the programme's stated strength instead when
        that is what eta should use, and say so.
        """
        crv = connector_curve(points, **curve_kw)
        k = crv[0][1] / crv[0][0]
        qu = max(f for _, f in crv) if qu_kip is None else qu_kip
        return cls(x_in, k, qu, n, "curve", curve=crv)


@dataclass
class RebarLayer:
    """Longitudinal slab reinforcement layer. ``depth_in`` is the depth of
    the layer centroid below the TOP surface of the slab."""

    area_in2: float
    depth_in: float
    fy_ksi: float = 60.0


@dataclass
class Specimen:
    """A simply supported steel-concrete composite beam test.

    Steel I-section with separate top and bottom flanges and per-plate yield
    stress (rolled W or welded plate; rolled-shape root fillets are not
    represented, see :func:`equivalent_plate_thicknesses`). Solid concrete
    slab of ``slab_width_in`` x ``slab_thickness_in``; its soffit sits
    ``haunch_height_in`` above the top flange. If ``haunch_width_in`` > 0 the
    haunch is concrete of that width (grout or cast haunch) and is part of
    the slab; if 0 the gap is void (offset only).
    """

    label: str
    span_in: float
    steel_depth_in: float
    top_flange_width_in: float
    top_flange_thickness_in: float
    bot_flange_width_in: float
    bot_flange_thickness_in: float
    web_thickness_in: float
    fy_top_flange_ksi: float
    fy_web_ksi: float
    fy_bot_flange_ksi: float
    slab_width_in: float
    slab_thickness_in: float
    fc_ksi: float
    connectors: list = field(default_factory=list)
    load_pattern: str = "midspan"
    load_offset_in: Optional[float] = None
    bearing_length_in: float = 0.0
    section_type: str = "W"
    ec_ksi: Optional[float] = None
    es_ksi: float = E_STEEL_KSI
    haunch_height_in: float = 0.0
    haunch_width_in: float = 0.0
    rebar: list = field(default_factory=list)
    overhang_in: float = 0.0
    source: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.load_pattern not in LOAD_PATTERNS:
            raise ValueError(f"load_pattern must be one of {LOAD_PATTERNS}")
        pos = ("span_in", "steel_depth_in", "top_flange_width_in",
               "top_flange_thickness_in", "bot_flange_width_in",
               "bot_flange_thickness_in", "web_thickness_in",
               "fy_top_flange_ksi", "fy_web_ksi", "fy_bot_flange_ksi",
               "slab_width_in", "slab_thickness_in", "fc_ksi", "es_ksi")
        for name in pos:
            if not getattr(self, name) > 0.0:
                raise ValueError(f"{name} must be > 0")
        if self.top_flange_thickness_in + self.bot_flange_thickness_in >= self.steel_depth_in:
            raise ValueError("flange thicknesses exceed the steel depth")
        if self.load_pattern == "two_point":
            a = self.load_offset_in
            if a is None or not (0.0 < a <= 0.5 * self.span_in):
                raise ValueError("two_point loading needs 0 < load_offset_in <= span/2")
        if self.load_pattern == "udl" and self.bearing_length_in:
            raise ValueError("bearing_length_in applies to point loads only")
        if self.bearing_length_in < 0.0 or self.haunch_height_in < 0.0 \
                or self.haunch_width_in < 0.0 or self.overhang_in < 0.0:
            raise ValueError("bearing, haunch and overhang must be >= 0")
        if self.ec_ksi is not None and self.ec_ksi <= 0.0:
            raise ValueError("ec_ksi must be > 0 when given")
        for c in self.connectors:
            if c.x_in < -self.overhang_in - 1e-9 or c.x_in > self.span_in + self.overhang_in + 1e-9:
                raise ValueError(f"connector at x = {c.x_in} lies outside the beam")
        for r in self.rebar:
            if not (0.0 < r.depth_in < self.slab_thickness_in):
                raise ValueError("rebar depth must lie inside the slab")


def i_section(depth_in: float, flange_width_in: float, flange_thickness_in: float,
              web_thickness_in: float, fy_ksi: float, *,
              bot_flange_width_in: Optional[float] = None,
              bot_flange_thickness_in: Optional[float] = None,
              fy_web_ksi: Optional[float] = None,
              fy_bot_flange_ksi: Optional[float] = None) -> dict:
    """Keyword block for the steel part of :class:`Specimen`; symmetric by
    default, any bottom-flange or web value may be given separately."""
    return dict(
        steel_depth_in=depth_in,
        top_flange_width_in=flange_width_in,
        top_flange_thickness_in=flange_thickness_in,
        bot_flange_width_in=flange_width_in if bot_flange_width_in is None else bot_flange_width_in,
        bot_flange_thickness_in=(flange_thickness_in if bot_flange_thickness_in is None
                                 else bot_flange_thickness_in),
        web_thickness_in=web_thickness_in,
        fy_top_flange_ksi=fy_ksi,
        fy_web_ksi=fy_ksi if fy_web_ksi is None else fy_web_ksi,
        fy_bot_flange_ksi=fy_ksi if fy_bot_flange_ksi is None else fy_bot_flange_ksi,
    )


def connector_rows(span_in: float, pitch_in: float, k_kip_per_in: float,
                   qu_kip: float, *, per_row: int = 1,
                   n_rows: Optional[int] = None,
                   end_distance_in: Optional[float] = None,
                   backbone: str = "ollgaard",
                   slip_plateau_in: float = 6.0 * IN_PER_MM) -> list:
    """Uniform rows of connectors at ``pitch_in``.

    * ``n_rows`` and ``end_distance_in`` both given: rows start at
      ``end_distance_in`` from the left support.
    * only ``n_rows``: the group is centred on mid-span.
    * only ``end_distance_in`` (or neither, default pitch / 2): as many rows
      as fit between the two end distances, centred on mid-span.
    """
    if pitch_in <= 0.0:
        raise ValueError("pitch must be > 0")
    if n_rows is None:
        e = 0.5 * pitch_in if end_distance_in is None else end_distance_in
        n_rows = int(math.floor((span_in - 2.0 * e) / pitch_in + 1e-9)) + 1
        x0 = 0.5 * (span_in - (n_rows - 1) * pitch_in)
    elif end_distance_in is None:
        x0 = 0.5 * (span_in - (n_rows - 1) * pitch_in)
    else:
        x0 = end_distance_in
    return [Connector(x0 + i * pitch_in, k_kip_per_in, qu_kip, per_row,
                      backbone, slip_plateau_in) for i in range(n_rows)]


def curve_connector_rows(span_in: float, pitch_in: float, points: Sequence, *,
                         per_row: int = 1, n_rows: Optional[int] = None,
                         end_distance_in: Optional[float] = None,
                         qu_kip: Optional[float] = None, **curve_kw) -> list:
    """:func:`connector_rows` for a measured backbone: the same row layout
    rules, every connector built by :meth:`Connector.from_curve` from the
    same ``points`` (per connector, cleaned once)."""
    proto = Connector.from_curve(0.0, points, n=per_row, qu_kip=qu_kip, **curve_kw)
    rows = connector_rows(span_in, pitch_in, proto.k_kip_per_in, proto.qu_kip,
                          per_row=per_row, n_rows=n_rows, end_distance_in=end_distance_in)
    return [Connector(r.x_in, proto.k_kip_per_in, proto.qu_kip, per_row, "curve",
                      curve=proto.curve) for r in rows]


def connector_curve(points: Sequence, *, slip_scale: float = 1.0, force_scale: float = 1.0,
                    k_initial_kip_per_in: Optional[float] = None,
                    max_points: Optional[int] = None, tail: str = "hold") -> tuple:
    """Clean a digitised load-slip curve into the backbone of a ``"curve"``
    connector: ((slip_in, force_kip), ...), origin implied.

    Steps, in order:

    1. Scale (``slip_scale`` source slip unit -> in, e.g. ``IN_PER_MM``;
       ``force_scale`` source force unit -> kip, e.g. ``KIP_PER_KN``), drop
       NaN rows, sort by slip, drop points at slip <= 0 and any leading
       points with force <= 0 (digitising noise at the origin); average
       forces at duplicate slips. A negative force after that is an error.
       Softening (descending) branches are kept.
    2. ``k_initial_kip_per_in`` (optional): replace the toe by the line
       F = k s from the origin to its first intersection with the curve at
       s > 0, dropping the points before it. If the curve starts stiffer
       than k this softens the toe; if it starts softer (seating slack) this
       removes the slack. No intersection raises ``ValueError``. Report it
       as an assumption whenever used.
    3. ``max_points`` (optional): greedy simplification to at most that
       many points (origin and the step-4 tail point not counted). The
       first point, the peak and the last point are always kept; points
       are then added one at a time where the current piecewise-linear
       curve deviates most in force. Use it on dense raw traces (a few
       hundred points of digitising jitter make Newton iterations stall on
       tangent jumps); level-based curves rarely need it.
    4. ``tail``: ``"hold"`` (default) appends a flat point past the last
       slip so the force is held at its last value beyond the measured
       range; ``"extend"`` lets OpenSees continue the last slope.

    The beam model builds OpenSees ``MultiLinear`` (symmetric in slip
    direction) from the result; for monotonic slip it is the same envelope
    law as the ``Hysteretic`` material of the built-in backbones (verified
    to machine precision in tests/test_partial_connection.py).
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be a sequence of (slip, force) pairs")
    pts = pts[np.all(np.isfinite(pts), axis=1)] * np.array([slip_scale, force_scale])
    pts = pts[np.argsort(pts[:, 0], kind="stable")]
    pts = pts[pts[:, 0] > 0.0]
    pos = np.nonzero(pts[:, 1] > 0.0)[0]
    if pos.size == 0:
        raise ValueError("curve has no point with slip > 0 and force > 0")
    pts = pts[pos[0]:]
    if np.any(pts[:, 1] < 0.0):
        raise ValueError("negative force after the first positive point")
    tol = 1e-9 * pts[-1, 0]
    merged = [[pts[0, 0], [pts[0, 1]]]]
    for s, f in pts[1:]:
        if s - merged[-1][0] <= tol:
            merged[-1][1].append(f)
        else:
            merged.append([s, [f]])
    pts = np.array([[s, float(np.mean(fs))] for s, fs in merged])

    if k_initial_kip_per_in is not None:
        k = float(k_initial_kip_per_in)
        if k <= 0.0:
            raise ValueError("k_initial_kip_per_in must be > 0")
        g = pts[:, 1] - k * pts[:, 0]
        if g[0] != 0.0:
            s0 = np.sign(g[0])
            idx = np.nonzero(np.sign(g) != s0)[0]
            if idx.size == 0:
                raise ValueError("the line of slope k_initial never meets the curve")
            i = int(idx[0])
            if g[i] == 0.0:
                s_star = pts[i, 0]
                rest = pts[i + 1:]
            else:
                t = g[i - 1] / (g[i - 1] - g[i])
                s_star = pts[i - 1, 0] + t * (pts[i, 0] - pts[i - 1, 0])
                rest = pts[i:]
            pts = np.vstack([[s_star, k * s_star], rest])

    if max_points is not None and pts.shape[0] > max_points:
        if max_points < 3:
            raise ValueError("max_points must be >= 3")
        keep = {0, int(np.argmax(pts[:, 1])), pts.shape[0] - 1}
        full = np.vstack([[0.0, 0.0], pts])
        while len(keep) < max_points:
            idx = sorted(keep)
            xs = np.concatenate([[0.0], pts[idx, 0]])
            ys = np.concatenate([[0.0], pts[idx, 1]])
            dev = np.abs(full[1:, 1] - np.interp(full[1:, 0], xs, ys))
            dev[idx] = -1.0
            j = int(np.argmax(dev))
            if dev[j] <= 1e-12 * pts[:, 1].max():
                break
            keep.add(j)
        pts = pts[sorted(keep)]

    if tail == "hold":
        s_last = pts[-1, 0]
        pts = np.vstack([pts, [s_last + max(s_last, 1.0), pts[-1, 1]]])
    elif tail != "extend":
        raise ValueError("tail must be 'hold' or 'extend'")
    if pts[0, 1] <= 0.0:
        raise ValueError("cleaned curve starts at zero force")
    return tuple((float(s), float(f)) for s, f in pts)


# ====================================================== section geometry

def _steel_plates(spec: Specimen) -> list:
    """Plates with y measured DOWN from the top of the top flange."""
    d = spec.steel_depth_in
    tft, tfb = spec.top_flange_thickness_in, spec.bot_flange_thickness_in
    return [
        dict(name="top_flange", b=spec.top_flange_width_in, y0=0.0, y1=tft,
             fy=spec.fy_top_flange_ksi),
        dict(name="web", b=spec.web_thickness_in, y0=tft, y1=d - tfb,
             fy=spec.fy_web_ksi),
        dict(name="bot_flange", b=spec.bot_flange_width_in, y0=d - tfb, y1=d,
             fy=spec.fy_bot_flange_ksi),
    ]


def _rect_props(parts: list) -> tuple[float, float, float]:
    """Area, centroid and own second moment of stacked rectangles
    (dicts with b, y0, y1)."""
    a = sum(p["b"] * (p["y1"] - p["y0"]) for p in parts)
    if a <= 0.0:
        return 0.0, 0.0, 0.0
    y = sum(p["b"] * (p["y1"] - p["y0"]) * 0.5 * (p["y0"] + p["y1"]) for p in parts) / a
    i = 0.0
    for p in parts:
        h = p["y1"] - p["y0"]
        ap = p["b"] * h
        i += p["b"] * h ** 3 / 12.0 + ap * (0.5 * (p["y0"] + p["y1"]) - y) ** 2
    return a, y, i


def steel_area_in2(spec: Specimen) -> float:
    return _rect_props(_steel_plates(spec))[0]


def steel_yield_force_kip(spec: Specimen) -> float:
    """A_s F_y with each plate at its own yield stress."""
    return sum(p["b"] * (p["y1"] - p["y0"]) * p["fy"] for p in _steel_plates(spec))


def equivalent_plate_thicknesses(depth_in: float, flange_width_in: float,
                                 area_in2: float, inertia_in4: float) -> tuple[float, float]:
    """Flange and web thickness (t_f, t_w) of a doubly symmetric three-plate
    I that reproduces a handbook area and major-axis second moment. Use it
    when a rolled section's tabulated A and I (which include root fillets
    and flange taper) are the documented properties."""
    from scipy.optimize import brentq

    def resid(tf: float) -> float:
        hw = depth_in - 2.0 * tf
        tw = (area_in2 - 2.0 * flange_width_in * tf) / hw
        i = tw * hw ** 3 / 12.0 + 2.0 * (flange_width_in * tf ** 3 / 12.0
                                         + flange_width_in * tf * (0.5 * (depth_in - tf)) ** 2)
        return i - inertia_in4

    tf_hi = 0.999 * area_in2 / (2.0 * flange_width_in)
    tf = brentq(resid, 1e-4, tf_hi)
    tw = (area_in2 - 2.0 * flange_width_in * tf) / (depth_in - 2.0 * tf)
    return float(tf), float(tw)


# ================================================== transformed section

@dataclass
class TransformedSection:
    """Elastic section properties. Depths are measured DOWN from the slab
    top. Second moments are in steel units (concrete divided by n)."""

    es_ksi: float
    ec_ksi: float
    n: float
    ec_source: str
    I_full_in4: float
    EI_full_kip_in2: float
    y_na_from_slab_top_in: float
    na_in_concrete: bool
    A_steel_in2: float
    I_steel_in4: float
    EI_steel_kip_in2: float
    y_steel_centroid_from_slab_top_in: float
    A_concrete_in2: float
    I_concrete_own_in4: float           # concrete units, own centroid
    y_concrete_centroid_from_slab_top_in: float
    EI_noncomposite_kip_in2: float      # E_c I_c + E_s I_s (+ rebar with slab)
    include_rebar: bool


def section_moduli(spec: Specimen) -> tuple[float, float, str]:
    """(E_s, E_c, source of E_c) for the transformed section."""
    if spec.ec_ksi is not None:
        return spec.es_ksi, spec.ec_ksi, "specimen ec_ksi"
    return spec.es_ksi, aci_concrete_modulus_ksi(spec.fc_ksi), "ACI 1802 sqrt(fc), aashto.py"


def beam_model_concrete_modulus_ksi(spec: Specimen) -> float:
    """Initial tangent of the beam-model Concrete02: ``ec_ksi`` if given,
    else 2 f'c / 0.002 = 1000 f'c as in the fibre database."""
    return spec.ec_ksi if spec.ec_ksi is not None else 2.0 * spec.fc_ksi / abs(EPS_C0_DATABASE)


def _concrete_parts(spec: Specimen) -> list:
    """Slab (and concrete haunch), y down from the slab top."""
    t, h = spec.slab_thickness_in, spec.haunch_height_in
    parts = [dict(b=spec.slab_width_in, y0=0.0, y1=t)]
    if h > 0.0 and spec.haunch_width_in > 0.0:
        parts.append(dict(b=spec.haunch_width_in, y0=t, y1=t + h))
    return parts


def transformed_section(spec: Specimen, *, es_ksi: Optional[float] = None,
                        ec_ksi: Optional[float] = None,
                        include_rebar: bool = False) -> TransformedSection:
    """Full-composite transformed section, bare steel and non-composite pair.

    AASHTO reading, as src/validation/aashto.transformed_section_properties
    computes it for the database: uncracked concrete over the full slab width
    as tested, n = E_s / E_c. For a doubly symmetric section with no haunch
    the result is identical to that function (tested). Deck reinforcement is
    ignored by default, as there; ``include_rebar=True`` adds each layer at
    its own area (E_s), without deducting displaced concrete, which is how the
    fibre sections treat it. ``es_ksi`` / ``ec_ksi`` override the specimen.
    """
    es0, ec0, src = section_moduli(spec)
    es = es0 if es_ksi is None else es_ksi
    ec = ec0 if ec_ksi is None else ec_ksi
    if ec_ksi is not None:
        src = "override"
    n = es / ec

    a_c, y_c, i_c = _rect_props(_concrete_parts(spec))
    a_s, ys_local, i_s = _rect_props(_steel_plates(spec))
    y_s = spec.slab_thickness_in + spec.haunch_height_in + ys_local

    terms = [(a_c / n, y_c, i_c / n), (a_s, y_s, i_s)]
    if include_rebar:
        terms += [(r.area_in2, r.depth_in, 0.0) for r in spec.rebar]
    a_tot = sum(t[0] for t in terms)
    y_na = sum(t[0] * t[1] for t in terms) / a_tot
    i_tr = sum(t[2] + t[0] * (t[1] - y_na) ** 2 for t in terms)

    # Non-composite: slab (+ its rebar if included) and steel about own axes.
    if include_rebar and spec.rebar:
        slab_terms = [(a_c / n, y_c, i_c / n)] + [(r.area_in2, r.depth_in, 0.0) for r in spec.rebar]
        a_sl = sum(t[0] for t in slab_terms)
        y_sl = sum(t[0] * t[1] for t in slab_terms) / a_sl
        i_sl_steel_units = sum(t[2] + t[0] * (t[1] - y_sl) ** 2 for t in slab_terms)
        ei_slab = es * i_sl_steel_units
    else:
        ei_slab = ec * i_c
    concrete_depth = spec.slab_thickness_in + (spec.haunch_height_in if spec.haunch_width_in > 0 else 0.0)
    return TransformedSection(
        es_ksi=es, ec_ksi=ec, n=n, ec_source=src,
        I_full_in4=i_tr, EI_full_kip_in2=es * i_tr,
        y_na_from_slab_top_in=y_na, na_in_concrete=bool(y_na < concrete_depth),
        A_steel_in2=a_s, I_steel_in4=i_s, EI_steel_kip_in2=es * i_s,
        y_steel_centroid_from_slab_top_in=y_s,
        A_concrete_in2=a_c, I_concrete_own_in4=i_c,
        y_concrete_centroid_from_slab_top_in=y_c,
        EI_noncomposite_kip_in2=ei_slab + es * i_s,
        include_rebar=include_rebar,
    )


def first_yield_moment_full_composite(spec: Specimen) -> float:
    """Elastic full-composite moment at which a steel flange first reaches
    F_y (kip-in). Information only; partial interaction yields earlier."""
    ts = transformed_section(spec)
    y_top = spec.slab_thickness_in + spec.haunch_height_in
    y_bot = y_top + spec.steel_depth_in
    m_bot = spec.fy_bot_flange_ksi * ts.I_full_in4 / (y_bot - ts.y_na_from_slab_top_in)
    if ts.y_na_from_slab_top_in > y_top + 1e-9:
        m_top = spec.fy_top_flange_ksi * ts.I_full_in4 / (ts.y_na_from_slab_top_in - y_top)
        return min(m_bot, m_top)
    return m_bot


# ============================================================ loading

def _point_load_positions(spec: Specimen) -> list:
    if spec.load_pattern == "midspan":
        return [0.5 * spec.span_in]
    if spec.load_pattern == "two_point":
        a = spec.load_offset_in
        return [a] if abs(a - 0.5 * spec.span_in) < 1e-9 else [a, spec.span_in - a]
    return []


def midspan_moment(spec: Specimen, load: float) -> float:
    """Mid-span (= maximum) moment, kip-in, for ``load`` in the module's
    convention (total kip for point patterns, kip/in for udl)."""
    L = spec.span_in
    if spec.load_pattern == "midspan":
        return load * (0.25 * L - 0.125 * spec.bearing_length_in)
    if spec.load_pattern == "two_point":
        a = spec.load_offset_in
        if abs(a - 0.5 * L) < 1e-9:           # both loads at mid-span
            return load * (0.25 * L - 0.125 * spec.bearing_length_in)
        return 0.5 * load * a
    return load * L * L / 8.0


def _defl_unit_point(x: float, L: float, EI: float) -> float:
    """Mid-span deflection of a simply supported beam from a unit point
    load at x (0 <= x <= L)."""
    xx = min(x, L - x)
    return xx * (3.0 * L * L - 4.0 * xx * xx) / (48.0 * EI)


def elastic_deflection(spec: Specimen, EI_kip_in2: float, load: float) -> float:
    """Closed-form FLEXURAL mid-span deflection (in), simply supported.

    midspan    W (8 L^3 - 4 L c^2 + c^3) / (384 EI)   (c = bearing length;
               c = 0 gives W L^3 / 48 EI)
    two_point  each W/2 at a: (W/2) a (3 L^2 - 4 a^2) / (24 EI), with a
               uniform spread over c integrated exactly
    udl        5 w L^4 / (384 EI)
    Overhangs carry no load and do not change the mid-span value.
    """
    L = spec.span_in
    if spec.load_pattern == "udl":
        return 5.0 * load * L ** 4 / (384.0 * EI_kip_in2)
    c = spec.bearing_length_in
    pts = _point_load_positions(spec)
    w_each = load / len(pts)
    if c <= 0.0:
        return sum(w_each * _defl_unit_point(x, L, EI_kip_in2) for x in pts)
    # Uniform patch of length c at each point; the kernel is cubic on each
    # side of mid-span, so 3-point Gauss on each piece is exact.
    g = np.polynomial.legendre.leggauss(3)
    total = 0.0
    for x in pts:
        lo, hi = x - 0.5 * c, x + 0.5 * c
        pieces = [(lo, min(hi, 0.5 * L)), (max(lo, 0.5 * L), hi)]
        for p0, p1 in pieces:
            if p1 <= p0:
                continue
            xm, hl = 0.5 * (p0 + p1), 0.5 * (p1 - p0)
            total += (w_each / c) * hl * sum(
                wi * _defl_unit_point(xm + hl * ti, L, EI_kip_in2)
                for ti, wi in zip(g[0], g[1]))
    return total


def shear_deflection(spec: Specimen, load: float, *,
                     G_ksi: Optional[float] = None,
                     shear_area_in2: Optional[float] = None,
                     steel_shear_fraction: float = 1.0) -> float:
    """Mid-span shear deflection (in), reported SEPARATELY from flexure.

    For any load symmetric about mid-span, int V v dx / (G A_v) reduces to
    M_mid / (G A_v). Default A_v is the steel web between the flanges and
    G = E_s / (2 (1 + 0.3)). ``steel_shear_fraction`` is the share of the
    vertical shear the web carries (Balakrishnan 1963, typed p. 17-18, uses
    0.68, the slab carrying the rest).
    """
    if G_ksi is None:
        G_ksi = spec.es_ksi / (2.0 * (1.0 + NU_STEEL))
    if shear_area_in2 is None:
        shear_area_in2 = spec.web_thickness_in * (
            spec.steel_depth_in - spec.top_flange_thickness_in - spec.bot_flange_thickness_in)
    return steel_shear_fraction * midspan_moment(spec, load) / (G_ksi * shear_area_in2)


# ============================================ M_p, C_f and eta (plastic)

def compression_force_capacity(spec: Specimen) -> float:
    """C_f = min(0.85 f'c A_c, A_s F_y) (AISC 360), kip. A_c is the slab
    b x t_s only (no haunch, no reinforcement); A_s F_y sums each plate at
    its own yield stress."""
    return min(0.85 * spec.fc_ksi * spec.slab_width_in * spec.slab_thickness_in,
               steel_yield_force_kip(spec))


def _in_region(x: float, x_end: float, L: float, tol: float) -> float:
    """Weight of a connector at x in the region [0, x_end] measured from one
    support: 1 inside, 0.5 exactly at mid-span when x_end = L/2, else 0."""
    if x < -tol or x > x_end + tol:
        return 0.0
    if abs(x_end - 0.5 * L) <= tol and abs(x - 0.5 * L) <= tol:
        return 0.5
    return 1.0


def eta_plastic_details(spec: Specimen, *, two_point_region: str = "shear_span",
                        include_overhang: bool = False) -> dict:
    """AISC eta = sum(Q_u) / C_f with the region convention made explicit.

    Region counted, from EACH support towards the section of maximum moment:
    * ``midspan`` and ``udl``: [0, L/2]; a connector exactly at mid-span is
      shared equally by the two halves (weight 0.5).
    * ``two_point``: AISC counts connectors between the point of maximum
      and zero moment. With a constant-moment zone the governing section is
      the load point, so the default ``two_point_region="shear_span"``
      counts [0, a] (a connector at the load point counts fully).
      ``"half_span"`` counts [0, L/2] instead (the beam-model read-out
      region); report which was used.
    Connectors outside the span (overhangs) are excluded unless
    ``include_overhang``. The SMALLER of the two sides is used.
    """
    L = spec.span_in
    tol = 1e-6 * L
    if spec.load_pattern == "two_point":
        if two_point_region not in ("shear_span", "half_span"):
            raise ValueError("two_point_region must be 'shear_span' or 'half_span'")
        x_end = spec.load_offset_in if two_point_region == "shear_span" else 0.5 * L
        region = two_point_region
    else:
        x_end, region = 0.5 * L, "half_span"
    left = right = 0.0
    for c in spec.connectors:
        q = c.n * c.qu_kip
        xl, xr = c.x_in, L - c.x_in
        if include_overhang:
            xl, xr = max(xl, 0.0), max(xr, 0.0)
        left += q * _in_region(xl, x_end, L, tol)
        right += q * _in_region(xr, x_end, L, tol)
    cf = compression_force_capacity(spec)
    sum_q = min(left, right)
    return dict(eta=sum_q / cf, sum_qu_kip=sum_q, sum_qu_left_kip=left,
                sum_qu_right_kip=right, cf_kip=cf, region=region,
                region_end_in=x_end,
                concrete_crushing_kip=0.85 * spec.fc_ksi * spec.slab_width_in * spec.slab_thickness_in,
                steel_yield_kip=steel_yield_force_kip(spec))


def eta_plastic(spec: Specimen, **kw) -> float:
    """Degree of composite action sum(Q_u)/C_f; see :func:`eta_plastic_details`."""
    return eta_plastic_details(spec, **kw)["eta"]


def plastic_moment_details(spec: Specimen, eta: Optional[float] = None) -> dict:
    """The database M_p with every intermediate value; see
    :func:`plastic_moment_database_definition`."""
    if eta is None:
        eta = eta_plastic(spec)
    eta_used = min(max(float(eta), 0.0), 1.0)
    t_s = spec.slab_thickness_in
    b_eff = spec.slab_width_in * eta_used
    plates = _steel_plates(spec)
    c_steel = sum(p["b"] * (p["y1"] - p["y0"]) * p["fy"] for p in plates)
    y_force = sum(p["b"] * (p["y1"] - p["y0"]) * p["fy"] * 0.5 * (p["y0"] + p["y1"])
                  for p in plates) / c_steel
    a_block = c_steel / (0.85 * spec.fc_ksi * b_eff) if b_eff > 0.0 else t_s
    a_block = min(a_block, t_s)
    y_steel = t_s + spec.haunch_height_in + y_force
    return dict(mp_kip_in=c_steel * (y_steel - 0.5 * a_block), eta_input=float(eta),
                eta_used=eta_used, b_eff_in=b_eff, c_steel_kip=c_steel,
                a_block_in=a_block, block_capped_at_slab=bool(a_block >= t_s),
                y_steel_force_centroid_from_slab_top_in=y_steel,
                beam_model_full_width_mp_kip_in=_mp_full_width(spec))


def plastic_moment_database_definition(spec: Specimen, eta: Optional[float] = None) -> float:
    """M_p (kip-in) exactly as the database normalises M / M_p.

    Source: src/data_generation/composite_section.py,
    ``_estimate_plastic_moment_steel_i``, called by ``_build_steel_i_with_deck``
    with the effective-width-SCALED deck b_eff = deck_width * eta_c; the
    parquet column ``mp_estimate_kip_in`` carries it and
    src/data_generation/generate_dataset.py ``to_rows`` sets
    ``moment_ratio = moment / mp_estimate_kip_in``. That is the M/M_p the
    manuscript regimes (<= 0.4, <= 0.6) and Table tab:design-correction use.

        C = A_s F_y ;  a = min(C / (0.85 f'c b eta_c), t_s)
        M_p = C (t_s + d_s / 2 - a / 2)

    It is NOT src/beam_model.estimate_plastic_moment, which uses the full
    width b; that value is returned by :func:`plastic_moment_details` as
    ``beam_model_full_width_mp_kip_in`` for reference only. The two agree
    at eta_c = 1 and whenever the block is not width-limited.

    Deck reinforcement is ignored (as in the database). When the block
    exceeds the slab the database caps a at t_s and keeps C = A_s F_y; that
    is reproduced, not corrected. Generalisations needed for test specimens,
    each reducing to the database formula for a doubly symmetric,
    single-F_y, haunch-free section: C sums each plate at its own F_y; the
    steel lever arm is taken to the yield-force centroid of the steel
    (d_s / 2 when symmetric); a haunch height is added to the lever arm.
    ``eta`` defaults to :func:`eta_plastic` and is clamped to [0, 1] (the
    database samples eta_c in [0.25, 1.0]; an over-designed connection is
    read at full width).
    """
    return plastic_moment_details(spec, eta)["mp_kip_in"]


def _mp_full_width(spec: Specimen) -> float:
    t_s = spec.slab_thickness_in
    plates = _steel_plates(spec)
    c = sum(p["b"] * (p["y1"] - p["y0"]) * p["fy"] for p in plates)
    y_force = sum(p["b"] * (p["y1"] - p["y0"]) * p["fy"] * 0.5 * (p["y0"] + p["y1"])
                  for p in plates) / c
    a = min(c / (0.85 * spec.fc_ksi * spec.slab_width_in), t_s)
    return c * (t_s + spec.haunch_height_in + y_force - 0.5 * a)


def rho_l(spec: Specimen, convention: str = "gross") -> float:
    """Longitudinal slab reinforcement ratio (fraction).

    ``"gross"``    sum(A_r) / (b t_s), the engineering reading.
    ``"database"`` sum(A_r) / (eta_c b t_s): the fibre database refers
                   rho to the eta-scaled modelled deck
                   (composite_section.py, a_layer = 0.5 rho b_eff t_s).
    """
    a_r = sum(r.area_in2 for r in spec.rebar)
    gross = a_r / (spec.slab_width_in * spec.slab_thickness_in)
    if convention == "gross":
        return gross
    if convention == "database":
        eta = min(max(eta_plastic(spec), 1e-9), 1.0)
        return gross / eta
    raise ValueError("convention must be 'gross' or 'database'")


# ===================================================== R_EI design table

def r_ei_regime(m_over_mp: float) -> str:
    """``"service"`` (M/M_p <= 0.4), ``"extended"`` (<= 0.6) or ``"beyond"``."""
    if not np.isfinite(m_over_mp) or m_over_mp < 0.0:
        raise ValueError("m_over_mp must be finite and >= 0")
    if m_over_mp <= M_OVER_MP_SERVICE + 1e-12:
        return "service"
    if m_over_mp <= M_OVER_MP_EXTENDED + 1e-12:
        return "extended"
    return "beyond"


def r_ei_table(eta: float, m_over_mp: float, rho_l: float, *,
               regime: Optional[str] = None, rho_policy: str = "clamp") -> float:
    """R_EI from Table tab:design-correction with the manuscript's rule.

    * Regime by M/M_p: <= 0.4 service, (0.4, 0.6] extended elastic, above
      0.6 NaN. ``regime="service"|"extended"`` forces a column instead.
    * Anchors at eta = 0.375, 0.600, 0.800, 0.950; linear in eta between
      anchors; held flat below 0.375 (an upper bound there, manuscript
      step 4) and above 0.950.
    * Linear in rho_l (a FRACTION) between 0 and 0.007. Above 0.007,
      ``rho_policy="clamp"`` (default) holds the 0.7 % column, the
      conservative reading given the saturation the manuscript reports;
      ``"extrapolate"`` extrapolates linearly up to 0.010 (the manuscript's
      stated limit) and returns NaN beyond. Results are capped at 1.00.
    """
    if regime is None:
        regime = r_ei_regime(m_over_mp)
        if regime == "beyond":
            return float("nan")
    elif regime not in ("service", "extended"):
        raise ValueError("regime must be 'service' or 'extended'")
    if rho_l < 0.0 or rho_l > 0.05:
        raise ValueError("rho_l is a fraction (0.007 = 0.7 %); got %r" % rho_l)
    lo = np.interp(eta, R_EI_ETA_ANCHORS, R_EI_TABLE[(regime, 0.000)])
    hi = np.interp(eta, R_EI_ETA_ANCHORS, R_EI_TABLE[(regime, 0.007)])
    if rho_l > 0.007:
        if rho_policy == "clamp":
            rho_l_eff = 0.007
        elif rho_policy == "extrapolate":
            if rho_l > 0.010 + 1e-12:
                return float("nan")
            rho_l_eff = rho_l
        else:
            raise ValueError("rho_policy must be 'clamp' or 'extrapolate'")
    else:
        rho_l_eff = rho_l
    val = lo + (hi - lo) * rho_l_eff / 0.007
    return float(min(val, 1.0))


# ======================================================== beam model

_MAT_CONC = 1
_MAT_STEEL_BASE = 10
_MAT_REBAR_BASE = 30
_MAT_CONN_BASE = 100_000
_SEC_DECK = 1
_SEC_STEEL = 2
_TRANSF = 1
_INTEG_DECK = 1
_INTEG_STEEL = 2
_DECK_NODE = 1_000_000
_STEEL_NODE = 2_000_000
_DECK_ELE = 3_000_000
_STEEL_ELE = 4_000_000
_CONN_ELE = 6_000_000
_N_LOBATTO = 5


@dataclass
class BeamModelResult:
    """Beam-model response at each requested load (input order).

    Arrays are NaN and ``converged`` False for loads not reached.
    ``eta_emergent`` is the slab force at the governing section (sum of
    connector forces from the slab end to it, both halves averaged) / C_f,
    over the same region as :func:`eta_plastic` (shear span for two-point
    loads); ``eta_emergent_half_span`` always reads to mid-span.
    """

    label: str
    load: np.ndarray
    moment_kip_in: np.ndarray
    deflection_in: np.ndarray
    end_slip_in: np.ndarray             # at the slab ends (= supports if no overhang)
    support_slip_in: np.ndarray
    max_slip_in: np.ndarray
    curvature_mid_1_per_in: np.ndarray  # steel chain, mid-span
    curvature_mid_deck_1_per_in: np.ndarray
    steel_strain_bottom_mid: np.ndarray
    steel_strain_top_mid: np.ndarray
    first_yield: np.ndarray             # a mid-span flange fibre at or past F_y/E_s
    eta_emergent: np.ndarray
    eta_emergent_half_span: np.ndarray
    connector_force_max_kip: np.ndarray  # per connector, worst position
    converged: np.ndarray
    n_elements: int
    node_x_in: np.ndarray
    cf_kip: float
    path: Optional[dict] = None


def _mesh(spec: Specimen, target_elements: int) -> np.ndarray:
    """Node x-coordinates: all key points kept exactly, intervals
    subdivided so no element exceeds span / target_elements."""
    L, o = spec.span_in, spec.overhang_in
    keys = [0.0, L, 0.5 * L]
    if o > 0.0:
        keys += [-o, L + o]
    c = spec.bearing_length_in
    for x in _point_load_positions(spec):
        keys.append(x)
        if c > 0.0:
            keys += [x - 0.5 * c, x + 0.5 * c]
    keys += [cn.x_in for cn in spec.connectors]
    keys = np.sort(np.asarray(keys, dtype=float))
    tol = 1e-6 * L
    uniq = [keys[0]]
    for x in keys[1:]:
        if x - uniq[-1] > tol:
            uniq.append(x)
    h_max = L / float(target_elements)
    xs = [uniq[0]]
    for x0, x1 in zip(uniq[:-1], uniq[1:]):
        m = max(1, int(math.ceil((x1 - x0) / h_max - 1e-9)))
        xs += list(x0 + (x1 - x0) * np.arange(1, m + 1) / m)
    return np.asarray(xs)


def _node_index(xs: np.ndarray, x: float) -> int:
    i = int(np.argmin(np.abs(xs - x)))
    if abs(xs[i] - x) > 1e-6 * max(abs(xs[-1] - xs[0]), 1.0):
        raise RuntimeError(f"no node at x = {x}")
    return i


def _hysteretic_points(backbone: str, q_ult: float, k_init: float,
                       slip_plateau: float) -> list:
    """The three (slip, force) envelope points of the built-in laws, exactly
    as sheehan_udl._connector_material and src/beam_model.py define them."""
    if backbone == "discco":
        d1 = q_ult / k_init
        d2 = max(slip_plateau, 1.5 * d1)
        d3 = 2.0 * d2
        q1, q2, q3 = q_ult, q_ult, 0.8 * q_ult
    elif backbone == "ollgaard":
        q1 = 0.5 * q_ult
        d1 = q1 / k_init
        q2, d2 = 0.9 * q_ult, max(0.05, 2.0 * d1)
        q3, d3 = q_ult, max(0.30, 2.0 * d2)
    else:
        raise ValueError(f"no three-point envelope for backbone {backbone!r}")
    return [(d1, q1), (d2, q2), (d3, q3)]


def backbone_points(c: Connector) -> np.ndarray:
    """Envelope of ONE connector as (slip_in, force_kip) rows, origin
    implied, continued past the last row with the last slope."""
    if c.backbone == "curve":
        return np.asarray(c.curve, dtype=float)
    if c.backbone == "elastic":
        return np.array([[1.0, c.k_kip_per_in]])
    return np.asarray(_hysteretic_points(c.backbone, c.qu_kip, c.k_kip_per_in,
                                         c.slip_plateau_in), dtype=float)


def _eval_backbone(s: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Piecewise-linear envelope through the origin and ``pts``, extended
    past the last point with the last slope (as OpenSees does)."""
    xs = np.concatenate([[0.0], pts[:, 0]])
    ys = np.concatenate([[0.0], pts[:, 1]])
    out = np.interp(s, xs, ys)
    slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    beyond = s > xs[-1]
    out[beyond] = ys[-1] + slope * (s[beyond] - xs[-1])
    return out


def _connector_material(tag: int, backbone: str, q_ult: float, k_init: float,
                        slip_plateau: float, points: Optional[np.ndarray] = None) -> None:
    if backbone == "multilinear":
        flat = [float(v) for row in points for v in row]
        ops.uniaxialMaterial("MultiLinear", tag, *flat)
        return
    if backbone == "elastic":
        ops.uniaxialMaterial("Elastic", tag, k_init)
        return
    (d1, q1), (d2, q2), (d3, q3) = _hysteretic_points(backbone, q_ult, k_init, slip_plateau)
    ops.uniaxialMaterial("Hysteretic", tag,
                         q1, d1, q2, d2, q3, d3,
                         -q1, -d1, -q2, -d2, -q3, -d3,
                         1.0, 1.0, 0.0, 0.0, 0.0)


def _build(spec: Specimen, target_elements: int) -> dict:
    if not spec.connectors:
        raise ValueError("beam_model needs at least one connector")
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    # ---- materials: as sheehan_udl.py / composite_section.py
    fc = spec.fc_ksi
    ec = beam_model_concrete_modulus_ksi(spec)
    eps_c0 = -2.0 * fc / ec
    eps_cu = min(EPS_CU, 2.0 * eps_c0)
    ft = 0.21 * math.sqrt(fc)
    ets = 0.05 * 57.0 * math.sqrt(fc * 1000.0)
    ops.uniaxialMaterial("Concrete02", _MAT_CONC, -fc, eps_c0, -0.2 * fc, eps_cu, 0.1, ft, ets)

    steel_tags = {}
    for p in _steel_plates(spec):
        if p["fy"] not in steel_tags:
            tag = _MAT_STEEL_BASE + len(steel_tags)
            ops.uniaxialMaterial("Steel02", tag, p["fy"], spec.es_ksi, 0.01, 20.0, 0.925, 0.15)
            steel_tags[p["fy"]] = tag
    rebar_tags = {}
    for r in spec.rebar:
        if r.fy_ksi not in rebar_tags:
            tag = _MAT_REBAR_BASE + len(rebar_tags)
            ops.uniaxialMaterial("Steel02", tag, r.fy_ksi, spec.es_ksi, 0.01, 20.0, 0.925, 0.15)
            rebar_tags[r.fy_ksi] = tag

    # ---- sections, both about the interface line y = 0 (top of steel)
    h, t = spec.haunch_height_in, spec.slab_thickness_in
    b = spec.slab_width_in
    ops.section("Fiber", _SEC_DECK, "-noCentroid")
    ops.patch("rect", _MAT_CONC, 40, 1, h, -0.5 * b, h + t, 0.5 * b)
    if h > 0.0 and spec.haunch_width_in > 0.0:
        bh = spec.haunch_width_in
        ops.patch("rect", _MAT_CONC, max(4, int(math.ceil(40 * h / t))), 1,
                  0.0, -0.5 * bh, h, 0.5 * bh)
    for r in spec.rebar:
        y = h + t - r.depth_in
        ops.layer("straight", rebar_tags[r.fy_ksi], 1, r.area_in2, y, -0.5 * b, y, 0.5 * b)

    ops.section("Fiber", _SEC_STEEL, "-noCentroid")
    nfib = {"top_flange": 8, "web": 40, "bot_flange": 8}
    for p in _steel_plates(spec):
        ops.patch("rect", steel_tags[p["fy"]], nfib[p["name"]], 1,
                  -p["y1"], -0.5 * p["b"], -p["y0"], 0.5 * p["b"])
    ops.beamIntegration("Lobatto", _INTEG_DECK, _SEC_DECK, _N_LOBATTO)
    ops.beamIntegration("Lobatto", _INTEG_STEEL, _SEC_STEEL, _N_LOBATTO)
    ops.geomTransf("Linear", _TRANSF)

    # ---- nodes, ties, supports
    xs = _mesh(spec, target_elements)
    n_node = len(xs)
    n_ele = n_node - 1
    i_left = _node_index(xs, 0.0)
    i_right = _node_index(xs, spec.span_in)
    i_mid = _node_index(xs, 0.5 * spec.span_in)
    for i, x in enumerate(xs):
        ops.node(_DECK_NODE + i, float(x), 0.0)
        ops.node(_STEEL_NODE + i, float(x), 0.0)
        # Support nodes get uy = 0 on both chords directly (no tie), as in
        # src/beam_model.py and sheehan_udl.py (Transformation handler).
        if i not in (i_left, i_right):
            ops.equalDOF(_STEEL_NODE + i, _DECK_NODE + i, 2)
    ops.fix(_STEEL_NODE + i_left, 1, 1, 0)
    ops.fix(_STEEL_NODE + i_right, 0, 1, 0)
    ops.fix(_DECK_NODE + i_left, 0, 1, 0)
    ops.fix(_DECK_NODE + i_right, 0, 1, 0)

    for e in range(n_ele):
        ops.element("forceBeamColumn", _DECK_ELE + e, _DECK_NODE + e, _DECK_NODE + e + 1,
                    _TRANSF, _INTEG_DECK)
        ops.element("forceBeamColumn", _STEEL_ELE + e, _STEEL_NODE + e, _STEEL_NODE + e + 1,
                    _TRANSF, _INTEG_STEEL)

    # ---- connectors, lumped only with others at the SAME position
    # A group of one built-in law keeps the Hysteretic (or Elastic) material
    # of sheehan_udl.py with summed k and Q_u (the envelope scales exactly).
    # A group holding a measured curve, or mixing laws, is the exact sum of
    # the members' envelopes (each continued with its last slope) as one
    # MultiLinear material.
    groups: dict = {}
    for c in spec.connectors:
        i = _node_index(xs, c.x_in)
        g = groups.setdefault(i, dict(k=0.0, q=0.0, n=0, backbone=c.backbone,
                                      plateau=c.slip_plateau_in, members=[]))
        if g["backbone"] != c.backbone or c.backbone == "curve" \
                or abs(g["plateau"] - c.slip_plateau_in) > 1e-12:
            g["backbone"] = "multilinear"
        g["k"] += c.n * c.k_kip_per_in
        g["q"] += c.n * c.qu_kip
        g["n"] += c.n
        g["members"].append(c)
    conn_nodes = sorted(groups)
    for j, i in enumerate(conn_nodes):
        g = groups[i]
        pts = None
        if g["backbone"] == "multilinear":
            curves = [(backbone_points(c), c.n) for c in g["members"]]
            grid = np.unique(np.concatenate([p[:, 0] for p, _ in curves]))
            force = sum(n * _eval_backbone(grid, p) for p, n in curves)
            pts = np.column_stack([grid, force])
            g["points"] = pts
        _connector_material(_MAT_CONN_BASE + j, g["backbone"], g["q"], g["k"], g["plateau"], pts)
        ops.element("zeroLength", _CONN_ELE + j, _STEEL_NODE + i, _DECK_NODE + i,
                    "-mat", _MAT_CONN_BASE + j, "-dir", 1)

    # ---- reference load pattern (load factor = load in module convention)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    nodal = np.zeros(n_node)
    if spec.load_pattern == "udl":
        _lump_patch(nodal, xs, 0.0, spec.span_in, 1.0)
    else:
        pts = _point_load_positions(spec)
        for x in pts:
            w = 1.0 / len(pts)
            c = spec.bearing_length_in
            if c > 0.0:
                _lump_patch(nodal, xs, x - 0.5 * c, x + 0.5 * c, w / c)
            else:
                nodal[_node_index(xs, x)] += w
    for i in np.nonzero(nodal)[0]:
        ops.load(_STEEL_NODE + int(i), 0.0, -float(nodal[i]), 0.0)

    return dict(xs=xs, n_ele=n_ele, i_left=i_left, i_right=i_right, i_mid=i_mid,
                conn_nodes=conn_nodes, groups=groups)


def _lump_patch(nodal: np.ndarray, xs: np.ndarray, x0: float, x1: float, q: float) -> None:
    """Consistent (half-element) lumping of a uniform load q over [x0, x1];
    x0 and x1 are mesh nodes."""
    tol = 1e-9 * max(abs(xs[-1] - xs[0]), 1.0)
    for e in range(len(xs) - 1):
        a, b = xs[e], xs[e + 1]
        if a >= x0 - tol and b <= x1 + tol:
            nodal[e] += 0.5 * q * (b - a)
            nodal[e + 1] += 0.5 * q * (b - a)


def _emergent_weight(x: float, x_end: float, L: float, tol: float) -> float:
    if x > x_end + tol:
        return 0.0
    if abs(x - 0.5 * L) <= tol and abs(x_end - 0.5 * L) <= tol:
        return 0.5
    return 1.0


def _advance(dlam: float, depth: int = 0) -> bool:
    """One load increment: Newton, modified Newton, bisection line search,
    then halving to a floor of dlam / 16 (as sheehan_udl._advance)."""
    for algo in (("Newton",), ("ModifiedNewton",), ("NewtonLineSearch", "-type", "Bisection")):
        ops.algorithm(*algo)
        if ops.analyze(1) == 0:
            ops.algorithm("Newton")
            return True
    ops.algorithm("Newton")
    if depth >= 4:
        return False
    half = 0.5 * dlam
    ops.integrator("LoadControl", half)
    ok = _advance(half, depth + 1) and _advance(half, depth + 1)
    ops.integrator("LoadControl", dlam)
    return ok


def beam_model(spec: Specimen, loads: Sequence[float], *,
               target_elements: int = 80, n_increments: int = 100,
               tol: float = 1e-8, max_iter: int = 60,
               record_path: bool = False) -> BeamModelResult:
    """Two-chain discrete-connector fibre beam model under load control.

    Formulation as src/validation/sheehan_udl.py and src/beam_model.py:
    deck and steel ``forceBeamColumn`` chains (5 Gauss-Lobatto points) on a
    shared reference line at the top of the steel, both fibre sections
    ``-noCentroid``; ``equalDOF`` on DOF 2 at every non-support node; one
    ``zeroLength`` connector (direction 1) at every connector POSITION with
    that position's connectors (not smeared); Concrete02 slab, Steel02
    plates and bars; steel pinned / roller, both chords restrained
    vertically at the supports.

    Mesh: every connector position, support, load point, bearing edge and
    mid-span is a node; intervals are subdivided so that no element is
    longer than span / ``target_elements``. The default 80 is mesh-converged
    to well under 1 % in deflection (tests/test_partial_connection.py:
    change against 160 below 0.01 % for a point-loaded beam with studs at
    12 in and 0.04 % for the Sheehan UDL beam with discrete studs, both past
    connector nonlinearity). Connectors at the same position are lumped
    into one spring: one built-in law keeps the Hysteretic material with
    summed k and Q_u; a measured curve or mixed laws become the exact sum of
    the envelopes as one MultiLinear material.

    ``loads`` in the module convention (total kip, or kip/in for udl), any
    order, all > 0. The path runs monotonically to the largest load in at
    least ``n_increments`` equal increments, landing exactly on every
    requested load. Stops at the first increment that fails.
    """
    loads_in = np.asarray(loads, dtype=float)
    if loads_in.ndim != 1 or loads_in.size == 0 or np.any(loads_in <= 0.0):
        raise ValueError("loads must be a non-empty 1-D sequence of positive values")
    targets = np.unique(loads_in)
    lay = _build(spec, target_elements)
    xs, i_mid = lay["xs"], lay["i_mid"]
    L = spec.span_in
    cf = compression_force_capacity(spec)
    ey_bot = spec.fy_bot_flange_ksi / spec.es_ksi
    ey_top = spec.fy_top_flange_ksi / spec.es_ksi
    e_mid = i_mid - 1 if i_mid > 0 else 0          # element ending at mid-span node
    sec_mid = _N_LOBATTO if i_mid > 0 else 1
    tol_x = 1e-6 * L
    x_region = (spec.load_offset_in if spec.load_pattern == "two_point" else 0.5 * L)

    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("UmfPack")
    ops.test("NormDispIncr", tol, max_iter)
    ops.algorithm("Newton")
    dlam_max = targets[-1] / float(n_increments)
    ops.integrator("LoadControl", dlam_max)
    ops.analysis("Static")

    def snapshot() -> dict:
        slips = np.array([ops.nodeDisp(_DECK_NODE + i, 1) - ops.nodeDisp(_STEEL_NODE + i, 1)
                          for i in range(len(xs))])
        defo_s = ops.eleResponse(_STEEL_ELE + e_mid, "section", sec_mid, "deformation")
        defo_d = ops.eleResponse(_DECK_ELE + e_mid, "section", sec_mid, "deformation")
        eps0, kap = float(defo_s[0]), float(defo_s[1])
        # fibre strain = eps0 - y kappa, y up from the top of the steel
        e_bot = eps0 + spec.steel_depth_in * kap
        e_top = eps0
        forces, per = {}, 0.0
        for j, i in enumerate(lay["conn_nodes"]):
            s = ops.eleResponse(_CONN_ELE + j, "material", 1, "stress")
            f = float(s[0]) if s else 0.0
            forces[i] = f
            per = max(per, abs(f) / lay["groups"][i]["n"])

        def slab_force(x_end: float) -> tuple[float, float]:
            # Slab force at x_end from each slab end: connectors from the
            # free end (overhang included) up to x_end; a connector exactly
            # at mid-span is shared by the two halves.
            left = right = 0.0
            for i, f in forces.items():
                left += _emergent_weight(xs[i], x_end, L, tol_x) * f
                right += _emergent_weight(L - xs[i], x_end, L, tol_x) * f
            return abs(left), abs(right)

        lft, rgt = slab_force(x_region)
        lh, rh = slab_force(0.5 * L)
        return dict(
            deflection=-ops.nodeDisp(_STEEL_NODE + i_mid, 2),
            end_slip=max(abs(slips[0]), abs(slips[-1])),
            support_slip=max(abs(slips[lay["i_left"]]), abs(slips[lay["i_right"]])),
            max_slip=float(np.abs(slips).max()),
            kappa=kap, kappa_deck=float(defo_d[1]),
            e_bot=e_bot, e_top=e_top,
            yielded=bool(e_bot >= ey_bot or -e_top >= ey_top or e_top >= ey_top),
            eta=0.5 * (lft + rgt) / cf, eta_half=0.5 * (lh + rh) / cf,
            fmax=per)

    keys = ("deflection", "end_slip", "support_slip", "max_slip", "kappa", "kappa_deck",
            "e_bot", "e_top", "eta", "eta_half", "fmax")
    rec = {k: np.full(targets.size, np.nan) for k in keys}
    yielded = np.zeros(targets.size, dtype=bool)
    conv = np.zeros(targets.size, dtype=bool)
    path = {"load": [0.0], "deflection_in": [0.0], "end_slip_in": [0.0],
            "max_slip_in": [0.0]} if record_path else None

    lam = 0.0
    ok = True
    for t_idx, target in enumerate(targets):
        m = max(1, int(math.ceil((target - lam) / dlam_max - 1e-9)))
        d = (target - lam) / m
        ops.integrator("LoadControl", d)
        for _ in range(m):
            if not _advance(d):
                ok = False
                break
            if record_path:
                snap = snapshot()
                path["load"].append(ops.getTime())
                path["deflection_in"].append(snap["deflection"])
                path["end_slip_in"].append(snap["end_slip"])
                path["max_slip_in"].append(snap["max_slip"])
        if not ok:
            break
        lam = ops.getTime()
        snap = snapshot()
        for k in keys:
            rec[k][t_idx] = snap[k]
        yielded[t_idx] = snap["yielded"]
        conv[t_idx] = True
    ops.wipe()

    order = np.searchsorted(targets, loads_in)
    moments = np.array([midspan_moment(spec, w) for w in loads_in])
    if path is not None:
        path = {k: np.asarray(v) for k, v in path.items()}
    return BeamModelResult(
        label=spec.label, load=loads_in, moment_kip_in=moments,
        deflection_in=rec["deflection"][order], end_slip_in=rec["end_slip"][order],
        support_slip_in=rec["support_slip"][order], max_slip_in=rec["max_slip"][order],
        curvature_mid_1_per_in=rec["kappa"][order],
        curvature_mid_deck_1_per_in=rec["kappa_deck"][order],
        steel_strain_bottom_mid=rec["e_bot"][order], steel_strain_top_mid=rec["e_top"][order],
        first_yield=yielded[order], eta_emergent=rec["eta"][order],
        eta_emergent_half_span=rec["eta_half"][order],
        connector_force_max_kip=rec["fmax"][order], converged=conv[order],
        n_elements=lay["n_ele"], node_x_in=xs, cf_kip=cf, path=path)


def mesh_convergence(spec: Specimen, load: float,
                     target_elements: Sequence[int] = (40, 80, 160)) -> dict:
    """Mid-span deflection at ``load`` for several mesh densities, and the
    relative change of each against the finest."""
    defl = {}
    for n in target_elements:
        r = beam_model(spec, [load], target_elements=n)
        defl[n] = float(r.deflection_in[0])
    finest = defl[max(target_elements)]
    return dict(deflection_in=defl,
                rel_change_vs_finest={n: abs(v / finest - 1.0) for n, v in defl.items()})


# ============================================ per-specimen comparison

def compare_specimen(spec: Specimen, loads: Sequence[float],
                     delta_meas_in: Sequence[float], *,
                     rho_l_value: Optional[float] = None,
                     linearity_limit_load: Optional[float] = None,
                     first_yield_load: Optional[float] = None,
                     run_beam_model: bool = True,
                     beam_kwargs: Optional[dict] = None,
                     eta_kwargs: Optional[dict] = None,
                     shear: Optional[dict] = None):
    """One row per measured load level, every quantity in the brief.

    Columns: load, moment_kip_in, mp_db_kip_in, m_over_mp, regime,
    eta_plastic, rho_l, delta_meas_in, delta_full_in, delta_steel_in,
    delta_noncomposite_in, delta_shear_in (NaN unless ``shear`` is given; it
    is NOT added to delta_full), delta_beam_in, R_meas = delta_full /
    delta_meas, Delta_meas = 1 - R_meas, R_beam = delta_full / delta_beam,
    R_EI (table, NaN above M/M_p = 0.6), delta_REI_in = delta_full / R_EI,
    beam_first_yield, beam_converged, beam_end_slip_in, beam_eta_emergent,
    in_elastic_range, exclusion_reason.

    A level is excluded from the stiffness statistics when M/M_p > 0.6,
    when load > ``linearity_limit_load`` (programme's reported limit), or
    past first yield: ``first_yield_load`` if the programme reports one,
    otherwise the beam model's mid-span flange fibre reaching F_y.
    ``rho_l_value`` defaults to :func:`rho_l` (gross).
    """
    import pandas as pd

    loads = np.asarray(loads, dtype=float)
    dmeas = np.asarray(delta_meas_in, dtype=float)
    if loads.shape != dmeas.shape:
        raise ValueError("loads and delta_meas_in must have the same length")
    ts = transformed_section(spec)
    eta = eta_plastic(spec, **(eta_kwargs or {}))
    mp = plastic_moment_database_definition(spec, eta)
    rho = rho_l(spec) if rho_l_value is None else rho_l_value

    beam = beam_model(spec, loads, **(beam_kwargs or {})) if run_beam_model else None
    rows = []
    for k, (w, dm) in enumerate(zip(loads, dmeas)):
        mom = midspan_moment(spec, w)
        mr = mom / mp
        regime = r_ei_regime(mr)
        d_full = elastic_deflection(spec, ts.EI_full_kip_in2, w)
        d_steel = elastic_deflection(spec, ts.EI_steel_kip_in2, w)
        d_nc = elastic_deflection(spec, ts.EI_noncomposite_kip_in2, w)
        d_shear = shear_deflection(spec, w, **shear) if shear is not None else float("nan")
        rei = r_ei_table(eta, mr, rho)
        d_beam = float(beam.deflection_in[k]) if beam is not None else float("nan")
        reasons = []
        if mr > M_OVER_MP_EXTENDED + 1e-12:
            reasons.append("M/M_p > 0.6")
        if linearity_limit_load is not None and w > linearity_limit_load * (1 + 1e-9):
            reasons.append("past reported linearity limit")
        if first_yield_load is not None:
            if w >= first_yield_load * (1 - 1e-9):
                reasons.append("past reported first yield")
        elif beam is not None and bool(beam.first_yield[k]):
            reasons.append("past first yield (beam model)")
        if beam is not None and not bool(beam.converged[k]):
            reasons.append("beam model not converged")
        r_meas = d_full / dm if dm > 0 else float("nan")
        rows.append(dict(
            label=spec.label, load=w, moment_kip_in=mom, mp_db_kip_in=mp, m_over_mp=mr,
            regime=regime, eta_plastic=eta, rho_l=rho, delta_meas_in=dm,
            delta_full_in=d_full, delta_steel_in=d_steel, delta_noncomposite_in=d_nc,
            delta_shear_in=d_shear, delta_beam_in=d_beam,
            R_meas=r_meas, Delta_meas=1.0 - r_meas,
            R_beam=d_full / d_beam if d_beam and np.isfinite(d_beam) else float("nan"),
            R_EI=rei, delta_REI_in=d_full / rei if np.isfinite(rei) else float("nan"),
            beam_first_yield=bool(beam.first_yield[k]) if beam is not None else None,
            beam_converged=bool(beam.converged[k]) if beam is not None else None,
            beam_end_slip_in=float(beam.end_slip_in[k]) if beam is not None else float("nan"),
            beam_eta_emergent=float(beam.eta_emergent[k]) if beam is not None else float("nan"),
            in_elastic_range=not any(r for r in reasons if r != "beam model not converged"),
            exclusion_reason="; ".join(reasons),
            n_modular=ts.n, ec_ksi=ts.ec_ksi, es_ksi=ts.es_ksi,
            I_full_in4=ts.I_full_in4, I_steel_in4=ts.I_steel_in4,
        ))
    return pd.DataFrame(rows)


def summarise_comparison(df) -> dict:
    """Service-load and extended-elastic summary of a
    :func:`compare_specimen` table, over in-range levels only.

    For each regime subset (service: M/M_p <= 0.4; extended: all in-range
    levels with M/M_p <= 0.6): mean of the per-level ratios, and a
    least-squares secant value R_ls = s_full / s_meas where s is the slope
    of deflection against load fitted through the origin (for delta_full,
    exactly linear, s_full = delta_full / load). R_EI is the table value in
    that regime (it does not vary with load inside a regime).
    """
    out = {}
    ok = df[df.in_elastic_range]
    for name, sub in (("service", ok[ok.regime == "service"]),
                      ("extended", ok[ok.m_over_mp <= M_OVER_MP_EXTENDED + 1e-12])):
        if sub.empty:
            out[name] = dict(n_levels=0)
            continue
        w = sub.load.to_numpy()
        s_meas = float(np.sum(w * sub.delta_meas_in) / np.sum(w * w))
        s_full = float(np.mean(sub.delta_full_in / w))
        entry = dict(
            n_levels=int(len(sub)), load_max=float(w.max()),
            m_over_mp_max=float(sub.m_over_mp.max()),
            R_meas_mean=float(sub.R_meas.mean()), R_meas_ls=s_full / s_meas,
            Delta_meas_mean=float(sub.Delta_meas.mean()),
            R_beam_mean=float(sub.R_beam.mean()) if sub.R_beam.notna().any() else float("nan"),
            R_EI=float(r_ei_table(float(sub.eta_plastic.iloc[0]), 0.0, float(sub.rho_l.iloc[0]),
                                  regime=name)),
            eta_plastic=float(sub.eta_plastic.iloc[0]), rho_l=float(sub.rho_l.iloc[0]))
        if sub.R_beam.notna().any():
            s_beam = float(np.sum(w * sub.delta_beam_in) / np.sum(w * w))
            entry["R_beam_ls"] = s_full / s_beam
        out[name] = entry
    return out
