"""Windowed-slope stiffness comparison for partial-connection beam tests.

Revision 2 method, applied identically to every specimen of the solid-slab
partial-connection validation set (scripts/validate_partial_<key>.py).

PRIMARY quantities (all flexural, to match R_EI, a ratio of flexural EI)
------------------------------------------------------------------------
1. K_meas: least-squares slope of load on mid-span deflection, intercept
   free, over the elastic window. C_meas = 1 / K_meas.
2. Shear compliance C_sh for the load pattern
   (:func:`partial_connection.shear_deflection`: the steel web carries the
   whole shear, G = E_s / 2.6), identical for the full-composite section
   and the tested beam. Primary shear area: the clear web t_w (d - 2 t_f);
   alternative: the full-depth web t_w d.
   C_flex,meas = C_meas - C_sh;  C_flex,full = 1 / K_full,flex (transformed
   section, ACI E_c unless a measured E_c is supplied, flexure only);
   R_meas,flex = C_flex,full / C_flex,meas.
3. R_beam_aci = K_beam / K_full,flex, with the beam model run at the SAME
   concrete modulus as K_full (``ec_ksi`` set to the transformed-section
   E_c). The beam model has no shear deformation, so this is already a
   flexural ratio.
4. R_EI: mean of :func:`partial_connection.r_ei_table` evaluated at each
   in-window point's M/M_p (database M_p), with the mid-window value and
   the window range.

SECONDARY quantities (kept)
---------------------------
R_meas = K_meas / K_full,flex (shear not removed); the secant from zero
(least-squares line through the origin over the window) in both the plain
and the flexural form; R_beam_E0 with the database Concrete02 initial
tangent 1000 f'c in the beam model.

Nothing here is fitted to a test; this module only organises calls to
src/validation/partial_connection.py (which it does not modify).

Load convention is the shared module's: total load (kip) for point
patterns, kip/in for udl. ``moment_offset_kip_in`` adds a moment that acts
on the COMPOSITE section but is not in the load record (a propped
specimen's self-weight) to M/M_p only; it does not change any stiffness.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.validation import partial_connection as pc


# ----------------------------------------------------------- regressions

def slope_intercept_free(load: np.ndarray, defl: np.ndarray) -> dict:
    """Least-squares load = K defl + b. Returns K, b, the standard error of K
    and the deflection intercept -b/K (the implied zero offset)."""
    load = np.asarray(load, float)
    defl = np.asarray(defl, float)
    n = load.size
    if n < 2:
        return dict(K=float("nan"), b=float("nan"), se=float("nan"),
                    defl_intercept_in=float("nan"), n=int(n))
    A = np.column_stack([defl, np.ones(n)])
    coef, *_ = np.linalg.lstsq(A, load, rcond=None)
    k, b = float(coef[0]), float(coef[1])
    se = float("nan")
    if n > 2:
        resid = load - A @ coef
        s2 = float(resid @ resid) / (n - 2)
        cov = s2 * np.linalg.inv(A.T @ A)
        se = float(math.sqrt(max(cov[0, 0], 0.0)))
    return dict(K=k, b=b, se=se, defl_intercept_in=-b / k if k else float("nan"), n=int(n))


def secant_through_origin(load: np.ndarray, defl: np.ndarray) -> float:
    """Least-squares load = K defl through the origin."""
    load = np.asarray(load, float)
    defl = np.asarray(defl, float)
    if load.size == 0:
        return float("nan")
    return float(np.sum(load * defl) / np.sum(defl * defl))


def flexural_ratio(k_meas: float, c_full_flex: float, c_shear: float) -> float:
    """R_flex = C_flex,full / (1/K_meas - C_sh); NaN if the flexural
    compliance left after removing shear is not positive."""
    c_flex = 1.0 / k_meas - c_shear
    return c_full_flex / c_flex if c_flex > 0.0 else float("nan")


# ------------------------------------------------------------ window rule

WINDOW_FRACTION = 0.25
WINDOW_TOLERANCE = 0.95     # nominal load steps (e.g. 9.87 kips on a 10 kip step of a 40 kip record)


def default_window(load: np.ndarray, elastic: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Stiffness window: elastic points with load >= 25 % of the highest
    elastic load (5 % tolerance for nominal load steps). Returns the mask,
    the highest elastic load and the threshold."""
    load = np.asarray(load, float)
    elastic = np.asarray(elastic, bool)
    p_el = float(load[elastic].max())
    thr = WINDOW_FRACTION * WINDOW_TOLERANCE * p_el
    return elastic & (load >= thr), p_el, thr


# ---------------------------------------------- multi-point load patterns

@dataclass
class MultiPointSpecimen(pc.Specimen):
    """A :class:`partial_connection.Specimen` loaded by equal point loads at
    ``point_positions_in`` (from the left support), symmetric about
    mid-span; ``load`` is the TOTAL of the point loads, each carrying
    load / n. ``load_pattern`` stays "midspan" only to pass the shared
    validation; every loading function of the shared module is routed to
    the positions below by :func:`_install_multipoint` (no edit of the
    shared module: its module-level helpers are wrapped at import time and
    fall through unchanged for every other specimen). The mid-span moment
    must be the maximum moment (true for symmetric patterns used here)."""

    point_positions_in: tuple = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.bearing_length_in:
            raise ValueError("MultiPointSpecimen: bearing_length_in not supported")
        xs = sorted(float(x) for x in self.point_positions_in)
        if len(xs) < 1 or xs[0] <= 0.0 or xs[-1] >= self.span_in:
            raise ValueError("point positions must lie inside the span")
        if not np.allclose(xs, sorted(self.span_in - x for x in xs), atol=1e-6):
            raise ValueError("point positions must be symmetric about mid-span")
        self.point_positions_in = tuple(xs)


def _install_multipoint() -> None:
    if getattr(pc, "_multipoint_installed", False):
        return
    orig_positions = pc._point_load_positions
    orig_moment = pc.midspan_moment

    def positions(spec):
        if isinstance(spec, MultiPointSpecimen):
            return list(spec.point_positions_in)
        return orig_positions(spec)

    def moment(spec, load):
        if isinstance(spec, MultiPointSpecimen):
            L = spec.span_in
            xs = spec.point_positions_in
            w = load / len(xs)
            # left reaction = load / 2 (symmetric); M(L/2) = R L/2 - sum w (L/2 - x) for x < L/2
            return 0.5 * load * 0.5 * L - sum(w * (0.5 * L - x) for x in xs if x < 0.5 * L - 1e-9)
        return orig_moment(spec, load)

    pc._point_load_positions = positions
    pc.midspan_moment = moment
    pc._multipoint_installed = True


_install_multipoint()


def verify_multipoint_rigid(spec: MultiPointSpecimen, load: float, *, k_rigid: float = 1.0e7,
                            pitch_in: float = 3.0) -> dict:
    """Check the multi-point wrapper: beam model with near-rigid elastic
    connectors at a close pitch, concrete modulus equal to the transformed
    section's, no rebar, against the closed-form superposition
    sum (W/n) x (3 L^2 - 4 x^2) / (48 EI) of the transformed section."""
    ts = pc.transformed_section(spec)
    L = spec.span_in
    n = int(round(L / pitch_in))
    conns = [pc.Connector(i * L / n, k_rigid, 1.0e6, 1, "elastic") for i in range(n + 1)]
    sp = dataclasses.replace(spec, connectors=conns, rebar=[], ec_ksi=ts.ec_ksi)
    res = pc.beam_model(sp, [load])
    closed = sum((load / len(sp.point_positions_in)) * min(x, L - x) * (3 * L * L - 4 * min(x, L - x) ** 2)
                 / (48.0 * ts.EI_full_kip_in2) for x in sp.point_positions_in)
    module = pc.elastic_deflection(sp, ts.EI_full_kip_in2, load)
    m_closed = 0.5 * load * 0.5 * L - sum((load / len(sp.point_positions_in)) * (0.5 * L - x)
                                            for x in sp.point_positions_in if x < 0.5 * L)
    return dict(load=load, closed_form_in=closed, module_elastic_deflection_in=module,
                beam_model_rigid_in=float(res.deflection_in[0]),
                beam_over_closed=float(res.deflection_in[0]) / closed,
                moment_wrapper_kip_in=pc.midspan_moment(sp, load), moment_closed_kip_in=m_closed)


def sub_window_ratios(case: "Case", mask: np.ndarray, label: str, c_full: float,
                      c_sh_clear: float, c_sh_full: float) -> dict:
    """R_meas and R_meas,flex (slope and secant) over another subset of
    points (for example readings above first slip, or a second record)."""
    mask = np.asarray(mask, bool)
    w = case.loads[mask]
    dm = case.delta_meas_in[mask]
    if w.size < 2:
        return dict(label=label, n_points=int(w.size), note="fewer than 2 points")
    fit = slope_intercept_free(w, dm)
    ks = secant_through_origin(w, dm)
    return dict(label=label, n_points=int(w.size),
                source_load_range=[float(case.source_load[mask].min()), float(case.source_load[mask].max())],
                K_meas=fit["K"], R_meas=fit["K"] * c_full,
                R_meas_flex=flexural_ratio(fit["K"], c_full, c_sh_clear),
                R_meas_flex_full_depth_web=flexural_ratio(fit["K"], c_full, c_sh_full),
                R_meas_secant=ks * c_full, R_meas_flex_secant=flexural_ratio(ks, c_full, c_sh_clear))


# ------------------------------------------------------------ case data

@dataclass
class Case:
    """One specimen, ready for :func:`analyse`.

    ``loads`` module convention (total kip); ``source_load`` and
    ``source_load_unit`` keep the record's own quantity for the CSV;
    ``elastic`` is the data file's flag; ``window`` the stiffness window
    (a subset of elastic); ``defl_uncertainty_in`` the stated absolute
    deflection uncertainty of a reading. ``shear_area_clear_in2`` =
    t_w (d - 2 t_f) and ``shear_area_full_in2`` = t_w d from the documented
    (handbook or measured) section dimensions. ``classification`` carries
    programme, slab_type, connection_system, scale and role for collation.
    """

    key: str
    spec: pc.Specimen
    loads: np.ndarray
    source_load: np.ndarray
    source_load_unit: str
    delta_meas_in: np.ndarray
    elastic: np.ndarray
    window: np.ndarray
    window_rule: str
    rho_l: float
    shear_area_clear_in2: float
    shear_area_full_in2: float
    classification: dict = field(default_factory=dict)
    rho_l_note: str = ""
    eta_kwargs: dict = field(default_factory=dict)
    eta_spec: Optional[pc.Specimen] = None      # connector set counted for eta (default: spec)
    eta_note: str = ""
    moment_offset_kip_in: float = 0.0
    moment_offset_note: str = ""
    defl_uncertainty_in: float = 0.0
    connector_note: str = ""
    flags: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    run_beam: bool = True

    def __post_init__(self) -> None:
        for name in ("loads", "source_load", "delta_meas_in"):
            setattr(self, name, np.asarray(getattr(self, name), float))
        self.elastic = np.asarray(self.elastic, bool)
        self.window = np.asarray(self.window, bool)
        n = self.loads.size
        if not all(a.size == n for a in (self.source_load, self.delta_meas_in,
                                         self.elastic, self.window)):
            raise ValueError("per-load arrays must have equal length")
        if np.any(self.window & ~self.elastic):
            raise ValueError("window must be a subset of the elastic points")


# ------------------------------------------------------------ helpers

def web_shear_areas(d_in: float, tf_in: float, tw_in: float) -> tuple[float, float]:
    """(clear web t_w (d - 2 t_f), full-depth web t_w d)."""
    return tw_in * (d_in - 2.0 * tf_in), tw_in * d_in


def unit_compliances(spec: pc.Specimen, a_clear: float, a_full: float) -> dict:
    """Deflection per unit load: full-composite, steel and non-composite
    flexure, and the shear term for both shear areas."""
    ts = pc.transformed_section(spec)
    return dict(
        ts=ts, full=pc.elastic_deflection(spec, ts.EI_full_kip_in2, 1.0),
        steel=pc.elastic_deflection(spec, ts.EI_steel_kip_in2, 1.0),
        noncomposite=pc.elastic_deflection(spec, ts.EI_noncomposite_kip_in2, 1.0),
        shear_clear=pc.shear_deflection(spec, 1.0, shear_area_in2=a_clear),
        shear_full=pc.shear_deflection(spec, 1.0, shear_area_in2=a_full))


def aci_spec(spec: pc.Specimen) -> pc.Specimen:
    """The specimen with the beam-model concrete modulus set to the
    transformed-section E_c (no change if a measured E_c is already given)."""
    if spec.ec_ksi is not None:
        return spec
    return dataclasses.replace(spec, ec_ksi=pc.transformed_section(spec).ec_ksi)


def m_over_mp(spec: pc.Specimen, loads: Sequence[float], mp: float, offset: float = 0.0) -> np.ndarray:
    return np.array([(pc.midspan_moment(spec, w) + offset) / mp for w in loads])


def r_ei_over(eta: float, mr: Sequence[float], rho: float) -> np.ndarray:
    return np.array([pc.r_ei_table(eta, float(m), rho) for m in mr])


def window_mid_load(case: Case) -> float:
    w = case.loads[case.window]
    return 0.5 * (float(w.min()) + float(w.max()))


def beam_window_stats(case: Case, spec: pc.Specimen, k_full: float,
                      extra_loads: Sequence[float] = (), **beam_kw) -> dict:
    """Beam model at the window loads (+ mid-window load + extras):
    K_beam slope and secant, R_beam, eta_emergent at mid-window."""
    w = case.loads[case.window]
    mid = window_mid_load(case)
    targets = np.unique(np.concatenate([w, [mid], np.asarray(extra_loads, float)]))
    res = pc.beam_model(spec, list(targets), **beam_kw)
    lookup = {float(t): i for i, t in enumerate(res.load)}
    dw = np.array([res.deflection_in[lookup[float(x)]] for x in w])
    i_mid = lookup[float(mid)]
    ok = bool(np.all(np.isfinite(dw)))
    fit = slope_intercept_free(w, dw) if ok else dict(K=float("nan"))
    k_sec = secant_through_origin(w, dw) if ok else float("nan")
    yielded = [float(x) for x, y in zip(res.load, res.first_yield) if y and x <= w.max() + 1e-9]
    return dict(
        K_beam=fit["K"], K_beam_secant=k_sec, R_beam=fit["K"] / k_full,
        R_beam_secant=k_sec / k_full, converged_all=ok,
        beam_concrete_modulus_ksi=pc.beam_model_concrete_modulus_ksi(spec),
        eta_emergent_mid=float(res.eta_emergent[i_mid]),
        eta_emergent_half_span_mid=float(res.eta_emergent_half_span[i_mid]),
        end_slip_mid_in=float(res.end_slip_in[i_mid]),
        connector_force_max_mid_kip=float(res.connector_force_max_kip[i_mid]),
        beam_first_yield_loads_in_window=yielded,
        result=res)


# ------------------------------------------------------------ analysis

def analyse(case: Case, *, beam_kwargs: Optional[dict] = None) -> tuple[pd.DataFrame, dict]:
    """Per-load table and specimen summary for one :class:`Case`."""
    spec = case.spec
    bk = beam_kwargs or {}
    u = unit_compliances(spec, case.shear_area_clear_in2, case.shear_area_full_in2)
    ts = u["ts"]
    c_full = u["full"]
    k_full, k_steel, k_nc = 1.0 / c_full, 1.0 / u["steel"], 1.0 / u["noncomposite"]

    eta_d = pc.eta_plastic_details(case.eta_spec or spec, **case.eta_kwargs)
    eta_d["note"] = case.eta_note
    eta = eta_d["eta"]
    mp_d = pc.plastic_moment_details(spec, eta)
    mp = mp_d["mp_kip_in"]
    mr = m_over_mp(spec, case.loads, mp, case.moment_offset_kip_in)
    rho = case.rho_l

    over = case.window & (mr > pc.M_OVER_MP_EXTENDED + 1e-12)
    if np.any(over):
        # in place, so that later sensitivity runs on this case use the same window
        case.window = case.window & ~over
        kept = case.source_load[case.window]
        msg = (f"{int(over.sum())} point(s) dropped for M/M_p > 0.6 (database M_p); window now "
               f"{kept.min():.4g}-{kept.max():.4g} ({case.source_load_unit}), {int(case.window.sum())} points")
        case.window_rule = f"{case.window_rule}; {msg}"
        case.notes.append(msg)

    w = case.loads[case.window]
    dm = case.delta_meas_in[case.window]
    fit = slope_intercept_free(w, dm)
    k_meas, k_sec = fit["K"], secant_through_origin(w, dm)
    mid = window_mid_load(case)
    mr_mid = float((pc.midspan_moment(spec, mid) + case.moment_offset_kip_in) / mp)
    mr_win = mr[case.window]
    rei_win = r_ei_over(eta, mr_win, rho)

    # reading uncertainty on K_meas: 2 x slope SE and 2u / deflection span
    span = float(dm.max() - dm.min()) if dm.size > 1 else float("nan")
    rel_reading = (2.0 * case.defl_uncertainty_in / span) if span and span > 0 else 0.0
    se_rel = fit["se"] / k_meas if np.isfinite(fit["se"]) else 0.0
    k_unc = k_meas * math.sqrt((2.0 * se_rel) ** 2 + rel_reading ** 2)

    # ---- flexural ratios
    r_flex = flexural_ratio(k_meas, c_full, u["shear_clear"])
    r_flex_fullweb = flexural_ratio(k_meas, c_full, u["shear_full"])
    r_flex_sec = flexural_ratio(k_sec, c_full, u["shear_clear"])
    r_flex_sec_fullweb = flexural_ratio(k_sec, c_full, u["shear_full"])
    r_flex_k_unc = sorted([flexural_ratio(k_meas - k_unc, c_full, u["shear_clear"]),
                           flexural_ratio(k_meas + k_unc, c_full, u["shear_clear"])])

    # ---- beam model: primary at the transformed-section E_c
    beam_stats, beam_e0 = {}, {}
    beam_defl = np.full(case.loads.size, np.nan)
    beam_defl_e0 = np.full(case.loads.size, np.nan)
    if case.run_beam:
        ctx = case.loads[(mr <= pc.M_OVER_MP_EXTENDED + 1e-12)]
        beam_stats = beam_window_stats(case, aci_spec(spec), k_full, extra_loads=ctx, **bk)
        res = beam_stats.pop("result")
        lk = {float(t): i for i, t in enumerate(res.load)}
        for i, x in enumerate(case.loads):
            if float(x) in lk:
                beam_defl[i] = res.deflection_in[lk[float(x)]]
        if spec.ec_ksi is None:
            beam_e0 = beam_window_stats(case, spec, k_full, **bk)
            res0 = beam_e0.pop("result")
            lk0 = {float(t): i for i, t in enumerate(res0.load)}
            for i, x in enumerate(case.loads):
                if float(x) in lk0:
                    beam_defl_e0[i] = res0.deflection_in[lk0[float(x)]]

    rows = []
    for i, x in enumerate(case.loads):
        rows.append(dict(
            specimen=spec.label, load=x, load_unit="kip (total applied, module convention)",
            source_load=case.source_load[i], source_load_unit=case.source_load_unit,
            moment_kip_in=pc.midspan_moment(spec, x) + case.moment_offset_kip_in,
            m_over_mp=mr[i], regime=pc.r_ei_regime(mr[i]),
            delta_meas_in=case.delta_meas_in[i], delta_full_in=x * c_full,
            delta_steel_in=x * u["steel"], delta_noncomposite_in=x * u["noncomposite"],
            delta_shear_clear_web_in=x * u["shear_clear"], delta_shear_full_web_in=x * u["shear_full"],
            delta_beam_in=beam_defl[i], delta_beam_E0_in=beam_defl_e0[i],
            R_EI_at_load=pc.r_ei_table(eta, mr[i], rho),
            elastic=bool(case.elastic[i]), in_window=bool(case.window[i])))
    df = pd.DataFrame(rows)

    k_beam = beam_stats.get("K_beam", float("nan"))
    checks = {
        "K_steel<=K_meas (within uncertainty)": bool(k_steel <= k_meas + k_unc),
        "K_meas<=K_full (within uncertainty)": bool(k_meas <= k_full + k_unc),
        "K_steel<=K_meas (nominal)": bool(k_steel <= k_meas),
        "K_meas<=K_full (nominal)": bool(k_meas <= k_full),
        "flexural compliance positive after shear removal": bool(np.isfinite(r_flex) and np.isfinite(r_flex_fullweb)),
    }
    if case.run_beam:
        checks["K_noncomposite<=K_beam_aci<=K_full"] = bool(k_nc <= k_beam <= k_full)

    summary = dict(
        specimen=spec.label, source=spec.source, classification=dict(case.classification),
        window=dict(rule=case.window_rule, n_points=int(case.window.sum()),
                    load_min=float(w.min()), load_max=float(w.max()), load_mid=mid,
                    source_load_min=float(case.source_load[case.window].min()),
                    source_load_max=float(case.source_load[case.window].max()),
                    source_load_unit=case.source_load_unit,
                    m_over_mp_min=float(mr_win.min()), m_over_mp_max=float(mr_win.max()),
                    m_over_mp_mid=mr_mid),
        K_units="kip of total applied load (kip/in for udl) per in of mid-span deflection",
        # ---------------- primary
        R_meas_flex=r_flex,
        R_meas_flex_range_shear_area=sorted([r_flex, r_flex_fullweb]),
        R_meas_flex_range_K_uncertainty=r_flex_k_unc,
        R_beam_aci=beam_stats.get("R_beam", float("nan")),
        R_EI_window_mean=float(np.nanmean(rei_win)),
        R_EI=dict(window_mean=float(np.nanmean(rei_win)), mid_window=pc.r_ei_table(eta, mr_mid, rho),
                  window_min=float(np.nanmin(rei_win)), window_max=float(np.nanmax(rei_win)),
                  regime_mid=pc.r_ei_regime(mr_mid)),
        eta_plastic=eta, rho_l=rho,
        # ---------------- secondary
        R_meas=k_meas / k_full, Delta_meas=1.0 - k_meas / k_full,
        R_meas_secant=k_sec / k_full,
        R_meas_flex_secant=r_flex_sec,
        R_meas_flex_secant_range_shear_area=sorted([r_flex_sec, r_flex_sec_fullweb]),
        R_beam_E0=beam_e0.get("R_beam", float("nan")) if beam_e0 else float("nan"),
        R_beam_E0_note=("beam model with Concrete02 E_0 = 1000 f'c" if beam_e0
                        else "not applicable: measured E_c used in both the section and the beam model"),
        # ---------------- stiffnesses and section
        K_meas=dict(slope=k_meas, slope_standard_error=fit["se"], uncertainty_used_in_checks=k_unc,
                    implied_deflection_zero_offset_in=fit["defl_intercept_in"], secant_from_zero=k_sec),
        K_full=k_full, K_steel=k_steel, K_noncomposite=k_nc, K_beam_aci=k_beam,
        K_beam_E0=beam_e0.get("K_beam", float("nan")) if beam_e0 else float("nan"),
        shear=dict(G_ksi=spec.es_ksi / 2.6, area_clear_web_in2=case.shear_area_clear_in2,
                   area_full_depth_web_in2=case.shear_area_full_in2,
                   C_sh_clear_over_C_full=u["shear_clear"] / c_full,
                   C_sh_full_over_C_full=u["shear_full"] / c_full,
                   assumption="steel web carries the whole vertical shear, same term for section and test"),
        section=dict(n_modular=ts.n, ec_ksi=ts.ec_ksi, ec_source=ts.ec_source, es_ksi=ts.es_ksi,
                     I_full_in4=ts.I_full_in4, I_steel_in4=ts.I_steel_in4,
                     y_na_from_slab_top_in=ts.y_na_from_slab_top_in),
        beam_aci=beam_stats, beam_E0=beam_e0,
        eta_plastic_details=eta_d,
        eta_emergent_mid_window=beam_stats.get("eta_emergent_mid", float("nan")),
        rho_l_note=case.rho_l_note,
        M_p=dict(mp_db_kip_in=mp, details=mp_d, moment_offset_kip_in=case.moment_offset_kip_in,
                 moment_offset_note=case.moment_offset_note),
        connector_note=case.connector_note,
        flags=list(case.flags), notes=list(case.notes),
        sanity_checks=checks,
    )
    return df, summary


# ------------------------------------------------------------ sensitivity

def sensitivity_beam(case: Case, label: str, spec_variant: pc.Specimen,
                     k_full: float, **beam_kw) -> dict:
    """Primary R_beam (ACI E_c in the beam model) of a specimen variant
    (connector layout or law) over the same window; K_full is the base
    value (the section is unchanged)."""
    st = beam_window_stats(case, aci_spec(spec_variant), k_full, **beam_kw)
    st.pop("result")
    return dict(label=label, R_beam=st["R_beam"], K_beam=st["K_beam"],
                eta_emergent_mid=st["eta_emergent_mid"])


def sensitivity_rei(case: Case, label: str, eta: float, rho: Optional[float] = None,
                    spec: Optional[pc.Specimen] = None) -> dict:
    """R_EI (window mean, mid-window, window range) at another eta (and
    rho), with M_p recomputed at that eta by the database definition."""
    sp = spec or case.spec
    rho = case.rho_l if rho is None else rho
    mp = pc.plastic_moment_database_definition(sp, eta)
    w = case.loads[case.window]
    mid = window_mid_load(case)
    mr_mid = (pc.midspan_moment(sp, mid) + case.moment_offset_kip_in) / mp
    rw = r_ei_over(eta, m_over_mp(sp, w, mp, case.moment_offset_kip_in), rho)
    return dict(label=label, eta=eta, rho_l=rho, m_over_mp_mid=mr_mid,
                R_EI_window_mean=float(np.nanmean(rw)),
                R_EI_mid=pc.r_ei_table(eta, mr_mid, rho),
                R_EI_window_min=float(np.nanmin(rw)), R_EI_window_max=float(np.nanmax(rw)))


def ranges(summary: dict, beam_cases: list, rei_cases: list) -> dict:
    """Collect base and variant values into min/max ranges."""
    def mm(v):
        v = [x for x in v if x is not None and np.isfinite(x)]
        return [min(v), max(v)] if v else None
    return dict(
        R_beam_aci_range=mm([summary["R_beam_aci"]] + [c["R_beam"] for c in beam_cases]),
        R_EI_window_mean_range=mm([summary["R_EI"]["window_mean"]] + [c["R_EI_window_mean"] for c in rei_cases]),
        R_EI_mid_range=mm([summary["R_EI"]["mid_window"]] + [c["R_EI_mid"] for c in rei_cases]),
        R_EI_window_envelope=mm([summary["R_EI"]["window_min"], summary["R_EI"]["window_max"]]
                                + [c["R_EI_window_min"] for c in rei_cases]
                                + [c["R_EI_window_max"] for c in rei_cases]),
        beam_cases=beam_cases, rei_cases=rei_cases,
        note="R_meas does not depend on these inputs")


def print_summary(s: dict) -> None:
    """One compact console block per specimen."""
    f = lambda v, p=3: "nan" if v is None or not np.isfinite(v) else f"{v:.{p}f}"
    print(f"{s['specimen']}: {s['window']['rule']}")
    print(f"   PRIMARY R_meas,flex {f(s['R_meas_flex'])} [shear area {f(s['R_meas_flex_range_shear_area'][0])}-"
          f"{f(s['R_meas_flex_range_shear_area'][1])}]  R_beam_aci {f(s['R_beam_aci'])}  "
          f"R_EI mean {f(s['R_EI']['window_mean'])} (mid {f(s['R_EI']['mid_window'])}, "
          f"{f(s['R_EI']['window_min'], 2)}-{f(s['R_EI']['window_max'], 2)})  eta {f(s['eta_plastic'])}  "
          f"M/Mp mid {f(s['window']['m_over_mp_mid'], 2)}")
    print(f"   secondary R_meas {f(s['R_meas'])} sec {f(s['R_meas_secant'])} flex-sec {f(s['R_meas_flex_secant'])}  "
          f"R_beam_E0 {f(s['R_beam_E0'])}  K_meas {f(s['K_meas']['slope'], 1)} K_full {f(s['K_full'], 1)} "
          f"K_steel {f(s['K_steel'], 1)} K_beam_aci {f(s['K_beam_aci'], 1)}  "
          f"C_sh/C_full {f(s['shear']['C_sh_clear_over_C_full'])}  checks ok {all(s['sanity_checks'].values())}")
    sens = s.get("sensitivity")
    if sens:
        for c in sens["beam_cases"]:
            print(f"      beam: {c['label']}: R_beam_aci {f(c['R_beam'])}")
        for c in sens["rei_cases"]:
            print(f"      R_EI: {c['label']}: eta {f(c['eta'])} mean {f(c['R_EI_window_mean'])}")


def to_jsonable(o):
    """Recursively convert numpy scalars/arrays for json.dump."""
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [to_jsonable(v) for v in o.tolist()]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if not np.isfinite(v) else v
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o
