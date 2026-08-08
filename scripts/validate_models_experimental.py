"""Experimental validation driver: OpenSeesPy models vs published tests.

Compares BOTH numerical models against the 24 measured (M, phi, slip)
points digitised from three laboratory programmes (Chapman &
Balakrishnan 1964; Nie & Cai 2003; Ansourian 1982) in
``data/experimental/literature_tests.csv``:

(a) the section-level fibre model (``src.data_generation.moment_curvature
    .analyze``), which represents partial composite action by scaling the
    effective deck width by ``eta_c`` and predicts M-phi only; and
(b) the beam-level model with discrete shear-connector springs
    (``src.beam_model.analyze_beam``), which predicts interface slip
    explicitly.

Each CSV row is one measured POINT, not a curve, so the comparison is:
run the model, interpolate the predicted curvature (and slip, beam model)
at the measured moment on the RISING branch of the computed M-phi
(M-slip) curve, and compare against the measured values.

Interpolation caveats handled explicitly (never silently clipped):

* ``beyond_model_peak`` — the measured moment exceeds the model's peak
  moment; phi(M) does not exist there, so the prediction is NaN and the
  specimen is flagged. All three test programmes loaded specimens to
  collapse, and the fibre model (bilinear hardening b = 0.01, unconfined
  Concrete02, no deck rebar) underpredicts collapse moments by ~5-20%,
  so this flag fires for most at-ultimate points. That is a strength
  statement, not a stiffness one, which is why the driver also reports:
* ``M_at_phi`` — the SUPPLEMENTARY, well-conditioned inverse metric:
  predicted moment at the measured curvature. Near the plastic plateau
  dphi/dM diverges (a 1% moment error maps to >100% curvature error)
  while dM/dphi stays finite, so M(phi_meas) is the meaningful fidelity
  measure for plateau points and is reported alongside phi(M).
* ``near_model_peak`` — measured moment within 3% of the model peak;
  phi(M) is interpolable but ill-conditioned, so treat with caution.

BEAM-MODEL STATUS: preliminary. The beam model is under concurrent
debugging for a suspected moment-assembly/curvature bug (symptom:
implausibly flexible response at high eta_c). It is invoked through the
single entry point :func:`run_beam_model` so the whole beam column-block
of the outputs can be regenerated after the fix by re-running this
script. All beam-model output columns carry a ``beam_`` prefix and a
``beam_model_status = preliminary`` marker.

Stud layouts for the beam model come from the CSV where available
(Chapman & Balakrishnan notes give total stud counts; assumed 3/4-in
studs in pairs) and are otherwise synthesised to match the specimen's
``composite_action`` following ``connector_layout_for_eta()`` in
paper2-beam-level/src/validation/section_vs_beam.py.

Units: the CSV is kip / inch / ksi throughout, matching the model units
(verified against src/validation/experimental.py, which feeds the same
columns to the surrogate unconverted).

Outputs
-------
reports/model_validation/per_specimen.csv
reports/model_validation/summary.csv
figures/validation/fig_exp_mphi_overlay.png
figures/validation/fig_exp_slip_parity.png

Usage
-----
/opt/anaconda3/envs/ops_x86/bin/python scripts/validate_models_experimental.py \
    [--skip-beam] [--n-steps-section 160] [--n-steps-beam 100]
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.beam_model import (BeamParams, analyze_beam,  # noqa: E402
                            _compression_force_capacity, stud_strength_kip)
from src.data_generation.lhs_sampler import SectionParams  # noqa: E402
from src.data_generation.moment_curvature import analyze  # noqa: E402
from src.utils import plotting  # noqa: E402

BEAM_MODEL_STATUS = "final"  # centroid bug fixed ('-noCentroid' on both fiber sections)
_E_STEEL_KSI = 29_000.0

PROGRAMME_LABELS = {
    "Chapman_Balakrishnan_1964": "Chapman & Balakrishnan (1964)",
    "Nie_Cai_2003": "Nie & Cai (2003)",
    "Ansourian_1982": "Ansourian (1982)",
}


# ---------------------------------------------------------------------------
# CSV loading (kip / in / ksi — same convention the models use)
# ---------------------------------------------------------------------------

def load_tests(csv_path: Path) -> pd.DataFrame:
    """Load the literature-test CSV. Prefers the project loader (adds
    mp_estimate etc.); falls back to a plain read if the surrogate stack
    (torch) is unavailable in the current environment."""
    try:
        from src.validation.experimental import load_experimental_csv
        return load_experimental_csv(csv_path)
    except Exception as exc:  # pragma: no cover - env-dependent
        print(f"[warn] project loader unavailable ({exc}); plain pd.read_csv")
        return pd.read_csv(csv_path)


# ---------------------------------------------------------------------------
# interpolation helpers
# ---------------------------------------------------------------------------

def rising_branch(moment: np.ndarray, *others: np.ndarray):
    """Truncate the sweep at the moment peak and drop any non-monotonic
    wiggles so np.interp sees a strictly increasing abscissa."""
    if moment.size == 0:
        return (moment,) + others
    i_peak = int(np.argmax(moment))
    m = moment[: i_peak + 1]
    keep = np.zeros(m.size, dtype=bool)
    run_max = -np.inf
    for k, v in enumerate(m):
        if v > run_max:
            keep[k] = True
            run_max = v
    return (m[keep],) + tuple(o[: i_peak + 1][keep] for o in others)


def interp_at_moment(m_target: float, moment: np.ndarray, y: np.ndarray):
    """Interpolate y(M) on the rising branch.

    Returns (value, flag) with flag in {"", "beyond_model_peak",
    "near_model_peak", "no_curve"}. Extrapolation beyond the peak is
    reported as NaN + flag, never clipped."""
    m_rise, y_rise = rising_branch(moment, y)
    if m_rise.size < 3:
        return float("nan"), "no_curve"
    peak = float(m_rise[-1])
    if m_target > peak:
        return float("nan"), "beyond_model_peak"
    val = float(np.interp(m_target, m_rise, y_rise))
    if m_target > 0.97 * peak:
        return val, "near_model_peak"
    return val, ""


def interp_moment_at(phi_target: float, phi: np.ndarray, moment: np.ndarray):
    """Supplementary inverse metric: predicted moment at the measured
    curvature, using the full (possibly softening) curve. phi is
    monotonic by construction of the displacement-controlled sweep."""
    if phi.size < 3 or not np.isfinite(phi_target):
        return float("nan"), "no_curve"
    if phi_target > phi[-1]:
        return float("nan"), "beyond_model_curve"
    return float(np.interp(phi_target, phi, moment)), ""


def rel_err_pct(pred: float, meas: float) -> float:
    if not (np.isfinite(pred) and np.isfinite(meas)) or meas == 0.0:
        return float("nan")
    return 100.0 * (pred - meas) / meas


# ---------------------------------------------------------------------------
# model entry points
# ---------------------------------------------------------------------------

def section_params_from_row(row: pd.Series) -> SectionParams:
    """Map one CSV row to the section model's parameter record.
    eta_c = composite_action (the width-scaling representation).
    pcb_* fields are inert placeholders (W sections only in this CSV)."""
    return SectionParams(
        sample_id=0,
        section_type=str(row.section_type),
        span_in=float(row.span_in),
        deck_thickness_in=float(row.deck_thickness_in),
        deck_width_in=float(row.deck_width_in),
        girder_spacing_in=float(row.girder_spacing_in),
        fc_deck_ksi=float(row.fc_deck_ksi),
        composite_action=float(row.composite_action),
        shear_stud_stiffness_ratio=float(row.shear_stud_stiffness_ratio),
        fy_ksi=float(row.fy_ksi),
        steel_depth_in=float(row.steel_depth_in),
        flange_width_in=float(row.flange_width_in),
        flange_thickness_in=float(row.flange_thickness_in),
        web_thickness_in=float(row.web_thickness_in),
        pcb_depth_in=30.0, pcb_fc_ksi=6.0, pcb_prestress_ratio=0.5,
    )


def run_section_model(row: pd.Series, n_steps: int):
    """Single clean entry point for the section-level model. Sweeps to
    40x the yield-curvature scale so the post-peak branch is captured
    (the labs pushed well past peak; 12x default would truncate)."""
    p = section_params_from_row(row)
    d_total = p.deck_thickness_in + p.steel_depth_in
    phi_y_scale = 2.0 * p.fy_ksi / (_E_STEEL_KSI * d_total)
    return analyze(p, n_steps=n_steps, curvature_max=40.0 * phi_y_scale)


_STUD_COUNT_RE = re.compile(r"(\d+)\s+studs")


def derive_stud_layout(row: pd.Series,
                       n_studs_hint: int | None = None) -> tuple[BeamParams, str]:
    """Build BeamParams with a connector layout derived from the CSV.

    * Chapman & Balakrishnan notes give total stud counts ("A6 32 studs
      ..."): use them directly — studs in pairs (per the 1964 paper's
      two-per-row arrangement), assumed 3/4-in diameter, uniform pitch.
    * Otherwise synthesise (diameter, studs/row, pitch) so the AISC
      layout ratio min(SumQn, Cf)/Cf matches the CSV composite_action,
      mirroring connector_layout_for_eta() in
      paper2-beam-level/src/validation/section_vs_beam.py.

    Returns (params, provenance-string).
    """
    base = BeamParams(
        span_in=float(row.span_in),
        deck_thickness_in=float(row.deck_thickness_in),
        deck_width_in=float(row.deck_width_in),
        fc_deck_ksi=float(row.fc_deck_ksi),
        fy_ksi=float(row.fy_ksi),
        steel_depth_in=float(row.steel_depth_in),
        flange_width_in=float(row.flange_width_in),
        flange_thickness_in=float(row.flange_thickness_in),
        web_thickness_in=float(row.web_thickness_in),
        section_type=str(row.section_type),
        ks_ratio=float(row.shear_stud_stiffness_ratio),
        # Match the section model's bare deck so the comparison isolates
        # the partial-interaction representation (same choice as
        # paper2's section_vs_beam.py).
        deck_rho_long=0.0,
    )
    note = str(row.get("notes", "") or "")
    m = _STUD_COUNT_RE.search(note)
    n_total = int(m.group(1)) if m else n_studs_hint
    if n_total:
        n_row = 2
        n_rows = max(2, n_total // n_row)
        pitch = base.span_in / n_rows
        p = BeamParams(**{**asdict(base), "stud_diameter_in": 0.75,
                          "n_studs_per_row": n_row, "stud_pitch_in": pitch})
        return p, (f"notes: {n_total} studs -> {n_rows} rows x {n_row} "
                   f"@ {pitch:.1f} in (0.75-in dia assumed)")
    p = _connector_layout_for_eta(base, float(row.composite_action))
    return p, (f"synthesised for eta_c={row.composite_action:.2f}: "
               f"{p.n_studs_per_row} x {p.stud_diameter_in:.3f}-in "
               f"@ {p.stud_pitch_in:.1f} in")


def _connector_layout_for_eta(p: BeamParams, target_eta: float) -> BeamParams:
    """Re-implementation of paper2-beam-level connector_layout_for_eta():
    pick (stud diameter, studs/row, pitch) so min(SumQn, Cf)/Cf over the
    half span matches target_eta, keeping pitch in the practical
    4-30 in range where possible."""
    cf = _compression_force_capacity(p)
    sum_qn_needed = target_eta * cf
    best = None
    for d_stud in (0.75, 0.875, 0.625):
        q_per_stud = stud_strength_kip(d_stud, p.fc_deck_ksi)
        for n_row in (2, 1, 3):
            q_per_row = q_per_stud * n_row
            n_rows_total = sum_qn_needed / max(q_per_row, 1e-9) * 2.0
            if n_rows_total < 2:
                continue
            pitch = p.span_in / n_rows_total
            if 4.0 <= pitch <= 30.0:
                return BeamParams(**{**asdict(p), "stud_diameter_in": d_stud,
                                     "n_studs_per_row": n_row,
                                     "stud_pitch_in": pitch})
            penalty = abs(pitch - 12.0)
            if best is None or penalty < best[0]:
                best = (penalty, d_stud, n_row, pitch)
    _, d_stud, n_row, pitch = best
    pitch = min(30.0, max(4.0, pitch))
    return BeamParams(**{**asdict(p), "stud_diameter_in": d_stud,
                         "n_studs_per_row": n_row, "stud_pitch_in": pitch})


def run_beam_model(row: pd.Series, n_steps: int,
                   n_studs_hint: int | None = None):
    """Single clean entry point for the (PRELIMINARY) beam-level model.
    Everything beam-related funnels through here so the beam block can be
    re-run wholesale once the moment-assembly bug is fixed.

    Returns (BeamResult, BeamParams, layout_note)."""
    p, layout_note = derive_stud_layout(row, n_studs_hint)
    res = analyze_beam(p, n_steps=n_steps)
    return res, p, layout_note


# ---------------------------------------------------------------------------
# per-specimen evaluation
# ---------------------------------------------------------------------------

def specimen_key(row: pd.Series) -> tuple:
    """Cache key: the physical specimen (geometry + materials only, NOT
    notes). CB-Ax-40T and CB-Ax-ULT rows are the same beam at two load
    levels — analyse it once and share the notes-derived stud layout."""
    return (row.section_type, row.span_in, row.deck_thickness_in,
            row.deck_width_in, row.fc_deck_ksi, row.composite_action,
            row.shear_stud_stiffness_ratio, row.fy_ksi, row.steel_depth_in,
            row.flange_width_in, row.flange_thickness_in, row.web_thickness_in)


def stud_count_by_specimen(df: pd.DataFrame) -> dict:
    """Scan all rows' notes for explicit stud counts and map them onto the
    specimen geometry key, so load-level rows whose own note omits the
    count (e.g. CB-Ax-ULT) still get the physical layout."""
    counts: dict = {}
    for _, row in df.iterrows():
        m = _STUD_COUNT_RE.search(str(row.get("notes", "") or ""))
        if m:
            counts.setdefault(specimen_key(row), int(m.group(1)))
    return counts


def evaluate(df: pd.DataFrame, n_steps_section: int, n_steps_beam: int,
             skip_beam: bool) -> tuple[pd.DataFrame, dict, dict]:
    """Run both models on every specimen; return the per-specimen table
    plus caches of the raw curves for plotting."""
    sec_cache: dict = {}
    beam_cache: dict = {}
    stud_counts = stud_count_by_specimen(df)
    records = []

    for idx, row in df.iterrows():
        key = specimen_key(row)
        rec = {
            "test_id": row.test_id,
            "programme": row.source,
            "measured_moment_kip_in": row.measured_moment_kip_in,
            "measured_curvature_1_per_in": row.measured_curvature_1_per_in,
            "measured_slip_in": row.measured_slip_in,
        }
        flags = []
        note = str(row.get("notes", "") or "")
        if "continuous" in note.lower():
            flags.append("continuous_span_vs_simply_supported_model")

        # ---- section-level model -------------------------------------
        if key not in sec_cache:
            try:
                t0 = time.time()
                res = run_section_model(row, n_steps_section)
                sec_cache[key] = (res, time.time() - t0, None)
            except Exception as exc:
                traceback.print_exc()
                sec_cache[key] = (None, 0.0, f"{type(exc).__name__}: {exc}")
        sec_res, sec_dt, sec_err = sec_cache[key]

        if sec_res is not None and sec_res.converged_steps >= 3:
            m, phi = sec_res.moment, sec_res.curvature
            rec["section_peak_moment_kip_in"] = float(m.max())
            rec["section_converged_steps"] = int(sec_res.converged_steps)
            phi_pred, fl = interp_at_moment(row.measured_moment_kip_in, m, phi)
            if fl:
                flags.append(f"section_{fl}")
            rec["section_phi_at_M_1_per_in"] = phi_pred
            rec["section_phi_rel_err_pct"] = rel_err_pct(
                phi_pred, row.measured_curvature_1_per_in)
            m_pred, fl2 = interp_moment_at(
                row.measured_curvature_1_per_in, phi, m)
            if fl2 and np.isfinite(row.measured_curvature_1_per_in):
                flags.append(f"section_Mphi_{fl2}")
            rec["section_M_at_phi_kip_in"] = m_pred
            rec["section_M_rel_err_pct"] = rel_err_pct(
                m_pred, row.measured_moment_kip_in)
            rec["section_peak_over_measured"] = (
                float(m.max()) / row.measured_moment_kip_in)
        else:
            flags.append("section_model_failed")
            rec["section_error"] = sec_err or "too few converged steps"

        # ---- beam-level model (PRELIMINARY) ---------------------------
        rec["beam_model_status"] = BEAM_MODEL_STATUS
        if not skip_beam:
            if key not in beam_cache:
                try:
                    t0 = time.time()
                    bres, bp, layout_note = run_beam_model(
                        row, n_steps_beam, stud_counts.get(key))
                    beam_cache[key] = (bres, bp, layout_note,
                                       time.time() - t0, None)
                except Exception as exc:
                    traceback.print_exc()
                    beam_cache[key] = (None, None, "", 0.0,
                                       f"{type(exc).__name__}: {exc}")
            bres, bp, layout_note, b_dt, b_err = beam_cache[key]

            if bres is not None and bres.converged_steps >= 3:
                mb, phib = bres.moment, bres.curvature
                slipb = bres.max_slip_in  # peak over span ~ end slip
                rec["beam_stud_layout"] = layout_note
                rec["beam_converged_steps"] = int(bres.converged_steps)
                rec["beam_peak_moment_kip_in"] = float(mb.max())
                rec["beam_eta_c_emergent"] = float(bres.eta_c_emergent)
                phi_b, flb = interp_at_moment(
                    row.measured_moment_kip_in, mb, phib)
                if flb:
                    flags.append(f"beam_{flb}")
                rec["beam_phi_at_M_1_per_in"] = phi_b
                rec["beam_phi_rel_err_pct"] = rel_err_pct(
                    phi_b, row.measured_curvature_1_per_in)
                slip_b, fls = interp_at_moment(
                    row.measured_moment_kip_in, mb, slipb)
                if fls and np.isfinite(row.measured_slip_in):
                    flags.append(f"beam_slip_{fls}")
                rec["beam_slip_at_M_in"] = slip_b
                rec["beam_slip_rel_err_pct"] = rel_err_pct(
                    slip_b, row.measured_slip_in)
                if np.isfinite(slip_b) and slip_b > 1.0:
                    # >1 in of interface slip is physically impossible for
                    # these lab beams — symptom of the known beam-model bug.
                    flags.append("beam_slip_implausible")
                if bres.converged_steps < n_steps_beam:
                    flags.append("beam_early_termination")
            else:
                flags.append("beam_model_failed")
                rec["beam_error"] = b_err or "too few converged steps"

        rec["flags"] = ";".join(flags)
        records.append(rec)
        print(f"[{idx + 1:2d}/{len(df)}] {row.test_id:10s} "
              f"flags=[{rec['flags']}]", flush=True)

    return pd.DataFrame(records), sec_cache, beam_cache


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------

def _ape_stats(err_pct: pd.Series) -> dict:
    e = err_pct.dropna().abs()
    return {
        "n": int(len(e)),
        "mape_pct": float(e.mean()) if len(e) else float("nan"),
        "median_ape_pct": float(e.median()) if len(e) else float("nan"),
        "max_ape_pct": float(e.max()) if len(e) else float("nan"),
    }


def summarise(per: pd.DataFrame) -> pd.DataFrame:
    """MAPE / median APE by programme and by model/quantity, plus
    all-programme rows."""
    metrics = [
        ("section", "phi_at_M", "section_phi_rel_err_pct", "final"),
        ("section", "M_at_phi", "section_M_rel_err_pct", "final"),
        ("beam", "phi_at_M", "beam_phi_rel_err_pct", BEAM_MODEL_STATUS),
        ("beam", "slip_at_M", "beam_slip_rel_err_pct", BEAM_MODEL_STATUS),
    ]
    rows = []
    groups = [("ALL", per)] + [(g, s) for g, s in per.groupby("programme")]
    for gname, sub in groups:
        for model, qty, col, status in metrics:
            if col not in sub.columns:
                continue
            stats = _ape_stats(sub[col])
            rows.append({"programme": gname, "model": model,
                         "quantity": qty, "status": status, **stats})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_mphi_overlay(df: pd.DataFrame, sec_cache: dict, beam_cache: dict,
                     out_path: Path) -> None:
    """One panel per programme: section-model M-phi curves (solid, one
    colour per specimen), beam-model curves (thin dashed grey,
    preliminary), and measured points (filled markers, colour-matched to
    their specimen's section curve)."""
    plotting.apply_paper_style()
    programmes = list(dict.fromkeys(df.source))
    fig, axes = plt.subplots(1, len(programmes),
                             figsize=(plotting.COL_DOUBLE_IN, 2.9))
    axes = np.atleast_1d(axes)
    for ax, prog in zip(axes, programmes):
        sub = df[df.source == prog]
        colour_of: dict = {}
        plotted: set = set()
        x_data_max = 0.0  # section + measured extent; beam curves from the
        #                   buggy model can be orders too flexible and must
        #                   not dictate the axis range.
        for _, row in sub.iterrows():
            key = specimen_key(row)
            if key not in colour_of:
                label = (row.test_id.rsplit("-", 1)[0]
                         if prog.startswith("Chapman") else row.test_id)
                colour_of[key] = (len(colour_of), label)
            ci, label = colour_of[key]
            color = plotting.COLORS[ci % len(plotting.COLORS)]
            sec_res = sec_cache.get(key, (None,))[0]
            if sec_res is not None and sec_res.converged_steps >= 3:
                if (key, "s") not in plotted:  # plot each curve once
                    ax.plot(sec_res.curvature * 1e3, sec_res.moment / 12.0,
                            "-", color=color, lw=1.2, label=label, zorder=2)
                    plotted.add((key, "s"))
                    x_data_max = max(x_data_max,
                                     float(sec_res.curvature.max()) * 1e3)
            btuple = beam_cache.get(key)
            if btuple and btuple[0] is not None and (key, "b") not in plotted:
                bres = btuple[0]
                if bres.converged_steps >= 3:
                    ax.plot(bres.curvature * 1e3, bres.moment / 12.0, "--",
                            color="0.55", lw=0.8, zorder=1)
                plotted.add((key, "b"))
            if np.isfinite(row.measured_curvature_1_per_in):
                ax.plot(row.measured_curvature_1_per_in * 1e3,
                        row.measured_moment_kip_in / 12.0,
                        "o", mfc=color, mec="black", mew=0.6, ms=5, zorder=3)
                x_data_max = max(x_data_max,
                                 row.measured_curvature_1_per_in * 1e3)
        if x_data_max > 0:
            ax.set_xlim(0.0, 1.1 * x_data_max)
        ax.set_title(PROGRAMME_LABELS.get(prog, prog), fontsize=9)
        ax.set_xlabel(r"Curvature $\varphi \times 10^{3}$ (1/in)")
        if ax is axes[0]:
            ax.set_ylabel("Moment (kip-ft)")
        ax.legend(fontsize=6, loc="lower right", framealpha=0.9)
    handles = [plt.Line2D([], [], color="black", lw=1.2, ls="-"),
               plt.Line2D([], [], color="0.55", lw=0.8, ls="--"),
               plt.Line2D([], [], color="white", marker="o", mfc="0.5",
                          mec="black", ls="none", ms=5)]
    fig.legend(handles, ["section model (width-scaled $\\eta_c$)",
                         "beam model (discrete connectors)", "measured point"],
               loc="upper center", ncol=3, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    plotting.savefig(fig, out_path)
    print(f"[fig] {out_path}")


def fig_slip_parity(per: pd.DataFrame, out_path: Path) -> None:
    """Beam-model predicted vs measured interface slip (log-log parity).
    Clearly labelled preliminary pending the beam-model fix."""
    plotting.apply_paper_style()
    fig, ax = plt.subplots(figsize=(plotting.COL_SINGLE_IN,
                                    plotting.COL_SINGLE_IN))
    sub = per.dropna(subset=["measured_slip_in", "beam_slip_at_M_in"]) \
        if "beam_slip_at_M_in" in per.columns else per.iloc[0:0]
    markers = {"Chapman_Balakrishnan_1964": "o",
               "Nie_Cai_2003": "s", "Ansourian_1982": "^"}
    if len(sub):
        vals = np.concatenate([sub.measured_slip_in.to_numpy(),
                               sub.beam_slip_at_M_in.to_numpy()])
        lo = max(np.nanmin(vals) * 0.5, 1e-5)
        hi = np.nanmax(vals) * 2.0
    else:
        lo, hi = 1e-4, 1.0
    grid = np.array([lo, hi])
    ax.plot(grid, grid, "-", color="black", lw=0.8, label="1:1")
    ax.fill_between(grid, grid * 0.5, grid * 2.0, color="0.85", alpha=0.5,
                    label=r"$\pm$2x band", zorder=0)
    for i, (prog, s) in enumerate(sub.groupby("programme")):
        ax.plot(s.measured_slip_in, s.beam_slip_at_M_in,
                markers.get(prog, "o"), ls="none", ms=5,
                color=plotting.COLORS[i % len(plotting.COLORS)],
                mec="black", mew=0.5,
                label=PROGRAMME_LABELS.get(prog, prog))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Measured end slip (in)")
    ax.set_ylabel("Beam-model slip at measured M (in)")
    ax.set_title("Interface slip parity — beam model", fontsize=8)
    ax.legend(fontsize=6, loc="upper left")
    ax.set_aspect("equal")
    plotting.savefig(fig, out_path)
    print(f"[fig] {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", default=str(
        REPO / "data/experimental/literature_tests.csv"))
    ap.add_argument("--reports", default=str(REPO / "reports/model_validation"))
    ap.add_argument("--figs", default=str(REPO / "figures/validation"))
    ap.add_argument("--n-steps-section", type=int, default=160)
    ap.add_argument("--n-steps-beam", type=int, default=100)
    ap.add_argument("--skip-beam", action="store_true",
                    help="Section-level validation only (e.g. while the "
                         "beam-model bug fix is in flight).")
    args = ap.parse_args()

    reports = Path(args.reports)
    figs = Path(args.figs)
    reports.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    df = load_tests(Path(args.csv))
    print(f"[load] {len(df)} test points, "
          f"{df.source.nunique()} programmes — units kip/in/ksi")

    per, sec_cache, beam_cache = evaluate(
        df, args.n_steps_section, args.n_steps_beam, args.skip_beam)

    per_path = reports / "per_specimen.csv"
    per.to_csv(per_path, index=False)
    print(f"[out] {per_path}")

    summary = summarise(per)
    sum_path = reports / "summary.csv"
    summary.to_csv(sum_path, index=False)
    print(f"[out] {sum_path}")
    print("\n=== error summary (MAPE / median APE, % ) ===")
    print(summary.to_string(index=False))

    fig_mphi_overlay(df, sec_cache, beam_cache,
                     figs / "fig_exp_mphi_overlay.png")
    fig_slip_parity(per, figs / "fig_exp_slip_parity.png")

    # Qualitative verdict for the console log
    n_beyond = per["flags"].str.contains("section_beyond_model_peak").sum()
    print("\n=== notes ===")
    print(f"* {n_beyond}/{len(per)} points exceed the section model's peak "
          "moment (tests loaded to collapse; fibre model has no heavy "
          "strain hardening) -> phi(M) flagged, M(phi) reported instead.")
    print("* Beam-model columns regenerated with the fixed model "
          "('-noCentroid' fiber sections).")


if __name__ == "__main__":
    main()
