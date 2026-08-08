"""Interface-slip BRACKETING: force-based vs displacement-based connector chains.

This script compares the two connector-kinematics implementations in
this repository on the four simply supported Nie & Cai (2003) beams
(NC-SCB1..4), and persists both side by side.

An earlier revision claimed the two chains BRACKET the measured
interface slip, the force-based chain under-predicting and the
displacement-based chain over-predicting. That claim was WITHDRAWN:
both chains in fact under-predict the measured end slips, by 52 to 74
per cent and 42 to 69 per cent respectively. What the comparison now
supports is narrower and is what Section 4.5 reports: the kinematic
coupling is not the source of the spread, since tying rotation in
addition to vertical translation changes the predicted slip by under
0.1 per cent, and the residual difference between the chains is the
assumed connector backbone.

The two implementations
-----------------------
force-based        ``src.beam_model``
                   forceBeamColumn chains, ``equalDOF`` on DOF 2 only
                   (vertical translation shared, rotation independent),
                   original Ollgaard backbone, no slip-based stop,
                   sweep to span/50.
displacement-based ``paper2-beam-level/src/beam_model``
                   dispBeamColumn chains, ``equalDOF`` on DOFs 2 and 3
                   (vertical translation AND rotation shared), revised
                   Ollgaard backbone (peak at s = q/k + 0.10 in, mild
                   post-peak drop), ``stop_at_slip_in = 0.30``, sweep to
                   span/35.

Both carry the ``-noCentroid`` fiber-section fix; without it the
zeroLength connectors are inert and every slip number is meaningless.
The script asserts this before running anything.

Method
------
Each specimen is built ONCE (geometry, materials and synthesised stud
layout come from ``scripts/validate_models_experimental.derive_stud_layout``)
and the SAME ``BeamParams`` record is handed to both chains, so the only
difference between the two predictions is the connector-kinematics
implementation. Predicted slip is the max-over-span interface slip
interpolated on the RISING branch of the computed M-slip curve at the
measured moment -- exactly the convention (``interp_at_moment``) that
produced the ``beam_slip_at_M_in`` column of
``reports/model_validation/per_specimen.csv``. If the measured moment
exceeds a chain's peak moment the prediction is NaN and the specimen is
flagged ``beyond_model_peak``; it is never extrapolated or clipped.

Outputs
-------
reports/model_validation/connector_chains.csv
reports/model_validation/connector_chains_summary.json

Usage
-----
/opt/anaconda3/envs/ops_x86/bin/python scripts/validate_connector_chains.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Reuse the force-based chain and the driver machinery verbatim.
import src.beam_model as fb_model  # noqa: E402
from scripts.validate_models_experimental import (  # noqa: E402
    derive_stud_layout, interp_at_moment, rel_err_pct, specimen_key,
    stud_count_by_specimen)

FORCE_MODEL_PATH = REPO / "src/beam_model.py"
DISP_MODEL_PATH = REPO / "paper2-beam-level/src/beam_model.py"
DEFAULT_PROGRAMME = "Nie_Cai_2003"


def load_disp_model():
    """Import paper2-beam-level/src/beam_model.py under its own module
    name. It shares the dotted path ``src.beam_model`` with the
    force-based model, so a plain import would collide; it has no
    intra-package imports, so a file-path import is safe."""
    spec = importlib.util.spec_from_file_location(
        "paper2_beam_model", DISP_MODEL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["paper2_beam_model"] = mod
    spec.loader.exec_module(mod)
    return mod


def assert_no_centroid() -> dict:
    """Both fiber sections in BOTH chains must be declared with
    ``-noCentroid``; otherwise OpenSees re-references each section to its
    own centroid, the coincident deck/steel nodes stop measuring true
    interface slip, and every result below is invalid."""
    status = {}
    for name, path in (("force_based", FORCE_MODEL_PATH),
                       ("displacement_based", DISP_MODEL_PATH)):
        text = path.read_text()
        n = text.count('"-noCentroid"')
        status[name] = {"path": str(path), "noCentroid_occurrences": n}
        if n < 2:
            raise SystemExit(
                f"[abort] {path} declares '-noCentroid' on {n} fiber "
                "section(s), expected 2 (deck + steel). Without the fix "
                "the connectors are inert and slip results are invalid.")
    return status


def run_chain(analyze, params, n_steps: int, tag: str):
    """Run one chain on one specimen. Returns (result, seconds, error)."""
    t0 = time.time()
    try:
        res = analyze(params, n_steps=n_steps)
        return res, time.time() - t0, None
    except Exception as exc:  # pragma: no cover - convergence dependent
        traceback.print_exc()
        print(f"[error] {tag}: {type(exc).__name__}: {exc}", flush=True)
        return None, time.time() - t0, f"{type(exc).__name__}: {exc}"


def extract_slip(res, m_meas: float, n_steps: int, prefix: str):
    """Predicted slip at the measured moment + provenance columns."""
    out, flags = {}, []
    if res is None or res.converged_steps < 3:
        flags.append(f"{prefix}_model_failed")
        return {f"{prefix}_slip_at_M_in": float("nan")}, flags
    m, s = res.moment, res.max_slip_in
    out[f"{prefix}_converged_steps"] = int(res.converged_steps)
    out[f"{prefix}_peak_moment_kip_in"] = float(m.max())
    out[f"{prefix}_max_slip_reached_in"] = float(np.max(s))
    out[f"{prefix}_eta_c_emergent"] = float(res.eta_c_emergent)
    slip, fl = interp_at_moment(m_meas, m, s)
    if fl:
        flags.append(f"{prefix}_{fl}")
    if res.converged_steps < n_steps:
        flags.append(f"{prefix}_early_termination")
    out[f"{prefix}_slip_at_M_in"] = slip
    return out, flags


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", default=str(
        REPO / "data/experimental/literature_tests.csv"))
    ap.add_argument("--reports", default=str(REPO / "reports/model_validation"))
    ap.add_argument("--programme", default=DEFAULT_PROGRAMME)
    ap.add_argument("--n-steps-beam", type=int, default=100,
                    help="matches validate_models_experimental.py's default")
    args = ap.parse_args()

    centroid_status = assert_no_centroid()
    print("[check] '-noCentroid' present on both fiber sections in both "
          "chains", flush=True)

    db_model = load_disp_model()

    df = pd.read_csv(args.csv)
    sub = df[(df.source == args.programme)
             & df.measured_slip_in.notna()].copy()
    # Simply supported specimens only: both chains model a simply
    # supported beam, so the continuous CCB beams are out of scope.
    excluded = sub[sub.notes.astype(str).str.contains("continuous", case=False)]
    sub = sub.drop(excluded.index)
    print(f"[load] {len(sub)} {args.programme} specimens with measured slip: "
          f"{', '.join(sub.test_id)}", flush=True)
    if len(excluded):
        print(f"[skip] continuous-span specimens: "
              f"{', '.join(excluded.test_id)}", flush=True)

    stud_counts = stud_count_by_specimen(df)
    records = []
    for _, row in sub.iterrows():
        params, layout_note = derive_stud_layout(
            row, stud_counts.get(specimen_key(row)))
        # Same geometry AND same stud layout into both chains: the field
        # sets of the two BeamParams dataclasses are identical, so the
        # only difference is the connector-kinematics implementation.
        fb_params = params
        db_params = db_model.BeamParams(**asdict(params))
        assert asdict(fb_params) == asdict(db_params)

        rec = {
            "test_id": row.test_id,
            "programme": row.source,
            "measured_moment_kip_in": float(row.measured_moment_kip_in),
            "measured_slip_in": float(row.measured_slip_in),
            "stud_layout": layout_note,
            "stud_diameter_in": params.stud_diameter_in,
            "n_studs_per_row": params.n_studs_per_row,
            "stud_pitch_in": params.stud_pitch_in,
            "ks_ratio": params.ks_ratio,
        }
        flags = []

        fb_res, fb_dt, fb_err = run_chain(
            fb_model.analyze_beam, fb_params, args.n_steps_beam,
            f"{row.test_id} force-based")
        fb_cols, fb_flags = extract_slip(
            fb_res, rec["measured_moment_kip_in"], args.n_steps_beam, "force")
        rec.update(fb_cols)
        flags += fb_flags
        if fb_err:
            rec["force_error"] = fb_err

        db_res, db_dt, db_err = run_chain(
            db_model.analyze_beam, db_params, args.n_steps_beam,
            f"{row.test_id} displacement-based")
        db_cols, db_flags = extract_slip(
            db_res, rec["measured_moment_kip_in"], args.n_steps_beam, "disp")
        rec.update(db_cols)
        flags += db_flags
        if db_err:
            rec["disp_error"] = db_err

        rec["force_slip_err_pct"] = rel_err_pct(
            rec.get("force_slip_at_M_in", float("nan")),
            rec["measured_slip_in"])
        rec["disp_slip_err_pct"] = rel_err_pct(
            rec.get("disp_slip_at_M_in", float("nan")),
            rec["measured_slip_in"])
        brackets = (np.isfinite(rec["force_slip_err_pct"])
                    and np.isfinite(rec["disp_slip_err_pct"])
                    and rec["force_slip_err_pct"] < 0.0
                    and rec["disp_slip_err_pct"] > 0.0)
        rec["brackets_measured"] = bool(brackets)
        if not brackets:
            flags.append("no_bracket")
        rec["force_runtime_s"] = round(fb_dt, 2)
        rec["disp_runtime_s"] = round(db_dt, 2)
        rec["flags"] = ";".join(flags)
        records.append(rec)
        print(f"[{row.test_id}] measured {rec['measured_slip_in']:.4f} in | "
              f"force {rec.get('force_slip_at_M_in', float('nan')):.5f} in "
              f"({rec['force_slip_err_pct']:+.1f}%) | disp "
              f"{rec.get('disp_slip_at_M_in', float('nan')):.5f} in "
              f"({rec['disp_slip_err_pct']:+.1f}%) flags=[{rec['flags']}]",
          flush=True)

    per = pd.DataFrame(records)
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    csv_path = reports / "connector_chains.csv"
    per.to_csv(csv_path, index=False)
    print(f"[out] {csv_path}")

    def span(series: pd.Series) -> dict:
        v = series.dropna()
        return {"n": int(len(v)),
                "min": float(v.min()) if len(v) else None,
                "max": float(v.max()) if len(v) else None}

    fe, de = per.force_slip_err_pct.dropna(), per.disp_slip_err_pct.dropna()
    preds = pd.concat([per.force_slip_at_M_in, per.disp_slip_at_M_in]).dropna()
    summary = {
        "generated_by": "scripts/validate_connector_chains.py",
        "programme": args.programme,
        "n_steps_beam": args.n_steps_beam,
        "specimens": list(per.test_id),
        "no_centroid_check": centroid_status,
        "implementations": {
            "force_based": {
                "module": "src/beam_model.py",
                "element": "forceBeamColumn",
                "equalDOF": [2],
            },
            "displacement_based": {
                "module": "paper2-beam-level/src/beam_model.py",
                "element": "dispBeamColumn",
                "equalDOF": [2, 3],
                "stop_at_slip_in": 0.30,
            },
        },
        "measured_slip_in": span(per.measured_slip_in),
        "force_based_err_pct": span(per.force_slip_err_pct),
        "displacement_based_err_pct": span(per.disp_slip_err_pct),
        "predicted_slip_in_both_chains": span(preds),
        # The two bracketing ranges, in the sign convention the
        # manuscript uses (under-prediction negative, over positive).
        "under_prediction_range_pct": (
            [float(fe.max()), float(fe.min())] if len(fe) else None),
        "over_prediction_range_pct": (
            [float(de.min()), float(de.max())] if len(de) else None),
        "brackets_all_specimens": bool(per.brackets_measured.all()),
        "n_bracketed": int(per.brackets_measured.sum()),
        # Constants for scripts/figs/make_fig_experimental_validation.py
        # (read these instead of hard-coding OVER_LO / OVER_HI).
        "OVER_LO": float(1.0 + de.min() / 100.0) if len(de) else None,
        "OVER_HI": float(1.0 + de.max() / 100.0) if len(de) else None,
        "UNDER_LO": float(1.0 + fe.max() / 100.0) if len(fe) else None,
        "UNDER_HI": float(1.0 + fe.min() / 100.0) if len(fe) else None,
    }
    json_path = reports / "connector_chains_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[out] {json_path}")

    print("\n=== bracketing summary ===")
    print(per[["test_id", "measured_slip_in", "force_slip_at_M_in",
               "force_slip_err_pct", "disp_slip_at_M_in",
               "disp_slip_err_pct", "brackets_measured", "flags"]]
          .to_string(index=False))
    print(f"\nforce-based (under):        {summary['under_prediction_range_pct']} %")
    print(f"displacement-based (over):  {summary['over_prediction_range_pct']} %")
    print(f"predicted slip span:        "
          f"{summary['predicted_slip_in_both_chains']} in")


if __name__ == "__main__":
    main()
