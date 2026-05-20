#!/usr/bin/env python
"""Permutation feature importance for the composite-girder surrogate.

For each input feature the feature column is randomly shuffled in the
held-out test set and the resulting drop in R^2 is recorded. A feature
the model genuinely uses produces a large R^2 drop when shuffled; an
uninformative feature produces a drop near zero.

By default this runs on the deployed 15-feature surrogate and reports a
model-interpretability figure: which design variables drive the
predicted curvature. Features whose importance is below ``--min-show``
(span, girder spacing, and the section-type indicator -- all
near-zero, as expected for a purely sectional quantity) are listed in
the JSON output but omitted from the bar chart for clarity.

Usage:
    python scripts/feature_importance.py \
        --checkpoint weights/best.pt \
        --data data/raw/full_50k.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.surrogate import CompositeGirderSurrogate
from src.utils.normalize import FeatureNormalizer, TARGET_COLUMNS
from src.utils.plotting import apply_paper_style, savefig, COLORS, COL_DOUBLE_IN
from scripts.train_surrogate import _split_by_sample, _resolve_path, _r2


# Display labels for the feature columns.
_LABELS = {
    "span_in": r"$L$  (span)",
    "deck_thickness_in": r"$t_s$  (deck thickness)",
    "deck_width_in": r"$b_{\mathrm{eff}}$  (deck width)",
    "girder_spacing_in": r"$S$  (girder spacing)",
    "fc_deck_ksi": r"$f_c'$  (deck strength)",
    "composite_action": r"$\eta_c$  (composite action)",
    "shear_stud_stiffness_ratio": r"$K_s$  (stud stiffness)",
    "fy_ksi": r"$f_y$  (steel yield)",
    "steel_depth_in": r"$d_s$  (steel depth)",
    "flange_width_in": r"$b_f$  (flange width)",
    "flange_thickness_in": r"$t_f$  (flange thickness)",
    "web_thickness_in": r"$t_w$  (web thickness)",
    "total_depth_in": r"$d_{\mathrm{total}}$  (total depth)",
    "moment_ratio": r"$M/M_p$  (moment ratio)",
}

# Physical grouping of the features, used to colour-code the bar chart.
_CATEGORY = {
    "moment_ratio": "Applied load",
    "web_thickness_in": "Steel section",
    "fy_ksi": "Steel section",
    "steel_depth_in": "Steel section",
    "flange_thickness_in": "Steel section",
    "flange_width_in": "Steel section",
    "deck_width_in": "Concrete deck",
    "fc_deck_ksi": "Concrete deck",
    "deck_thickness_in": "Concrete deck",
    "composite_action": "Composite section",
    "total_depth_in": "Composite section",
    "span_in": "Global geometry",
    "girder_spacing_in": "Global geometry",
}

# Legend order and a muted, journal-friendly palette.
_CAT_ORDER = ["Applied load", "Steel section", "Concrete deck",
              "Composite section", "Global geometry"]
_CAT_COLOR = {
    "Applied load":      "#E08A3C",   # amber
    "Steel section":     "#3B6B9A",   # steel blue
    "Concrete deck":     "#4F9D69",   # green
    "Composite section": "#8E6BAE",   # purple
    "Global geometry":   "#9AA0A6",   # grey
    "Section type":      "#9AA0A6",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="weights/best.pt",
                   help="Surrogate checkpoint (default: deployed 15-feature model).")
    p.add_argument("--data", required=True)
    p.add_argument("--config", default="configs/training.yaml")
    p.add_argument("--out", default="paper/figures/fig_feature_importance.png")
    p.add_argument("--repeats", type=int, default=5,
                   help="Permutation repeats per feature (averaged).")
    p.add_argument("--target", default="curvature_1_per_in",
                   choices=TARGET_COLUMNS)
    p.add_argument("--min-show", type=float, default=0.02,
                   help="Features with mean R^2 drop below this are kept "
                        "in the JSON but omitted from the bar chart.")
    args = p.parse_args()

    cfg = yaml.safe_load(open(_resolve_path(args.config)))
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    df = pd.read_parquet(_resolve_path(args.data))
    _, _, test_df = _split_by_sample(df, cfg["splits"], seed)
    print(f"[data] test rows={len(test_df)} samples={test_df['sample_id'].nunique()}")

    # The normaliser state is read from the checkpoint so the probed
    # feature set matches the model (15- or 17-feature).
    ckpt = torch.load(_resolve_path(args.checkpoint), map_location=device,
                      weights_only=False)
    normalizer = FeatureNormalizer()
    normalizer.load_state_dict(ckpt["normalizer_state"])
    feat_cols = list(normalizer.feature_columns)
    sect_types = list(normalizer.section_types)

    model = CompositeGirderSurrogate(
        input_dim=normalizer.input_dim,
        output_dim=normalizer.output_dim,
        width=cfg["model"]["width"],
        n_blocks=cfg["model"]["n_blocks"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    has_ks = "shear_stud_stiffness_ratio" in feat_cols
    print(f"[model] {normalizer.input_dim}-feature checkpoint "
          f"({len(feat_cols)} continuous + {len(sect_types)} one-hot); "
          f"K_s {'present' if has_ks else 'absent'} in input")

    X = normalizer.transform_features(test_df).astype(np.float32)
    # R^2 is invariant under the affine min-max target transform, so the
    # importance ranking is identical in normalised or physical space.
    Y = normalizer.transform_targets(test_df).astype(np.float32)
    tgt_idx = TARGET_COLUMNS.index(args.target)

    def predict(x_in: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model(torch.from_numpy(x_in).to(device)).cpu().numpy()

    base_r2 = _r2(Y[:, tgt_idx], predict(X)[:, tgt_idx])
    print(f"[baseline] {args.target} R2 = {base_r2:.4f}")

    rng = np.random.default_rng(seed)
    n_cont = len(feat_cols)
    results: dict[str, tuple[float, float]] = {}

    # continuous features -- one column each
    for j, col in enumerate(feat_cols):
        drops = []
        for _ in range(args.repeats):
            xp = X.copy()
            xp[:, j] = X[rng.permutation(len(X)), j]
            drops.append(base_r2 - _r2(Y[:, tgt_idx], predict(xp)[:, tgt_idx]))
        results[_LABELS.get(col, col)] = (float(np.mean(drops)),
                                          float(np.std(drops)))

    # section-type one-hot block -- permuted jointly
    if sect_types:
        drops = []
        block = slice(n_cont, n_cont + len(sect_types))
        for _ in range(args.repeats):
            xp = X.copy()
            xp[:, block] = X[rng.permutation(len(X)), block]
            drops.append(base_r2 - _r2(Y[:, tgt_idx], predict(xp)[:, tgt_idx]))
        results["section type (one-hot)"] = (float(np.mean(drops)),
                                             float(np.std(drops)))

    order = sorted(results, key=lambda k: results[k][0], reverse=True)
    print(f"\n[permutation importance -- {args.target} R2 drop]")
    for k in order:
        m, s = results[k]
        flag = "" if m >= args.min_show else "  (omitted from chart)"
        print(f"  {k:30s}  {m:+.4f} +/- {s:.4f}{flag}")

    # ---- bar chart (paper style, colour-coded by feature group) ----
    apply_paper_style()
    label_cat = {_LABELS[c]: _CATEGORY[c] for c in _CATEGORY if c in _LABELS}
    label_cat["section type (one-hot)"] = "Section type"

    shown = [k for k in order if results[k][0] >= args.min_show]
    omitted = [k for k in order if results[k][0] < args.min_show]
    means = [results[k][0] for k in shown]
    errs = [results[k][1] for k in shown]
    bar_colors = [_CAT_COLOR.get(label_cat.get(k, ""), COLORS[0]) for k in shown]
    xmax = max(means) * 1.24

    fig, ax = plt.subplots(figsize=(COL_DOUBLE_IN, 0.47 * len(shown) + 1.7))
    yp = np.arange(len(shown))[::-1]
    # faint vertical grid behind the bars
    ax.grid(axis="x", color="0.85", linewidth=0.6, zorder=0)
    ax.grid(axis="y", visible=False)
    ax.barh(yp, means, xerr=errs, height=0.70, color=bar_colors,
            edgecolor="white", linewidth=0.9, zorder=3,
            error_kw={"linewidth": 0.9, "ecolor": "0.30", "capsize": 2.5,
                      "zorder": 4})
    ax.set_yticks(yp)
    ax.set_yticklabels(shown)
    ax.set_ylim(-0.7, len(shown) - 0.3)
    ax.set_xlabel(r"Permutation importance: drop in curvature $R^2$ "
                  r"when the feature is shuffled")
    ax.set_xlim(0, xmax)
    ax.tick_params(axis="y", length=0)
    # value labels at the bar ends (clip_on off so none are hidden)
    for y, m in zip(yp, means):
        ax.text(m + xmax * 0.013, y, f"{m:.2f}", va="center", ha="left",
                fontsize=8.5, color="0.2", zorder=5, clip_on=False)
    # feature-group legend, lower-right (clear of the short bottom bars)
    cats_present = [c for c in _CAT_ORDER
                    if c in {label_cat.get(k) for k in shown}]
    handles = [plt.Rectangle((0, 0), 1, 1, fc=_CAT_COLOR[c], ec="white",
                             lw=0.8) for c in cats_present]
    leg = ax.legend(handles, cats_present, loc="lower right", frameon=True,
                    framealpha=0.96, edgecolor="0.8", title="Feature group",
                    fontsize=8.5, title_fontsize=9, borderpad=0.7,
                    handlelength=1.3, handleheight=1.1)
    fig.tight_layout()

    # baseline-R^2 note placed directly above the legend box
    fig.canvas.draw()
    lb = leg.get_window_extent().transformed(ax.transAxes.inverted())
    ax.text(lb.x1, lb.y1 + 0.025,
            f"baseline curvature $R^2 = {base_r2:.3f}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            color="0.35", clip_on=False,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.85",
                      lw=0.6))

    out = _resolve_path(args.out)
    savefig(fig, out)
    print(f"[done] {out}")
    if omitted:
        print(f"[note] omitted from chart (importance < {args.min_show}): "
              + ", ".join(omitted))

    json_out = out.with_suffix(".json")
    with open(json_out, "w") as f:
        json.dump({"checkpoint": str(args.checkpoint),
                   "target": args.target,
                   "baseline_r2": base_r2,
                   "repeats": args.repeats,
                   "min_show": args.min_show,
                   "importance": {k: {"mean_r2_drop": results[k][0],
                                      "std": results[k][1],
                                      "shown": results[k][0] >= args.min_show}
                                  for k in order}}, f, indent=2)
    print(f"[done] {json_out}")


if __name__ == "__main__":
    main()
