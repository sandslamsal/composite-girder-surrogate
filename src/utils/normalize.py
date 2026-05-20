"""Feature and target normalisation for the composite-girder surrogate.

Physical features use fixed ranges from configs/data_gen.yaml so the network sees
inputs on roughly the same scale across train/val/test/inference. Targets use
min-max from the training split. Section type is one-hot encoded.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import yaml


# Ordered input features the surrogate consumes (post-encoding). Total
# input dimension = 13 continuous + 2 one-hot = 15.
#  - 12 continuous design features matching proposal Table 2 (+ a couple of
#    composite-related extras the model benefits from)
#  - 1 load-step feature (moment_ratio)
#  - 2 one-hot section-type indicators (W, plate)
#
# Note: the shear-stud stiffness ratio K_s/K_max is NOT an NN input. It is
# sampled in the LHS design and stored in the dataset (used by the Nie & Cai
# cross-validation), but it has zero influence on the OpenSeesPy fiber-section
# labels, which represent partial composite action solely through the
# effective deck-width scaling by eta_c. Feeding it would only add a ghost
# feature the network learns to ignore. See paper Section 3 and the Table 1
# footnote.
FEATURE_COLUMNS: List[str] = [
    "span_in",
    "deck_thickness_in",
    "deck_width_in",
    "girder_spacing_in",
    "fc_deck_ksi",
    "composite_action",
    "fy_ksi",
    "steel_depth_in",
    "flange_width_in",
    "flange_thickness_in",
    "web_thickness_in",
    "total_depth_in",
    "moment_ratio",
]

# concrete_I is deferred to v2 (see configs/data_gen.yaml) and never appears
# in the dataset, so it is excluded from the one-hot encoding rather than
# carried as an always-zero column.
SECTION_TYPES: List[str] = ["W", "plate"]

TARGET_COLUMNS: List[str] = [
    "neutral_axis_in",
    "curvature_1_per_in",
]


def _load_data_gen_cfg(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _physical_ranges(cfg: dict) -> Dict[str, tuple]:
    """Map feature column -> (min, max) for the physical features we know
    a-priori from the LHS sampling ranges. Anything missing falls back to
    min-max from the training data.
    """
    steel = cfg.get("steel", {})
    # girder_spacing in cfg is in ft but data is in inches -> convert
    spacing_min_in = cfg["girder_spacing_ft"]["min"] * 12.0
    spacing_max_in = cfg["girder_spacing_ft"]["max"] * 12.0
    flange_w_min = steel["flange_width_ratio"]["min"] * steel["depth_in"]["min"]
    flange_w_max = steel["flange_width_ratio"]["max"] * steel["depth_in"]["max"]
    flange_t_min = steel["flange_thickness_ratio"]["min"] * flange_w_min
    flange_t_max = steel["flange_thickness_ratio"]["max"] * flange_w_max
    web_t_min = steel["web_thickness_ratio"]["min"] * steel["depth_in"]["min"]
    web_t_max = steel["web_thickness_ratio"]["max"] * steel["depth_in"]["max"]
    deck_w_min = cfg["deck_width_ratio"]["min"] * spacing_min_in
    deck_w_max = cfg["deck_width_ratio"]["max"] * spacing_max_in

    return {
        "span_in": (cfg["span_ft"]["min"] * 12.0, cfg["span_ft"]["max"] * 12.0),
        "deck_thickness_in": (cfg["deck_thickness_in"]["min"], cfg["deck_thickness_in"]["max"]),
        "deck_width_in": (deck_w_min, deck_w_max),
        "girder_spacing_in": (spacing_min_in, spacing_max_in),
        "fc_deck_ksi": (cfg["fc_ksi"]["min"], cfg["fc_ksi"]["max"]),
        "composite_action": (cfg["composite_action"]["min"], cfg["composite_action"]["max"]),
        # shear_stud_stiffness_ratio is intentionally absent: it is sampled in
        # the LHS and stored in the dataset, but is not an NN input feature.
        "fy_ksi": (steel["fy_ksi"]["min"], steel["fy_ksi"]["max"]),
        "steel_depth_in": (steel["depth_in"]["min"], steel["depth_in"]["max"]),
        "flange_width_in": (flange_w_min, flange_w_max),
        "flange_thickness_in": (flange_t_min, flange_t_max),
        "web_thickness_in": (web_t_min, web_t_max),
        # total_depth_in ~ steel_depth + deck_thickness (upper-bound)
        "total_depth_in": (
            steel["depth_in"]["min"] + cfg["deck_thickness_in"]["min"],
            steel["depth_in"]["max"] + cfg["deck_thickness_in"]["max"],
        ),
        "moment_ratio": (0.0, cfg["analysis"]["moment_ratio_max"]),
    }


class FeatureNormalizer:
    """Min-max normalises features (physical ranges where possible, data-driven
    else) and targets (data-driven). Section type expanded to a 2-way one-hot.

    State is serialisable via ``state_dict``/``load_state_dict`` so a fitted
    normaliser can be saved alongside a checkpoint.
    """

    def __init__(self, data_gen_yaml: str | Path | None = None):
        self.feature_columns: List[str] = list(FEATURE_COLUMNS)
        self.section_types: List[str] = list(SECTION_TYPES)
        self.target_columns: List[str] = list(TARGET_COLUMNS)
        self.feature_min: Dict[str, float] = {}
        self.feature_max: Dict[str, float] = {}
        self.target_min: Dict[str, float] = {}
        self.target_max: Dict[str, float] = {}
        self._fitted = False
        if data_gen_yaml is not None:
            self._physical = _physical_ranges(_load_data_gen_cfg(data_gen_yaml))
        else:
            self._physical = {}

    # ---- fit / transform -------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "FeatureNormalizer":
        for col in self.feature_columns:
            if col in self._physical:
                lo, hi = self._physical[col]
            else:
                lo = float(df[col].min())
                hi = float(df[col].max())
            # guard zero-range
            if hi - lo < 1e-12:
                hi = lo + 1.0
            self.feature_min[col] = float(lo)
            self.feature_max[col] = float(hi)

        for col in self.target_columns:
            if col == "moment_kip_in":
                # per-row scaled by mp_estimate; store the normalised
                # range so downstream code can still query target_scale().
                self.target_min[col] = 0.0
                self.target_max[col] = 1.5
                continue
            lo = float(df[col].min())
            hi = float(df[col].max())
            if hi - lo < 1e-12:
                hi = lo + 1.0
            self.target_min[col] = lo
            self.target_max[col] = hi
        self._fitted = True
        return self

    def _ensure_fit(self):
        if not self._fitted:
            raise RuntimeError("FeatureNormalizer.fit must be called first")

    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        """Return an (N, D) float32 array where D = len(FEATURE_COLUMNS) + len(SECTION_TYPES)."""
        self._ensure_fit()
        cols = []
        for c in self.feature_columns:
            lo, hi = self.feature_min[c], self.feature_max[c]
            cols.append(((df[c].to_numpy() - lo) / (hi - lo)).astype(np.float32))
        # one-hot section_type
        st = df["section_type"].to_numpy()
        for s in self.section_types:
            cols.append((st == s).astype(np.float32))
        return np.stack(cols, axis=1)

    def transform_targets(self, df: pd.DataFrame) -> np.ndarray:
        """Targets in normalised space.

        ``moment_kip_in`` uses *per-row* scaling by ``mp_estimate_kip_in`` so
        each section's moment target lives in roughly ``[0, 1.2]`` rather
        than spanning ~3 orders of magnitude across small and large
        sections. Without this, MSE loss is dominated by the largest
        sections and the model never learns moment for small sections.
        All other targets use train-set min-max.
        """
        self._ensure_fit()
        cols = []
        for c in self.target_columns:
            if c == "moment_kip_in":
                mp = df["mp_estimate_kip_in"].to_numpy().astype(np.float32)
                # guard tiny mp_est; happens in degenerate LHS draws
                mp_safe = np.where(mp > 1.0, mp, 1.0)
                cols.append((df[c].to_numpy().astype(np.float32) / mp_safe))
            else:
                lo, hi = self.target_min[c], self.target_max[c]
                cols.append(((df[c].to_numpy() - lo) / (hi - lo)).astype(np.float32))
        return np.stack(cols, axis=1)

    def inverse_transform_targets(
        self,
        arr: np.ndarray | torch.Tensor,
        mp_estimate_kip_in: np.ndarray | torch.Tensor | None = None,
    ) -> np.ndarray:
        """Map normalised targets back to physical units.

        ``mp_estimate_kip_in`` must be supplied (per row) when inverting
        the moment column, since that target uses per-row scaling. The
        other targets use the stored min-max.
        """
        self._ensure_fit()
        is_tensor = isinstance(arr, torch.Tensor)
        if is_tensor:
            arr_np = arr.detach().cpu().numpy()
        else:
            arr_np = np.asarray(arr)

        if mp_estimate_kip_in is None:
            mp = None
        elif isinstance(mp_estimate_kip_in, torch.Tensor):
            mp = mp_estimate_kip_in.detach().cpu().numpy()
        else:
            mp = np.asarray(mp_estimate_kip_in)

        out = np.empty_like(arr_np, dtype=np.float64)
        for j, c in enumerate(self.target_columns):
            if c == "moment_kip_in":
                if mp is None:
                    raise ValueError(
                        "inverse_transform_targets needs mp_estimate_kip_in "
                        "to invert the moment column"
                    )
                out[:, j] = arr_np[:, j] * mp
            else:
                lo, hi = self.target_min[c], self.target_max[c]
                out[:, j] = arr_np[:, j] * (hi - lo) + lo
        return out

    def target_scale(self) -> np.ndarray:
        """(hi - lo) per target, in physical units. Useful for the equilibrium
        proxy loss which lives in physical-moment space.
        """
        self._ensure_fit()
        return np.array(
            [self.target_max[c] - self.target_min[c] for c in self.target_columns],
            dtype=np.float32,
        )

    def target_offset(self) -> np.ndarray:
        self._ensure_fit()
        return np.array(
            [self.target_min[c] for c in self.target_columns], dtype=np.float32
        )

    # ---- I/O -------------------------------------------------------------
    @property
    def input_dim(self) -> int:
        return len(self.feature_columns) + len(self.section_types)

    @property
    def output_dim(self) -> int:
        return len(self.target_columns)

    def state_dict(self) -> dict:
        return {
            "feature_columns": self.feature_columns,
            "section_types": self.section_types,
            "target_columns": self.target_columns,
            "feature_min": self.feature_min,
            "feature_max": self.feature_max,
            "target_min": self.target_min,
            "target_max": self.target_max,
            "physical": self._physical,
            "_fitted": self._fitted,
        }

    def load_state_dict(self, state: dict) -> None:
        self.feature_columns = list(state["feature_columns"])
        self.section_types = list(state["section_types"])
        self.target_columns = list(state["target_columns"])
        self.feature_min = dict(state["feature_min"])
        self.feature_max = dict(state["feature_max"])
        self.target_min = dict(state["target_min"])
        self.target_max = dict(state["target_max"])
        self._physical = dict(state.get("physical", {}))
        self._fitted = bool(state.get("_fitted", True))
