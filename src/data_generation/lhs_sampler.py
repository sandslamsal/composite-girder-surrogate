"""Latin Hypercube Sampling over the composite-girder design space.

Produces a list of :class:`SectionParams` records that the OpenSeesPy
section builder consumes. The LHS draws a single unit hypercube of width
equal to the number of continuous design dimensions plus one extra
dimension that is quantised to pick the categorical section type, so the
stratification property holds across both continuous and discrete axes.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml
from scipy.stats import qmc


# ---------------------------------------------------------------------------
# Continuous LHS columns. Order is fixed so reseeding the sampler reproduces
# the same sample set.
# ---------------------------------------------------------------------------
_CONTINUOUS_KEYS: tuple[str, ...] = (
    "span_ft",
    "deck_thickness_in",
    "deck_width_ratio",
    "girder_spacing_ft",
    "fc_ksi",
    "composite_action",
    "shear_stud_stiffness_ratio",
    "steel.fy_ksi",
    "steel.depth_in",
    "steel.flange_width_ratio",
    "steel.flange_thickness_ratio",
    "steel.web_thickness_ratio",
    "concrete_i.depth_in",
    "concrete_i.fc_beam_ksi",
    "concrete_i.prestress_ratio",
)

# The categorical column is appended last; it is quantised onto the section
# types listed in the config.
_N_DIMS = len(_CONTINUOUS_KEYS) + 1


@dataclass
class SectionParams:
    """One sampled composite girder section.

    Only the fields relevant to the chosen :attr:`section_type` are
    physically meaningful; the others are still carried so that downstream
    code can record the full LHS row for reproducibility.
    """

    # bookkeeping
    sample_id: int
    section_type: str  # "W" | "plate" | "concrete_I"

    # shared
    span_in: float
    deck_thickness_in: float
    deck_width_in: float          # b_eff (already scaled by girder spacing)
    girder_spacing_in: float      # S
    fc_deck_ksi: float
    composite_action: float       # eta_c in [0.25, 1.0]
    shear_stud_stiffness_ratio: float  # K_s / K_max

    # steel I-section geometry (used when section_type in {W, plate})
    fy_ksi: float
    steel_depth_in: float
    flange_width_in: float
    flange_thickness_in: float
    web_thickness_in: float

    # precast concrete-I geometry (used when section_type == "concrete_I")
    pcb_depth_in: float
    pcb_fc_ksi: float
    pcb_prestress_ratio: float

    def as_record(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def sample(config: dict, n: int, seed: int | None = None) -> list[SectionParams]:
    """Draw ``n`` Latin Hypercube samples from the config-defined space.

    Parameters
    ----------
    config
        Parsed contents of ``configs/data_gen.yaml``.
    n
        Number of sections to draw.
    seed
        Optional override for ``config['random_seed']``.
    """
    if seed is None:
        seed = config.get("random_seed", 0)

    rng = np.random.default_rng(seed)
    sampler = qmc.LatinHypercube(d=_N_DIMS, seed=rng)
    u = sampler.random(n)  # shape (n, _N_DIMS), values in (0, 1)

    bounds = _flatten_bounds(config)
    cont = np.zeros((n, len(_CONTINUOUS_KEYS)))
    for j, key in enumerate(_CONTINUOUS_KEYS):
        lo, hi = bounds[key]
        cont[:, j] = lo + (hi - lo) * u[:, j]

    # Categorical: quantise the last column to section_types index.
    section_types = config["section_types"]
    section_idx = np.floor(u[:, -1] * len(section_types)).astype(int)
    section_idx = np.clip(section_idx, 0, len(section_types) - 1)

    samples: list[SectionParams] = []
    for i in range(n):
        row = dict(zip(_CONTINUOUS_KEYS, cont[i]))
        spacing_in = row["girder_spacing_ft"] * 12.0
        b_eff_in = row["deck_width_ratio"] * spacing_in
        steel_depth = row["steel.depth_in"]
        b_f = row["steel.flange_width_ratio"] * steel_depth
        t_f = row["steel.flange_thickness_ratio"] * b_f
        t_w = row["steel.web_thickness_ratio"] * steel_depth

        samples.append(
            SectionParams(
                sample_id=i,
                section_type=section_types[section_idx[i]],
                span_in=row["span_ft"] * 12.0,
                deck_thickness_in=row["deck_thickness_in"],
                deck_width_in=b_eff_in,
                girder_spacing_in=spacing_in,
                fc_deck_ksi=row["fc_ksi"],
                composite_action=row["composite_action"],
                shear_stud_stiffness_ratio=row["shear_stud_stiffness_ratio"],
                fy_ksi=row["steel.fy_ksi"],
                steel_depth_in=steel_depth,
                flange_width_in=b_f,
                flange_thickness_in=t_f,
                web_thickness_in=t_w,
                pcb_depth_in=row["concrete_i.depth_in"],
                pcb_fc_ksi=row["concrete_i.fc_beam_ksi"],
                pcb_prestress_ratio=row["concrete_i.prestress_ratio"],
            )
        )

    return samples


def to_dataframe(samples: Iterable[SectionParams]):
    """Convenience helper; lazy import keeps pandas off the hot import path."""
    import pandas as pd

    return pd.DataFrame([s.as_record() for s in samples])


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------

def _flatten_bounds(config: dict) -> dict[str, tuple[float, float]]:
    """Walk the config and emit a flat ``key -> (lo, hi)`` map for the LHS
    columns. Nested keys use a dotted path (e.g. ``steel.fy_ksi``)."""
    out: dict[str, tuple[float, float]] = {}
    for key in _CONTINUOUS_KEYS:
        node: dict = config
        for part in key.split("."):
            node = node[part]
        out[key] = (float(node["min"]), float(node["max"]))
    return out
