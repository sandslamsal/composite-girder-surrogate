"""Deep-ensemble inference wrapper.

A 5-member ensemble is the recommended drop-in replacement for MC-Dropout
when honest epistemic uncertainty matters
(MC-Dropout systematically under-estimates uncertainty;
deep ensembles are well-calibrated).

Each member is an independently-initialised :class:`CompositeGirderPINN`
trained from a different random seed but on the same data split.  At
inference we average the deterministic forward passes from each member
to get the prediction, and use the per-sample standard deviation across
members as the epistemic uncertainty band.

Members are produced by running ``scripts/train_pinn.py`` once per
seed; ``scripts/train_ensemble.py`` orchestrates the loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from src.models.inference import PINNPredictor
from src.utils.normalize import TARGET_COLUMNS


class EnsemblePredictor:
    """Aggregate predictions across a small set of independently-trained
    PINN members. Same public API as :class:`PINNPredictor`."""

    def __init__(self, members: Iterable[PINNPredictor]) -> None:
        self.members = list(members)
        if not self.members:
            raise ValueError("EnsemblePredictor needs at least one member")

    @classmethod
    def from_directory(
        cls, ensemble_dir: str | Path,
        checkpoint_name: str = "best.pt",
        device: torch.device | None = None,
    ) -> "EnsemblePredictor":
        """Load every ``member_*/best.pt`` under ``ensemble_dir``."""
        ensemble_dir = Path(ensemble_dir)
        member_dirs = sorted(
            d for d in ensemble_dir.iterdir()
            if d.is_dir() and d.name.startswith("member_")
        )
        if not member_dirs:
            raise FileNotFoundError(
                f"no member_* subdirs found under {ensemble_dir}"
            )
        predictors = [
            PINNPredictor.load(d / checkpoint_name, device=device)
            for d in member_dirs
        ]
        return cls(predictors)

    @property
    def n_members(self) -> int:
        return len(self.members)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mean prediction across members (point estimate)."""
        per_member = [m.predict(df).to_numpy() for m in self.members]
        stacked = np.stack(per_member, axis=0)
        mean = stacked.mean(axis=0)
        return pd.DataFrame(mean, columns=TARGET_COLUMNS, index=df.index)

    def predict_with_uncertainty(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mean and standard deviation across members per target.

        Output columns are
        ``{target}_mean`` and ``{target}_std`` (matching the schema of
        :meth:`PINNPredictor.predict_with_uncertainty`)."""
        per_member = [m.predict(df).to_numpy() for m in self.members]
        stacked = np.stack(per_member, axis=0)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0, ddof=1) if len(self.members) > 1 else np.zeros_like(mean)
        out = {}
        for j, name in enumerate(TARGET_COLUMNS):
            out[f"{name}_mean"] = mean[:, j]
            out[f"{name}_std"] = std[:, j]
        return pd.DataFrame(out, index=df.index)
